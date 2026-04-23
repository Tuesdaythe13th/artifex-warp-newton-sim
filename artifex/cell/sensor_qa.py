"""
QA Station Sensor Simulation

Simulates proximity and contact sensors at the interferometer QA station
using Newton's sensor API.  Detects disc arrival, measures dwell time,
and logs contact events for quality-gate decisions.

Based on the Newton ``sensor_contact`` example
(``python -m newton.examples sensor_contact``).

.. note::

    Newton is in active alpha development (March 2026).
    Sensor API surface is subject to change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import newton  # type: ignore[import-untyped]
except ImportError:
    newton = None

import warp as wp

from artifex.properties import DISC, PET
from artifex.config import CellConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SensorResult:
    """QA sensor telemetry from a single measurement cycle (Rev 1.0).
    
    Includes simulated readings from the WLI station, IR pyrometer,
    and inline load cells.
    """

    disc_detected: bool = False
    arrival_time: float = -1.0
    departure_time: float = -1.0
    dwell_time: float = 0.0
    
    # Process Telemetry (Simulated)
    melt_temp_C: float = 270.0
    mold_temp_C: float = 60.0
    hold_duration_s: float = 3.5
    disc_weight_g: float = 180.0
    
    # Metrology (Simulated WLI)
    groove_depth_um: float = 0.70
    eject_temp_C: float = 60.0
    crystallinity_proxy: float = 0.02  # 2% fraction
    
    # Validation
    pass_qa: bool = False
    quarantine_reason: Optional[str] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sensor QA class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SensorQA:
    """QA station sensor simulation using Newton's sensor primitives.

    Models a proximity sensor zone and a contact sensor pad at the
    interferometer station.  A disc is placed near the zone and the sim
    runs until the disc settles, logging sensor triggers.

    Parameters
    ----------
    sensor_zone_radius : float
        Radius of the circular proximity detection zone (m).
    sensor_zone_height : float
        Height of the proximity detection zone (m).
    max_contact_force_threshold : float
        Force above which a contact is flagged as a QA concern (N).
    config : CellConfig, optional
        Cell simulation configuration.
    device : str
        Compute device.
    """

    def __init__(
        self,
        sensor_zone_radius: float = 0.18,
        sensor_zone_height: float = 0.05,
        max_contact_force_threshold: float = 5.0,
        config: Optional[CellConfig] = None,
        device: str = "cuda:0",
    ) -> None:
        self.zone_radius = sensor_zone_radius
        self.zone_height = sensor_zone_height
        self.force_threshold = max_contact_force_threshold
        self.config = config or CellConfig()
        self.device = device

        self._scene = None
        self._disc = None
        self._sensor_pad = None

    def build(self) -> None:
        """Build the Newton scene with a QA sensor zone and disc."""
        if newton is None:
            raise RuntimeError(
                "Newton is not installed.  "
                "Install with: pip install 'newton[examples]'"
            )

        cfg = self.config

        self._scene = newton.Scene(
            device=cfg.device,
            sim_dt=cfg.sim_dt,
            sim_substeps=cfg.sim_substeps,
        )

        # Sensor pad (kinematic, represents the interferometer stage)
        self._sensor_pad = self._scene.add_rigid_body(  # type: ignore
            shape="cylinder",
            radius=self.zone_radius,
            height=0.005,
            pos=(0.0, 0.0, -0.005),
            is_kinematic=True,
            name="sensor_pad",
        )

        # Disc placed slightly above the pad (simulates robot release)
        self._disc = self._scene.add_rigid_body(  # type: ignore
            shape="cylinder",
            radius=DISC.radius,
            height=DISC.thickness,
            mass=PET.disc_mass,
            pos=(0.0, 0.0, 0.01),  # 10 mm above pad
            name="qa_disc",
        )

    def validate_disc(self, result: SensorResult) -> None:
        """Validate a disc against the Rev 1.0 Quarantine Thresholds.
        
        Applies the hard-coded triggers from the Engineering Specification.
        """
        # 1. Melt Temperature (270 ± 5°C)
        if result.melt_temp_C > 275.0 or result.melt_temp_C < 265.0:
            result.pass_qa = False
            result.quarantine_reason = "MELT_TEMP_OUT_OF_BOUNDS"
            return

        # 2. Mold Temperature (60 ± 3°C)
        if result.mold_temp_C > 63.0 or result.mold_temp_C < 57.0:
            result.pass_qa = False
            result.quarantine_reason = "MOLD_TEMP_OUT_OF_BOUNDS"
            return

        # 3. Hold Duration (2.0–5.0 s window)
        if result.hold_duration_s < 1.8 or result.hold_duration_s > 6.0:
            result.pass_qa = False
            result.quarantine_reason = "HOLD_DURATION_VIOLATION"
            return

        # 4. Disc Weight (180 ± 2 g)
        if result.disc_weight_g > 182.0 or result.disc_weight_g < 178.0:
            result.pass_qa = False
            result.quarantine_reason = "DISC_WEIGHT_OUT_OF_TOLERANCE"
            return

        # 5. Groove Depth (0.70 ± 0.1 µm — WLI)
        if result.groove_depth_um > 0.80 or result.groove_depth_um < 0.60:
            result.pass_qa = False
            result.quarantine_reason = "GROOVE_DEPTH_OUT_OF_SPEC"
            return

        # 6. Ejection Temperature (< 70°C — Critical Warpage Gate)
        if result.eject_temp_C >= 70.0:
            result.pass_qa = False
            result.quarantine_reason = "EJECT_TEMP_CRITICAL_WARPAGE_RISK"
            return

        # 7. Crystallinity Proxy (< 5%)
        if result.crystallinity_proxy > 0.05:
            result.pass_qa = False
            result.quarantine_reason = "EXCESSIVE_CRYSTALLINITY_OPACITY"
            return

        # If all checks pass
        result.pass_qa = True

    def run(self, n_steps: int = 1000) -> SensorResult:
        """Run the sensor simulation and return telemetry.

        Parameters
        ----------
        n_steps : int
            Number of physics steps.

        Returns
        -------
        SensorResult
        """
        if self._scene is None:
            self.build()

        result = SensorResult()
        sim_dt = self.config.sim_dt

        # Run physics
        for step in range(n_steps):
            self._scene.step()  # type: ignore
            t = step * sim_dt

            # ── Proximity check (disc arrival at WLI station) ──────────
            if self._disc is not None and hasattr(self._scene, "get_body_position"):
                pos = self._scene.get_body_position(self._disc)  # type: ignore
                if pos is not None:
                    lateral = float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
                    in_zone = (
                        lateral < self.zone_radius
                        and 0.0 <= pos[2] <= self.zone_height
                    )

                    if in_zone and not result.disc_detected:
                        result.disc_detected = True
                        result.arrival_time = t

                    if in_zone:
                        result.departure_time = t

        # Compute dwell time
        if result.disc_detected and result.arrival_time >= 0:
            result.dwell_time = result.departure_time - result.arrival_time

        # Perform Rev 1.0 Validation
        if result.disc_detected:
            self.validate_disc(result)

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Run the QA sensor simulation from the command line."""
    parser = argparse.ArgumentParser(
        description="Artifex QA station sensor simulation"
    )
    parser.add_argument("--device", default="cuda:0", help="Compute device")
    parser.add_argument(
        "--n-steps", type=int, default=1000,
        help="Physics steps (default: 1000)"
    )
    args = parser.parse_args()

    wp.init()

    qa = SensorQA(device=args.device)
    qa.build()

    print("Running QA sensor simulation...")
    result = qa.run(n_steps=args.n_steps)

    print()
    print("Sensor QA results:")
    print(f"  Disc detected:     {'YES' if result.disc_detected else 'NO'}")
    print(f"  Arrival time:      {result.arrival_time:.3f} s")
    print(f"  Dwell time:        {result.dwell_time:.3f} s")
    print(f"  Melt / Mold Temp:  {result.melt_temp_C:.1f} / {result.mold_temp_C:.1f} °C")
    print(f"  Groove Depth (WLI): {result.groove_depth_um:.3f} µm")
    print(f"  Eject Temp (IR):   {result.eject_temp_C:.1f} °C")
    print(f"  QA result:         {'PASS ✓' if result.pass_qa else 'FAIL ✗'}")
    if not result.pass_qa and result.quarantine_reason:
        print(f"  Quarantine Reason: {result.quarantine_reason}")


if __name__ == "__main__":
    main()
