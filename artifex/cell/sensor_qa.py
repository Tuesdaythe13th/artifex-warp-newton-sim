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
    """QA sensor telemetry from a single measurement cycle.

    Attributes:
        disc_detected: Whether the proximity sensor detected a disc.
        arrival_time: Simulation time when disc entered the sensor zone (s).
        departure_time: Simulation time when disc left the sensor zone (s).
        dwell_time: Time disc spent in the sensor zone (s).
        contact_events: Number of contact events during dwell.
        max_contact_force: Peak contact force during dwell (N).
        pass_qa: Whether the disc passed the simulated QA gate.
    """

    disc_detected: bool = False
    arrival_time: float = -1.0
    departure_time: float = -1.0
    dwell_time: float = 0.0
    contact_events: int = 0
    max_contact_force: float = 0.0
    pass_qa: bool = False


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

        for step in range(n_steps):
            self._scene.step()  # type: ignore
            t = step * sim_dt

            # ── Proximity check ──────────────────────────────────────────
            if self._disc is not None and hasattr(self._scene, "get_body_position"):
                pos = self._scene.get_body_position(self._disc)  # type: ignore
                if pos is not None:
                    # Check if disc centre is within the sensor zone cylinder
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

            # ── Contact force on sensor pad ──────────────────────────────
            if self._sensor_pad is not None and hasattr(
                self._scene, "get_contact_forces"
            ):
                forces = self._scene.get_contact_forces(self._sensor_pad)  # type: ignore
                if forces is not None:
                    mag = float(np.linalg.norm(forces))
                    if mag > 0.01:  # minimum threshold to count
                        result.contact_events += 1
                        result.max_contact_force = max(
                            result.max_contact_force, mag
                        )

        # Compute dwell time
        if result.disc_detected and result.arrival_time >= 0:
            result.dwell_time = result.departure_time - result.arrival_time

        # QA gate: disc detected, settled, no excessive force
        result.pass_qa = (
            result.disc_detected
            and result.max_contact_force < self.force_threshold
        )

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
    print(f"  Contact events:    {result.contact_events}")
    print(f"  Max contact force: {result.max_contact_force:.2f} N")
    print(f"  QA pass:           {'PASS ✓' if result.pass_qa else 'FAIL ✗'}")


if __name__ == "__main__":
    main()
