"""
Single Artifex Manufacturing Cell — Newton Scene

Builds and runs a single injection-molding cell as a Newton simulation
scene containing:

    • UR10 robot arm (articulation loaded from URDF)
    • Injection-molded disc (rigid body with T-dependent material)
    • Press platen + ejector (kinematic bodies)
    • Conveyor belt
    • QA interferometer station (target pose)
    • Disc stack / packaging zone

The scene implements the eject → pick → transfer → inspect → stack
workflow described in the Artifex spec.

.. note::

    Newton is in active alpha development as of March 2026.  Its API
    surface (especially the Isaac Lab integration path) is experimental
    and subject to change.  This module targets the documented public
    API from ``newton-physics/newton`` on GitHub.

References
----------
* Newton — https://github.com/newton-physics/newton
* Newton examples — ``python -m newton.examples robot_ur10``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import newton  # type: ignore[import-untyped]
except ImportError:
    newton = None  # Allow import without Newton installed (for type-checking)

import warp as wp

from artifex.properties import PET, DISC
from artifex.config import CellConfig
from artifex.cell.material_coupling import MaterialCoupling


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cell layout constants (metres)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRESS_POS = (0.0, 0.0, 0.0)  # Press centre
ROBOT_BASE_POS = (0.4, 0.0, 0.0)  # Robot base offset from press
QA_STATION_POS = (0.8, 0.3, 0.0)  # Interferometer station
STACK_POS = (0.8, -0.3, 0.0)  # Packaging stack position
CONVEYOR_POS = (0.6, 0.0, 0.0)  # Conveyor belt centre


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CellMetrics:
    """Performance metrics from a single cell simulation run.

    Attributes:
        cycle_time: Total eject-to-stack time (s).
        max_groove_contact_force: Peak contact force on groove area (N).
        max_disc_acceleration: Peak disc acceleration during transfer (m/s²).
        scrap_events: Number of scrap-risk events detected.
        eject_temperature: Disc temperature at ejection (°C).
        handling_risk: Risk class at ejection.
    """

    cycle_time: float = 0.0
    max_groove_contact_force: float = 0.0
    max_disc_acceleration: float = 0.0
    scrap_events: int = 0
    eject_temperature: float = 0.0
    handling_risk: str = "UNKNOWN"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scene builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ArtifexCellScene:
    """Single Artifex manufacturing cell, built on Newton.

    Parameters
    ----------
    config : CellConfig, optional
        Cell simulation parameters.
    device : str
        Compute device.
    """

    def __init__(
        self,
        config: Optional[CellConfig] = None,
        device: str = "cuda:0",
    ) -> None:
        self.config = config or CellConfig()
        self.device = device

        self._coupling = MaterialCoupling(device=device)

        # Newton scene handles (set during build)
        self._scene = None
        self._robot = None
        self._disc = None
        self._press = None
        self._conveyor = None
        self._metrics = CellMetrics()

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Construct the Newton scene with all cell elements.

        Raises
        ------
        RuntimeError
            If the ``newton`` package is not installed.
        """
        if newton is None:
            raise RuntimeError(
                "Newton is not installed.  "
                "Install with: pip install 'newton[examples]'"
            )

        cfg = self.config

        # ── Create scene ─────────────────────────────────────────────────
        self._scene = newton.Scene(
            device=cfg.device,
            sim_dt=cfg.sim_dt,
            sim_substeps=cfg.sim_substeps,
        )
        scene = self._scene

        # ── Robot arm (UR10 from URDF) ───────────────────────────────────
        self._robot = scene.add_articulation(  # type: ignore
            urdf_path=cfg.robot_urdf,
            pos=ROBOT_BASE_POS,
            name="ur10_arm",
        )

        # ── Disc (rigid body — cylinder) ─────────────────────────────────
        mat_state = self._coupling.evaluate_scalar(cfg.disc_eject_temperature)
        contact_params = MaterialCoupling.newton_contact_params(mat_state)

        self._disc = scene.add_rigid_body(  # type: ignore
            shape="cylinder",
            radius=DISC.radius,
            height=DISC.thickness,
            mass=PET.disc_mass,
            pos=(PRESS_POS[0], PRESS_POS[1], DISC.thickness / 2),
            name="disc",
            ke=contact_params["ke"],
            kd=contact_params["kd"],
            mu=contact_params["mu"],
        )

        # ── Press platen (kinematic) ─────────────────────────────────────
        self._press = scene.add_rigid_body(  # type: ignore
            shape="box",
            half_extents=(0.2, 0.2, 0.05),
            pos=(PRESS_POS[0], PRESS_POS[1], -0.05),
            is_kinematic=True,
            name="press_platen",
        )

        # ── Conveyor belt ────────────────────────────────────────────────
        self._conveyor = scene.add_conveyor(  # type: ignore
            pos=CONVEYOR_POS,
            length=0.6,
            width=0.2,
            speed=cfg.conveyor_speed,
            name="conveyor",
        )

        # ── QA station (static fixture) ──────────────────────────────────
        scene.add_rigid_body(  # type: ignore
            shape="box",
            half_extents=(0.1, 0.1, 0.02),
            pos=QA_STATION_POS,
            is_kinematic=True,
            name="qa_station",
        )

        # ── Stack fixture ────────────────────────────────────────────────
        scene.add_rigid_body(  # type: ignore
            shape="box",
            half_extents=(0.16, 0.16, 0.01),
            pos=STACK_POS,
            is_kinematic=True,
            name="stack_base",
        )

        # Record eject material state
        self._metrics.eject_temperature = cfg.disc_eject_temperature
        self._metrics.handling_risk = mat_state.risk.name

    # ── Stepping ─────────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the Newton scene by one time step."""
        if self._scene is None:
            raise RuntimeError("Call build() before step().")
        self._scene.step()  # type: ignore

    def run(self, n_steps: Optional[int] = None) -> CellMetrics:
        """Run a full eject → pick → transfer → inspect → stack cycle.

        Parameters
        ----------
        n_steps : int, optional
            Override number of simulation steps.  Defaults to enough
            steps to fill the configured cycle time.

        Returns
        -------
        CellMetrics
            Cell-level performance metrics.
        """
        if self._scene is None:
            self.build()

        cfg = self.config
        if n_steps is None:
            # Enough steps for one full cycle time (~20 s at 120 Hz)
            n_steps = int(25.0 / cfg.sim_dt)

        max_force = 0.0
        max_accel = 0.0
        scrap: int = 0

        for i in range(n_steps):
            self.step()

            # ── Log contact forces on the disc ───────────────────────────
            if self._disc is not None and hasattr(self._scene, "get_contact_forces"):
                forces = self._scene.get_contact_forces(self._disc)  # type: ignore
                if forces is not None:
                    force_mag = float(np.linalg.norm(forces))
                    max_force = max(max_force, force_mag)

                    # Scrap if groove-area sees excessive force
                    if force_mag > cfg.gripper_max_force:
                        scrap += 1

            # ── Log disc acceleration ────────────────────────────────────
            if self._disc is not None and hasattr(self._scene, "get_body_acceleration"):
                accel = self._scene.get_body_acceleration(self._disc)  # type: ignore
                if accel is not None:
                    accel_mag = float(np.linalg.norm(accel))
                    max_accel = max(max_accel, accel_mag)

        self._metrics.cycle_time = n_steps * cfg.sim_dt
        self._metrics.max_groove_contact_force = max_force
        self._metrics.max_disc_acceleration = max_accel
        self._metrics.scrap_events = scrap

        return self._metrics

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> CellMetrics:
        """Return the last recorded cell metrics."""
        return self._metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Run the single-cell Newton simulation from the command line."""
    parser = argparse.ArgumentParser(
        description="Artifex single-cell Newton simulation"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Compute device (default: cuda:0)"
    )
    parser.add_argument(
        "--eject-temp", type=float, default=85.0,
        help="Disc ejection temperature in °C (default: 85)"
    )
    parser.add_argument(
        "--n-steps", type=int, default=None,
        help="Override step count (default: auto from cycle time)"
    )
    args = parser.parse_args()

    wp.init()

    config = CellConfig(
        disc_eject_temperature=args.eject_temp,
        device=args.device,
    )

    cell = ArtifexCellScene(config=config, device=args.device)

    print("Building cell scene...")
    cell.build()

    print("Running eject → pick → transfer → inspect → stack cycle...")
    metrics = cell.run(n_steps=args.n_steps)

    print()
    print("Cell metrics:")
    print(f"  Cycle time:            {metrics.cycle_time:.2f} s")
    print(f"  Max groove force:      {metrics.max_groove_contact_force:.2f} N")
    print(f"  Max disc acceleration: {metrics.max_disc_acceleration:.2f} m/s²")
    print(f"  Scrap events:          {metrics.scrap_events}")
    print(f"  Eject temperature:     {metrics.eject_temperature:.1f} °C")
    print(f"  Handling risk:         {metrics.handling_risk}")


if __name__ == "__main__":
    main()
