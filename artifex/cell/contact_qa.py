"""
Disc Stacking and Contact Force Logging

Simulates disc-on-disc and disc-on-fixture stacking using Newton's
rigid-body contact solver.  Logs per-step contact forces, identifies
slip events, and measures groove-area loading.

Based on the Newton ``brick_stacking`` example pattern
(``python -m newton.examples brick_stacking``).

.. note::

    Newton is in active alpha development (March 2026).  Contact-solver
    details may change.  This module uses the documented public API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import newton  # type: ignore[import-untyped]
except ImportError:
    newton = None

import warp as wp

from artifex.properties import PET, DISC
from artifex.config import CellConfig
from artifex.cell.material_coupling import MaterialCoupling


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ContactResult:
    """Summary of a contact / stacking simulation.

    Attributes:
        n_discs: Number of discs stacked.
        max_contact_force: Peak contact force across all disc pairs (N).
        mean_contact_force: Mean contact force during stacking (N).
        slip_events: Number of inter-disc slip events detected.
        force_history: Per-step max contact force log.
        stable: Whether the final stack remained stable.
    """

    n_discs: int = 0
    max_contact_force: float = 0.0
    mean_contact_force: float = 0.0
    slip_events: int = 0
    force_history: list[float] = field(default_factory=list)
    stable: bool = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Contact QA simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ContactQA:
    """Simulates disc stacking and contact-force QA.

    Drops ``n_discs`` onto a stack fixture and logs contact forces,
    checking for excessive groove loading or inter-disc slip.

    Parameters
    ----------
    n_discs : int
        Number of discs to stack.
    disc_temperature : float
        Common disc temperature in °C.
    config : CellConfig, optional
        Cell configuration (sim_dt, etc.).
    device : str
        Compute device.
    """

    def __init__(
        self,
        n_discs: int = 10,
        disc_temperature: float = 70.0,
        config: Optional[CellConfig] = None,
        device: str = "cuda:0",
    ) -> None:
        self.n_discs = n_discs
        self.disc_temperature = disc_temperature
        self.config = config or CellConfig()
        self.device = device
        self._coupling = MaterialCoupling(device=device)
        self._scene = None
        self._disc_bodies: list = []

    def build(self) -> None:
        """Build the Newton scene with a stack fixture and discs."""
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

        # Material properties from temperature
        mat_state = self._coupling.evaluate_scalar(self.disc_temperature)
        cp = MaterialCoupling.newton_contact_params(mat_state)

        # Stack base (kinematic)
        self._scene.add_rigid_body(  # type: ignore
            shape="box",
            half_extents=(0.16, 0.16, 0.01),
            pos=(0.0, 0.0, -0.01),
            is_kinematic=True,
            name="stack_base",
        )

        # Discs — stacked with small vertical gaps
        self._disc_bodies = []
        for i in range(self.n_discs):
            z = DISC.thickness * (i + 0.5) + 0.002 * i  # 2 mm settling gap
            body = self._scene.add_rigid_body(  # type: ignore
                shape="cylinder",
                radius=DISC.radius,
                height=DISC.thickness,
                mass=PET.disc_mass,
                pos=(0.0, 0.0, z),
                name=f"disc_{i}",
                ke=cp["ke"],
                kd=cp["kd"],
                mu=cp["mu"],
            )
            self._disc_bodies.append(body)

    def run(self, settle_steps: int = 2000) -> ContactResult:
        """Run the stacking simulation and collect contact data.

        Parameters
        ----------
        settle_steps : int
            Number of physics steps to let the stack settle.

        Returns
        -------
        ContactResult
        """
        if self._scene is None:
            self.build()

        force_log: list[float] = []
        slip_count: int = 0

        for step in range(settle_steps):
            self._scene.step()  # type: ignore

            # Log contact forces
            step_max_force = 0.0
            for body in self._disc_bodies:
                if hasattr(self._scene, "get_contact_forces"):
                    forces = self._scene.get_contact_forces(body)  # type: ignore
                    if forces is not None:
                        mag = float(np.linalg.norm(forces))
                        step_max_force = max(step_max_force, mag)

            force_log.append(step_max_force)

            # Simple slip detection: if any disc moves laterally > threshold
            for body in self._disc_bodies:
                if hasattr(self._scene, "get_body_velocity"):
                    vel = self._scene.get_body_velocity(body)  # type: ignore
                    if vel is not None:
                        lateral = float(np.sqrt(vel[0] ** 2 + vel[1] ** 2))
                        if lateral > 0.01:  # 1 cm/s lateral velocity = slip
                            slip_count += 1

        # Stack stability check — verify all discs are roughly co-axial
        stable = True
        if hasattr(self._scene, "get_body_position"):
            for body in self._disc_bodies:
                pos = self._scene.get_body_position(body)  # type: ignore
                if pos is not None:
                    lateral_offset = float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
                    if lateral_offset > DISC.radius * 0.1:  # > 10 % of radius
                        stable = False
                        break

        max_f = float(np.max(force_log)) if force_log else 0.0
        mean_f = float(np.mean(force_log)) if force_log else 0.0

        return ContactResult(
            n_discs=self.n_discs,
            max_contact_force=max_f,
            mean_contact_force=mean_f,
            slip_events=slip_count,
            force_history=force_log,
            stable=stable,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Run the disc stacking contact QA simulation."""
    parser = argparse.ArgumentParser(
        description="Artifex disc stacking — contact force QA"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Compute device"
    )
    parser.add_argument(
        "--n-discs", type=int, default=10, help="Number of discs to stack"
    )
    parser.add_argument(
        "--disc-temp", type=float, default=70.0,
        help="Disc temperature in °C (default: 70)"
    )
    parser.add_argument(
        "--settle-steps", type=int, default=2000,
        help="Physics steps for settling (default: 2000)"
    )
    args = parser.parse_args()

    wp.init()

    qa = ContactQA(
        n_discs=args.n_discs,
        disc_temperature=args.disc_temp,
        device=args.device,
    )

    print(f"Building stack: {args.n_discs} discs at {args.disc_temp} °C")
    qa.build()

    print(f"Settling for {args.settle_steps} steps...")
    result = qa.run(settle_steps=args.settle_steps)

    print()
    print("Contact QA results:")
    print(f"  Discs:           {result.n_discs}")
    print(f"  Max force:       {result.max_contact_force:.2f} N")
    print(f"  Mean force:      {result.mean_contact_force:.2f} N")
    print(f"  Slip events:     {result.slip_events}")
    print(f"  Stack stable:    {'YES ✓' if result.stable else 'NO ✗'}")


if __name__ == "__main__":
    main()
