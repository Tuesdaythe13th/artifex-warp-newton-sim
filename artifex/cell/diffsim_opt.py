"""
Differentiable Trajectory Optimization for Robot Handling

Uses Warp's automatic differentiation (through Newton's differentiable
simulation path) to optimise the robot's pick-transfer-place trajectory.
The cost function balances:

    • Cycle time (minimise)
    • Peak groove-area contact force (constrain)
    • Peak disc acceleration during transfer (constrain)

Based on Newton's ``diffsim_ball`` / ``diffsim_drone`` example patterns
(``python -m newton.examples diffsim_ball``).

.. note::

    Newton's differentiable simulation is in active development
    (March 2026).  Gradient paths through contact may not be fully
    stable in all configurations.  This module is a starting template
    for Phase 3+ optimisation once the baseline handling sim is verified.

References
----------
* Newton DiffSim — https://github.com/newton-physics/newton
* Warp autodiff — https://nvidia.github.io/warp/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

import warp as wp

from artifex.properties import PET
from artifex.config import DiffSimConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trajectory parameterisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class TrajectoryParams:
    """Parameterised robot trajectory for pick-transfer-place.

    These are the design variables that the optimiser tunes.

    Attributes:
        pick_speed: End-effector approach speed during pick (m/s).
        transfer_speed: End-effector speed during transfer (m/s).
        place_speed: End-effector descent speed during placement (m/s).
        grip_force: Gripper clamping force (N).
        transfer_height: Height above table during transfer (m).
        approach_angle: Approach angle for pick (radians from vertical).
    """

    pick_speed: float = 0.15
    transfer_speed: float = 0.4
    place_speed: float = 0.10
    grip_force: float = 8.0
    transfer_height: float = 0.10
    approach_angle: float = 0.0  # vertical


@dataclass
class TrajectoryOptResult:
    """Result of trajectory optimisation.

    Attributes:
        optimal_params: Best trajectory parameters found.
        final_cost: Cost at convergence.
        cost_history: Cost per iteration.
        cycle_time: Estimated cycle time at optimal params (s).
        max_contact_force: Estimated max groove contact force (N).
        max_acceleration: Estimated max disc acceleration (m/s²).
        converged: Whether the optimiser converged.
    """

    optimal_params: TrajectoryParams
    final_cost: float
    cost_history: list[float]
    cycle_time: float
    max_contact_force: float
    max_acceleration: float
    converged: bool


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Differentiable cost kernels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@wp.kernel
def trajectory_cost_kernel(
    params: wp.array(dtype=float),  # type: ignore
    cost: wp.array(dtype=float),  # type: ignore
    w_time: float,
    w_force: float,
    w_accel: float,
    disc_mass: float,
):
    """Compute a proxy cost from trajectory parameters.

    params layout:
        [0] pick_speed, [1] transfer_speed, [2] place_speed,
        [3] grip_force, [4] transfer_height

    Proxy model (simplified kinematics, not full Newton sim):
        cycle_time ∝ distances / speeds
        contact_force ∝ grip_force
        acceleration ∝ speed² / height  (centripetal-like proxy)
    """
    # Distances (approximate cell layout)
    d_pick = 0.15  # m — approach distance
    d_transfer = 0.50  # m — transfer distance
    d_place = 0.10  # m — descent distance

    pick_spd = wp.max(params[0], 0.01)
    xfer_spd = wp.max(params[1], 0.01)
    place_spd = wp.max(params[2], 0.01)
    grip_f = params[3]
    xfer_h = wp.max(params[4], 0.02)

    # Cycle time
    t_cycle = d_pick / pick_spd + d_transfer / xfer_spd + d_place / place_spd

    # Contact force proxy (grip force on disc)
    f_contact = grip_f

    # Acceleration proxy (speed change over transfer height)
    a_proxy = (xfer_spd * xfer_spd) / xfer_h

    # Weighted cost
    total = (
        w_time * t_cycle
        + w_force * wp.max(f_contact - 15.0, 0.0)  # penalty above 15 N
        + w_accel * wp.max(a_proxy - 5.0, 0.0)  # penalty above 5 m/s²
    )

    cost[0] = total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optimiser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TrajectoryOptimizer:
    """Differentiable trajectory optimiser using ``wp.Tape()``.

    In Phase 3+, this should be upgraded to optimise through a full
    Newton simulation forward pass.  The current version uses a
    kinematic proxy cost for rapid iteration during architecture
    validation.

    Parameters
    ----------
    config : DiffSimConfig, optional
        Optimisation hyperparameters.
    initial_params : TrajectoryParams, optional
        Starting trajectory parameters.
    device : str
        Warp compute device.
    """

    def __init__(
        self,
        config: Optional[DiffSimConfig] = None,
        initial_params: Optional[TrajectoryParams] = None,
        device: str = "cuda:0",
    ) -> None:
        self.config = config or DiffSimConfig()
        self.initial = initial_params or TrajectoryParams()
        self.device = device

    def optimize(self) -> TrajectoryOptResult:
        """Run gradient descent on the trajectory cost.

        Returns
        -------
        TrajectoryOptResult
        """
        cfg = self.config

        # Pack initial params into a Warp array
        p0 = [
            self.initial.pick_speed,
            self.initial.transfer_speed,
            self.initial.place_speed,
            self.initial.grip_force,
            self.initial.transfer_height,
        ]
        params = wp.array(p0, dtype=float, device=self.device, requires_grad=True)

        cost_history: list[float] = []

        for iteration in range(cfg.max_iterations):
            cost = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)

            tape = wp.Tape()
            with tape:
                wp.launch(
                    kernel=trajectory_cost_kernel,
                    dim=1,
                    inputs=[
                        params,
                        cost,
                        cfg.weight_cycle_time,
                        cfg.weight_contact_force,
                        cfg.weight_acceleration,
                        PET.disc_mass,
                    ],
                    device=self.device,
                )

            cost_val = float(cost.numpy()[0])
            cost_history.append(cost_val)

            # Backprop
            tape.backward(cost)

            # Gradient step
            grad = params.grad.numpy()
            p_np = params.numpy()
            p_np -= cfg.lr * grad

            # Clamp to physically reasonable bounds
            p_np[0] = np.clip(p_np[0], 0.05, 1.0)  # type: ignore
            p_np[1] = np.clip(p_np[1], 0.05, 2.0)  # type: ignore
            p_np[2] = np.clip(p_np[2], 0.02, 0.5)  # type: ignore
            p_np[3] = np.clip(p_np[3], 2.0, 25.0)  # type: ignore
            p_np[4] = np.clip(p_np[4], 0.02, 0.30)  # type: ignore

            params = wp.array(
                p_np.tolist(), dtype=float, device=self.device, requires_grad=True,
            )

            tape.zero()

            # Convergence
            if iteration > 5 and abs(cost_history[-1] - cost_history[-2]) < 1e-6:
                break

        # Extract final params
        final = params.numpy()
        opt_params = TrajectoryParams(
            pick_speed=float(final[0]),
            transfer_speed=float(final[1]),
            place_speed=float(final[2]),
            grip_force=float(final[3]),
            transfer_height=float(final[4]),
        )

        # Estimate cycle time / force / accel from final params
        d_pick, d_transfer, d_place = 0.15, 0.50, 0.10
        ct = d_pick / max(opt_params.pick_speed, 0.01) + \
             d_transfer / max(opt_params.transfer_speed, 0.01) + \
             d_place / max(opt_params.place_speed, 0.01)

        return TrajectoryOptResult(
            optimal_params=opt_params,
            final_cost=cost_history[-1],
            cost_history=cost_history,
            cycle_time=ct,
            max_contact_force=opt_params.grip_force,
            max_acceleration=(opt_params.transfer_speed ** 2)
            / max(opt_params.transfer_height, 0.02),
            converged=len(cost_history) < self.config.max_iterations,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Run trajectory optimisation from the command line."""
    parser = argparse.ArgumentParser(
        description="Artifex differentiable trajectory optimiser"
    )
    parser.add_argument("--device", default="cuda:0", help="Compute device")
    parser.add_argument(
        "--max-iter", type=int, default=200,
        help="Max iterations (default: 200)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    args = parser.parse_args()

    wp.init()

    cfg = DiffSimConfig(lr=args.lr, max_iterations=args.max_iter)
    opt = TrajectoryOptimizer(config=cfg, device=args.device)

    print("Running trajectory optimisation...")
    result = opt.optimize()

    print()
    p = result.optimal_params
    print("Optimal trajectory:")
    print(f"  Pick speed:       {p.pick_speed:.3f} m/s")
    print(f"  Transfer speed:   {p.transfer_speed:.3f} m/s")
    print(f"  Place speed:      {p.place_speed:.3f} m/s")
    print(f"  Grip force:       {p.grip_force:.1f} N")
    print(f"  Transfer height:  {p.transfer_height:.3f} m")
    print()
    print(f"Estimated cycle time:    {result.cycle_time:.2f} s")
    print(f"Max contact force:       {result.max_contact_force:.1f} N")
    print(f"Max acceleration:        {result.max_acceleration:.1f} m/s²")
    print(f"Final cost:              {result.final_cost:.4f}")
    print(f"Converged:               {'YES' if result.converged else 'NO'}")


if __name__ == "__main__":
    main()
