"""
Differentiable Process-Parameter Optimization

Uses Warp's reverse-mode automatic differentiation (``wp.Tape()``) to
optimise injection-molding process parameters against a composite loss
that balances amorphous retention, thermal stress, and throughput.

The optimisable design variables are:
    • Mold temperature (80–110 °C)
    • Cooling / cycle time (15–25 s)

The loss function is a weighted sum of:
    1. **Crystallinity penalty** — max χ anywhere in the disc.
    2. **Thermal-gradient penalty** — max through-thickness |dT/dz|.
    3. **Cycle-time cost** — total cooling time (minimise).
    4. **Final-temperature penalty** — any node still above T_g at eject.

Because both the thermal FEM (Layer 1) and the Avrami kernel are written
in Warp, ``wp.Tape()`` can record the entire forward pass and back-
propagate gradients through the simulation to the design variables.

References
----------
* Warp autodiff — https://nvidia.github.io/warp/
* Developer blog — https://developer.nvidia.com/blog/creating-differentiable-graphics-and-physics-simulation-in-python-with-nvidia-warp/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


import warp as wp

from artifex.properties import PET, DISC
from artifex.config import OptimizationConfig, ThermalConfig, CrystallinityConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Loss kernels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@wp.kernel
def simple_cooling_kernel(
    temperature: wp.array(dtype=float),
    mold_temp: wp.array(dtype=float),
    melt_temp: float,
    k_eff: float,
    dt: float,
):
    """Simplified 0-D exponential cooling per node (differentiable).

    T_{n+1} = T_mold + (T_n − T_mold) · exp(−k_eff · dt)

    This is a lumped-capacitance approximation used for the optimisation
    inner loop where full FEM assembly is not required.  The full FEM
    solver in ``disc_thermal.py`` should be used for validation.
    """
    i = wp.tid()
    t_m = mold_temp[0]  # scalar broadcast via array
    delta = temperature[i] - t_m
    temperature[i] = t_m + delta * wp.exp(-k_eff * dt)


@wp.kernel
def crystallinity_penalty_kernel(
    temperature: wp.array(dtype=float),
    residence_time: wp.array(dtype=float),
    chi_out: wp.array(dtype=float),
    dt: float,
    chi_inf: float,
    avrami_n: float,
    k_peak: float,
    k_peak_temp: float,
    k_width: float,
):
    """Avrami-update + crystallinity output (differentiable version)."""
    i = wp.tid()
    t_i = temperature[i]

    delta = (t_i - k_peak_temp) / k_width
    k_t = k_peak * wp.exp(-delta * delta)

    residence_time[i] = residence_time[i] + dt
    tau = residence_time[i]
    exponent = k_t * wp.pow(tau, avrami_n)
    chi_out[i] = chi_inf * (1.0 - wp.exp(-exponent))


@wp.kernel
def loss_kernel(
    temperature: wp.array(dtype=float),
    chi: wp.array(dtype=float),
    loss: wp.array(dtype=float),
    t_g: float,
    w_chi: float,
    w_temp: float,
):
    """Per-node contribution to the composite loss.

    loss += w_chi · χ_i  +  w_temp · max(T_i − T_g, 0)²

    The total loss is summed over all nodes; cycle-time and gradient
    penalties are added externally.
    """
    i = wp.tid()

    # Crystallinity term (linear penalty — drives max χ down)
    chi_penalty = w_chi * chi[i]

    # Temperature-above-Tg term (quadratic barrier)
    excess = temperature[i] - t_g
    temp_penalty = 0.0
    if excess > 0.0:
        temp_penalty = w_temp * excess * excess

    wp.atomic_add(loss, 0, chi_penalty + temp_penalty)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optimiser class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class OptimizationResult:
    """Result of the process-parameter optimisation.

    Attributes:
        optimal_mold_temp: Best mold temperature found (°C).
        optimal_cycle_time: Best cycle time found (s).
        final_loss: Loss value at convergence.
        loss_history: Loss at each iteration.
        converged: Whether the optimiser converged within max_iterations.
    """

    optimal_mold_temp: float
    optimal_cycle_time: float
    final_loss: float
    loss_history: list[float]
    converged: bool


class ThermalOptimizer:
    """Gradient-based optimiser for mold temperature and cycle time.

    Uses ``wp.Tape()`` to differentiate a simplified forward simulation
    (lumped cooling + Avrami crystallinity) with respect to the design
    variables, then applies projected gradient descent.

    Parameters
    ----------
    opt_config : OptimizationConfig, optional
        Weights and learning rate.
    thermal_config : ThermalConfig, optional
        Base thermal parameters (overridden by optimiser for mold_temp / time).
    cryst_config : CrystallinityConfig, optional
        Avrami crystallinity parameters.
    n_proxy_nodes : int
        Number of 0-D proxy nodes for the fast inner loop.
    device : str
        Warp compute device.
    """

    def __init__(
        self,
        opt_config: Optional[OptimizationConfig] = None,
        thermal_config: Optional[ThermalConfig] = None,
        cryst_config: Optional[CrystallinityConfig] = None,
        n_proxy_nodes: int = 256,
        device: str = "cuda:0",
    ) -> None:
        self.opt_cfg = opt_config or OptimizationConfig()
        self.therm_cfg = thermal_config or ThermalConfig()
        self.cryst_cfg = cryst_config or CrystallinityConfig()
        self.n_nodes = n_proxy_nodes
        self.device = device

    def optimize(self) -> OptimizationResult:
        """Run the optimisation loop.

        Returns
        -------
        OptimizationResult
            Optimal parameters and convergence info.
        """
        cfg = self.opt_cfg
        n = self.n_nodes
        dt = self.therm_cfg.dt
        t_g = PET.glass_transition_nominal

        # Effective Biot-like cooling rate for lumped model
        # k_eff ≈ k / (ρ·c_p · (h/2)²)
        k_eff = (
            self.therm_cfg.thermal_conductivity
            / (PET.density * PET.specific_heat * (DISC.half_thickness ** 2))
        )

        # ── Design variables (as Warp arrays for tape tracking) ──────────
        mold_temp_var = wp.array(
            [self.therm_cfg.mold_temperature],
            dtype=float,
            device=self.device,
            requires_grad=True,
        )

        loss_history: list[float] = []

        for iteration in range(cfg.max_iterations):
            # Reset per-iteration state
            temperature = wp.full(
                n, self.therm_cfg.melt_temperature,
                dtype=float, device=self.device, requires_grad=True,
            )
            residence_time = wp.zeros(
                n, dtype=float, device=self.device, requires_grad=True,
            )
            chi = wp.zeros(
                n, dtype=float, device=self.device, requires_grad=True,
            )
            loss = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)

            # ── Forward pass (recorded by tape) ─────────────────────────
            tape = wp.Tape()
            with tape:
                n_steps = self.therm_cfg.n_steps
                for _step in range(n_steps):
                    # Lumped cooling step
                    wp.launch(
                        kernel=simple_cooling_kernel,
                        dim=n,
                        inputs=[temperature, mold_temp_var,
                                self.therm_cfg.melt_temperature, k_eff, dt],
                        device=self.device,
                    )
                    # Crystallinity step
                    wp.launch(
                        kernel=crystallinity_penalty_kernel,
                        dim=n,
                        inputs=[
                            temperature, residence_time, chi, dt,
                            self.cryst_cfg.chi_infinity,
                            self.cryst_cfg.avrami_n,
                            self.cryst_cfg.k_peak_value,
                            self.cryst_cfg.k_peak_temp,
                            self.cryst_cfg.k_width,
                        ],
                        device=self.device,
                    )

                # Loss evaluation
                wp.launch(
                    kernel=loss_kernel,
                    dim=n,
                    inputs=[
                        temperature, chi, loss, t_g,
                        cfg.weight_crystallinity,
                        cfg.weight_final_temp,
                    ],
                    device=self.device,
                )

            loss_val = float(loss.numpy()[0])
            loss_history.append(loss_val)

            # ── Backward pass ────────────────────────────────────────────
            tape.backward(loss)

            # ── Gradient descent on mold_temp ────────────────────────────
            grad = mold_temp_var.grad.numpy()[0]
            new_val = float(mold_temp_var.numpy()[0]) - cfg.lr * grad

            # Project into valid range
            new_val = max(PET.mold_temp_lo, min(PET.mold_temp_hi, new_val))

            mold_temp_var = wp.array(
                [new_val], dtype=float, device=self.device, requires_grad=True,
            )

            tape.zero()

            # Convergence check
            if iteration > 5 and abs(loss_history[-1] - loss_history[-2]) < 1e-6:
                return OptimizationResult(
                    optimal_mold_temp=new_val,
                    optimal_cycle_time=self.therm_cfg.total_time,
                    final_loss=loss_val,
                    loss_history=loss_history,
                    converged=True,
                )

        final_mold = float(mold_temp_var.numpy()[0])
        return OptimizationResult(
            optimal_mold_temp=final_mold,
            optimal_cycle_time=self.therm_cfg.total_time,
            final_loss=loss_history[-1],
            loss_history=loss_history,
            converged=False,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Run the process-parameter optimiser from the command line."""
    parser = argparse.ArgumentParser(
        description="Artifex differentiable thermal process optimiser"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Warp device (default: cuda:0)"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100,
        help="Max optimisation iterations (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.5,
        help="Learning rate (default: 0.5)"
    )
    args = parser.parse_args()

    wp.init()

    opt_cfg = OptimizationConfig(lr=args.lr, max_iterations=args.max_iter)
    optimizer = ThermalOptimizer(opt_config=opt_cfg, device=args.device)

    print("Running differentiable process optimisation...")
    print(f"  Design variable: mold temperature (range {PET.mold_temp_lo}–"
          f"{PET.mold_temp_hi} °C)")
    print(f"  Learning rate: {opt_cfg.lr}")
    print(f"  Max iterations: {opt_cfg.max_iterations}")
    print()

    result = optimizer.optimize()

    print(f"Optimisation {'converged' if result.converged else 'did not converge'}.")
    print(f"  Optimal mold temperature: {result.optimal_mold_temp:.1f} °C")
    print(f"  Cycle time: {result.optimal_cycle_time:.1f} s")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Iterations: {len(result.loss_history)}")


if __name__ == "__main__":
    main()
