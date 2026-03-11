"""
Transient Heat Diffusion FEM for Injection-Molded Disc Cooling

Solves the heat equation on a thin 3D disc geometry using NVIDIA Warp FEM.
The governing PDE is:

    ρ c_p ∂T/∂t = ∇·(k ∇T)

with Dirichlet boundary conditions on mold-contact faces (T = T_mold) and
convective conditions on free edges.  Time integration uses implicit Euler
(backward Euler) for unconditional stability at moderate time steps.

Based on the Warp FEM diffusion example pattern
(``warp/examples/fem/example_diffusion_3d.py``).

References
----------
* NVIDIA Warp docs — https://nvidia.github.io/warp/
* Warp FEM module — warp.fem
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

import warp as wp
import warp.fem as fem

from artifex.properties import PET, DISC
from artifex.config import ThermalConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEM integrands
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@fem.integrand
def mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
    rho_cp: float,
):
    """Mass-matrix integrand:  ρ·c_p · u · v."""
    return rho_cp * u(s) * v(s)


@fem.integrand
def diffusion_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
    k: float,
):
    """Stiffness (diffusion) integrand:  k · ∇u · ∇v."""
    return k * wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def rhs_form(
    s: fem.Sample,
    v: fem.Field,
    temperature: fem.Field,
    rho_cp: float,
):
    """Right-hand-side integrand for implicit Euler:  ρ·c_p · T^n · v."""
    return rho_cp * temperature(s) * v(s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Boundary condition helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@fem.integrand
def dirichlet_projector(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
    penalty: float,
):
    """Penalty Dirichlet BC integrand:  penalty · u · v  on boundary."""
    return penalty * u(s) * v(s)


@fem.integrand
def dirichlet_rhs(
    s: fem.Sample,
    v: fem.Field,
    t_boundary: float,
    penalty: float,
):
    """Penalty Dirichlet RHS:  penalty · T_mold · v  on boundary."""
    return penalty * t_boundary * v(s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulation class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ThermalResult:
    """Container for a single time-step's thermal output.

    Attributes:
        time: Simulation time in seconds.
        temperature: Temperature field as a 1-D numpy array (°C per node).
        min_temp: Minimum temperature across all nodes.
        max_temp: Maximum temperature across all nodes.
        mean_temp: Volume-averaged temperature.
    """

    time: float
    temperature: np.ndarray
    min_temp: float
    max_temp: float
    mean_temp: float


class DiscThermalSim:
    """GPU-accelerated transient thermal simulation of a PET disc.

    Uses Warp FEM with implicit Euler time stepping on a ``Grid3D`` geometry
    representing the disc (305 mm diameter × 1.9 mm thick).

    Parameters
    ----------
    config : ThermalConfig
        Simulation configuration (grid resolution, time step, BCs, etc.).
    device : str
        Warp compute device, e.g. ``"cuda:0"`` or ``"cpu"``.
    """

    def __init__(
        self,
        config: Optional[ThermalConfig] = None,
        device: str = "cuda:0",
    ) -> None:
        self.config = config or ThermalConfig()
        self.device = device

        # Derived constants
        self._rho_cp = PET.density * PET.specific_heat  # ρ·c_p  [J/(m³·K)]
        self._k = self.config.thermal_conductivity  # k  [W/(m·K)]

        # Penalty coefficient for Dirichlet BCs (large relative to k/h²)
        self._penalty = 1.0e6

        # State
        self._geometry: Optional[fem.Grid3D] = None
        self._scalar_space = None
        self._temperature_field = None
        self._temperature_dofs: Optional[wp.array] = None
        self._time = 0.0

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Build the FEM geometry, function space, and initial condition.

        Creates a ``Grid3D`` with the disc's physical extents and the
        configured resolution.  Initialises the temperature field uniformly
        at the melt temperature.
        """
        cfg = self.config
        r = DISC.radius  # half-diameter in metres
        h = DISC.thickness

        # Grid3D maps x,y to the disc plane, z to through-thickness
        self._geometry = fem.Grid3D(
            res=wp.vec3i(cfg.grid_res_xy, cfg.grid_res_xy, cfg.grid_res_z),
            bounds_lo=wp.vec3(-r, -r, 0.0),
            bounds_hi=wp.vec3(r, r, h),
        )

        # Q1 (trilinear) scalar function space
        self._scalar_space = fem.make_polynomial_space(
            self._geometry, degree=1
        )

        # Allocate DOF array and fill with T_melt (initial condition)
        n_dofs = self._scalar_space.node_count()
        self._temperature_dofs = wp.zeros(n_dofs, dtype=float, device=self.device)
        wp.launch(
            kernel=_fill_constant,
            dim=n_dofs,
            inputs=[self._temperature_dofs, float(cfg.melt_temperature)],
            device=self.device,
        )

        self._time = 0.0

    # ── Single time step ─────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the thermal solution by one implicit-Euler time step.

        Assembles  (M + dt·K) T^{n+1} = M·T^n  with penalty Dirichlet BCs
        on the top and bottom faces (mold contact), then solves via CG.
        """
        cfg = self.config
        dt = cfg.dt
        geo = self._geometry
        space = self._scalar_space

        # Current temperature as a discrete field
        domain = fem.Cells(geometry=geo)
        temperature_field = fem.make_restriction(
            space=space, domain=domain, device=self.device
        )
        temperature_field.dof_values = self._temperature_dofs

        # Trial / test spaces
        test = fem.make_test(space=space, domain=domain)
        trial = fem.make_trial(space=space, domain=domain)

        # ── LHS: (M + dt·K) ─────────────────────────────────────────────
        lhs_mass = fem.integrate(
            mass_form,
            fields={"u": trial, "v": test},
            values={"rho_cp": self._rho_cp},
        )
        lhs_diff = fem.integrate(
            diffusion_form,
            fields={"u": trial, "v": test},
            values={"k": self._k * dt},
        )

        # ── RHS: M · T^n ────────────────────────────────────────────────
        rhs = fem.integrate(
            rhs_form,
            fields={"v": test, "temperature": temperature_field},
            values={"rho_cp": self._rho_cp},
            output_dtype=float,
        )

        # ── Boundary conditions (penalty Dirichlet on z=0 and z=h) ───────
        boundary = fem.BoundarySides(geo)
        bd_test = fem.make_test(space=space, domain=boundary)
        bd_trial = fem.make_trial(space=space, domain=boundary)

        bc_lhs = fem.integrate(
            dirichlet_projector,
            fields={"u": bd_trial, "v": bd_test},
            values={"penalty": self._penalty},
        )
        bc_rhs = fem.integrate(
            dirichlet_rhs,
            fields={"v": bd_test},
            values={
                "t_boundary": float(cfg.mold_temperature),
                "penalty": self._penalty,
            },
            output_dtype=float,
        )

        # Combine system matrices
        # NOTE: fem.integrate returns BSR matrices; addition is via
        # Warp's sparse-matrix utilities.
        system_matrix = fem.bsr_axpy(lhs_mass, lhs_diff)
        system_matrix = fem.bsr_axpy(system_matrix, bc_lhs)

        # Combine RHS vectors
        rhs_combined = wp.clone(rhs)
        wp.launch(
            kernel=_axpy_vectors,
            dim=len(rhs_combined),
            inputs=[rhs_combined, bc_rhs, 1.0],
            device=self.device,
        )

        # ── Solve (CG) ──────────────────────────────────────────────────
        result = wp.zeros_like(self._temperature_dofs)
        fem.bsr_cg(system_matrix, rhs_combined, x=result, tol=1e-6)

        self._temperature_dofs = result
        self._time += dt

    # ── Multi-step run ───────────────────────────────────────────────────────

    def run(self, callback=None) -> list[ThermalResult]:
        """Run the full cooling simulation.

        Parameters
        ----------
        callback : callable, optional
            Called with ``(step_index, ThermalResult)`` after each step
            for real-time monitoring or USD export.

        Returns
        -------
        list[ThermalResult]
            One result per time step.
        """
        if self._geometry is None:
            self.setup()

        results: list[ThermalResult] = []
        n_steps = self.config.n_steps

        for i in range(n_steps):
            self.step()
            result = self.get_result()
            results.append(result)
            if callback is not None:
                callback(i, result)

        return results

    # ── Query ────────────────────────────────────────────────────────────────

    def get_result(self) -> ThermalResult:
        """Snapshot the current temperature field as a ``ThermalResult``."""
        t_np = self._temperature_dofs.numpy()
        return ThermalResult(
            time=self._time,
            temperature=t_np.copy(),
            min_temp=float(np.min(t_np)),
            max_temp=float(np.max(t_np)),
            mean_temp=float(np.mean(t_np)),
        )

    def get_temperature_field(self) -> np.ndarray:
        """Return the nodal temperature field as a numpy array (°C)."""
        return self._temperature_dofs.numpy().copy()

    def is_below_tg(self, margin: float = 0.0) -> bool:
        """Check if all nodes have cooled below T_g – margin."""
        t_np = self._temperature_dofs.numpy()
        return bool(np.max(t_np) < PET.glass_transition_nominal - margin)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper kernels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@wp.kernel
def _fill_constant(field: wp.array(dtype=float), value: float):
    """Fill a 1-D array with a constant value."""
    i = wp.tid()
    field[i] = value


@wp.kernel
def _axpy_vectors(y: wp.array(dtype=float), x: wp.array(dtype=float), alpha: float):
    """y[i] += alpha * x[i]."""
    i = wp.tid()
    y[i] = y[i] + alpha * x[i]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Run the disc thermal simulation from the command line."""
    parser = argparse.ArgumentParser(
        description="Artifex disc thermal cooling simulation (Warp FEM)"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Warp device (default: cuda:0)"
    )
    parser.add_argument(
        "--mold-temp", type=float, default=95.0,
        help="Mold temperature in °C (default: 95)"
    )
    parser.add_argument(
        "--melt-temp", type=float, default=270.0,
        help="Melt temperature in °C (default: 270)"
    )
    parser.add_argument(
        "--total-time", type=float, default=20.0,
        help="Total cooling time in seconds (default: 20)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Time step in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--res-xy", type=int, default=128,
        help="Grid resolution in disc plane (default: 128)"
    )
    parser.add_argument(
        "--res-z", type=int, default=8,
        help="Grid resolution through thickness (default: 8)"
    )
    args = parser.parse_args()

    wp.init()

    config = ThermalConfig(
        grid_res_xy=args.res_xy,
        grid_res_z=args.res_z,
        dt=args.dt,
        total_time=args.total_time,
        mold_temperature=args.mold_temp,
        melt_temperature=args.melt_temp,
    )

    sim = DiscThermalSim(config=config, device=args.device)
    sim.setup()

    def _report(step_idx: int, result: ThermalResult) -> None:
        if step_idx % 10 == 0:
            print(
                f"  step {step_idx:4d}  t={result.time:6.2f}s  "
                f"T_min={result.min_temp:6.1f}°C  "
                f"T_max={result.max_temp:6.1f}°C  "
                f"T_mean={result.mean_temp:6.1f}°C"
            )

    print(f"Running disc thermal sim: {config.n_steps} steps, "
          f"dt={config.dt}s, total={config.total_time}s")
    print(f"  Mold T = {config.mold_temperature}°C, "
          f"Melt T = {config.melt_temperature}°C")
    print(f"  Grid: {config.grid_res_xy}×{config.grid_res_xy}×{config.grid_res_z}")
    print()

    results = sim.run(callback=_report)

    final = results[-1]
    below_tg = sim.is_below_tg()
    print()
    print(f"Final state at t={final.time:.1f}s:")
    print(f"  T_min  = {final.min_temp:.1f}°C")
    print(f"  T_max  = {final.max_temp:.1f}°C")
    print(f"  T_mean = {final.mean_temp:.1f}°C")
    print(f"  All below T_g ({PET.glass_transition_nominal:.0f}°C): "
          f"{'YES ✓' if below_tg else 'NO ✗'}")


if __name__ == "__main__":
    main()
