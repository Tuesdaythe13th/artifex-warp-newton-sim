"""
Avrami Crystallinity Proxy Kernel

Co-solves a reduced crystallinity state variable alongside the thermal
solution from :mod:`artifex.thermal.disc_thermal`.  The model tracks
how long each node's temperature resides inside the PET crystallisation
window and applies an Avrami-type transformation law:

    χ(t) = χ_∞ · (1 − exp(−K(T) · t^n))

where  K(T) = K_peak · exp(−((T − T_peak) / σ)²)

is a Gaussian rate function peaked between T_g and T_m.

The primary output is a per-node crystallinity fraction and a Go / No-Go
flag based on the ``max_allowed`` threshold (default 5 %). In the
standard Artifex process, rapid quench and the absence of nucleators
should keep χ well below this bound.

References
----------
* Warp custom kernel programming — https://nvidia.github.io/warp/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

import warp as wp

from artifex.config import CrystallinityConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Warp kernels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@wp.kernel
def avrami_step_kernel(
    temperature: wp.array(dtype=float),
    chi: wp.array(dtype=float),
    residence_time: wp.array(dtype=float),
    dt: float,
    chi_inf: float,
    avrami_n: float,
    k_peak: float,
    k_peak_temp: float,
    k_width: float,
):
    """Advance the Avrami crystallinity one time step per node.

    For each node i:
        1. Compute K(T_i) as a Gaussian centred on k_peak_temp.
        2. Accumulate residence time in the crystallisation window.
        3. Update χ_i via the Avrami equation.

    Parameters correspond one-to-one with :class:`CrystallinityConfig`.
    """
    i = wp.tid()

    t_i = temperature[i]

    # Temperature-dependent rate constant — Gaussian in T
    delta = (t_i - k_peak_temp) / k_width
    k_t = k_peak * wp.exp(-delta * delta)

    # Accumulate time spent in the crystallisation-susceptible window.
    # Only meaningful where K(T) is non-negligible, but the Gaussian
    # naturally decays outside the window so we can always accumulate.
    residence_time[i] = residence_time[i] + dt

    # Avrami equation
    tau = residence_time[i]
    exponent = k_t * wp.pow(tau, avrami_n)
    chi[i] = chi_inf * (1.0 - wp.exp(-exponent))


@wp.kernel
def _max_reduce_kernel(
    data: wp.array(dtype=float),
    result: wp.array(dtype=float),
):
    """Single-pass max reduction (thread 0 writes final result).

    NOTE: This is a naïve serial reduction for convenience.  For large
    arrays a parallel segmented reduction is preferred.
    """
    n = data.shape[0]
    max_val = data[0]
    for i in range(1, n):
        if data[i] > max_val:
            max_val = data[i]
    result[0] = max_val


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tracker class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CrystallinityResult:
    """Snapshot of the crystallinity state.

    Attributes:
        chi: Per-node crystallinity fraction as numpy array.
        max_chi: Maximum crystallinity across all nodes.
        mean_chi: Mean crystallinity.
        is_acceptable: True if max_chi < max_allowed threshold.
    """

    chi: np.ndarray
    max_chi: float
    mean_chi: float
    is_acceptable: bool


class CrystallinityTracker:
    """Tracks Avrami crystallinity evolution alongside a thermal simulation.

    Create an instance, call :meth:`setup` with the node count, then call
    :meth:`step` each time the temperature field is updated.

    Parameters
    ----------
    config : CrystallinityConfig, optional
        Avrami model parameters.  Defaults to PET-specific values.
    device : str
        Warp compute device.
    """

    def __init__(
        self,
        config: Optional[CrystallinityConfig] = None,
        device: str = "cuda:0",
    ) -> None:
        self.config = config or CrystallinityConfig()
        self.device = device

        self._chi: Optional[wp.array] = None
        self._residence_time: Optional[wp.array] = None

    def setup(self, n_nodes: int) -> None:
        """Allocate per-node state arrays, initialised to zero.

        Parameters
        ----------
        n_nodes : int
            Number of FEM nodes (must match the thermal field length).
        """
        self._chi = wp.zeros(n_nodes, dtype=float, device=self.device)
        self._residence_time = wp.zeros(n_nodes, dtype=float, device=self.device)

    def step(self, temperature: wp.array, dt: float) -> None:
        """Advance crystallinity by one time step.

        Parameters
        ----------
        temperature : wp.array(dtype=float)
            Current nodal temperature field from the thermal solver (°C).
        dt : float
            Time step in seconds.
        """
        if self._chi is None:
            raise RuntimeError("Call setup() before step().")

        cfg = self.config
        wp.launch(
            kernel=avrami_step_kernel,
            dim=len(self._chi),
            inputs=[
                temperature,
                self._chi,
                self._residence_time,
                dt,
                cfg.chi_infinity,
                cfg.avrami_n,
                cfg.k_peak_value,
                cfg.k_peak_temp,
                cfg.k_width,
            ],
            device=self.device,
        )

    def get_result(self) -> CrystallinityResult:
        """Return the current crystallinity state as a CPU-side snapshot."""
        if self._chi is None:
            raise RuntimeError("Call setup() before get_result().")

        chi_np = self._chi.numpy()
        max_chi = float(np.max(chi_np))
        mean_chi = float(np.mean(chi_np))

        return CrystallinityResult(
            chi=chi_np.copy(),
            max_chi=max_chi,
            mean_chi=mean_chi,
            is_acceptable=max_chi < self.config.max_allowed,
        )

    def is_acceptable(self) -> bool:
        """Quick check: is max crystallinity below the threshold?"""
        return self.get_result().is_acceptable


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Standalone crystallinity demo with a synthetic temperature ramp."""
    parser = argparse.ArgumentParser(
        description="Avrami crystallinity proxy — standalone demo"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Warp device (default: cuda:0)"
    )
    parser.add_argument(
        "--n-nodes", type=int, default=1000,
        help="Number of synthetic nodes (default: 1000)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Time step in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--total-time", type=float, default=20.0,
        help="Total simulation time (default: 20 s)"
    )
    args = parser.parse_args()

    wp.init()

    # Synthetic temperature ramp: starts at 270 °C, linearly cools to 60 °C
    n = args.n_nodes
    n_steps = int(args.total_time / args.dt)

    tracker = CrystallinityTracker(device=args.device)
    tracker.setup(n)

    print(f"Running crystallinity demo: {n} nodes, {n_steps} steps")
    print()

    for step in range(n_steps):
        t = step * args.dt
        # Linear ramp from melt to below Tg
        current_temp = 270.0 - (270.0 - 60.0) * (t / args.total_time)
        temp_array = wp.full(n, current_temp, dtype=float, device=args.device)

        tracker.step(temp_array, args.dt)

        if step % 20 == 0:
            result = tracker.get_result()
            print(
                f"  step {step:4d}  t={t:5.1f}s  "
                f"T={current_temp:6.1f}°C  "
                f"χ_max={result.max_chi:.4f}  "
                f"OK={'✓' if result.is_acceptable else '✗'}"
            )

    final = tracker.get_result()
    print()
    print(f"Final crystallinity: max={final.max_chi:.4f}, mean={final.mean_chi:.4f}")
    print(f"Acceptable (< {tracker.config.max_allowed:.0%}): "
          f"{'YES ✓' if final.is_acceptable else 'NO ✗'}")


if __name__ == "__main__":
    main()
