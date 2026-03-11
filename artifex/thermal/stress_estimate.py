"""
Thermoelastic Residual-Stress / Birefringence Proxy

Computes a simplified residual-stress metric from the through-thickness
temperature gradient produced by :mod:`artifex.thermal.disc_thermal`.
High thermal gradients in the z-direction during cooling correlate with
birefringence and groove distortion risk.

This is a *proxy* — it does not model optical birefringence directly but
provides a spatially resolved risk map that can be overlaid on the disc
to identify hotspots for groove distortion.

The Warp FEM elasticity examples (``example_mixed_elasticity.py``) show
the pattern for full thermoelastic solves; this module starts with a
lighter gradient-based approach suitable for Phase 1 / 2.

References
----------
* Warp FEM elasticity — https://nvidia.github.io/warp/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import warp as wp

from artifex.properties import DISC


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Warp kernels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@wp.kernel
def compute_z_gradient_kernel(
    temperature: wp.array(dtype=float),
    gradient_z: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dz: float,
):
    """Compute the through-thickness (z) temperature gradient per node.

    The field is assumed to be stored in a flattened array with ordering
    i + j*nx + k*nx*ny  (i=x, j=y, k=z).  Central differences are used
    for interior nodes; one-sided differences for faces.

    Parameters
    ----------
    temperature : array
        Flattened nodal temperatures  (length = nx * ny * nz).
    gradient_z : array
        Output:  |dT/dz| per node.
    nx, ny, nz : int
        Grid node counts along each axis.
    dz : float
        Grid spacing in z (metres).
    """
    tid = wp.tid()

    # Map flat index → (i, j, k)
    k = tid // (nx * ny)
    rem = tid - k * (nx * ny)
    j = rem // nx
    i = rem - j * nx

    if k == 0:
        # Forward difference on bottom face
        idx_up = i + j * nx + 1 * (nx * ny)
        grad = (temperature[idx_up] - temperature[tid]) / dz
    elif k == nz - 1:
        # Backward difference on top face
        idx_down = i + j * nx + (k - 1) * (nx * ny)
        grad = (temperature[tid] - temperature[idx_down]) / dz
    else:
        # Central difference
        idx_up = i + j * nx + (k + 1) * (nx * ny)
        idx_down = i + j * nx + (k - 1) * (nx * ny)
        grad = (temperature[idx_up] - temperature[idx_down]) / (2.0 * dz)

    # Store absolute gradient
    gradient_z[tid] = wp.abs(grad)


@wp.kernel
def thermal_stress_proxy_kernel(
    gradient_z: wp.array(dtype=float),
    stress_proxy: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    alpha_cte: float,
):
    """Map gradient magnitude to a stress-risk metric.

    A simple proxy:  σ_proxy = E(T) · α · |dT/dz| · h

    where E(T) is interpolated from the glassy/rubbery moduli, α is
    the coefficient of thermal expansion, and h is the disc thickness.

    This gives units of Pa — physically interpretable as a first-order
    stress scale induced by the through-thickness gradient.
    """
    i = wp.tid()
    t_c = temperature[i]

    # Sigmoid modulus interpolation (matching properties.py)
    t_g = 75.0  # nominal T_g
    width = 5.0
    sigmoid = 1.0 / (1.0 + wp.exp(-(t_c - t_g) / width))
    e_mod = 2.5e9 + (1.0e9 - 2.5e9) * sigmoid

    stress_proxy[i] = e_mod * alpha_cte * gradient_z[i] * 0.0019  # h = 1.9 mm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class StressResult:
    """Snapshot of the residual-stress proxy.

    Attributes:
        gradient_z: Through-thickness |dT/dz| per node (°C/m).
        stress_proxy: Stress-risk metric per node (Pa).
        max_gradient: Maximum |dT/dz| across the disc.
        max_stress: Maximum stress proxy value.
        mean_stress: Mean stress proxy value.
    """

    gradient_z: np.ndarray
    stress_proxy: np.ndarray
    max_gradient: float
    max_stress: float
    mean_stress: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Estimator class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StressEstimator:
    """Computes a through-thickness stress / birefringence risk proxy.

    Takes the structured temperature field from :class:`DiscThermalSim`
    and produces a spatially resolved stress-risk map.

    Parameters
    ----------
    nx : int
        Grid node count along x (disc plane).
    ny : int
        Grid node count along y (disc plane).
    nz : int
        Grid node count along z (through-thickness).
    dz : float
        Node spacing in z in metres.
    alpha_cte : float
        Coefficient of thermal expansion in 1/K.
        PET ≈ 60–70 × 10⁻⁶  /K.
    device : str
        Warp compute device.
    """

    # PET linear CTE (approximately 65 × 10⁻⁶ /K for amorphous PET)
    DEFAULT_ALPHA_CTE: float = 65.0e-6

    def __init__(
        self,
        nx: int = 129,
        ny: int = 129,
        nz: int = 9,
        dz: Optional[float] = None,
        alpha_cte: float = DEFAULT_ALPHA_CTE,
        device: str = "cuda:0",
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dz = dz if dz is not None else DISC.thickness / max(nz - 1, 1)
        self.alpha_cte = alpha_cte
        self.device = device

        n_total = nx * ny * nz
        self._gradient_z = wp.zeros(n_total, dtype=float, device=device)
        self._stress_proxy = wp.zeros(n_total, dtype=float, device=device)

    def compute(self, temperature: wp.array) -> StressResult:
        """Compute the stress proxy from a nodal temperature field.

        Parameters
        ----------
        temperature : wp.array(dtype=float)
            Flat nodal temperature array with shape (nx*ny*nz,), ordered
            as i + j*nx + k*nx*ny.

        Returns
        -------
        StressResult
            Gradient and stress-proxy snapshot.
        """
        n_total = self.nx * self.ny * self.nz

        # Step 1 — through-thickness gradient
        wp.launch(
            kernel=compute_z_gradient_kernel,
            dim=n_total,
            inputs=[
                temperature,
                self._gradient_z,
                self.nx,
                self.ny,
                self.nz,
                self.dz,
            ],
            device=self.device,
        )

        # Step 2 — stress proxy
        wp.launch(
            kernel=thermal_stress_proxy_kernel,
            dim=n_total,
            inputs=[
                self._gradient_z,
                self._stress_proxy,
                temperature,
                self.alpha_cte,
            ],
            device=self.device,
        )

        # Pull to CPU
        grad_np = self._gradient_z.numpy()
        stress_np = self._stress_proxy.numpy()

        return StressResult(
            gradient_z=grad_np.copy(),
            stress_proxy=stress_np.copy(),
            max_gradient=float(np.max(grad_np)),
            max_stress=float(np.max(stress_np)),
            mean_stress=float(np.mean(stress_np)),
        )
