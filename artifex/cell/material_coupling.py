"""
Temperature → Mechanical Property Coupling (Layer 1 → Layer 2)

Bridges the Warp thermal simulation output to Newton scene material
parameters.  At ejection time, the disc's temperature field maps to
local Young's modulus, friction coefficient, and a handling-risk class.

This coupling ensures that Newton contact simulations capture the real
handling-risk window: warmer (rubbery) PET is softer, more scuff-prone,
and has higher friction, while cooled (glassy) PET is harder and safer.

The current implementation uses the piecewise sigmoid interpolation from
:mod:`artifex.properties`.  A more detailed version could use Prony-
series viscoelasticity, but the sigmoid is sufficient for Phase 1–3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

import warp as wp

from artifex.properties import PET


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Risk classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HandlingRisk(Enum):
    """Handling-risk classification based on disc temperature."""

    LOW = auto()  # Below T_g — glassy, safe
    MEDIUM = auto()  # Near T_g — transitioning
    HIGH = auto()  # Above T_g — rubbery, scuff risk


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class MaterialState:
    """Mechanical properties of the disc at a given temperature state.

    Attributes:
        temperature: Representative temperature in °C.
        youngs_modulus: Effective Young's modulus in Pa.
        friction_coefficient: Static friction coefficient.
        risk: Handling-risk classification.
    """

    temperature: float
    youngs_modulus: float
    friction_coefficient: float
    risk: HandlingRisk


@dataclass
class DiscMaterialField:
    """Spatially resolved material state across the disc.

    Attributes:
        temperatures: Per-node temperatures (°C).
        youngs_moduli: Per-node Young's modulus (Pa).
        friction_coefficients: Per-node friction coefficients.
        max_temp: Hottest node temperature.
        min_temp: Coldest node temperature.
        overall_risk: Worst-case risk across all nodes.
    """

    temperatures: np.ndarray
    youngs_moduli: np.ndarray
    friction_coefficients: np.ndarray
    max_temp: float
    min_temp: float
    overall_risk: HandlingRisk


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Warp kernel for vectorized property evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@wp.kernel
def evaluate_material_kernel(
    temperature: wp.array(dtype=float),
    youngs_modulus: wp.array(dtype=float),
    friction: wp.array(dtype=float),
    e_glassy: float,
    e_rubbery: float,
    t_g: float,
    width: float,
):
    """Evaluate temperature-dependent material properties per node.

    Parameters match the sigmoid model in :class:`PETProperties`.
    """
    i = wp.tid()
    t_c = temperature[i]

    sigmoid = 1.0 / (1.0 + wp.exp(-(t_c - t_g) / width))

    # Young's modulus: glassy → rubbery transition
    youngs_modulus[i] = e_glassy + (e_rubbery - e_glassy) * sigmoid

    # Friction: increases with temperature (softer surface)
    friction[i] = 0.25 + 0.30 * sigmoid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Coupling class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MaterialCoupling:
    """Maps disc temperature to Newton-compatible material parameters.

    This is the bridge between Layer 1 (Warp thermal) and Layer 2
    (Newton robot cell).  Use :meth:`evaluate_scalar` for a single
    representative temperature, or :meth:`evaluate_field` for a
    spatially resolved material state.

    Parameters
    ----------
    device : str
        Warp compute device for GPU-side evaluation.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    @staticmethod
    def classify_risk(temperature_c: float) -> HandlingRisk:
        """Classify handling risk from a temperature value.

        Parameters
        ----------
        temperature_c : float
            Temperature in °C.

        Returns
        -------
        HandlingRisk
        """
        if temperature_c > PET.glass_transition_hi:
            return HandlingRisk.HIGH
        elif temperature_c >= PET.glass_transition_lo:
            return HandlingRisk.MEDIUM
        else:
            return HandlingRisk.LOW

    def evaluate_scalar(self, temperature_c: float) -> MaterialState:
        """Evaluate material properties at a single temperature.

        Parameters
        ----------
        temperature_c : float
            Disc temperature in °C.

        Returns
        -------
        MaterialState
        """
        return MaterialState(
            temperature=temperature_c,
            youngs_modulus=PET.youngs_modulus_at(temperature_c),
            friction_coefficient=PET.friction_coefficient_at(temperature_c),
            risk=self.classify_risk(temperature_c),
        )

    def evaluate_field(
        self,
        temperature_field: np.ndarray,
    ) -> DiscMaterialField:
        """Evaluate material properties across a nodal temperature field.

        Parameters
        ----------
        temperature_field : np.ndarray
            1-D array of per-node temperatures in °C.

        Returns
        -------
        DiscMaterialField
            Spatially resolved material state.
        """
        n = len(temperature_field)

        # Upload temperature to GPU
        temp_wp = wp.array(temperature_field, dtype=float, device=self.device)
        e_wp = wp.zeros(n, dtype=float, device=self.device)
        mu_wp = wp.zeros(n, dtype=float, device=self.device)

        wp.launch(
            kernel=evaluate_material_kernel,
            dim=n,
            inputs=[
                temp_wp,
                e_wp,
                mu_wp,
                PET.youngs_modulus_glassy,
                PET.youngs_modulus_rubbery,
                PET.glass_transition_nominal,
                5.0,  # sigmoid width
            ],
            device=self.device,
        )

        e_np = e_wp.numpy()
        mu_np = mu_wp.numpy()

        max_temp = float(np.max(temperature_field))
        min_temp = float(np.min(temperature_field))
        overall_risk = self.classify_risk(max_temp)

        return DiscMaterialField(
            temperatures=temperature_field.copy(),
            youngs_moduli=e_np.copy(),
            friction_coefficients=mu_np.copy(),
            max_temp=max_temp,
            min_temp=min_temp,
            overall_risk=overall_risk,
        )

    @staticmethod
    def newton_contact_params(state: MaterialState) -> dict[str, float]:
        """Convert a MaterialState to Newton-compatible contact params.

        Returns a dict suitable for setting up Newton rigid-body contact
        material properties.

        Parameters
        ----------
        state : MaterialState

        Returns
        -------
        dict
            Keys: ``ke`` (contact stiffness), ``kd`` (contact damping),
            ``mu`` (friction coefficient).
        """
        # Contact stiffness scales with Young's modulus
        # (heuristic: ke ∝ E / disc_thickness)
        ke = state.youngs_modulus * 1e-3  # scale to Newton contact units
        kd = 0.1 * ke  # 10 % critical damping
        return {
            "ke": ke,
            "kd": kd,
            "mu": state.friction_coefficient,
        }
