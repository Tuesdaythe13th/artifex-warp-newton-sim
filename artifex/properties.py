"""
PET Material Properties & Disc Geometry Constants

All values from the validated Artifex Labs reference table for amorphous
recycled PET (r-PET) used in injection-molded record manufacturing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PETProperties:
    """Thermo-mechanical properties of amorphous recycled PET.

    Attributes:
        density: Density in kg/m³ (amorphous r-PET).
        specific_heat: Specific heat capacity in J/(kg·K) near T_g.
        thermal_conductivity_lo: Low end of thermal conductivity in W/(m·K).
        thermal_conductivity_hi: High end of thermal conductivity in W/(m·K).
        glass_transition_lo: Lower bound of T_g in °C.
        glass_transition_hi: Upper bound of T_g in °C.
        melt_temp_lo: Lower bound of injection melt temperature in °C.
        melt_temp_hi: Upper bound of injection melt temperature in °C.
        mold_temp_lo: Lower bound of mold surface temperature in °C.
        mold_temp_hi: Upper bound of mold surface temperature in °C.
        youngs_modulus_glassy: Young's modulus below T_g in Pa.
        youngs_modulus_rubbery: Young's modulus above T_g in Pa.
        disc_mass: Mass of a 12-inch LP disc in kg.
    """

    # --- Thermal ---
    density: float = 1350.0  # kg/m³
    specific_heat: float = 1200.0  # J/(kg·K)
    thermal_conductivity_lo: float = 0.15  # W/(m·K)
    thermal_conductivity_hi: float = 0.24  # W/(m·K)

    # --- Phase transitions ---
    glass_transition_lo: float = 70.0  # °C
    glass_transition_hi: float = 80.0  # °C
    melt_temp_lo: float = 250.0  # °C
    melt_temp_hi: float = 290.0  # °C
    mold_temp_lo: float = 80.0  # °C
    mold_temp_hi: float = 110.0  # °C

    # --- Mechanical ---
    youngs_modulus_glassy: float = 2.5e9  # Pa (2.2–3.0 GPa, nominal)
    youngs_modulus_rubbery: float = 1.0e9  # Pa (~1.0 GPa above T_g)

    # --- Mass ---
    disc_mass: float = 0.140  # kg (140 g, 12-inch LP)

    @property
    def thermal_conductivity_nominal(self) -> float:
        """Nominal thermal conductivity — geometric mean of range."""
        return math.sqrt(self.thermal_conductivity_lo * self.thermal_conductivity_hi)

    @property
    def glass_transition_nominal(self) -> float:
        """Nominal T_g — midpoint of range in °C."""
        return 0.5 * (self.glass_transition_lo + self.glass_transition_hi)

    @property
    def melt_temp_nominal(self) -> float:
        """Nominal melt temperature — midpoint of range in °C."""
        return 0.5 * (self.melt_temp_lo + self.melt_temp_hi)

    @property
    def mold_temp_nominal(self) -> float:
        """Nominal mold temperature — midpoint of range in °C."""
        return 0.5 * (self.mold_temp_lo + self.mold_temp_hi)

    def youngs_modulus_at(self, temperature_c: float) -> float:
        """Interpolate Young's modulus as a function of temperature.

        Uses a smooth sigmoid transition centred on T_g.

        Args:
            temperature_c: Temperature in °C.

        Returns:
            Young's modulus in Pa.
        """
        t_g = self.glass_transition_nominal
        # Width of the sigmoid transition zone (~5 °C half-width)
        width = 5.0
        sigmoid = 1.0 / (1.0 + math.exp(-(temperature_c - t_g) / width))
        return (
            self.youngs_modulus_glassy
            + (self.youngs_modulus_rubbery - self.youngs_modulus_glassy) * sigmoid
        )

    def friction_coefficient_at(self, temperature_c: float) -> float:
        """Estimate friction coefficient as a function of temperature.

        Warmer (rubbery) PET has higher friction due to softer surface.

        Args:
            temperature_c: Temperature in °C.

        Returns:
            Static friction coefficient (dimensionless).
        """
        t_g = self.glass_transition_nominal
        width = 5.0
        sigmoid = 1.0 / (1.0 + math.exp(-(temperature_c - t_g) / width))
        # Glassy ≈ 0.25, rubbery ≈ 0.55
        return 0.25 + 0.30 * sigmoid


@dataclass(frozen=True)
class DiscGeometry:
    """Dimensional parameters of a 12-inch injection-molded disc.

    Attributes:
        diameter: Disc outer diameter in metres.
        thickness: Disc thickness in metres.
        groove_depth_target: Target groove depth in metres (0.70 µm).
        groove_depth_tolerance: Groove depth tolerance ± in metres (0.03 µm).
    """

    diameter: float = 0.305  # m (305 mm)
    thickness: float = 0.0019  # m (1.9 mm, midpoint of 1.8–2.0)
    groove_depth_target: float = 0.70e-6  # m (0.70 µm)
    groove_depth_tolerance: float = 0.03e-6  # m (±0.03 µm)

    @property
    def radius(self) -> float:
        """Half-diameter in metres."""
        return self.diameter / 2.0

    @property
    def half_thickness(self) -> float:
        """Half-thickness in metres."""
        return self.thickness / 2.0


@dataclass(frozen=True)
class ProcessWindow:
    """Validated Artifex injection-molding process window.

    Attributes:
        cycle_time_lo: Min cycle time in seconds.
        cycle_time_hi: Max cycle time in seconds.
        max_crystallinity: Maximum allowable crystallinity fraction.
        fleet_size: Number of manufacturing cells.
        uptime_target: Target uptime fraction.
    """

    cycle_time_lo: float = 15.0  # s
    cycle_time_hi: float = 25.0  # s
    max_crystallinity: float = 0.05  # 5 %
    fleet_size: int = 48
    uptime_target: float = 0.85  # 85 %

    @property
    def target_daily_output(self) -> float:
        """Records per day at midpoint cycle time and target uptime."""
        cycle_mid = 0.5 * (self.cycle_time_lo + self.cycle_time_hi)
        seconds_per_day = 86_400.0
        return self.fleet_size * (seconds_per_day / cycle_mid) * self.uptime_target


# ── Module-level singletons ──────────────────────────────────────────────────
PET = PETProperties()
DISC = DiscGeometry()
PROCESS = ProcessWindow()
