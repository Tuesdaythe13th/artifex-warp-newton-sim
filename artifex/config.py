"""
Simulation Configuration

Dataclass-based configuration for Warp FEM grid resolution, time stepping,
and Newton scene parameters.  Serializable to / from dict for YAML I/O.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 1 — Warp FEM thermal config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ThermalConfig:
    """Configuration for the disc thermal FEM simulation.

    Attributes:
        grid_res_xy: Grid cells along x and y (maps to disc diameter).
        grid_res_z: Grid cells along z (through-thickness).
        dt: Time step in seconds for the heat-equation solve.
        total_time: Total cooling simulation time in seconds.
        mold_temperature: Mold surface temperature BC in °C.
        melt_temperature: Initial melt temperature in °C.
        thermal_conductivity: PET thermal conductivity in W/(m·K).
        convection_coeff: Free-edge convective coefficient in W/(m²·K).
        ambient_temperature: Ambient air temperature in °C.
    """

    # Grid resolution
    grid_res_xy: int = 128
    grid_res_z: int = 8

    # Time stepping
    dt: float = 0.1  # seconds
    total_time: float = 20.0  # seconds (midpoint of 15–25 s window)

    # Boundary conditions
    mold_temperature: float = 95.0  # °C (midpoint of 80–110)
    melt_temperature: float = 270.0  # °C (midpoint of 250–290)
    thermal_conductivity: float = 0.20  # W/(m·K) (nominal)
    convection_coeff: float = 10.0  # W/(m²·K) — natural convection
    ambient_temperature: float = 25.0  # °C

    @property
    def n_steps(self) -> int:
        """Number of time steps in the simulation."""
        return int(self.total_time / self.dt)


@dataclass
class CrystallinityConfig:
    """Configuration for the Avrami crystallinity proxy.

    Attributes:
        chi_infinity: Ultimate (equilibrium) crystallinity fraction.
        avrami_n: Avrami exponent.
        k_peak_temp: Temperature at which K(T) peaks in °C.
        k_peak_value: Maximum value of the rate constant K.
        k_width: Width (σ) of the Gaussian K(T) curve in °C.
        max_allowed: Maximum allowable crystallinity (Go/No-Go threshold).
    """

    chi_infinity: float = 0.45  # PET max crystallinity ~45 %
    avrami_n: float = 3.0
    k_peak_temp: float = 170.0  # °C — peak between T_g and T_m for PET
    k_peak_value: float = 0.01  # 1/s^n — deliberately low for rapid-quench
    k_width: float = 30.0  # °C — Gaussian width
    max_allowed: float = 0.05  # 5 % — acoustic transparency threshold


@dataclass
class OptimizationConfig:
    """Configuration for differentiable process-parameter optimization.

    Attributes:
        lr: Learning rate for gradient descent.
        max_iterations: Maximum number of optimisation steps.
        weight_crystallinity: Loss weight for max-crystallinity penalty.
        weight_gradient: Loss weight for max through-thickness gradient.
        weight_cycle_time: Loss weight for cycle time minimisation.
        weight_final_temp: Loss weight for final-temp-above-T_g penalty.
    """

    lr: float = 0.5
    max_iterations: int = 100
    weight_crystallinity: float = 10.0
    weight_gradient: float = 5.0
    weight_cycle_time: float = 1.0
    weight_final_temp: float = 8.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 2 — Newton robot cell config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CellConfig:
    """Configuration for a single Artifex manufacturing cell.

    Attributes:
        robot_urdf: Path to the UR10 URDF file.
        sim_dt: Physics time step in seconds for Newton.
        sim_substeps: Newton sub-steps per time step.
        disc_eject_temperature: Disc temperature at ejection in °C.
        gripper_max_force: Maximum gripper force in N.
        conveyor_speed: Conveyor belt speed in m/s.
        stack_height_limit: Max discs in a packaging stack.
        device: Compute device string (e.g. ``"cuda:0"``).
    """

    robot_urdf: str = "ur10.urdf"
    sim_dt: float = 1.0 / 120.0  # 120 Hz
    sim_substeps: int = 4
    disc_eject_temperature: float = 85.0  # °C — from Layer 1
    gripper_max_force: float = 20.0  # N
    conveyor_speed: float = 0.3  # m/s
    stack_height_limit: int = 25
    device: str = "cuda:0"


@dataclass
class DiffSimConfig:
    """Configuration for differentiable trajectory optimization.

    Attributes:
        lr: Learning rate.
        max_iterations: Max optimisation iterations.
        weight_cycle_time: Cost weight for cycle time.
        weight_contact_force: Cost weight for max groove contact force.
        weight_acceleration: Cost weight for disc acceleration.
    """

    lr: float = 1e-3
    max_iterations: int = 200
    weight_cycle_time: float = 1.0
    weight_contact_force: float = 10.0
    weight_acceleration: float = 5.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 3 — Isaac Lab config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class IsaacConfig:
    """Configuration for the Isaac Lab RL environment.

    Attributes:
        num_envs: Number of parallel environments for training.
        episode_length: Max steps per episode.
        domain_rand_thermal_noise: σ for T_eject randomisation in °C.
        domain_rand_material_var: Fractional variation in material props.
        domain_rand_robot_cal: Positional calibration noise in m.
    """

    num_envs: int = 48  # Matches fleet size
    episode_length: int = 500
    domain_rand_thermal_noise: float = 5.0  # °C σ
    domain_rand_material_var: float = 0.05  # ± 5 %
    domain_rand_robot_cal: float = 0.002  # 2 mm σ


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Composite config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ArtifexConfig:
    """Aggregate configuration container for the full simulation stack.

    Combines Layer 1, Layer 2, and Layer 3 configs into one serializable
    object.
    """

    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    crystallinity: CrystallinityConfig = field(default_factory=CrystallinityConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    cell: CellConfig = field(default_factory=CellConfig)
    diffsim: DiffSimConfig = field(default_factory=DiffSimConfig)
    isaac: IsaacConfig = field(default_factory=IsaacConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a nested dict (suitable for YAML dump)."""
        return dataclasses.asdict(self)  # type: ignore

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifexConfig":
        """Deserialise from a nested dict."""
        thermal_data: dict[str, Any] = data.get("thermal", {})
        cryst_data: dict[str, Any] = data.get("crystallinity", {})
        opt_data: dict[str, Any] = data.get("optimization", {})
        cell_data: dict[str, Any] = data.get("cell", {})
        diff_data: dict[str, Any] = data.get("diffsim", {})
        isaac_data: dict[str, Any] = data.get("isaac", {})
        return cls(
            thermal=ThermalConfig(**thermal_data),  # type: ignore
            crystallinity=CrystallinityConfig(**cryst_data),  # type: ignore
            optimization=OptimizationConfig(**opt_data),  # type: ignore
            cell=CellConfig(**cell_data),  # type: ignore
            diffsim=DiffSimConfig(**diff_data),  # type: ignore
            isaac=IsaacConfig(**isaac_data),  # type: ignore
        )
