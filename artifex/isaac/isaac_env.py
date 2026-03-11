"""
Isaac Lab Environment for RL-Based Adaptive Pick Timing

Wraps the Newton single-cell scene as an Isaac Lab task environment
for reinforcement-learning policy training.  The agent controls
pick timing and gripper force, aiming to minimise cycle time while
constraining scuff risk and maintaining QA pass rates.

.. important::

    As of March 2026, Newton's integration into Isaac Lab is documented
    as an **experimental feature**.  The observation / action space
    wrappers below follow the documented Isaac Lab task API, but the
    Newton physics backend binding may undergo breaking changes.

    See: https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html

References
----------
* Isaac Lab — https://isaac-sim.github.io/IsaacLab/
* Newton integration — experimental feature docs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from artifex.properties import PET, DISC
from artifex.config import IsaacConfig, CellConfig
from artifex.cell.material_coupling import MaterialCoupling, HandlingRisk


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Observation / action spaces (defined independently of Isaac Lab
# so the module can be imported without Isaac Lab installed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ObservationSpec:
    """Observation space for the Artifex pick-timing task.

    Each observation is a vector of:
        [0]  disc_temperature          — °C, normalised to [0, 1]
        [1]  youngs_modulus_ratio      — E(T)/E_glassy, ∈ [0, 1]
        [2]  friction_coefficient      — from material coupling
        [3]  time_since_eject          — seconds, normalised
        [4]  robot_ee_distance_to_disc — metres
        [5]  disc_velocity_magnitude   — m/s
        [6]  grip_force_current        — N, normalised
        [7]  qa_contact_force_history  — max force in last N steps
    """

    dim: int = 8  # observation vector length
    low: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    high: np.ndarray = field(default_factory=lambda: np.ones(8, dtype=np.float32))


@dataclass
class ActionSpec:
    """Action space for the Artifex pick-timing task.

    Actions are continuous:
        [0]  pick_trigger         — ∈ [0, 1], threshold to initiate pick
        [1]  grip_force_fraction  — ∈ [0, 1], fraction of max gripper force
        [2]  transfer_speed_frac  — ∈ [0, 1], fraction of max transfer speed
    """

    dim: int = 3
    low: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    high: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain randomization config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class DomainRandomization:
    """Parameters for domain randomization during training.

    Attributes:
        thermal_noise_std: σ for eject-temperature perturbation (°C).
        material_variation: Fractional variation in PET properties.
        robot_calibration_noise: Positional noise for robot base (m).
        conveyor_speed_variation: Fractional variation in conveyor speed.
    """

    thermal_noise_std: float = 5.0  # °C
    material_variation: float = 0.05  # ± 5 %
    robot_calibration_noise: float = 0.002  # 2 mm
    conveyor_speed_variation: float = 0.10  # ± 10 %

    def sample_eject_temperature(
        self, base_temp: float, rng: np.random.Generator
    ) -> float:
        """Sample a randomised eject temperature."""
        return float(base_temp + rng.normal(0.0, self.thermal_noise_std))

    def sample_material_scale(self, rng: np.random.Generator) -> float:
        """Sample a multiplicative material-property scale factor."""
        return float(1.0 + rng.uniform(-self.material_variation, self.material_variation))

    def sample_robot_offset(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a 3-D robot base calibration offset (metres)."""
        return rng.normal(0.0, self.robot_calibration_noise, size=3).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reward function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_reward(
    cycle_time: float,
    max_contact_force: float,
    qa_passed: bool,
    handling_risk: HandlingRisk,
    target_cycle_time: float = 20.0,
    force_threshold: float = 15.0,
) -> float:
    """Compute the per-episode reward for the pick-timing task.

    Reward structure:
        +1.0 base reward for QA pass
        +0.5 bonus for cycle time < target
        −1.0 penalty for QA fail
        −0.5 penalty for HIGH handling risk at pick
        −0.3 penalty for excessive contact force

    Parameters
    ----------
    cycle_time : float
        Total cycle time in seconds.
    max_contact_force : float
        Peak contact force on groove area in Newtons.
    qa_passed : bool
        Whether the disc passed the QA gate.
    handling_risk : HandlingRisk
        Risk class at the moment of pick.
    target_cycle_time : float
        Target cycle time for bonus.
    force_threshold : float
        Force above which a penalty is applied.

    Returns
    -------
    float
        Scalar reward.
    """
    reward = 0.0

    # QA outcome
    if qa_passed:
        reward += 1.0
    else:
        reward -= 1.0

    # Cycle-time bonus
    if cycle_time < target_cycle_time:
        reward += 0.5 * (1.0 - cycle_time / target_cycle_time)

    # Risk penalty
    if handling_risk == HandlingRisk.HIGH:
        reward -= 0.5
    elif handling_risk == HandlingRisk.MEDIUM:
        reward -= 0.1

    # Contact-force penalty
    if max_contact_force > force_threshold:
        reward -= 0.3 * (max_contact_force - force_threshold) / force_threshold

    return reward


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Environment class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ArtifexIsaacEnv:
    """Isaac Lab task environment for Artifex pick-timing RL.

    This class defines the observation/action interface, reward function,
    reset logic, and domain randomization.  It is designed to be
    registered as an Isaac Lab task and backed by a Newton physics scene.

    .. note::

        The actual Isaac Lab base class (``DirectRLEnv``) should be
        subclassed once the Newton integration is stable.  This
        standalone version can be used for offline prototyping and
        policy-architecture validation.

    Parameters
    ----------
    isaac_config : IsaacConfig, optional
        RL environment configuration.
    cell_config : CellConfig, optional
        Newton cell configuration.
    seed : int
        Random seed for domain randomization.
    """

    def __init__(
        self,
        isaac_config: Optional[IsaacConfig] = None,
        cell_config: Optional[CellConfig] = None,
        seed: int = 42,
    ) -> None:
        self.isaac_cfg = isaac_config or IsaacConfig()
        self.cell_cfg = cell_config or CellConfig()
        self.rng = np.random.default_rng(seed)

        self.obs_spec = ObservationSpec()
        self.act_spec = ActionSpec()
        self.domain_rand = DomainRandomization(
            thermal_noise_std=self.isaac_cfg.domain_rand_thermal_noise,
            material_variation=self.isaac_cfg.domain_rand_material_var,
            robot_calibration_noise=self.isaac_cfg.domain_rand_robot_cal,
        )

        self._coupling = MaterialCoupling(device="cpu")
        self._step_count = 0
        self._episode_count = 0

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self.isaac_cfg.num_envs

    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environments and return initial observations.

        Parameters
        ----------
        env_ids : np.ndarray, optional
            Indices of environments to reset.  If None, resets all.

        Returns
        -------
        np.ndarray
            Initial observations, shape ``(num_envs, obs_dim)``.
        """
        n = self.num_envs if env_ids is None else len(env_ids)

        obs = np.zeros((n, self.obs_spec.dim), dtype=np.float32)

        for i in range(n):
            # Randomised eject temperature
            t_eject = self.domain_rand.sample_eject_temperature(
                self.cell_cfg.disc_eject_temperature, self.rng
            )
            mat = self._coupling.evaluate_scalar(t_eject)

            obs[i, 0] = (t_eject - PET.mold_temp_lo) / (
                PET.melt_temp_hi - PET.mold_temp_lo
            )  # normalised temperature
            obs[i, 1] = mat.youngs_modulus / PET.youngs_modulus_glassy
            obs[i, 2] = mat.friction_coefficient
            obs[i, 3] = 0.0  # time since eject
            obs[i, 4] = 0.3  # initial EE distance (arbitrary)
            obs[i, 5] = 0.0  # disc velocity
            obs[i, 6] = 0.0  # grip force
            obs[i, 7] = 0.0  # contact force history

        self._step_count = 0
        self._episode_count += 1
        return obs

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Take one environment step.

        Parameters
        ----------
        actions : np.ndarray
            Actions, shape ``(num_envs, act_dim)``.

        Returns
        -------
        tuple
            (observations, rewards, terminated, truncated, info)
        """
        n = self.num_envs
        obs = np.zeros((n, self.obs_spec.dim), dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        info: dict[str, Any] = {}

        self._step_count += 1

        for i in range(n):
            # Simplified step logic — placeholder for Newton stepping
            pick_trigger = actions[i, 0]
            grip_frac = actions[i, 1]

            grip_force = grip_frac * self.cell_cfg.gripper_max_force
            t_eject = self.cell_cfg.disc_eject_temperature
            mat = self._coupling.evaluate_scalar(t_eject)

            # Proxy observations
            obs[i, 0] = (t_eject - PET.mold_temp_lo) / (
                PET.melt_temp_hi - PET.mold_temp_lo
            )
            obs[i, 1] = mat.youngs_modulus / PET.youngs_modulus_glassy
            obs[i, 2] = mat.friction_coefficient
            obs[i, 3] = float(self._step_count) / self.isaac_cfg.episode_length
            obs[i, 6] = grip_force / self.cell_cfg.gripper_max_force

            # Proxy reward
            rewards[i] = compute_reward(
                cycle_time=self._step_count * 0.1,
                max_contact_force=grip_force,
                qa_passed=pick_trigger > 0.5,
                handling_risk=mat.risk,
            )

            # Truncation
            if self._step_count >= self.isaac_cfg.episode_length:
                truncated[i] = True

        return obs, rewards, terminated, truncated, info
