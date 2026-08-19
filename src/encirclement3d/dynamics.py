"""Rate-limited inertial dynamics backend for the 3D containment task."""

from __future__ import annotations

from typing import Any

import numpy as np

from .environment import Encirclement3DEnv


class InertialEncirclement3DEnv(Encirclement3DEnv):
    """A lightweight translational backend with delay, drag, and mass variation.

    The public action remains a desired velocity so existing rule and CBF
    controllers can be compared without changing their API. Internally, the
    delayed desired velocity is converted to a thrust-like acceleration, drag
    is applied, and the state is integrated with semi-implicit Euler dynamics.
    """

    def __init__(self, config: dict[str, Any], obstacle_count: int, target_speed_scale: float = 1.0):
        super().__init__(config, obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
        dynamics = config.get("dynamics", {})
        self.action_delay_steps = int(dynamics.get("action_delay_steps", 1))
        self.velocity_response_time = float(dynamics.get("velocity_response_time", 0.35))
        self.nominal_mass = float(dynamics.get("defender_nominal_mass", 1.0))
        self.nominal_drag = float(dynamics.get("defender_nominal_drag", 0.35))
        self.mass_variation = float(dynamics.get("defender_mass_variation", 0.15))
        self.drag_variation = float(dynamics.get("defender_drag_variation", 0.20))
        self.defender_accelerations = np.zeros_like(self.defender_positions)
        self.defender_masses = np.ones(self.n_defenders, dtype=np.float64) * self.nominal_mass
        self.defender_drag_coefficients = np.ones(self.n_defenders, dtype=np.float64) * self.nominal_drag
        self.action_queue: list[np.ndarray] = []

    def reset(self, seed: int, record_history: bool = False) -> dict[str, Any]:
        observation = super().reset(seed=seed, record_history=record_history)
        self.defender_accelerations.fill(0.0)
        self.defender_masses = self.nominal_mass * (
            1.0 + self.rng.uniform(-self.mass_variation, self.mass_variation, size=self.n_defenders)
        )
        self.defender_drag_coefficients = self.nominal_drag * (
            1.0 + self.rng.uniform(-self.drag_variation, self.drag_variation, size=self.n_defenders)
        )
        zero_action = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.action_queue = [zero_action.copy() for _ in range(self.action_delay_steps)]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        observation = super().observe()
        observation.update(
            {
                "defender_accelerations": self.defender_accelerations.copy(),
                "defender_masses": self.defender_masses.copy(),
                "defender_drag_coefficients": self.defender_drag_coefficients.copy(),
                "action_delay_steps": self.action_delay_steps,
            }
        )
        return observation

    def _apply_defender_actions(self, defender_actions: np.ndarray) -> None:
        if self.action_delay_steps > 0:
            self.action_queue.append(defender_actions.copy())
            effective_action = self.action_queue.pop(0)
        else:
            effective_action = defender_actions

        max_acceleration = float(self.agents["defender_max_acceleration"])
        desired_acceleration = (
            effective_action - self.defender_velocities
        ) / max(self.velocity_response_time, 1e-6)
        drag_acceleration = -self.defender_drag_coefficients[:, None] * self.defender_velocities
        acceleration = (desired_acceleration + drag_acceleration) / self.defender_masses[:, None]
        acceleration_norm = np.linalg.norm(acceleration, axis=1, keepdims=True)
        acceleration *= np.minimum(1.0, max_acceleration / np.maximum(acceleration_norm, 1e-9))
        self.defender_accelerations = acceleration

        self.defender_velocities += self.defender_accelerations * self.dt
        self.defender_velocities = self._clip_rows(
            self.defender_velocities,
            float(self.agents["defender_max_speed"]),
        )
        self.defender_positions += self.defender_velocities * self.dt
        self._enforce_world_bounds(self.defender_positions, self.defender_velocities)
