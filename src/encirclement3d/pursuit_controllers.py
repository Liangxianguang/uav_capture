"""Rule and safety baselines for capture-radius pursuit-evasion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pursuit_env import CaptureRadiusPursuit3DEnv, TETRAHEDRON_DIRECTIONS, _unit


@dataclass(frozen=True)
class PursuitSafetyDiagnostics:
    action_correction_norm: float
    minimum_barrier_value: float


class PursuitCBFSafetyFilter:
    """A local-information CBF projection for obstacles, teammates, and bounds.

    The target is deliberately excluded from the barrier constraints because
    approaching it is the task objective. The filter sees the same obstacle and
    teammate data as the decentralized pursuit controllers.
    """

    def __init__(self, env: CaptureRadiusPursuit3DEnv) -> None:
        self.env = env
        self.gamma = float(env.task.get("cbf_gamma", 0.25))
        self.margin = float(env.pursuit["safety_margin"])

    def filter(
        self,
        desired_actions: np.ndarray,
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, PursuitSafetyDiagnostics]:
        desired = self.env._clip_rows(
            np.asarray(desired_actions, dtype=np.float64),
            float(self.env.agents["defender_max_speed"]),
        )
        safe = desired.copy()
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        radius = float(self.env.agents["drone_radius"])
        barriers: list[float] = []

        for _ in range(4):
            for index, position in enumerate(positions):
                for obstacle in self.env.obstacles:
                    clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                    barrier = clearance - radius - self.margin
                    lower_bound = -self.gamma * barrier / self.env.dt
                    projection = float(np.sum(normal * safe[index]))
                    if projection < lower_bound:
                        safe[index] += (lower_bound - projection) * normal
                    barriers.append(barrier)
                for axis in range(3):
                    lower_barrier = position[axis] - self.env.lower[axis] - radius - self.margin
                    upper_barrier = self.env.upper[axis] - position[axis] - radius - self.margin
                    if safe[index, axis] < -self.gamma * lower_barrier / self.env.dt:
                        safe[index, axis] = -self.gamma * lower_barrier / self.env.dt
                    if safe[index, axis] > self.gamma * upper_barrier / self.env.dt:
                        safe[index, axis] = self.gamma * upper_barrier / self.env.dt
                    barriers.extend([lower_barrier, upper_barrier])
            for first in range(self.env.n_defenders):
                for second in range(first + 1, self.env.n_defenders):
                    delta = positions[first] - positions[second]
                    distance = float(np.linalg.norm(delta))
                    normal = _unit(delta, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
                    barrier = distance - (2.0 * radius + self.margin)
                    lower_bound = -self.gamma * barrier / self.env.dt
                    relative_projection = float(np.sum(normal * (safe[first] - safe[second])))
                    if relative_projection < lower_bound:
                        correction = 0.5 * (lower_bound - relative_projection) * normal
                        safe[first] += correction
                        safe[second] -= correction
                    barriers.append(barrier)
            safe = self.env._clip_rows(safe, float(self.env.agents["defender_max_speed"]))

        return safe, PursuitSafetyDiagnostics(
            action_correction_norm=float(np.mean(np.linalg.norm(safe - desired, axis=1))),
            minimum_barrier_value=float(min(barriers)) if barriers else float("inf"),
        )


class _PursuitController:
    def __init__(self, env: CaptureRadiusPursuit3DEnv) -> None:
        self.env = env
        self.max_speed = float(env.agents["defender_max_speed"])
        self.gain = float(env.task.get("slot_tracking_gain", 4.0))
        self.obstacle_distance = float(env.pursuit["controller_obstacle_avoidance_distance"])
        self.obstacle_gain = float(env.pursuit["controller_obstacle_avoidance_gain"])
        self.inter_agent_distance = float(env.pursuit["controller_inter_agent_distance"])
        self.inter_agent_gain = float(env.pursuit["controller_inter_agent_gain"])

    def _avoidance(self, desired: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        corrected = np.asarray(desired, dtype=np.float64).copy()
        for index, position in enumerate(positions):
            for obstacle in self.env.obstacles:
                clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                if clearance < self.obstacle_distance:
                    corrected[index] += normal * (self.obstacle_distance - clearance) * self.obstacle_gain
            for other_index, other_position in enumerate(positions):
                if index == other_index:
                    continue
                delta = position - other_position
                distance = float(np.linalg.norm(delta))
                if distance < self.inter_agent_distance:
                    corrected[index] += (
                        _unit(delta, fallback=TETRAHEDRON_DIRECTIONS[index])
                        * (self.inter_agent_distance - distance)
                        * self.inter_agent_gain
                    )
        return self.env._clip_rows(corrected, self.max_speed)

    @staticmethod
    def _team_prediction(observation: dict[str, Any], horizon_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
        velocities = np.asarray(observation["target_belief_velocities"], dtype=np.float64)
        ages = np.asarray(observation["message_age_steps"], dtype=np.float64)
        weights = 1.0 / (1.0 + ages)
        weights /= np.sum(weights)
        position = np.sum(beliefs * weights[:, None], axis=0)
        velocity = np.sum(velocities * weights[:, None], axis=0)
        return position + horizon_seconds * velocity, velocity


class PurePursuitController(_PursuitController):
    """Greedy baseline that pursues each defender's own target belief."""

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
        desired = self.gain * (beliefs - positions)
        return self._avoidance(desired, observation)


class PredictionPursuitController(_PursuitController):
    """Constant-velocity target predictor followed by local pursuit."""

    def __init__(self, env: CaptureRadiusPursuit3DEnv, horizon_seconds: float = 0.45) -> None:
        super().__init__(env)
        self.horizon_seconds = float(horizon_seconds)

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
        velocities = np.asarray(observation["target_belief_velocities"], dtype=np.float64)
        predicted = beliefs + self.horizon_seconds * velocities
        desired = self.gain * (predicted - positions) + velocities
        return self._avoidance(desired, observation)


class DynamicEncirclementController(_PursuitController):
    """Use delayed observations to divide agents into one interceptor and blockers."""

    def __init__(self, env: CaptureRadiusPursuit3DEnv, horizon_seconds: float = 0.55) -> None:
        super().__init__(env)
        self.horizon_seconds = float(horizon_seconds)
        self.interceptor_id: int | None = None

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        target, target_velocity = self._team_prediction(observation, self.horizon_seconds)
        distances = np.linalg.norm(positions - target[None, :], axis=1)
        if self.interceptor_id is None:
            self.interceptor_id = int(np.argmin(distances))
        interceptor = self.interceptor_id
        perimeter = float(
            np.clip(
                0.55 * np.median(distances),
                float(self.env.pursuit["capture_radius"]) + 0.35,
                2.6,
            )
        )
        desired = np.zeros_like(positions)
        for index, position in enumerate(positions):
            if index == interceptor:
                target_point = target
            else:
                target_point = target + TETRAHEDRON_DIRECTIONS[index] * perimeter
            desired[index] = self.gain * (target_point - position) + target_velocity
        return self._avoidance(desired, observation)


class SafetyFilteredPursuitController:
    """Wrap a pursuit policy with the local-information CBF filter."""

    def __init__(self, controller: _PursuitController) -> None:
        self.controller = controller
        self.safety_filter = PursuitCBFSafetyFilter(controller.env)
        self.last_diagnostics = PursuitSafetyDiagnostics(0.0, float("inf"))

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        desired = self.controller.act(observation)
        safe, self.last_diagnostics = self.safety_filter.filter(desired, observation)
        return safe
