"""Partially observable 3D capture-radius pursuit-evasion environment.

This module is intentionally separate from the tetrahedral containment
benchmark. A capture is a geometric event: one defender reaches the configured
capture radius without a safety failure. The policy observation never exposes
the target's true state; the target belief is formed only from local detections
and delayed teammate messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

TETRAHEDRON_DIRECTIONS = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)
TETRAHEDRON_DIRECTIONS /= np.linalg.norm(TETRAHEDRON_DIRECTIONS, axis=1, keepdims=True)


@dataclass(frozen=True)
class CylinderObstacle:
    """Static obstacle used by the pursuit benchmark.

    The historical name is retained for compatibility.  ``shape='cylinder'``
    uses ``radius``; ``shape='box'`` additionally uses ``half_extents_xy``.
    Keeping one observation-compatible record lets the benchmark introduce
    boxes and wall segments without changing the actor input dimension.
    """

    center_xy: np.ndarray
    radius: float
    height: float
    shape: str = "cylinder"
    half_extents_xy: np.ndarray | None = None


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is None:
        return np.zeros_like(vector)
    return fallback.copy()


_PURSUIT_DEFAULTS: dict[str, Any] = {
    "capture_radius": 0.80,
    "spawn_distance": 4.80,
    "detection_range": 7.50,
    "visibility_cosine_threshold": -1.0,
    "detection_dropout_probability": 0.15,
    "observation_noise_std": 0.03,
    "message_delay_steps": 2,
    "message_dropout_probability": 0.05,
    "communication_link_dropout_probability": 0.0,
    "maximum_message_age_steps": 60,
    "observation_delay_steps": 0,
    "detection_loss_burst_probability": 0.0,
    "detection_loss_burst_duration_steps": 1,
    "observation_confidence_decay": 0.97,
    "observation_covariance_growth": 0.25,
    "include_prediction_features": False,
    "include_uncertainty_features": False,
    "prediction_horizon_seconds": 0.55,
    "prediction_uncertainty_base": 0.08,
    "target_defender_avoidance_distance": 2.20,
    "target_defender_avoidance_gain": 4.00,
    "target_obstacle_avoidance_distance": 2.20,
    "target_obstacle_avoidance_gain": 3.50,
    "target_boundary_margin": 1.20,
    "target_boundary_gain": 7.00,
    "target_heading_persistence": 0.55,
    "target_motion_mode": "flee_persistence",
    "target_turn_interval_steps": 12,
    "target_s_curve_amplitude": 0.85,
    "target_s_curve_frequency": 0.18,
    "target_burst_period_steps": 30,
    "target_burst_duration_steps": 8,
    "target_burst_speed_scale": 1.25,
    "target_flee_gain": 1.00,
    "target_vertical_gain": 0.20,
    "controller_obstacle_avoidance_distance": 2.00,
    "controller_obstacle_avoidance_gain": 2.80,
    "controller_inter_agent_distance": 1.00,
    "controller_inter_agent_gain": 1.20,
    "safety_margin": 0.20,
    "capture_bonus": 25.00,
    "collision_penalty": 15.00,
    "progress_reward_weight": 3.00,
    "distance_reward_weight": 0.12,
    "coverage_reward_weight": 0.15,
    "max_observation_obstacles": 3,
    "obstacle_profile": "cylinders",
    "map_seed_offset": 0,
}

_TARGET_MOTION_MODES = {"flee_persistence", "random_turn", "s_curve", "burst", "boundary_escape"}
_OBSTACLE_PROFILES = {"cylinders", "boxes", "walls", "narrow_channels", "mixed"}


def pursuit_settings(task: dict[str, Any]) -> dict[str, float | int | bool]:
    configured = task.get("pursuit", {})
    if not isinstance(configured, dict):
        raise ValueError("task.pursuit must be a mapping.")
    unknown = sorted(set(configured).difference(_PURSUIT_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown task.pursuit settings: {', '.join(unknown)}")
    settings = {**_PURSUIT_DEFAULTS, **configured}
    positive = {
        "capture_radius",
        "spawn_distance",
        "detection_range",
        "target_defender_avoidance_distance",
        "target_obstacle_avoidance_distance",
        "target_boundary_margin",
        "controller_obstacle_avoidance_distance",
        "controller_inter_agent_distance",
        "max_observation_obstacles",
        "prediction_horizon_seconds",
    }
    for name in positive:
        if float(settings[name]) <= 0.0:
            raise ValueError(f"task.pursuit.{name} must be positive.")
    if not -1.0 <= float(settings["visibility_cosine_threshold"]) <= 1.0:
        raise ValueError("task.pursuit.visibility_cosine_threshold must be in [-1, 1].")
    for name in (
        "message_dropout_probability",
        "communication_link_dropout_probability",
        "detection_dropout_probability",
        "detection_loss_burst_probability",
    ):
        if not 0.0 <= float(settings[name]) < 1.0:
            raise ValueError(f"task.pursuit.{name} must be in [0, 1).")
    if (
        int(settings["message_delay_steps"]) < 0
        or int(settings["observation_delay_steps"]) < 0
        or int(settings["maximum_message_age_steps"]) <= 0
    ):
        raise ValueError("Message/observation delays must be non-negative and maximum message age must be positive.")
    if int(settings["detection_loss_burst_duration_steps"]) <= 0:
        raise ValueError("task.pursuit.detection_loss_burst_duration_steps must be positive.")
    if not 0.0 < float(settings["observation_confidence_decay"]) <= 1.0:
        raise ValueError("task.pursuit.observation_confidence_decay must be in (0, 1].")
    if float(settings["observation_covariance_growth"]) < 0.0:
        raise ValueError("task.pursuit.observation_covariance_growth must be non-negative.")
    if float(settings["prediction_uncertainty_base"]) < 0.0:
        raise ValueError("task.pursuit.prediction_uncertainty_base must be non-negative.")
    if str(settings["target_motion_mode"]) not in _TARGET_MOTION_MODES:
        raise ValueError(
            "task.pursuit.target_motion_mode must be one of "
            + ", ".join(sorted(_TARGET_MOTION_MODES))
            + "."
        )
    if str(settings["obstacle_profile"]) not in _OBSTACLE_PROFILES:
        raise ValueError(
            "task.pursuit.obstacle_profile must be one of "
            + ", ".join(sorted(_OBSTACLE_PROFILES))
            + "."
        )
    for name in (
        "target_turn_interval_steps",
        "target_burst_period_steps",
        "target_burst_duration_steps",
    ):
        if int(settings[name]) <= 0:
            raise ValueError(f"task.pursuit.{name} must be positive.")
    for name in ("target_s_curve_amplitude", "target_s_curve_frequency", "target_burst_speed_scale"):
        if float(settings[name]) < 0.0:
            raise ValueError(f"task.pursuit.{name} must be non-negative.")
    if int(settings["target_burst_duration_steps"]) > int(settings["target_burst_period_steps"]):
        raise ValueError("target_burst_duration_steps cannot exceed target_burst_period_steps.")
    if int(settings["map_seed_offset"]) < 0:
        raise ValueError("task.pursuit.map_seed_offset must be non-negative.")
    return settings


@dataclass(frozen=True)
class PursuitEpisodeMetrics:
    minimum_target_distance: float
    nearest_defender: int
    collision: bool
    physical_target_contact: bool
    min_clearance: float


@dataclass(frozen=True)
class _BeliefPacket:
    """Timestamped local detection or delayed teammate message."""

    delivery_step: int
    receiver: int
    source: int
    timestamp_step: int
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    covariance: np.ndarray
    via_message: bool


class CaptureRadiusPursuit3DEnv:
    """Cooperative 3D pursuit task with partial target observations.

    The environment stores target ground truth internally for simulation and a
    centralized critic, but observe exposes only defender states, obstacle
    geometry, local target beliefs, visibility flags, and message age.
    """

    def __init__(self, config: dict[str, Any], obstacle_count: int, target_speed_scale: float = 1.0):
        self.config = config
        self.world = config["world"]
        self.agents = config["agents"]
        self.task = config["task"]
        self.pursuit = pursuit_settings(self.task)
        self.obstacle_count = int(obstacle_count)
        self.target_speed_scale = float(target_speed_scale)
        self.n_defenders = int(self.agents["defenders"])
        if self.n_defenders != 4:
            raise ValueError("CaptureRadiusPursuit3DEnv currently requires four homogeneous defenders.")

        self.dt = float(self.world["dt"])
        self.max_steps = int(self.world["max_steps"])
        half_extent = float(self.world["half_extent_xy"])
        self.lower = np.array([-half_extent, -half_extent, float(self.world["minimum_altitude"])], dtype=np.float64)
        self.upper = np.array([half_extent, half_extent, float(self.world["height"])], dtype=np.float64)
        self.rng = np.random.default_rng()

        self.obstacles: list[CylinderObstacle] = []
        self.defender_positions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.defender_velocities = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.target_position = np.zeros(3, dtype=np.float64)
        self.target_velocity = np.zeros(3, dtype=np.float64)
        self.target_escape_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.target_belief_positions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.target_belief_velocities = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.target_visible = np.zeros(self.n_defenders, dtype=bool)
        self.target_observation_confidence = np.zeros(self.n_defenders, dtype=np.float64)
        self.target_observation_timestamps = np.full(self.n_defenders, -1, dtype=np.int64)
        self.target_observation_covariance = np.zeros((self.n_defenders, 3, 3), dtype=np.float64)
        self.detection_loss_burst_remaining = np.zeros(self.n_defenders, dtype=np.int64)
        self.message_age_steps = np.full(self.n_defenders, int(self.pursuit["maximum_message_age_steps"]), dtype=np.int64)
        self._message_queue: list[_BeliefPacket] = []

        self.step_count = 0
        self.collision_steps = 0
        self.world_violation_steps = 0
        self.min_clearance = float("inf")
        self.capture_time_seconds: float | None = None
        self.capturing_defender_id: int | None = None
        self.history: list[dict[str, np.ndarray | float | int]] = []

    def reset(self, seed: int, record_history: bool = False) -> dict[str, Any]:
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.collision_steps = 0
        self.world_violation_steps = 0
        self.min_clearance = float("inf")
        self.capture_time_seconds = None
        self.capturing_defender_id = None
        self.history = []
        self._message_queue = []

        self.target_position = np.array(
            [
                self.rng.uniform(-2.0, 2.0),
                self.rng.uniform(-2.0, 2.0),
                self.rng.uniform(3.0, min(7.0, self.upper[2] - 1.5)),
            ],
            dtype=np.float64,
        )
        self.target_velocity.fill(0.0)
        self.target_escape_direction = _unit(
            self.rng.normal(0.0, 1.0, size=3),
            fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        self.defender_positions = self.target_position + TETRAHEDRON_DIRECTIONS * float(self.pursuit["spawn_distance"])
        self.defender_positions += self.rng.normal(0.0, 0.20, size=self.defender_positions.shape)
        self.defender_positions = np.clip(self.defender_positions, self.lower + 0.6, self.upper - 0.6)
        self.defender_velocities.fill(0.0)
        map_seed_offset = int(self.pursuit["map_seed_offset"])
        if map_seed_offset:
            map_rng = self.rng
            self.rng = np.random.default_rng(seed + map_seed_offset)
            self.obstacles = self._sample_obstacles()
            self.rng = map_rng
        else:
            self.obstacles = self._sample_obstacles()

        self.target_belief_positions[:] = 0.0
        self.target_belief_velocities[:] = 0.0
        self.message_age_steps[:] = int(self.pursuit["maximum_message_age_steps"])
        self.target_observation_confidence.fill(0.0)
        self.target_observation_timestamps.fill(-1)
        self.target_observation_covariance[:] = np.eye(3, dtype=np.float64)[None, :, :] * float(
            self.pursuit["observation_covariance_growth"]
        )
        self.detection_loss_burst_remaining.fill(0)
        self._update_target_beliefs()
        if record_history:
            self._record_history()
        return self.observe()

    def observe(self) -> dict[str, Any]:
        """Return policy-safe partial observations without target ground truth."""
        predicted_positions, predicted_uncertainties = self._predict_target_beliefs()
        return {
            "defender_positions": self.defender_positions.copy(),
            "defender_velocities": self.defender_velocities.copy(),
            "obstacles": [
                {
                    "center_xy": obstacle.center_xy.copy(),
                    "radius": float(obstacle.radius),
                    "height": float(obstacle.height),
                    "shape": obstacle.shape,
                    "half_extents_xy": (
                        None
                        if obstacle.half_extents_xy is None
                        else obstacle.half_extents_xy.copy()
                    ),
                }
                for obstacle in self.obstacles
            ],
            "target_belief_positions": self.target_belief_positions.copy(),
            "target_belief_velocities": self.target_belief_velocities.copy(),
            "target_visible": self.target_visible.copy(),
            "target_observation_confidence": self.target_observation_confidence.copy(),
            "target_observation_timestamps": self.target_observation_timestamps.copy(),
            "target_observation_covariance": self.target_observation_covariance.copy(),
            "target_observation_age_steps": np.maximum(
                self.step_count - self.target_observation_timestamps,
                0,
            ).astype(np.int64),
            "message_age_steps": self.message_age_steps.copy(),
            "target_prediction_positions": predicted_positions,
            "target_prediction_uncertainties": predicted_uncertainties,
            "step": int(self.step_count),
        }

    def centralized_state(self) -> np.ndarray:
        """Training-only global state for a centralized critic."""
        extent = float(self.world["half_extent_xy"])
        max_obstacles = int(self.pursuit["max_observation_obstacles"])
        obstacle_features = np.zeros((max_obstacles, 5), dtype=np.float32)
        for index, obstacle in enumerate(sorted(self.obstacles, key=lambda item: float(item.radius))[:max_obstacles]):
            obstacle_features[index] = np.array(
                [
                    obstacle.center_xy[0] / extent,
                    obstacle.center_xy[1] / extent,
                    obstacle.radius / extent,
                    obstacle.height / extent,
                    1.0,
                ],
                dtype=np.float32,
            )
        values = np.concatenate(
            [
                (self.defender_positions / extent).reshape(-1),
                (self.defender_velocities / float(self.agents["defender_max_speed"])).reshape(-1),
                self.target_position / extent,
                self.target_velocity / float(self.agents["target_max_speed"]),
                obstacle_features.reshape(-1),
                np.array([self.step_count / max(self.max_steps, 1)], dtype=np.float32),
            ]
        )
        return values.astype(np.float32)

    def policy_observations(self, observation: dict[str, Any] | None = None) -> np.ndarray:
        """Encode fixed-size decentralized observations for the shared actor."""
        current = self.observe() if observation is None else observation
        positions = np.asarray(current["defender_positions"], dtype=np.float32)
        velocities = np.asarray(current["defender_velocities"], dtype=np.float32)
        beliefs = np.asarray(current["target_belief_positions"], dtype=np.float32)
        belief_velocities = np.asarray(current["target_belief_velocities"], dtype=np.float32)
        visible = np.asarray(current["target_visible"], dtype=np.float32)
        confidence = np.asarray(current["target_observation_confidence"], dtype=np.float32)
        covariance = np.asarray(current["target_observation_covariance"], dtype=np.float32)
        message_age = np.asarray(current["message_age_steps"], dtype=np.float32)
        prediction_positions = np.asarray(current["target_prediction_positions"], dtype=np.float32)
        prediction_uncertainties = np.asarray(current["target_prediction_uncertainties"], dtype=np.float32)
        obstacles = list(current["obstacles"])
        extent = float(self.world["half_extent_xy"])
        max_obstacles = int(self.pursuit["max_observation_obstacles"])
        rows: list[np.ndarray] = []
        for index in range(self.n_defenders):
            teammate_indices = [other for other in range(self.n_defenders) if other != index]
            relative_teammates = (positions[teammate_indices] - positions[index]).reshape(-1) / extent
            relative_teammate_velocities = (
                velocities[teammate_indices] - velocities[index]
            ).reshape(-1) / float(self.agents["defender_max_speed"])
            nearest = sorted(
                obstacles,
                key=lambda obstacle: float(
                    np.linalg.norm(np.asarray(obstacle["center_xy"], dtype=np.float32) - positions[index, :2])
                    - float(obstacle["radius"])
                ),
            )[:max_obstacles]
            obstacle_features = np.zeros((max_obstacles, 5), dtype=np.float32)
            for obstacle_index, obstacle in enumerate(nearest):
                center = np.asarray(obstacle["center_xy"], dtype=np.float32)
                obstacle_features[obstacle_index] = np.array(
                    [
                        (center[0] - positions[index, 0]) / extent,
                        (center[1] - positions[index, 1]) / extent,
                        (0.5 * float(obstacle["height"]) - positions[index, 2]) / extent,
                        float(obstacle["radius"]) / extent,
                        float(obstacle["height"]) / extent,
                    ],
                    dtype=np.float32,
                )
            rows.append(
                np.concatenate(
                    [
                        velocities[index] / float(self.agents["defender_max_speed"]),
                        (beliefs[index] - positions[index]) / extent,
                        belief_velocities[index] / float(self.agents["target_max_speed"]),
                        np.array(
                            [
                                visible[index],
                                min(
                                    message_age[index] / float(self.pursuit["maximum_message_age_steps"]),
                                    1.0,
                                ),
                            ],
                            dtype=np.float32,
                        ),
                        (
                            np.concatenate(
                                [
                                    np.array([confidence[index]], dtype=np.float32),
                                    np.clip(np.diag(covariance[index]) / max(extent**2, 1e-9), 0.0, 1.0),
                                ]
                            )
                            if bool(self.pursuit["include_uncertainty_features"])
                            else np.empty(0, dtype=np.float32)
                        ),
                        (
                            np.concatenate(
                                [
                                    (prediction_positions[index] - positions[index]) / extent,
                                    np.array([prediction_uncertainties[index] / extent], dtype=np.float32),
                                ]
                            )
                            if bool(self.pursuit["include_prediction_features"])
                            else np.empty(0, dtype=np.float32)
                        ),
                        relative_teammates,
                        relative_teammate_velocities,
                        obstacle_features.reshape(-1),
                    ]
                )
            )
        values = np.stack(rows).astype(np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError("Partial observation encoder emitted a non-finite value.")
        return values

    def prediction_feature_slice(self) -> slice:
        """Return the four-column prediction block in the actor observation.

        The block is deliberately fixed-width: three relative predicted
        position values followed by one scalar uncertainty. A learned
        predictor can replace this block without changing the frozen actor
        input dimension.
        """
        if not bool(self.pursuit["include_prediction_features"]):
            raise ValueError("Prediction features are disabled in this environment configuration.")
        start = 3 + 3 + 3 + 2
        if bool(self.pursuit["include_uncertainty_features"]):
            start += 4
        return slice(start, start + 4)

    def step(
        self,
        defender_actions: np.ndarray,
        record_history: bool = False,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        actions = np.asarray(defender_actions, dtype=np.float64)
        if actions.shape != (self.n_defenders, 3):
            raise ValueError(f"Expected actions with shape {(self.n_defenders, 3)}, got {actions.shape}.")
        previous_distance = self._target_distances().min()
        actions = self._clip_rows(actions, float(self.agents["defender_max_speed"]))
        self._apply_defender_actions(actions)

        target_action = self._target_action()
        self.target_velocity = self._move_toward_velocity(
            self.target_velocity[None, :],
            target_action[None, :],
            max_delta=float(self.agents["target_max_acceleration"]) * self.dt,
        )[0]
        self.target_position += self.target_velocity * self.dt
        self._enforce_world_bounds(self.target_position[None, :], self.target_velocity[None, :])

        self.step_count += 1
        self._update_target_beliefs()
        metrics = self._metrics()
        self.min_clearance = min(self.min_clearance, metrics.min_clearance)
        if metrics.collision:
            self.collision_steps += 1

        capture_event = bool(metrics.minimum_target_distance <= float(self.pursuit["capture_radius"]))
        # Boundary violations are safety failures just like obstacle and
        # teammate collisions.  The clamped state is useful for continuing a
        # diagnostic rollout, but it must not turn an earlier violation into a
        # Safe Capture later in the episode.
        safety_failure = bool(metrics.collision or self.world_violation_steps > 0)
        safe_capture = bool(capture_event and not safety_failure)
        if safe_capture:
            self.capture_time_seconds = float(self.step_count * self.dt)
            self.capturing_defender_id = int(metrics.nearest_defender)

        if safety_failure:
            termination_reason = "safety_failure"
        elif safe_capture:
            termination_reason = "safe_capture"
        else:
            termination_reason = "running"
        terminated = bool(safety_failure or safe_capture)
        truncated = bool(not terminated and self.step_count >= self.max_steps)
        if truncated:
            termination_reason = "timeout"

        progress = float(previous_distance - metrics.minimum_target_distance)
        coverage = self._coverage_score()
        reward_components = {
            "progress": float(self.pursuit["progress_reward_weight"]) * progress,
            "distance": -float(self.pursuit["distance_reward_weight"]) * metrics.minimum_target_distance,
            "coverage": float(self.pursuit["coverage_reward_weight"]) * coverage,
            "capture": float(self.pursuit["capture_bonus"]) if safe_capture else 0.0,
            "safety": -float(self.pursuit["collision_penalty"]) if safety_failure else 0.0,
        }
        reward = float(sum(reward_components.values()))

        if record_history:
            self._record_history()
        info = {
            "success": safe_capture,
            "safe_capture_success": safe_capture,
            "capture_event": capture_event,
            "capture_time_seconds": self.capture_time_seconds,
            "capturing_defender_id": self.capturing_defender_id,
            "nearest_target_distance": float(metrics.minimum_target_distance),
            "nearest_defender": int(metrics.nearest_defender),
            "relative_speed_at_capture": (
                float(np.linalg.norm(self.defender_velocities[metrics.nearest_defender] - self.target_velocity))
                if capture_event
                else None
            ),
            "collision": safety_failure,
            "collision_steps": int(self.collision_steps),
            "physical_target_contact": bool(metrics.physical_target_contact),
            "world_violation_steps": int(self.world_violation_steps),
            "min_clearance": float(metrics.min_clearance),
            "min_clearance_so_far": float(self.min_clearance),
            "termination_reason": termination_reason,
            "reward_components": reward_components,
            "target_visible_fraction": float(np.mean(self.target_visible)),
            "mean_message_age_steps": float(np.mean(self.message_age_steps)),
            "mean_observation_confidence": float(np.mean(self.target_observation_confidence)),
            "mean_observation_age_steps": float(
                np.mean(np.maximum(self.step_count - self.target_observation_timestamps, 0))
            ),
            "mean_observation_covariance_trace": float(
                np.mean(np.trace(self.target_observation_covariance, axis1=1, axis2=2))
            ),
            "capture_radius": float(self.pursuit["capture_radius"]),
        }
        return self.observe(), reward, terminated, truncated, info

    def _apply_defender_actions(self, actions: np.ndarray) -> None:
        # The first pursuit benchmark is a velocity-level task. This keeps the
        # CBF action constraints and executed motion identical; action delay
        # and vehicle dynamics belong to the later PyBullet transfer gate.
        self.defender_velocities = actions.copy()
        self.defender_positions += self.defender_velocities * self.dt
        self._enforce_world_bounds(self.defender_positions, self.defender_velocities)

    def _target_action(self) -> np.ndarray:
        desired = float(self.pursuit["target_heading_persistence"]) * self.target_escape_direction
        for defender_position in self.defender_positions:
            delta = self.target_position - defender_position
            distance = float(np.linalg.norm(delta))
            if distance < float(self.pursuit["target_defender_avoidance_distance"]):
                desired += (
                    _unit(delta)
                    * (float(self.pursuit["target_defender_avoidance_distance"]) - distance)
                    * float(self.pursuit["target_defender_avoidance_gain"])
                )
        for obstacle in self.obstacles:
            clearance, normal = self._cylinder_clearance_and_normal(self.target_position, obstacle)
            if clearance < float(self.pursuit["target_obstacle_avoidance_distance"]):
                desired += (
                    normal
                    * (float(self.pursuit["target_obstacle_avoidance_distance"]) - clearance)
                    * float(self.pursuit["target_obstacle_avoidance_gain"])
                )
        defender_centroid = self.defender_positions.mean(axis=0)
        desired += float(self.pursuit["target_flee_gain"]) * _unit(
            self.target_position - defender_centroid,
            fallback=self.target_escape_direction,
        )
        desired[2] += float(self.pursuit["target_vertical_gain"]) * np.sin(0.11 * self.step_count)
        margin = float(self.pursuit["target_boundary_margin"])
        for axis in range(3):
            if self.target_position[axis] < self.lower[axis] + margin:
                desired[axis] += float(self.pursuit["target_boundary_gain"])
            if self.target_position[axis] > self.upper[axis] - margin:
                desired[axis] -= float(self.pursuit["target_boundary_gain"])
        mode = str(self.pursuit["target_motion_mode"])
        if mode == "random_turn" and self.step_count > 0 and self.step_count % int(self.pursuit["target_turn_interval_steps"]) == 0:
            self.target_escape_direction = _unit(
                self.rng.normal(0.0, 1.0, size=3),
                fallback=self.target_escape_direction,
            )
            desired = 0.35 * desired + self.target_escape_direction
        elif mode == "s_curve":
            lateral = np.array(
                [-self.target_escape_direction[1], self.target_escape_direction[0], 0.0],
                dtype=np.float64,
            )
            lateral = _unit(lateral, fallback=np.array([0.0, 1.0, 0.0]))
            desired += lateral * float(self.pursuit["target_s_curve_amplitude"]) * np.sin(
                float(self.pursuit["target_s_curve_frequency"]) * self.step_count
            )
        elif mode == "burst":
            burst_period = int(self.pursuit["target_burst_period_steps"])
            burst_duration = int(self.pursuit["target_burst_duration_steps"])
            if self.step_count % burst_period < burst_duration:
                desired *= float(self.pursuit["target_burst_speed_scale"])
        elif mode == "boundary_escape":
            boundary_direction = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                distance_to_lower = self.target_position[axis] - self.lower[axis]
                distance_to_upper = self.upper[axis] - self.target_position[axis]
                if distance_to_lower < distance_to_upper:
                    boundary_direction[axis] -= 1.0 / max(distance_to_lower, 0.5)
                else:
                    boundary_direction[axis] += 1.0 / max(distance_to_upper, 0.5)
            desired += 0.35 * _unit(boundary_direction, fallback=self.target_escape_direction)

        direction = _unit(desired, fallback=self.target_escape_direction)
        self.target_escape_direction = direction
        speed_scale = self.target_speed_scale
        if mode == "burst":
            burst_period = int(self.pursuit["target_burst_period_steps"])
            burst_duration = int(self.pursuit["target_burst_duration_steps"])
            if self.step_count % burst_period < burst_duration:
                speed_scale *= float(self.pursuit["target_burst_speed_scale"])
        return direction * float(self.agents["target_max_speed"]) * speed_scale

    def _update_target_beliefs(self) -> None:
        self.target_visible[:] = False
        pending: list[_BeliefPacket] = []
        for packet in self._message_queue:
            if packet.delivery_step <= self.step_count:
                receiver = packet.receiver
                if packet.timestamp_step >= int(self.target_observation_timestamps[receiver]):
                    self.target_belief_positions[receiver] = packet.position
                    self.target_belief_velocities[receiver] = packet.velocity
                    self.target_observation_confidence[receiver] = float(packet.confidence)
                    self.target_observation_covariance[receiver] = packet.covariance
                    self.target_observation_timestamps[receiver] = int(packet.timestamp_step)
                    self.message_age_steps[receiver] = min(
                        max(self.step_count - packet.timestamp_step, 0),
                        int(self.pursuit["maximum_message_age_steps"]),
                    )
            else:
                pending.append(packet)
        self._message_queue = pending

        for index in range(self.n_defenders):
            if self._target_is_visible(index):
                noise = self.rng.normal(0.0, float(self.pursuit["observation_noise_std"]), size=3)
                position = self.target_position + noise
                velocity = self.target_velocity + self.rng.normal(
                    0.0,
                    float(self.pursuit["observation_noise_std"]) / max(self.dt, 1e-9),
                    size=3,
                )
                self.target_visible[index] = True
                distance = float(np.linalg.norm(self.target_position - self.defender_positions[index]))
                confidence = float(
                    np.clip(
                        1.0 - distance / max(float(self.pursuit["detection_range"]), 1e-9),
                        0.05,
                        1.0,
                    )
                )
                covariance = np.eye(3, dtype=np.float64) * max(
                    float(self.pursuit["observation_noise_std"]) ** 2,
                    1e-8,
                )
                packet = _BeliefPacket(
                    delivery_step=self.step_count + int(self.pursuit["observation_delay_steps"]),
                    receiver=index,
                    source=index,
                    timestamp_step=self.step_count,
                    position=position.copy(),
                    velocity=velocity.copy(),
                    confidence=confidence,
                    covariance=covariance,
                    via_message=False,
                )
                if packet.delivery_step <= self.step_count:
                    self._deliver_belief_packet(packet)
                else:
                    self._message_queue.append(packet)
                for receiver in range(self.n_defenders):
                    if receiver == index:
                        continue
                    if (
                        self.rng.random() < float(self.pursuit["message_dropout_probability"])
                        or self.rng.random() < float(self.pursuit["communication_link_dropout_probability"])
                    ):
                        continue
                    self._message_queue.append(
                        _BeliefPacket(
                            delivery_step=self.step_count + int(self.pursuit["message_delay_steps"]),
                            receiver=receiver,
                            source=index,
                            timestamp_step=self.step_count,
                            position=position.copy(),
                            velocity=velocity.copy(),
                            confidence=confidence,
                            covariance=covariance.copy(),
                            via_message=True,
                        )
                    )
            else:
                self.target_belief_positions[index] += self.target_belief_velocities[index] * self.dt
                self.message_age_steps[index] = min(
                    self.message_age_steps[index] + 1,
                    int(self.pursuit["maximum_message_age_steps"]),
                )
                self.target_observation_confidence[index] *= float(self.pursuit["observation_confidence_decay"])
                self.target_observation_covariance[index] += np.eye(3, dtype=np.float64) * float(
                    self.pursuit["observation_covariance_growth"]
                )

    def _deliver_belief_packet(self, packet: _BeliefPacket) -> None:
        """Apply one packet only if it is newer than the receiver's belief."""
        receiver = int(packet.receiver)
        if packet.timestamp_step < int(self.target_observation_timestamps[receiver]):
            return
        self.target_belief_positions[receiver] = packet.position
        self.target_belief_velocities[receiver] = packet.velocity
        self.target_observation_confidence[receiver] = float(packet.confidence)
        self.target_observation_covariance[receiver] = packet.covariance.copy()
        self.target_observation_timestamps[receiver] = int(packet.timestamp_step)
        self.message_age_steps[receiver] = min(
            max(self.step_count - packet.timestamp_step, 0),
            int(self.pursuit["maximum_message_age_steps"]),
        )

    def _predict_target_beliefs(self) -> tuple[np.ndarray, np.ndarray]:
        """Predict each local target belief without reading simulator target truth.

        This conservative constant-velocity predictor is the V1 interface for
        trajectory prediction. A learned predictor can replace its output as
        long as it consumes only the same per-defender belief history.
        """
        horizon = float(self.pursuit["prediction_horizon_seconds"])
        positions = self.target_belief_positions + horizon * self.target_belief_velocities
        ages = self.message_age_steps.astype(np.float64) * self.dt
        uncertainty = float(self.pursuit["prediction_uncertainty_base"]) + ages * float(
            self.agents["target_max_speed"]
        )
        return positions.copy(), uncertainty.astype(np.float64, copy=False)

    def _target_is_visible(self, defender_index: int) -> bool:
        if self.detection_loss_burst_remaining[defender_index] > 0:
            self.detection_loss_burst_remaining[defender_index] -= 1
            return False
        if self.rng.random() < float(self.pursuit["detection_loss_burst_probability"]):
            self.detection_loss_burst_remaining[defender_index] = max(
                int(self.pursuit["detection_loss_burst_duration_steps"]) - 1,
                0,
            )
            return False
        if self.rng.random() < float(self.pursuit["detection_dropout_probability"]):
            return False
        origin = self.defender_positions[defender_index]
        delta = self.target_position - origin
        distance = float(np.linalg.norm(delta))
        if distance > float(self.pursuit["detection_range"]):
            return False
        velocity = self.defender_velocities[defender_index]
        if np.linalg.norm(velocity) > 1e-6:
            heading_cosine = float(np.sum(_unit(velocity) * _unit(delta)))
            if heading_cosine < float(self.pursuit["visibility_cosine_threshold"]):
                return False
        return not any(self._segment_blocked_by_cylinder(origin, self.target_position, obstacle) for obstacle in self.obstacles)

    @staticmethod
    def _segment_blocked_by_cylinder(start: np.ndarray, end: np.ndarray, obstacle: CylinderObstacle) -> bool:
        if obstacle.shape != "cylinder":
            for fraction in np.linspace(0.0, 1.0, 41):
                point = start + fraction * (end - start)
                clearance, _normal = CaptureRadiusPursuit3DEnv._box_clearance_and_normal(point, obstacle)
                if clearance <= 0.0:
                    return True
            return False
        direction_xy = end[:2] - start[:2]
        squared_length = float(np.sum(direction_xy * direction_xy))
        if squared_length < 1e-12:
            return False
        interpolation = float(
            np.clip(
                np.sum((obstacle.center_xy - start[:2]) * direction_xy) / squared_length,
                0.0,
                1.0,
            )
        )
        closest_xy = start[:2] + interpolation * direction_xy
        if float(np.linalg.norm(closest_xy - obstacle.center_xy)) > obstacle.radius:
            return False
        height_at_closest = float(start[2] + interpolation * (end[2] - start[2]))
        return 0.0 <= height_at_closest <= obstacle.height

    def _metrics(self) -> PursuitEpisodeMetrics:
        target_distances = self._target_distances()
        nearest_defender = int(np.argmin(target_distances))
        radius = float(self.agents["drone_radius"])
        clearances: list[float] = []
        for position in self.defender_positions:
            for obstacle in self.obstacles:
                clearance, _normal = self._cylinder_clearance_and_normal(position, obstacle)
                clearances.append(clearance - radius)
        for first in range(self.n_defenders):
            for second in range(first + 1, self.n_defenders):
                clearances.append(
                    float(np.linalg.norm(self.defender_positions[first] - self.defender_positions[second]) - 2.0 * radius)
                )
        min_clearance = float(min(clearances)) if clearances else float("inf")
        return PursuitEpisodeMetrics(
            minimum_target_distance=float(target_distances[nearest_defender]),
            nearest_defender=nearest_defender,
            collision=bool(min_clearance < 0.0),
            physical_target_contact=bool(float(target_distances[nearest_defender]) <= 2.0 * radius),
            min_clearance=min_clearance,
        )

    def _coverage_score(self) -> float:
        vectors = self.defender_positions - self.target_position[None, :]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        # Four defenders make this tiny pairwise calculation. Elementwise
        # summation avoids dispatching to a BLAS/OpenMP runtime during a
        # PyTorch training process.
        pairwise = np.sum(vectors[:, None, :] * vectors[None, :, :], axis=2)
        upper = pairwise[np.triu_indices(self.n_defenders, k=1)]
        return float(np.clip(-np.mean(upper), -1.0, 1.0))

    def _target_distances(self) -> np.ndarray:
        return np.linalg.norm(self.defender_positions - self.target_position[None, :], axis=1)

    def _sample_obstacles(self) -> list[CylinderObstacle]:
        obstacles: list[CylinderObstacle] = []
        protected_points = np.vstack([self.defender_positions, self.target_position[None, :]])
        profile = str(self.pursuit["obstacle_profile"])
        for _ in range(self.obstacle_count):
            for _attempt in range(200):
                shape = profile
                if profile == "mixed":
                    shape = str(self.rng.choice(["cylinder", "box", "walls"]))
                elif profile == "narrow_channels":
                    shape = "wall"
                if shape == "cylinders":
                    shape = "cylinder"
                if shape == "boxes":
                    shape = "box"
                if shape == "walls":
                    shape = "wall"
                radius = float(self.rng.uniform(0.65, 1.15))
                height = float(self.rng.uniform(3.0, 7.0))
                center_xy = self.rng.uniform(-7.5, 7.5, size=2)
                half_extents_xy: np.ndarray | None = None
                if shape == "box":
                    half_extents_xy = self.rng.uniform(0.65, 1.5, size=2).astype(np.float64)
                    radius = float(np.max(half_extents_xy))
                elif shape == "wall":
                    long_extent = float(self.rng.uniform(3.0, 5.0))
                    short_extent = float(self.rng.uniform(0.25, 0.45))
                    if self.rng.random() < 0.5:
                        half_extents_xy = np.array([long_extent, short_extent], dtype=np.float64)
                    else:
                        half_extents_xy = np.array([short_extent, long_extent], dtype=np.float64)
                    radius = short_extent
                    if profile == "narrow_channels":
                        long_extent = float(self.rng.uniform(3.5, 5.5))
                        short_extent = float(self.rng.uniform(0.30, 0.40))
                        half_extents_xy = np.array([long_extent, short_extent], dtype=np.float64)
                        radius = short_extent
                candidate = CylinderObstacle(
                    center_xy=center_xy,
                    radius=radius,
                    height=height,
                    shape="cylinder" if shape == "cylinder" else shape,
                    half_extents_xy=half_extents_xy,
                )
                if self._obstacle_clear_of_points(candidate, protected_points) and all(
                    self._obstacle_horizontal_separation(candidate, existing) >= 1.0
                    for existing in obstacles
                ):
                    obstacles.append(candidate)
                    break
            else:
                raise RuntimeError("Unable to sample a non-overlapping pursuit obstacle layout.")
        return obstacles

    def _obstacle_clear_of_points(self, obstacle: CylinderObstacle, points: np.ndarray) -> bool:
        clearances = [self._obstacle_clearance(point, obstacle) for point in points]
        return bool(all(clearance > 1.8 for clearance in clearances))

    @staticmethod
    def _box_clearance_and_normal(position: np.ndarray, obstacle: CylinderObstacle) -> tuple[float, np.ndarray]:
        if obstacle.half_extents_xy is None:
            raise ValueError("Box/wall obstacle requires half_extents_xy.")
        center = np.array([obstacle.center_xy[0], obstacle.center_xy[1], obstacle.height * 0.5], dtype=np.float64)
        half = np.array([obstacle.half_extents_xy[0], obstacle.half_extents_xy[1], obstacle.height * 0.5], dtype=np.float64)
        delta = np.asarray(position, dtype=np.float64) - center
        signed = np.abs(delta) - half
        outside = np.maximum(signed, 0.0)
        outside_norm = float(np.linalg.norm(outside))
        if outside_norm > 1e-9:
            normal = outside * np.sign(delta)
            return outside_norm, _unit(normal, fallback=np.array([1.0, 0.0, 0.0]))
        penetration = float(np.min(-signed))
        axis = int(np.argmin(-signed))
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0 if delta[axis] >= 0.0 else -1.0
        return -penetration, normal

    def _obstacle_clearance(self, position: np.ndarray, obstacle: CylinderObstacle) -> float:
        if obstacle.shape == "cylinder":
            clearance, _normal = self._cylinder_clearance_and_normal(position, obstacle)
            return float(clearance)
        clearance, _normal = self._box_clearance_and_normal(position, obstacle)
        return float(clearance)

    @staticmethod
    def _obstacle_horizontal_separation(first: CylinderObstacle, second: CylinderObstacle) -> float:
        first_half = (
            np.array([first.radius, first.radius], dtype=np.float64)
            if first.half_extents_xy is None
            else np.asarray(first.half_extents_xy, dtype=np.float64)
        )
        second_half = (
            np.array([second.radius, second.radius], dtype=np.float64)
            if second.half_extents_xy is None
            else np.asarray(second.half_extents_xy, dtype=np.float64)
        )
        delta = np.abs(first.center_xy - second.center_xy) - first_half - second_half
        gap = np.maximum(delta, 0.0)
        if np.any(delta <= 0.0):
            # If the axis-aligned footprints overlap in one direction, use
            # the separating gap in the other direction.  This is less
            # conservative than bounding every wall by a large circle and
            # keeps narrow-channel maps sampleable.
            return float(np.max(gap))
        return float(np.linalg.norm(gap))

    def _cylinder_clearance_and_normal(self, position: np.ndarray, obstacle: CylinderObstacle) -> tuple[float, np.ndarray]:
        if obstacle.shape != "cylinder":
            return self._box_clearance_and_normal(position, obstacle)
        xy_delta = position[:2] - obstacle.center_xy
        xy_norm = float(np.linalg.norm(xy_delta))
        radial_normal = _unit(
            np.array([xy_delta[0], xy_delta[1], 0.0]),
            fallback=np.array([1.0, 0.0, 0.0]),
        )
        radial_gap = xy_norm - obstacle.radius
        if 0.0 <= position[2] <= obstacle.height:
            return radial_gap, radial_normal
        nearest_z = 0.0 if position[2] < 0.0 else obstacle.height
        vertical_gap = abs(position[2] - nearest_z)
        if radial_gap <= 0.0:
            normal = np.array([0.0, 0.0, -1.0 if position[2] < 0.0 else 1.0])
            return vertical_gap, normal
        clearance = float(np.hypot(radial_gap, vertical_gap))
        normal = _unit(radial_normal * radial_gap + np.array([0.0, 0.0, position[2] - nearest_z]))
        return clearance, normal

    def _enforce_world_bounds(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        for axis in range(3):
            below = positions[:, axis] < self.lower[axis]
            above = positions[:, axis] > self.upper[axis]
            if bool(np.any(below | above)):
                self.world_violation_steps += int(np.count_nonzero(below | above))
            positions[below, axis] = self.lower[axis]
            positions[above, axis] = self.upper[axis]
            velocities[below | above, axis] *= -0.4

    @staticmethod
    def _clip_rows(values: np.ndarray, max_norm: float) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values * np.minimum(1.0, max_norm / np.maximum(norms, 1e-9))

    @staticmethod
    def _move_toward_velocity(current: np.ndarray, desired: np.ndarray, max_delta: float) -> np.ndarray:
        delta = desired - current
        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        return current + delta * np.minimum(1.0, max_delta / np.maximum(delta_norm, 1e-9))

    def _record_history(self) -> None:
        self.history.append(
            {
                "defender_positions": self.defender_positions.copy(),
                "target_position": self.target_position.copy(),
                "belief_positions": self.target_belief_positions.copy(),
                "capture_radius": float(self.pursuit["capture_radius"]),
                "step": int(self.step_count),
            }
        )
