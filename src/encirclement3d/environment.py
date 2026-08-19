"""A deterministic kinematic benchmark for 3D UAV containment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .capture import (
    TetrahedralCageMetrics,
    resolve_sphere_tetrahedral_contact,
    tetrahedral_cage_metrics,
)
from .flexible_net import FlexibleNetMetrics, FlexibleTetrahedralNet, resolve_sphere_flexible_net_contact


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


_CAPTURE_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    # ``analytical`` preserves the original point-in-tetrahedron proxy.
    # ``rigid_contact`` treats the target as a sphere and resolves a swept
    # collision against the four planar faces after closure.
    "model": "analytical",
    "target_radius": 0.0,
    "contact_tolerance": 0.005,
    "contact_projection_iterations": 96,
    "compression_tolerance": 0.02,
    # Minimal four-panel mass-spring net configuration. These values are a
    # simulation baseline and must be calibrated before a hardware claim.
    "net_node_mass": 0.02,
    "net_spring_stiffness": 40.0,
    "net_spring_damping": 0.40,
    "net_drag_coefficient": 0.03,
    "net_gravity": 9.81,
    "net_substeps": 8,
    # 1 preserves the original four-center proxy; >=2 uses shared mesh nodes.
    "net_face_subdivisions": 1,
    # Per-spring initial tension. It must be calibrated before any hardware use.
    "net_spring_pretension": 0.0,
    "net_max_tension": 50.0,
    "net_max_strain": 0.40,
    "target_mass": 0.03,
    "closure_slot_tolerance": 0.65,
    "minimum_face_clearance": 0.20,
    "minimum_edge_length": 3.60,
    "maximum_edge_length": 5.40,
    "maximum_relative_speed": 2.00,
    "hold_seconds": 2.00,
    "escape_tolerance": 0.05,
    "encirclement_reward_weight": 1.00,
    "feasibility_reward_weight": 0.75,
    "closure_bonus": 4.00,
    "success_bonus": 20.00,
    "escape_penalty": 10.00,
    "collision_penalty": 5.00,
}


@dataclass(frozen=True)
class CylinderObstacle:
    center_xy: np.ndarray
    radius: float
    height: float


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is None:
        return np.zeros_like(vector)
    return fallback.copy()


def _capture_settings(task: dict[str, Any]) -> dict[str, Any]:
    """Validate the optional analytical or rigid-contact capture configuration."""
    configured = task.get("capture", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise ValueError("task.capture must be a mapping when provided.")
    unknown = sorted(set(configured).difference(_CAPTURE_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown task.capture settings: {', '.join(unknown)}")
    settings = {**_CAPTURE_DEFAULTS, **configured}
    model = str(settings["model"])
    if model not in {"analytical", "rigid_contact", "flexible_net"}:
        raise ValueError("task.capture.model must be analytical, rigid_contact, or flexible_net.")
    settings["model"] = model
    numeric_keys = set(settings).difference({"enabled", "model"})
    for key in numeric_keys:
        if float(settings[key]) < 0.0:
            raise ValueError(f"task.capture.{key} must be non-negative.")
    if float(settings["closure_slot_tolerance"]) <= 0.0:
        raise ValueError("task.capture.closure_slot_tolerance must be positive.")
    if float(settings["hold_seconds"]) <= 0.0:
        raise ValueError("task.capture.hold_seconds must be positive.")
    if float(settings["maximum_edge_length"]) <= float(settings["minimum_edge_length"]):
        raise ValueError("task.capture.maximum_edge_length must exceed minimum_edge_length.")
    if int(settings["contact_projection_iterations"]) <= 0:
        raise ValueError("task.capture.contact_projection_iterations must be positive.")
    if int(settings["net_substeps"]) <= 0:
        raise ValueError("task.capture.net_substeps must be positive.")
    if int(settings["net_face_subdivisions"]) != settings["net_face_subdivisions"] or int(
        settings["net_face_subdivisions"]
    ) <= 0:
        raise ValueError("task.capture.net_face_subdivisions must be a positive integer.")
    return settings


class Encirclement3DEnv:
    """Continuous 3D velocity-control environment with a moving target."""

    def __init__(self, config: dict[str, Any], obstacle_count: int, target_speed_scale: float = 1.0):
        self.config = config
        self.world = config["world"]
        self.agents = config["agents"]
        self.task = config["task"]
        self.capture = _capture_settings(self.task)
        self.obstacle_count = obstacle_count
        self.target_speed_scale = target_speed_scale
        self.n_defenders = int(self.agents["defenders"])
        if self.n_defenders != 4:
            raise ValueError("This v0 benchmark requires exactly four defenders for tetrahedral containment.")

        self.dt = float(self.world["dt"])
        self.max_steps = int(self.world["max_steps"])
        self.lower = np.array(
            [-float(self.world["half_extent_xy"]), -float(self.world["half_extent_xy"]), float(self.world["minimum_altitude"])],
            dtype=np.float64,
        )
        self.upper = np.array(
            [float(self.world["half_extent_xy"]), float(self.world["half_extent_xy"]), float(self.world["height"])],
            dtype=np.float64,
        )
        self.rng = np.random.default_rng()
        self.obstacles: list[CylinderObstacle] = []
        self.defender_positions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.defender_velocities = np.zeros_like(self.defender_positions)
        self.target_position = np.zeros(3, dtype=np.float64)
        self.target_velocity = np.zeros(3, dtype=np.float64)
        self.target_escape_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.step_count = 0
        self.hold_steps = 0
        self.encirclement_hold_steps = 0
        self.collision_steps = 0
        self.min_clearance = float("inf")
        self.capture_closed = False
        self.capture_escaped = False
        self.capture_structural_failure = False
        self.capture_hold_steps = 0
        self.capture_close_attempts = 0
        self.capture_closure_events = 0
        self.capture_close_rejected_steps = 0
        self.capture_escape_events = 0
        self.capture_compression_events = 0
        self.capture_net_contact_steps = 0
        self.capture_time_seconds: float | None = None
        self.capture_relative_speed_at_closure: float | None = None
        self.capture_min_face_clearance_while_closed = float("inf")
        self.capture_min_net_margin_while_closed = float("inf")
        self.last_capture_close_requested = False
        self.last_capture_close_accepted = False
        self.last_capture_escape_event = False
        self.last_capture_compression_event = False
        self.last_capture_net_contact = False
        self.last_capture_contact_face: int | None = None
        self.last_capture_contact_panel: int | None = None
        self.last_capture_contact_impulse = 0.0
        self.capture_peak_net_tension = 0.0
        self.capture_peak_net_strain = 0.0
        self.capture_peak_contact_impulse = 0.0
        self.flexible_net: FlexibleTetrahedralNet | None = None
        self.history: list[dict[str, np.ndarray]] = []

    def reset(self, seed: int, record_history: bool = False) -> dict[str, Any]:
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.hold_steps = 0
        self.encirclement_hold_steps = 0
        self.collision_steps = 0
        self.min_clearance = float("inf")
        self.capture_closed = False
        self.capture_escaped = False
        self.capture_structural_failure = False
        self.capture_hold_steps = 0
        self.capture_close_attempts = 0
        self.capture_closure_events = 0
        self.capture_close_rejected_steps = 0
        self.capture_escape_events = 0
        self.capture_compression_events = 0
        self.capture_net_contact_steps = 0
        self.capture_time_seconds = None
        self.capture_relative_speed_at_closure = None
        self.capture_min_face_clearance_while_closed = float("inf")
        self.capture_min_net_margin_while_closed = float("inf")
        self.last_capture_close_requested = False
        self.last_capture_close_accepted = False
        self.last_capture_escape_event = False
        self.last_capture_compression_event = False
        self.last_capture_net_contact = False
        self.last_capture_contact_face = None
        self.last_capture_contact_panel = None
        self.last_capture_contact_impulse = 0.0
        self.capture_peak_net_tension = 0.0
        self.capture_peak_net_strain = 0.0
        self.capture_peak_contact_impulse = 0.0
        self.flexible_net = None
        self.history = []

        self.target_position = np.array(
            [
                self.rng.uniform(-1.5, 1.5),
                self.rng.uniform(-1.5, 1.5),
                self.rng.uniform(3.0, 7.0),
            ],
            dtype=np.float64,
        )
        self.target_velocity.fill(0.0)
        self.target_escape_direction = _unit(
            self.rng.normal(0.0, 1.0, size=3),
            fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        self.defender_positions = self.target_position + TETRAHEDRON_DIRECTIONS * (
            float(self.task["encirclement_radius"]) + 2.2
        )
        self.defender_positions += self.rng.normal(0.0, 0.15, size=self.defender_positions.shape)
        self.defender_positions = np.clip(self.defender_positions, self.lower + 0.5, self.upper - 0.5)
        self.defender_velocities.fill(0.0)
        self.obstacles = self._sample_obstacles()
        if record_history:
            self._record_history()
        return self.observe()

    def observe(self) -> dict[str, Any]:
        """Truth-state observation for Phase 1 only."""
        return {
            "defender_positions": self.defender_positions.copy(),
            "defender_velocities": self.defender_velocities.copy(),
            "target_position": self.target_position.copy(),
            "target_velocity": self.target_velocity.copy(),
            "obstacles": [
                {
                    "center_xy": obstacle.center_xy.copy(),
                    "radius": obstacle.radius,
                    "height": obstacle.height,
                }
                for obstacle in self.obstacles
            ],
            "slot_positions": self.slot_positions.copy(),
            "capture": self._capture_observation(),
            "step": self.step_count,
        }

    @property
    def slot_positions(self) -> np.ndarray:
        return self.target_position + TETRAHEDRON_DIRECTIONS * float(self.task["encirclement_radius"])

    def step(
        self,
        defender_actions: np.ndarray,
        record_history: bool = False,
        close_cage: bool = False,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        previous_corners = self.defender_positions.copy()
        previous_target = self.target_position.copy()
        self._begin_capture_step()
        defender_actions = np.asarray(defender_actions, dtype=np.float64)
        if defender_actions.shape != (self.n_defenders, 3):
            raise ValueError(f"Expected actions with shape {(self.n_defenders, 3)}, got {defender_actions.shape}.")

        defender_actions = self._clip_rows(defender_actions, float(self.agents["defender_max_speed"]))
        self._apply_defender_actions(defender_actions)

        target_action = self._target_action()
        self.target_velocity = self._move_toward_velocity(
            self.target_velocity[None, :],
            target_action[None, :],
            max_delta=float(self.agents["target_max_acceleration"]) * self.dt,
        )[0]
        self.target_position += self.target_velocity * self.dt
        self._enforce_world_bounds(self.target_position[None, :], self.target_velocity[None, :])
        self._resolve_closed_cage_contact(previous_corners, previous_target)

        self.step_count += 1
        metrics = self._metrics()
        self.min_clearance = min(self.min_clearance, metrics["min_clearance"])
        if metrics["collision"]:
            self.collision_steps += 1

        task_outcome = self._update_task_state(metrics, close_cage=close_cage)
        success = bool(task_outcome["success"])
        terminated = bool(task_outcome["terminated"])
        truncated = self.step_count >= self.max_steps
        reward = float(task_outcome["reward"])

        if record_history:
            self._record_history()
        info = {
            **metrics,
            **task_outcome,
            "collision_steps": self.collision_steps,
            "min_clearance_so_far": self.min_clearance,
        }
        return self.observe(), reward, terminated, truncated, info

    def _apply_defender_actions(self, defender_actions: np.ndarray) -> None:
        """Advance defenders for one step; subclasses may provide dynamics."""
        self.defender_velocities = self._move_toward_velocity(
            self.defender_velocities,
            defender_actions,
            max_delta=float(self.agents["defender_max_acceleration"]) * self.dt,
        )
        self.defender_positions += self.defender_velocities * self.dt
        self._enforce_world_bounds(self.defender_positions, self.defender_velocities)

    def _sample_obstacles(self) -> list[CylinderObstacle]:
        obstacles: list[CylinderObstacle] = []
        protected_points = np.vstack([self.defender_positions, self.target_position[None, :]])
        for _ in range(self.obstacle_count):
            for _attempt in range(100):
                radius = float(self.rng.uniform(0.65, 1.15))
                height = float(self.rng.uniform(3.0, 7.0))
                center_xy = self.rng.uniform(-7.5, 7.5, size=2)
                candidate = CylinderObstacle(center_xy=center_xy, radius=radius, height=height)
                if self._obstacle_clear_of_points(candidate, protected_points) and all(
                    np.linalg.norm(candidate.center_xy - existing.center_xy)
                    >= candidate.radius + existing.radius + 1.0
                    for existing in obstacles
                ):
                    obstacles.append(candidate)
                    break
            else:
                raise RuntimeError("Unable to sample a non-overlapping obstacle layout.")
        return obstacles

    def _obstacle_clear_of_points(self, obstacle: CylinderObstacle, points: np.ndarray) -> bool:
        radial = np.linalg.norm(points[:, :2] - obstacle.center_xy[None, :], axis=1)
        vertical_overlap = points[:, 2] <= obstacle.height + 1.0
        return bool(np.all((radial > obstacle.radius + 2.0) | (~vertical_overlap)))

    def _target_action(self) -> np.ndarray:
        defender_centroid = self.defender_positions.mean(axis=0)
        flee = _unit(self.target_position - defender_centroid, fallback=self.target_escape_direction)
        vertical = 0.25 * np.sin(0.08 * self.step_count + 0.4)
        desired_dir = _unit(0.7 * self.target_escape_direction + 0.8 * flee + np.array([0.0, 0.0, vertical]))
        desired = desired_dir * float(self.agents["target_max_speed"]) * self.target_speed_scale

        for obstacle in self.obstacles:
            clearance, normal = self._cylinder_clearance_and_normal(self.target_position, obstacle)
            if clearance < float(self.task["obstacle_avoidance_distance"]):
                desired += normal * (float(self.task["obstacle_avoidance_distance"]) - clearance) * 2.5

        for defender_position in self.defender_positions:
            separation = self.target_position - defender_position
            clearance = float(np.linalg.norm(separation) - 2.0 * float(self.agents["drone_radius"]))
            if clearance < float(self.task["target_defender_avoidance_distance"]):
                desired += (
                    _unit(separation)
                    * (float(self.task["target_defender_avoidance_distance"]) - clearance)
                    * float(self.task["target_defender_avoidance_gain"])
                )

        # Keep the target far enough from the world boundary that all four
        # tetrahedral slots remain inside the feasible flight volume.
        margin = float(self.task["encirclement_radius"]) + float(self.task["slot_tolerance"]) + 0.6
        boundary_gain = 2.0 * float(self.agents["target_max_speed"]) * self.target_speed_scale
        for axis in range(3):
            if self.target_position[axis] < self.lower[axis] + margin:
                desired[axis] += boundary_gain
            if self.target_position[axis] > self.upper[axis] - margin:
                desired[axis] -= boundary_gain
        return self._clip_rows(desired[None, :], float(self.agents["target_max_speed"]) * self.target_speed_scale)[0]

    def _capture_observation(self) -> dict[str, int | bool | float | None]:
        """Expose capture state without changing the legacy policy features."""
        geometry = self._cage_metrics()
        flexible_metrics = self._current_flexible_net_metrics()
        relative_speed = np.linalg.norm(self.defender_velocities - self.target_velocity[None, :], axis=1)
        slot_error = np.linalg.norm(self.defender_positions - self.slot_positions, axis=1)
        face_clearance = (
            flexible_metrics.min_face_clearance if flexible_metrics is not None else geometry.min_face_clearance
        )
        net_margin = face_clearance - self._capture_target_radius()
        return {
            "enabled": bool(self.capture["enabled"]),
            "model": str(self.capture["model"]),
            "closed": bool(self.capture_closed),
            "escaped": bool(self.capture_escaped),
            "structural_failure": bool(self.capture_structural_failure),
            "target_inside": bool(geometry.target_inside),
            "max_slot_error": float(np.max(slot_error)),
            "min_face_clearance": float(face_clearance),
            "target_radius": float(self._capture_target_radius()),
            "net_margin": float(net_margin),
            "min_edge_length": float(geometry.min_edge_length),
            "max_edge_length": float(geometry.max_edge_length),
            "max_relative_speed": float(np.max(relative_speed)),
            "hold_steps": int(self.capture_hold_steps),
            "close_attempts": int(self.capture_close_attempts),
            "closure_events": int(self.capture_closure_events),
            "escape_events": int(self.capture_escape_events),
            "compression_events": int(self.capture_compression_events),
            "net_contact": bool(self.last_capture_net_contact),
            "net_contact_face": self.last_capture_contact_face,
            "net_contact_panel": self.last_capture_contact_panel,
            "net_contact_steps": int(self.capture_net_contact_steps),
            "max_net_tension": float(flexible_metrics.max_tension) if flexible_metrics is not None else 0.0,
            "max_net_strain": float(flexible_metrics.max_strain) if flexible_metrics is not None else 0.0,
            "peak_contact_impulse": (
                float(flexible_metrics.peak_contact_impulse) if flexible_metrics is not None else 0.0
            ),
            "time_seconds": self.capture_time_seconds,
        }

    def _cage_metrics(self) -> TetrahedralCageMetrics:
        """Return finite geometry values even after a degenerate collision state."""
        try:
            return tetrahedral_cage_metrics(self.defender_positions, self.target_position)
        except ValueError:
            return TetrahedralCageMetrics(
                face_clearances=np.full(4, -float("inf"), dtype=np.float64),
                edge_lengths=np.zeros(6, dtype=np.float64),
            )

    def _current_flexible_net_metrics(self) -> FlexibleNetMetrics | None:
        if self.capture["model"] != "flexible_net" or self.flexible_net is None:
            return None
        try:
            return self.flexible_net.metrics(self.target_position)
        except ValueError:
            return None

    def capture_close_feasible(self) -> bool:
        """Report whether the current state is safe to command virtual cage closure."""
        return self._capture_close_feasible(self._metrics())

    def _capture_close_feasible(self, metrics: dict[str, Any]) -> bool:
        if not bool(self.capture["enabled"]) or self.capture_escaped or self.capture_structural_failure:
            return False
        return bool(
            not bool(metrics["collision"])
            and np.all(metrics["slot_error"] <= float(self.capture["closure_slot_tolerance"]))
            and bool(metrics["cage_target_inside"])
            and float(metrics["cage_net_margin"]) >= float(self.capture["minimum_face_clearance"])
            and float(metrics["cage_min_edge_length"]) >= float(self.capture["minimum_edge_length"])
            and float(metrics["cage_max_edge_length"]) <= float(self.capture["maximum_edge_length"])
            and float(metrics["capture_max_relative_speed"]) <= float(self.capture["maximum_relative_speed"])
        )

    def _capture_feasibility_reward(self, metrics: dict[str, Any]) -> float:
        """Shape approach toward a closable cage without rewarding unsafe proximity."""
        if not bool(self.capture["enabled"]):
            return 0.0
        slot_quality = np.clip(
            1.0 - float(np.max(metrics["slot_error"])) / float(self.capture["closure_slot_tolerance"]),
            -1.0,
            1.0,
        )
        face_quality = np.clip(
            float(metrics["cage_net_margin"]) / max(float(self.capture["minimum_face_clearance"]), 1e-9),
            -1.0,
            1.0,
        )
        min_edge_quality = np.clip(
            float(metrics["cage_min_edge_length"]) / max(float(self.capture["minimum_edge_length"]), 1e-9),
            0.0,
            1.0,
        )
        max_edge_quality = np.clip(
            (float(self.capture["maximum_edge_length"]) - float(metrics["cage_max_edge_length"]))
            / max(float(self.capture["maximum_edge_length"]) - float(self.capture["minimum_edge_length"]), 1e-9),
            -1.0,
            1.0,
        )
        relative_speed_quality = np.clip(
            1.0 - float(metrics["capture_max_relative_speed"]) / max(float(self.capture["maximum_relative_speed"]), 1e-9),
            -1.0,
            1.0,
        )
        return float(
            float(self.capture["feasibility_reward_weight"])
            * np.mean([slot_quality, face_quality, min_edge_quality, max_edge_quality, relative_speed_quality])
        )

    def _update_task_state(self, metrics: dict[str, Any], *, close_cage: bool) -> dict[str, Any]:
        """Advance either the legacy containment task or the capture-proxy state machine."""
        covered = bool(np.all(metrics["slot_error"] <= float(self.task["slot_tolerance"])))
        self.hold_steps = self.hold_steps + 1 if covered else 0
        self.encirclement_hold_steps = self.hold_steps
        required_encirclement_hold_steps = int(round(float(self.task["hold_seconds"]) / self.dt))
        encirclement_success = self.hold_steps >= required_encirclement_hold_steps and self.collision_steps == 0

        capture_enabled = bool(self.capture["enabled"])
        self.last_capture_close_requested = bool(close_cage and capture_enabled)
        self.last_capture_close_accepted = False
        self.last_capture_escape_event = False
        self.last_capture_compression_event = False
        capture_ready = self._capture_close_feasible(metrics)

        if capture_enabled:
            if self.last_capture_close_requested and not self.capture_closed:
                self.capture_close_attempts += 1
                if capture_ready:
                    self.capture_closed = True
                    self.last_capture_close_accepted = True
                    self.capture_closure_events += 1
                    self.capture_relative_speed_at_closure = float(metrics["capture_max_relative_speed"])
                    if self.capture["model"] == "flexible_net":
                        # Initialise at the accepted geometry so the net never
                        # appears before a closure command has passed its gate.
                        self.flexible_net = FlexibleTetrahedralNet(
                            self.defender_positions,
                            node_mass=float(self.capture["net_node_mass"]),
                            spring_stiffness=float(self.capture["net_spring_stiffness"]),
                            spring_damping=float(self.capture["net_spring_damping"]),
                            drag_coefficient=float(self.capture["net_drag_coefficient"]),
                            gravity=float(self.capture["net_gravity"]),
                            substeps=int(self.capture["net_substeps"]),
                            face_subdivisions=int(self.capture["net_face_subdivisions"]),
                            spring_pretension=float(self.capture["net_spring_pretension"]),
                        )
                else:
                    self.capture_close_rejected_steps += 1

            if self.capture_closed:
                self.capture_min_face_clearance_while_closed = min(
                    self.capture_min_face_clearance_while_closed,
                    float(metrics["cage_min_face_clearance"]),
                )
                self.capture_min_net_margin_while_closed = min(
                    self.capture_min_net_margin_while_closed,
                    float(metrics["cage_net_margin"]),
                )
                rigid_contact_failure = bool(
                    self.capture["model"] == "rigid_contact"
                    and (
                        not bool(metrics["cage_sphere_contained"])
                        or float(metrics["cage_net_margin"]) < -float(self.capture["compression_tolerance"])
                    )
                )
                flexible_net_failure = bool(
                    self.capture["model"] == "flexible_net"
                    and self.flexible_net is not None
                    and (
                        not bool(metrics["cage_sphere_contained"])
                        or float(metrics["flexible_net_max_tension"]) > float(self.capture["net_max_tension"])
                        or float(metrics["flexible_net_max_strain"]) > float(self.capture["net_max_strain"])
                    )
                )
                if rigid_contact_failure or flexible_net_failure:
                    self.capture_closed = False
                    self.capture_structural_failure = True
                    self.capture_compression_events += 1
                    self.capture_hold_steps = 0
                    self.last_capture_compression_event = True
                elif float(metrics["cage_min_face_clearance"]) < -float(self.capture["escape_tolerance"]):
                    self.capture_closed = False
                    self.capture_escaped = True
                    self.capture_escape_events += 1
                    self.capture_hold_steps = 0
                    self.last_capture_escape_event = True
                else:
                    self.capture_hold_steps += 1

            required_capture_hold_steps = int(round(float(self.capture["hold_seconds"]) / self.dt))
            capture_success = bool(
                self.capture_closed
                and self.capture_hold_steps >= required_capture_hold_steps
                and self.collision_steps == 0
                and not self.capture_escaped
            )
            if capture_success and self.capture_time_seconds is None:
                self.capture_time_seconds = self.step_count * self.dt
            success = capture_success
            failure = bool(metrics["collision"] or self.capture_escaped or self.capture_structural_failure)
        else:
            capture_success = False
            success = encirclement_success
            failure = bool(metrics["collision"])

        if capture_enabled:
            reward_components = {
                "encirclement": -float(self.capture["encirclement_reward_weight"]) * float(metrics["mean_slot_error"]),
                "safety": -float(self.capture["collision_penalty"]) if metrics["collision"] else 0.0,
                "capture_feasibility": self._capture_feasibility_reward(metrics),
                "capture_closure": float(self.capture["closure_bonus"]) if self.last_capture_close_accepted else 0.0,
                "capture_escape": -float(self.capture["escape_penalty"]) if self.last_capture_escape_event else 0.0,
                "capture_structural_failure": (
                    -float(self.capture["escape_penalty"]) if self.last_capture_compression_event else 0.0
                ),
                "capture_success": float(self.capture["success_bonus"]) if success else 0.0,
            }
        else:
            # Preserve the original containment reward exactly for every frozen baseline.
            reward_components = {
                "encirclement": -float(metrics["mean_slot_error"]),
                "safety": -5.0 if metrics["collision"] else 0.0,
                "capture_feasibility": 0.0,
                "capture_closure": 0.0,
                "capture_escape": 0.0,
                "capture_structural_failure": 0.0,
                "capture_success": 20.0 if success else 0.0,
            }

        return {
            "success": bool(success),
            "failure": bool(failure),
            "terminated": bool(success or failure),
            "reward": float(sum(reward_components.values())),
            "reward_components": reward_components,
            "encirclement_success": bool(encirclement_success),
            "hold_steps": int(self.hold_steps),
            "encirclement_hold_steps": int(self.encirclement_hold_steps),
            "capture_enabled": capture_enabled,
            "capture_success": bool(capture_success),
            "capture_closed": bool(self.capture_closed),
            "capture_close_requested": bool(self.last_capture_close_requested),
            "capture_close_ready": bool(capture_ready),
            "capture_close_accepted": bool(self.last_capture_close_accepted),
            "capture_close_attempts": int(self.capture_close_attempts),
            "capture_closure_events": int(self.capture_closure_events),
            "capture_close_rejected_steps": int(self.capture_close_rejected_steps),
            "capture_escaped": bool(self.capture_escaped),
            "capture_escape_event": bool(self.last_capture_escape_event),
            "capture_escape_events": int(self.capture_escape_events),
            "capture_structural_failure": bool(self.capture_structural_failure),
            "capture_compression_event": bool(self.last_capture_compression_event),
            "capture_compression_events": int(self.capture_compression_events),
            "capture_net_contact": bool(self.last_capture_net_contact),
            "capture_net_contact_face": self.last_capture_contact_face,
            "capture_net_contact_panel": self.last_capture_contact_panel,
            "capture_net_contact_steps": int(self.capture_net_contact_steps),
            "capture_last_contact_impulse": float(self.last_capture_contact_impulse),
            "capture_peak_net_tension": float(self.capture_peak_net_tension),
            "capture_peak_net_strain": float(self.capture_peak_net_strain),
            "capture_peak_contact_impulse": float(self.capture_peak_contact_impulse),
            "capture_hold_steps": int(self.capture_hold_steps),
            "capture_time_seconds": self.capture_time_seconds,
            "capture_relative_speed_at_closure": self.capture_relative_speed_at_closure,
            "capture_min_face_clearance_while_closed": (
                float(self.capture_min_face_clearance_while_closed)
                if np.isfinite(self.capture_min_face_clearance_while_closed)
                else None
            ),
            "capture_min_net_margin_while_closed": (
                float(self.capture_min_net_margin_while_closed)
                if np.isfinite(self.capture_min_net_margin_while_closed)
                else None
            ),
        }

    def _metrics(self) -> dict[str, Any]:
        slot_error = np.linalg.norm(self.defender_positions - self.slot_positions, axis=1)
        cage = self._cage_metrics()
        flexible_metrics = self._current_flexible_net_metrics()
        relative_speed = np.linalg.norm(self.defender_velocities - self.target_velocity[None, :], axis=1)
        clearances: list[float] = []
        radius = float(self.agents["drone_radius"])
        for position in self.defender_positions:
            for obstacle in self.obstacles:
                clearance, _ = self._cylinder_clearance_and_normal(position, obstacle)
                clearances.append(clearance - radius)
        for i in range(self.n_defenders):
            for j in range(i + 1, self.n_defenders):
                clearances.append(float(np.linalg.norm(self.defender_positions[i] - self.defender_positions[j]) - 2.0 * radius))
        for position in self.defender_positions:
            clearances.append(float(np.linalg.norm(position - self.target_position) - 2.0 * radius))

        min_clearance = float(min(clearances)) if clearances else float("inf")
        face_clearance = (
            flexible_metrics.min_face_clearance if flexible_metrics is not None else cage.min_face_clearance
        )
        target_radius = self._capture_target_radius()
        net_margin = float(face_clearance - target_radius)
        return {
            "slot_error": slot_error,
            "mean_slot_error": float(np.mean(slot_error)),
            "min_clearance": min_clearance,
            "collision": min_clearance < 0.0,
            "cage_face_clearances": cage.face_clearances,
            "cage_edge_lengths": cage.edge_lengths,
            "cage_target_inside": cage.target_inside,
            "cage_min_face_clearance": cage.min_face_clearance,
            "cage_target_radius": float(target_radius),
            "cage_net_margin": net_margin,
            "cage_sphere_contained": bool(
                net_margin >= -float(self.capture["contact_tolerance"])
            ),
            "flexible_net_active": flexible_metrics is not None,
            "flexible_net_min_face_clearance": float(face_clearance),
            "flexible_net_max_tension": (
                float(flexible_metrics.max_tension) if flexible_metrics is not None else 0.0
            ),
            "flexible_net_max_strain": (
                float(flexible_metrics.max_strain) if flexible_metrics is not None else 0.0
            ),
            "flexible_net_peak_contact_impulse": (
                float(flexible_metrics.peak_contact_impulse) if flexible_metrics is not None else 0.0
            ),
            "cage_min_edge_length": cage.min_edge_length,
            "cage_max_edge_length": cage.max_edge_length,
            "capture_max_relative_speed": float(np.max(relative_speed)),
        }

    def _capture_target_radius(self) -> float:
        """Return a sphere radius for both contact-capable capture models."""
        return float(self.capture["target_radius"]) if self.capture["model"] in {"rigid_contact", "flexible_net"} else 0.0

    def _begin_capture_step(self) -> None:
        """Clear per-step contact diagnostics before physics advances."""
        self.last_capture_net_contact = False
        self.last_capture_contact_face = None
        self.last_capture_contact_panel = None
        self.last_capture_contact_impulse = 0.0

    def _resolve_closed_cage_contact(self, previous_corners: np.ndarray, previous_target: np.ndarray) -> None:
        """Resolve closed-cage contact for rigid or minimal flexible nets."""
        if not bool(self.capture["enabled"]) or not self.capture_closed:
            return
        if self.capture["model"] == "flexible_net":
            if self.flexible_net is None:
                # This is defensive only: normal closure creates the net in
                # ``_update_task_state`` at the end of the previous step.
                self.flexible_net = FlexibleTetrahedralNet(
                    previous_corners,
                    node_mass=float(self.capture["net_node_mass"]),
                    spring_stiffness=float(self.capture["net_spring_stiffness"]),
                    spring_damping=float(self.capture["net_spring_damping"]),
                    drag_coefficient=float(self.capture["net_drag_coefficient"]),
                    gravity=float(self.capture["net_gravity"]),
                    substeps=int(self.capture["net_substeps"]),
                    face_subdivisions=int(self.capture["net_face_subdivisions"]),
                    spring_pretension=float(self.capture["net_spring_pretension"]),
                )
            previous_net = self.flexible_net.advance(self.defender_positions, self.dt)
            proposed_velocity = (self.target_position - previous_target) / self.dt
            result = resolve_sphere_flexible_net_contact(
                previous_net,
                self.flexible_net.snapshot(),
                previous_target,
                self.target_position,
                radius=self._capture_target_radius(),
                tolerance=float(self.capture["contact_tolerance"]),
                max_projection_iterations=int(self.capture["contact_projection_iterations"]),
            )
            self.target_position = result.position
            self.target_velocity = (self.target_position - previous_target) / self.dt
            self.last_capture_net_contact = bool(result.contact)
            self.last_capture_contact_face = result.triangle_index
            self.last_capture_contact_panel = result.panel_index
            if result.contact:
                self.capture_net_contact_steps += 1
                impulse = float(
                    float(self.capture["target_mass"])
                    * np.linalg.norm(self.target_velocity - proposed_velocity)
                )
                self.last_capture_contact_impulse = impulse
                if result.panel_index is not None and result.inward_normal is not None:
                    self.flexible_net.apply_contact_impulse(
                        result.panel_index,
                        result.inward_normal,
                        impulse,
                        triangle_index=result.triangle_index,
                    )
            flexible_metrics = self._current_flexible_net_metrics()
            if flexible_metrics is not None:
                self.capture_peak_net_tension = max(
                    self.capture_peak_net_tension, float(flexible_metrics.max_tension)
                )
                self.capture_peak_net_strain = max(
                    self.capture_peak_net_strain, float(flexible_metrics.max_strain)
                )
                self.capture_peak_contact_impulse = max(
                    self.capture_peak_contact_impulse, float(flexible_metrics.peak_contact_impulse)
                )
            return
        if self.capture["model"] != "rigid_contact":
            return
            return
        result = resolve_sphere_tetrahedral_contact(
            previous_corners,
            self.defender_positions,
            previous_target,
            self.target_position,
            radius=self._capture_target_radius(),
            tolerance=float(self.capture["contact_tolerance"]),
            max_projection_iterations=int(self.capture["contact_projection_iterations"]),
        )
        self.target_position = result.position
        self.target_velocity = (self.target_position - previous_target) / self.dt
        self.last_capture_net_contact = bool(result.contact)
        self.last_capture_contact_face = result.contact_face_index
        if result.contact:
            self.capture_net_contact_steps += 1

    def _cylinder_clearance_and_normal(self, position: np.ndarray, obstacle: CylinderObstacle) -> tuple[float, np.ndarray]:
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

    @staticmethod
    def _clip_rows(values: np.ndarray, max_norm: float) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-9))
        return values * scale

    @staticmethod
    def _move_toward_velocity(current: np.ndarray, desired: np.ndarray, max_delta: float) -> np.ndarray:
        delta = desired - current
        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_delta / np.maximum(delta_norm, 1e-9))
        return current + delta * scale

    def _enforce_world_bounds(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        for axis in range(3):
            below = positions[:, axis] < self.lower[axis]
            above = positions[:, axis] > self.upper[axis]
            positions[below, axis] = self.lower[axis]
            positions[above, axis] = self.upper[axis]
            velocities[below | above, axis] *= -0.4

    def _record_history(self) -> None:
        self.history.append(
            {
                "defender_positions": self.defender_positions.copy(),
                "target_position": self.target_position.copy(),
                "slot_positions": self.slot_positions.copy(),
            }
        )
