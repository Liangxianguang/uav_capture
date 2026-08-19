"""PyBullet-backed defender dynamics for the 3D encirclement benchmark.

The non-cooperative target remains a kinematic task object. Defenders are
Crazyflie 2.x quadrotors simulated by the fixed gym-pybullet-drones source
recorded in ``EXTERNAL_SOURCES.md``. The public interface intentionally
matches :class:`Encirclement3DEnv`: actions are desired world-frame velocities
in metres per second, and task success is evaluated by the same tetrahedral
slot and collision criteria.
"""

from __future__ import annotations

import collections
import collections.abc
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .environment import Encirclement3DEnv
from .pybullet_softbody_net import (
    PyBulletSoftBodyConfig,
    TetrahedralSoftBodyMesh,
    attach_tetrahedral_softbody_anchors,
    load_tetrahedral_softbody,
    softbody_vertices,
    write_tetrahedral_softbody_obj,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYBULLET_DRONES_ROOT = PROJECT_ROOT / "third_party" / "gym-pybullet-drones-7688e7208a1572b1680736a3c0c9b93c379db3fe"


def _load_pybullet_dependencies():
    """Load the pinned vendored simulator without installing its RL extras."""
    if not PYBULLET_DRONES_ROOT.is_dir():
        raise RuntimeError(
            "The pinned gym-pybullet-drones source is missing. Expected: "
            f"{PYBULLET_DRONES_ROOT}"
        )
    # v1.0.0 predates Python 3.10 and still references collections.Mapping.
    # Keep this compatibility shim local rather than modifying vendored code.
    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping  # type: ignore[attr-defined]
    source_path = str(PYBULLET_DRONES_ROOT)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    import pybullet as pybullet
    from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
    # Import the controller after the environment modules have completed their
    # legacy package initialization; importing it first creates a circular
    # import through gym_pybullet_drones.envs.__init__ on Python 3.11.
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

    return pybullet, DroneModel, Physics, CtrlAviary, DSLPIDControl, VelocityAviary


class PyBulletEncirclement3DEnv(Encirclement3DEnv):
    """Encirclement task with four PyBullet Crazyflie defender vehicles.

    ``VelocityAviary`` internally maps a unit direction plus speed fraction to
    motor RPM through its cascaded PID controller. This wrapper maps the
    benchmark's velocity action back to that format and then reads the actual
    six-degree-of-freedom state after every control interval.
    """

    def __init__(self, config: dict[str, Any], obstacle_count: int, target_speed_scale: float = 1.0):
        super().__init__(config, obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
        settings = config.get("dynamics", {})
        self.pybullet_physics_name = str(settings.get("pybullet_physics", "pyb_drag")).lower()
        self.pybullet_control_mode = str(settings.get("pybullet_control_mode", "position_pid")).lower()
        self.pybullet_frequency = int(settings.get("pybullet_frequency", 240))
        self.pybullet_aggregate_steps = int(settings.get("pybullet_aggregate_steps", 24))
        self.pybullet_speed_limit = float(settings.get("pybullet_speed_limit", self.agents["defender_max_speed"]))
        self.pybullet_position_horizon = float(settings.get("pybullet_position_horizon", self.dt))
        self.pybullet_position_velocity_feedforward = float(
            settings.get("pybullet_position_velocity_feedforward", 0.0)
        )
        self.pybullet_command_max_acceleration = float(
            settings.get("pybullet_command_max_acceleration", 0.0)
        )
        self.pybullet_boundary_reference_margin = self._boundary_margin(
            settings.get("pybullet_boundary_reference_margin", 0.0)
        )
        self.pybullet_vertical_recovery_enabled = bool(settings.get("pybullet_vertical_recovery_enabled", False))
        self.pybullet_vertical_recovery_altitude = float(
            settings.get("pybullet_vertical_recovery_altitude", self.lower[2])
        )
        self.pybullet_vertical_recovery_descend_speed = float(
            settings.get("pybullet_vertical_recovery_descend_speed", 1.0)
        )
        self.pybullet_vertical_recovery_climb_height = float(
            settings.get("pybullet_vertical_recovery_climb_height", 2.0)
        )
        self.pybullet_vertical_emergency_enabled = bool(settings.get("pybullet_vertical_emergency_enabled", False))
        self.pybullet_vertical_emergency_braking_deceleration = float(
            settings.get("pybullet_vertical_emergency_braking_deceleration", 0.0)
        )
        self.pybullet_vertical_emergency_reaction_time = float(
            settings.get("pybullet_vertical_emergency_reaction_time", self.control_dt)
        )
        self.pybullet_vertical_emergency_margin = float(
            settings.get("pybullet_vertical_emergency_margin", 0.0)
        )
        self.pybullet_vertical_emergency_climb_height = float(
            settings.get("pybullet_vertical_emergency_climb_height", 1.0)
        )
        self.pybullet_attitude_recovery_enabled = bool(settings.get("pybullet_attitude_recovery_enabled", False))
        self.pybullet_attitude_recovery_max_tilt = float(
            settings.get("pybullet_attitude_recovery_max_tilt", 0.0)
        )
        self.pybullet_attitude_recovery_climb_height = float(
            settings.get("pybullet_attitude_recovery_climb_height", 1.0)
        )
        self.pybullet_gui = bool(settings.get("pybullet_gui", False))
        self.pybullet_deformable_world = bool(settings.get("pybullet_deformable_world", False))
        self.pybullet_softbody_net_enabled = bool(settings.get("pybullet_softbody_net_enabled", False))
        self.pybullet_softbody_net_face_subdivisions = int(
            settings.get("pybullet_softbody_net_face_subdivisions", 3)
        )
        self.pybullet_softbody_net_mass_kg = float(settings.get("pybullet_softbody_net_mass_kg", 0.0))
        self.pybullet_softbody_net_elastic_stiffness = float(
            settings.get("pybullet_softbody_net_elastic_stiffness", 0.0)
        )
        self.pybullet_softbody_net_damping_stiffness = float(
            settings.get("pybullet_softbody_net_damping_stiffness", 0.0)
        )
        self.pybullet_softbody_net_friction = float(settings.get("pybullet_softbody_net_friction", 0.5))
        mesh_directory = settings.get("pybullet_softbody_net_mesh_directory")
        self.pybullet_softbody_net_mesh_directory = (
            Path(mesh_directory).expanduser() if mesh_directory is not None else None
        )
        if self.pybullet_frequency <= 0 or self.pybullet_aggregate_steps <= 0:
            raise ValueError("PyBullet frequency and aggregate steps must be positive.")
        if self.pybullet_speed_limit <= 0.0:
            raise ValueError("pybullet_speed_limit must be positive.")
        if self.pybullet_position_horizon <= 0.0:
            raise ValueError("pybullet_position_horizon must be positive.")
        if self.pybullet_position_velocity_feedforward < 0.0:
            raise ValueError("pybullet_position_velocity_feedforward must be non-negative.")
        if self.pybullet_command_max_acceleration < 0.0:
            raise ValueError("pybullet_command_max_acceleration must be non-negative.")
        if self.pybullet_vertical_recovery_enabled:
            if not self.lower[2] < self.pybullet_vertical_recovery_altitude < self.upper[2]:
                raise ValueError("pybullet_vertical_recovery_altitude must lie inside the flight volume.")
            if self.pybullet_vertical_recovery_descend_speed <= 0.0:
                raise ValueError("pybullet_vertical_recovery_descend_speed must be positive.")
            if self.pybullet_vertical_recovery_climb_height <= 0.0:
                raise ValueError("pybullet_vertical_recovery_climb_height must be positive.")
        if self.pybullet_vertical_emergency_enabled:
            if self.pybullet_vertical_emergency_braking_deceleration <= 0.0:
                raise ValueError("pybullet_vertical_emergency_braking_deceleration must be positive.")
            if self.pybullet_vertical_emergency_reaction_time < 0.0:
                raise ValueError("pybullet_vertical_emergency_reaction_time must be non-negative.")
            if self.pybullet_vertical_emergency_margin < 0.0:
                raise ValueError("pybullet_vertical_emergency_margin must be non-negative.")
            if self.pybullet_vertical_emergency_climb_height <= 0.0:
                raise ValueError("pybullet_vertical_emergency_climb_height must be positive.")
        if self.pybullet_attitude_recovery_enabled:
            if not 0.0 < self.pybullet_attitude_recovery_max_tilt < np.pi / 2.0:
                raise ValueError("pybullet_attitude_recovery_max_tilt must lie in (0, pi/2).")
            if self.pybullet_attitude_recovery_climb_height <= 0.0:
                raise ValueError("pybullet_attitude_recovery_climb_height must be positive.")
        if np.any(self.pybullet_boundary_reference_margin < 0.0):
            raise ValueError("pybullet_boundary_reference_margin must be non-negative.")
        if np.any(self.lower + self.pybullet_boundary_reference_margin >= self.upper - self.pybullet_boundary_reference_margin):
            raise ValueError("pybullet_boundary_reference_margin leaves no feasible flight volume.")
        if self.pybullet_softbody_net_enabled:
            if not self.pybullet_deformable_world:
                raise ValueError("pybullet_softbody_net_enabled requires pybullet_deformable_world=true.")
            if not bool(self.capture["enabled"]) or self.capture["model"] != "flexible_net":
                raise ValueError("pybullet_softbody_net_enabled requires task.capture.model=flexible_net.")
            if self.pybullet_softbody_net_face_subdivisions < 2:
                raise ValueError("pybullet_softbody_net_face_subdivisions must be at least 2.")
            if self.pybullet_softbody_net_mass_kg <= 0.0:
                raise ValueError("pybullet_softbody_net_mass_kg must be positive.")
            if self.pybullet_softbody_net_elastic_stiffness <= 0.0:
                raise ValueError("pybullet_softbody_net_elastic_stiffness must be positive.")
            if self.pybullet_softbody_net_damping_stiffness <= 0.0:
                raise ValueError("pybullet_softbody_net_damping_stiffness must be positive.")
            if not 0.0 <= self.pybullet_softbody_net_friction <= 1.0:
                raise ValueError("pybullet_softbody_net_friction must lie in [0, 1].")
            if self.pybullet_softbody_net_mesh_directory is None:
                raise ValueError("pybullet_softbody_net_enabled requires pybullet_softbody_net_mesh_directory.")
            if float(self.capture["net_spring_pretension"]) != 0.0:
                raise ValueError(
                    "The PyBullet soft-body backend does not model pre-tension; set net_spring_pretension to zero."
                )
        control_dt = self.pybullet_aggregate_steps / self.pybullet_frequency
        if not np.isclose(control_dt, self.dt, rtol=0.0, atol=1e-12):
            raise ValueError(
                "world.dt must equal pybullet_aggregate_steps / pybullet_frequency; "
                f"got {self.dt} versus {control_dt}."
            )

        self.aviary: Any | None = None
        self.pybullet: Any | None = None
        self.obstacle_body_ids: list[int] = []
        self.physical_collision_steps = 0
        self.world_violation_steps = 0
        self.command_speed_sum = 0.0
        self.filtered_command_speed_sum = 0.0
        self.command_filter_correction_sum = 0.0
        self.realized_speed_sum = 0.0
        self.command_count = 0
        self.filtered_defender_actions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.last_requested_defender_actions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.last_executed_defender_actions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.pid_controllers: list[Any] = []
        self.boundary_governor_active_steps = 0
        self.boundary_governor_correction_sum = 0.0
        self.vertical_recovery_active_steps = 0
        self.vertical_recovery_agent_steps = 0
        self.last_vertical_recovery_mask = np.zeros(self.n_defenders, dtype=bool)
        self.vertical_emergency_active_steps = 0
        self.vertical_emergency_agent_steps = 0
        self.last_vertical_emergency_mask = np.zeros(self.n_defenders, dtype=bool)
        self.last_vertical_emergency_required_distance = np.zeros(self.n_defenders, dtype=np.float64)
        self.attitude_recovery_active_steps = 0
        self.attitude_recovery_agent_steps = 0
        self.last_attitude_recovery_mask = np.zeros(self.n_defenders, dtype=bool)
        self.last_attitude_tilt = np.zeros(self.n_defenders, dtype=np.float64)
        self.last_pid_target_positions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.pybullet_softbody_net_id: int | None = None
        self.pybullet_softbody_target_body_id: int | None = None
        self.pybullet_softbody_mesh: TetrahedralSoftBodyMesh | None = None
        self.pybullet_softbody_anchor_constraints: list[int] = []
        self.pybullet_softbody_episode_index = 0
        self.pybullet_softbody_seed: int | None = None
        self.pybullet_softbody_deployment_events = 0
        self.pybullet_softbody_target_contact_steps = 0
        self.pybullet_softbody_peak_target_normal_force = 0.0
        self.pybullet_softbody_min_target_contact_distance = float("inf")
        self.pybullet_softbody_max_anchor_error = 0.0
        self.last_pybullet_softbody_target_contact = False

    @property
    def control_dt(self) -> float:
        """Duration in seconds represented by one benchmark action."""
        return self.pybullet_aggregate_steps / self.pybullet_frequency

    def reset(self, seed: int, record_history: bool = False) -> dict[str, Any]:
        self.close()
        super().reset(seed=seed, record_history=False)
        self.pybullet_softbody_episode_index += 1
        self.pybullet_softbody_seed = int(seed)
        self.physical_collision_steps = 0
        self.world_violation_steps = 0
        self.command_speed_sum = 0.0
        self.filtered_command_speed_sum = 0.0
        self.command_filter_correction_sum = 0.0
        self.realized_speed_sum = 0.0
        self.command_count = 0
        self.filtered_defender_actions.fill(0.0)
        self.last_requested_defender_actions.fill(0.0)
        self.last_executed_defender_actions.fill(0.0)
        self.boundary_governor_active_steps = 0
        self.boundary_governor_correction_sum = 0.0
        self.vertical_recovery_active_steps = 0
        self.vertical_recovery_agent_steps = 0
        self.last_vertical_recovery_mask.fill(False)
        self.vertical_emergency_active_steps = 0
        self.vertical_emergency_agent_steps = 0
        self.last_vertical_emergency_mask.fill(False)
        self.last_vertical_emergency_required_distance.fill(0.0)
        self.attitude_recovery_active_steps = 0
        self.attitude_recovery_agent_steps = 0
        self.last_attitude_recovery_mask.fill(False)
        self.last_attitude_tilt.fill(0.0)
        self.last_pid_target_positions = self.defender_positions.copy()
        self._reset_softbody_state()
        self._create_aviary()
        self._add_obstacle_bodies()
        if self.pybullet_softbody_net_enabled:
            self._create_softbody_target_body()
        self._sync_defender_state()
        if record_history:
            self._record_history()
        return self.observe()

    def step(
        self,
        defender_actions: np.ndarray,
        record_history: bool = False,
        close_cage: bool = False,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.aviary is None:
            raise RuntimeError("Call reset() before step() in PyBulletEncirclement3DEnv.")
        previous_corners = self.defender_positions.copy()
        previous_target = self.target_position.copy()
        self._begin_capture_step()
        self._sync_softbody_target_body()
        requested_actions = np.asarray(defender_actions, dtype=np.float64)
        if requested_actions.shape != (self.n_defenders, 3):
            raise ValueError(f"Expected actions with shape {(self.n_defenders, 3)}, got {requested_actions.shape}.")
        requested_actions = self._clip_rows(requested_actions, float(self.agents["defender_max_speed"]))
        defender_actions = self._filter_defender_actions(requested_actions)
        # Hold the high-level command for one benchmark interval, while
        # recomputing the low-level controller at the 240 Hz physics rate.
        if self.pybullet_control_mode == "position_pid":
            unconstrained_references = self.defender_positions + defender_actions * self.pybullet_position_horizon
            target_positions = np.clip(
                unconstrained_references,
                self.lower[None, :] + self.pybullet_boundary_reference_margin[None, :],
                self.upper[None, :] - self.pybullet_boundary_reference_margin[None, :],
            )
            target_velocities = defender_actions * self.pybullet_position_velocity_feedforward
            recovery_agent_count = self._apply_vertical_recovery(target_positions, target_velocities)
            if recovery_agent_count:
                self.vertical_recovery_active_steps += 1
                self.vertical_recovery_agent_steps += recovery_agent_count
            emergency_agent_count = self._apply_vertical_emergency(target_positions, target_velocities)
            if emergency_agent_count:
                self.vertical_emergency_active_steps += 1
                self.vertical_emergency_agent_steps += emergency_agent_count
            attitude_recovery_agent_count = self._apply_attitude_recovery(target_positions, target_velocities)
            if attitude_recovery_agent_count:
                self.attitude_recovery_active_steps += 1
                self.attitude_recovery_agent_steps += attitude_recovery_agent_count
            reference_correction = float(np.linalg.norm(target_positions - unconstrained_references))
            self.last_pid_target_positions = target_positions.copy()
            if reference_correction > 1e-9:
                self.boundary_governor_active_steps += 1
                self.boundary_governor_correction_sum += reference_correction
            for _ in range(self.pybullet_aggregate_steps):
                rpm_action: dict[str, np.ndarray] = {}
                for index in range(self.n_defenders):
                    state = self.aviary._getDroneStateVector(index)
                    rpm, _position_error, _yaw_error = self.pid_controllers[index].computeControl(
                        control_timestep=1.0 / self.pybullet_frequency,
                        cur_pos=state[0:3],
                        cur_quat=state[3:7],
                        cur_vel=state[10:13],
                        cur_ang_vel=state[13:16],
                        target_pos=target_positions[index],
                        target_rpy=np.array([0.0, 0.0, state[9]]),
                        target_vel=target_velocities[index],
                    )
                    rpm_action[str(index)] = rpm
                self.aviary.step(rpm_action)
        elif self.pybullet_control_mode == "velocity_aviary":
            self.last_vertical_recovery_mask.fill(False)
            self.last_vertical_emergency_mask.fill(False)
            self.last_vertical_emergency_required_distance.fill(0.0)
            self.last_attitude_recovery_mask.fill(False)
            self.last_attitude_tilt.fill(0.0)
            self.last_pid_target_positions = self.defender_positions.copy()
            aviary_action = self.velocity_actions_to_aviary(defender_actions)
            for _ in range(self.pybullet_aggregate_steps):
                self.aviary.step(aviary_action)
        else:
            raise ValueError(
                f"Unknown pybullet_control_mode={self.pybullet_control_mode!r}; "
                "choose position_pid or velocity_aviary."
            )
        self._sync_defender_state()
        self._record_softbody_diagnostics()
        self.command_speed_sum += float(np.linalg.norm(requested_actions, axis=1).sum())
        self.filtered_command_speed_sum += float(np.linalg.norm(defender_actions, axis=1).sum())
        self.realized_speed_sum += float(np.linalg.norm(self.defender_velocities, axis=1).sum())
        self.command_count += self.n_defenders

        target_action = self._target_action()
        self.target_velocity = self._move_toward_velocity(
            self.target_velocity[None, :],
            target_action[None, :],
            max_delta=float(self.agents["target_max_acceleration"]) * self.dt,
        )[0]
        self.target_position += self.target_velocity * self.dt
        self._enforce_world_bounds(self.target_position[None, :], self.target_velocity[None, :])
        self._sync_softbody_target_body()
        self._resolve_closed_cage_contact(previous_corners, previous_target)

        self.step_count += 1
        metrics = self._metrics()
        physical_contact = self._has_physical_contact()
        world_violation = self._has_world_violation()
        if world_violation:
            self.world_violation_steps += 1
        if physical_contact:
            self.physical_collision_steps += 1
        bounds_clearance = self._boundary_clearance()
        metrics["min_clearance"] = min(float(metrics["min_clearance"]), bounds_clearance)
        metrics["collision"] = bool(metrics["collision"] or physical_contact or world_violation)
        self.min_clearance = min(self.min_clearance, float(metrics["min_clearance"]))
        if metrics["collision"]:
            self.collision_steps += 1

        task_outcome = self._update_task_state(metrics, close_cage=close_cage)
        if self.last_capture_close_accepted:
            self._deploy_softbody_net()
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
            "pybullet_physics": self.pybullet_physics_name,
            "pybullet_control_mode": self.pybullet_control_mode,
            "pybullet_control_dt": self.control_dt,
            "pybullet_speed_limit": self.pybullet_speed_limit,
            "pybullet_position_horizon": self.pybullet_position_horizon,
            "pybullet_position_velocity_feedforward": self.pybullet_position_velocity_feedforward,
            "pybullet_command_max_acceleration": self.pybullet_command_max_acceleration,
            "pybullet_vertical_recovery_enabled": self.pybullet_vertical_recovery_enabled,
            "pybullet_vertical_recovery_altitude": self.pybullet_vertical_recovery_altitude,
            "pybullet_vertical_recovery_descend_speed": self.pybullet_vertical_recovery_descend_speed,
            "pybullet_vertical_recovery_climb_height": self.pybullet_vertical_recovery_climb_height,
            "pybullet_vertical_emergency_enabled": self.pybullet_vertical_emergency_enabled,
            "pybullet_vertical_emergency_braking_deceleration": self.pybullet_vertical_emergency_braking_deceleration,
            "pybullet_vertical_emergency_reaction_time": self.pybullet_vertical_emergency_reaction_time,
            "pybullet_vertical_emergency_margin": self.pybullet_vertical_emergency_margin,
            "pybullet_vertical_emergency_climb_height": self.pybullet_vertical_emergency_climb_height,
            "pybullet_attitude_recovery_enabled": self.pybullet_attitude_recovery_enabled,
            "pybullet_attitude_recovery_max_tilt": self.pybullet_attitude_recovery_max_tilt,
            "pybullet_attitude_recovery_climb_height": self.pybullet_attitude_recovery_climb_height,
            "pybullet_boundary_reference_margin": self.pybullet_boundary_reference_margin.copy(),
            "physical_contact": physical_contact,
            "physical_collision_steps": self.physical_collision_steps,
            "world_violation": world_violation,
            "world_violation_steps": self.world_violation_steps,
            "mean_command_speed": self.command_speed_sum / max(self.command_count, 1),
            "mean_filtered_command_speed": self.filtered_command_speed_sum / max(self.command_count, 1),
            "mean_command_filter_correction": self.command_filter_correction_sum / max(self.step_count, 1),
            "mean_realized_speed": self.realized_speed_sum / max(self.command_count, 1),
            "boundary_governor_active_steps": self.boundary_governor_active_steps,
            "mean_boundary_governor_correction": self.boundary_governor_correction_sum / max(self.step_count, 1),
            "vertical_recovery_active_steps": self.vertical_recovery_active_steps,
            "vertical_recovery_agent_steps": self.vertical_recovery_agent_steps,
            "vertical_recovery_active_agents": self.last_vertical_recovery_mask.copy(),
            "vertical_emergency_active_steps": self.vertical_emergency_active_steps,
            "vertical_emergency_agent_steps": self.vertical_emergency_agent_steps,
            "vertical_emergency_active_agents": self.last_vertical_emergency_mask.copy(),
            "vertical_emergency_required_distance": self.last_vertical_emergency_required_distance.copy(),
            "attitude_recovery_active_steps": self.attitude_recovery_active_steps,
            "attitude_recovery_agent_steps": self.attitude_recovery_agent_steps,
            "attitude_recovery_active_agents": self.last_attitude_recovery_mask.copy(),
            "attitude_tilt": self.last_attitude_tilt.copy(),
            **self._softbody_info(),
        }
        return self.observe(), reward, terminated, truncated, info

    def velocity_actions_to_aviary(self, velocities: np.ndarray) -> dict[str, np.ndarray]:
        """Translate world-frame velocity targets to VelocityAviary's action API."""
        values = np.asarray(velocities, dtype=np.float64)
        speed = np.linalg.norm(values, axis=1)
        directions = np.divide(values, speed[:, None], out=np.zeros_like(values), where=speed[:, None] > 1e-9)
        fractions = np.clip(speed / self.pybullet_speed_limit, 0.0, 1.0)
        return {
            str(index): np.concatenate([directions[index], np.array([fractions[index]])]).astype(np.float32)
            for index in range(self.n_defenders)
        }

    def close(self) -> None:
        if self.aviary is not None:
            self.aviary.close()
        self.aviary = None
        self.pybullet = None
        self.obstacle_body_ids = []
        self._reset_softbody_state()

    def _create_aviary(self) -> None:
        pybullet, DroneModel, Physics, CtrlAviary, DSLPIDControl, VelocityAviary = _load_pybullet_dependencies()
        physics_name = self.pybullet_physics_name.upper()
        if not hasattr(Physics, physics_name):
            allowed = ", ".join(item.name.lower() for item in Physics)
            raise ValueError(f"Unknown pybullet_physics={self.pybullet_physics_name!r}; choose one of {allowed}.")
        self.pybullet = pybullet
        aviary_class = CtrlAviary if self.pybullet_control_mode == "position_pid" else VelocityAviary
        self.aviary = aviary_class(
            drone_model=DroneModel.CF2X,
            num_drones=self.n_defenders,
            neighbourhood_radius=float("inf"),
            initial_xyzs=self.defender_positions.copy(),
            initial_rpys=np.zeros((self.n_defenders, 3), dtype=np.float64),
            physics=getattr(Physics, physics_name),
            freq=self.pybullet_frequency,
            aggregate_phy_steps=1,
            gui=self.pybullet_gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
            deformable_world=self.pybullet_deformable_world,
        )
        if self.pybullet_control_mode == "velocity_aviary":
            # VelocityAviary defaults to 3% of URDF max speed. The benchmark
            # action is already specified in m/s, so use its declared limit.
            self.aviary.SPEED_LIMIT = self.pybullet_speed_limit
        elif self.pybullet_control_mode == "position_pid":
            self.pid_controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(self.n_defenders)]

    def _add_obstacle_bodies(self) -> None:
        if self.pybullet is None or self.aviary is None:
            raise RuntimeError("PyBullet aviary has not been created.")
        self.obstacle_body_ids = []
        for obstacle in self.obstacles:
            collision_shape = self.pybullet.createCollisionShape(
                self.pybullet.GEOM_CYLINDER,
                radius=float(obstacle.radius),
                height=float(obstacle.height),
                physicsClientId=self.aviary.CLIENT,
            )
            visual_shape = self.pybullet.createVisualShape(
                self.pybullet.GEOM_CYLINDER,
                radius=float(obstacle.radius),
                length=float(obstacle.height),
                rgbaColor=[0.45, 0.45, 0.45, 0.75],
                physicsClientId=self.aviary.CLIENT,
            )
            body_id = self.pybullet.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[float(obstacle.center_xy[0]), float(obstacle.center_xy[1]), float(obstacle.height) / 2.0],
                physicsClientId=self.aviary.CLIENT,
            )
            self.obstacle_body_ids.append(int(body_id))

    def _reset_softbody_state(self) -> None:
        """Forget handles invalidated when the PyBullet client is closed."""
        self.pybullet_softbody_net_id = None
        self.pybullet_softbody_target_body_id = None
        self.pybullet_softbody_mesh = None
        self.pybullet_softbody_anchor_constraints = []
        self.pybullet_softbody_deployment_events = 0
        self.pybullet_softbody_target_contact_steps = 0
        self.pybullet_softbody_peak_target_normal_force = 0.0
        self.pybullet_softbody_min_target_contact_distance = float("inf")
        self.pybullet_softbody_max_anchor_error = 0.0
        self.last_pybullet_softbody_target_contact = False

    def _create_softbody_target_body(self) -> None:
        """Create a kinematic collision target for native-net diagnostics.

        The benchmark target remains kinematic. This body supplies a collision
        surface to PyBullet's soft body only; it does not make the target's
        contact response or capture result physically validated.
        """
        if self.pybullet is None or self.aviary is None:
            raise RuntimeError("PyBullet aviary has not been created.")
        radius = float(self.capture["target_radius"])
        collision_shape = self.pybullet.createCollisionShape(
            self.pybullet.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self.aviary.CLIENT,
        )
        visual_shape = self.pybullet.createVisualShape(
            self.pybullet.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.85, 0.15, 0.15, 0.65],
            physicsClientId=self.aviary.CLIENT,
        )
        self.pybullet_softbody_target_body_id = int(
            self.pybullet.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=self.target_position.tolist(),
                physicsClientId=self.aviary.CLIENT,
            )
        )

    def _sync_softbody_target_body(self) -> None:
        if self.pybullet_softbody_target_body_id is None:
            return
        if self.pybullet is None or self.aviary is None:
            raise RuntimeError("PyBullet soft-body target exists without an aviary.")
        self.pybullet.resetBasePositionAndOrientation(
            self.pybullet_softbody_target_body_id,
            self.target_position.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.aviary.CLIENT,
        )

    def _deploy_softbody_net(self) -> None:
        """Deploy one native mesh after the analytical closure gate accepts."""
        if not self.pybullet_softbody_net_enabled or self.pybullet_softbody_net_id is not None:
            return
        if self.pybullet is None or self.aviary is None or self.pybullet_softbody_net_mesh_directory is None:
            raise RuntimeError("PyBullet soft-body deployment is not fully configured.")
        if self.pybullet_softbody_seed is None:
            raise RuntimeError("PyBullet soft-body deployment requires an episode seed.")
        mesh_path = self.pybullet_softbody_net_mesh_directory / (
            f"capture_net_seed{self.pybullet_softbody_seed:06d}_episode"
            f"{self.pybullet_softbody_episode_index:05d}.obj"
        )
        self.pybullet_softbody_mesh = write_tetrahedral_softbody_obj(
            mesh_path,
            self.defender_positions,
            face_subdivisions=self.pybullet_softbody_net_face_subdivisions,
        )
        self.pybullet_softbody_net_id = load_tetrahedral_softbody(
            self.pybullet,
            self.pybullet_softbody_mesh,
            PyBulletSoftBodyConfig(
                mass_kg=self.pybullet_softbody_net_mass_kg,
                spring_elastic_stiffness=self.pybullet_softbody_net_elastic_stiffness,
                spring_damping_stiffness=self.pybullet_softbody_net_damping_stiffness,
                friction_coefficient=self.pybullet_softbody_net_friction,
            ),
            physics_client_id=self.aviary.CLIENT,
        )
        self.pybullet_softbody_anchor_constraints = attach_tetrahedral_softbody_anchors(
            self.pybullet,
            self.pybullet_softbody_net_id,
            self.aviary.DRONE_IDS,
            physics_client_id=self.aviary.CLIENT,
        )
        self.pybullet_softbody_deployment_events += 1
        self._record_softbody_diagnostics()

    def _record_softbody_diagnostics(self) -> None:
        """Record native-softbody contact and anchor diagnostics without scoring capture."""
        self.last_pybullet_softbody_target_contact = False
        if self.pybullet_softbody_net_id is None:
            return
        if self.pybullet is None or self.aviary is None:
            raise RuntimeError("PyBullet soft-body exists without an aviary.")
        vertices = softbody_vertices(
            self.pybullet,
            self.pybullet_softbody_net_id,
            physics_client_id=self.aviary.CLIENT,
        )
        if len(vertices) < self.n_defenders:
            raise RuntimeError("PyBullet soft-body mesh has fewer vertices than anchor defenders.")
        anchor_error = float(np.max(np.linalg.norm(vertices[: self.n_defenders] - self.defender_positions, axis=1)))
        self.pybullet_softbody_max_anchor_error = max(self.pybullet_softbody_max_anchor_error, anchor_error)
        if self.pybullet_softbody_target_body_id is None:
            return
        contacts = self.pybullet.getContactPoints(
            bodyA=self.pybullet_softbody_target_body_id,
            bodyB=self.pybullet_softbody_net_id,
            physicsClientId=self.aviary.CLIENT,
        )
        self.last_pybullet_softbody_target_contact = bool(contacts)
        if not contacts:
            return
        self.pybullet_softbody_target_contact_steps += 1
        self.pybullet_softbody_peak_target_normal_force = max(
            self.pybullet_softbody_peak_target_normal_force,
            max(float(contact[9]) for contact in contacts),
        )
        self.pybullet_softbody_min_target_contact_distance = min(
            self.pybullet_softbody_min_target_contact_distance,
            min(float(contact[8]) for contact in contacts),
        )

    def _softbody_info(self) -> dict[str, Any]:
        """Expose native-softbody provenance and diagnostics separately from capture success."""
        return {
            "pybullet_softbody_net_enabled": self.pybullet_softbody_net_enabled,
            "pybullet_softbody_net_deployed": self.pybullet_softbody_net_id is not None,
            "pybullet_softbody_net_id": self.pybullet_softbody_net_id,
            "pybullet_softbody_target_body_id": self.pybullet_softbody_target_body_id,
            "pybullet_softbody_mesh_path": (
                str(self.pybullet_softbody_mesh.path) if self.pybullet_softbody_mesh is not None else None
            ),
            "pybullet_softbody_mesh_vertices": (
                self.pybullet_softbody_mesh.vertices if self.pybullet_softbody_mesh is not None else 0
            ),
            "pybullet_softbody_mesh_triangles": (
                self.pybullet_softbody_mesh.triangles if self.pybullet_softbody_mesh is not None else 0
            ),
            "pybullet_softbody_anchor_constraints": len(self.pybullet_softbody_anchor_constraints),
            "pybullet_softbody_deployment_events": self.pybullet_softbody_deployment_events,
            "pybullet_softbody_target_contact": self.last_pybullet_softbody_target_contact,
            "pybullet_softbody_target_contact_steps": self.pybullet_softbody_target_contact_steps,
            "pybullet_softbody_peak_target_normal_force": self.pybullet_softbody_peak_target_normal_force,
            "pybullet_softbody_min_target_contact_distance": (
                self.pybullet_softbody_min_target_contact_distance
                if np.isfinite(self.pybullet_softbody_min_target_contact_distance)
                else None
            ),
            "pybullet_softbody_max_anchor_error": self.pybullet_softbody_max_anchor_error,
        }

    def _sync_defender_state(self) -> None:
        if self.aviary is None:
            raise RuntimeError("PyBullet aviary has not been created.")
        self.defender_positions = np.asarray(self.aviary.pos, dtype=np.float64).copy()
        self.defender_velocities = np.asarray(self.aviary.vel, dtype=np.float64).copy()

    def _has_physical_contact(self) -> bool:
        if self.pybullet is None or self.aviary is None:
            return False
        drone_ids = {int(value) for value in self.aviary.DRONE_IDS}
        for contact in self.pybullet.getContactPoints(physicsClientId=self.aviary.CLIENT):
            body_a, body_b = int(contact[1]), int(contact[2])
            if body_a in drone_ids and body_b != body_a:
                return True
            if body_b in drone_ids and body_a != body_b:
                return True
        return False

    def _has_world_violation(self) -> bool:
        return bool(np.any(self.defender_positions < self.lower[None, :]) or np.any(self.defender_positions > self.upper[None, :]))

    def _boundary_clearance(self) -> float:
        return float(np.min(np.minimum(self.defender_positions - self.lower[None, :], self.upper[None, :] - self.defender_positions)))

    def _filter_defender_actions(self, requested_actions: np.ndarray) -> np.ndarray:
        """Apply an optional acceleration bound before the low-level PID.

        A zero value leaves the historical position-PID interface unchanged.
        Positive values limit the rate at which a kinematic-policy velocity
        request reaches the physical simulator, so the applied command remains
        within an explicit achievable-response envelope.
        """
        requested = np.asarray(requested_actions, dtype=np.float64)
        self.last_requested_defender_actions = requested.copy()
        if self.pybullet_command_max_acceleration == 0.0:
            executed = requested.copy()
        else:
            executed = self._move_toward_velocity(
                self.filtered_defender_actions,
                requested,
                max_delta=self.pybullet_command_max_acceleration * self.control_dt,
            )
            executed = self._clip_rows(executed, float(self.agents["defender_max_speed"]))
        self.filtered_defender_actions = executed.copy()
        self.last_executed_defender_actions = executed.copy()
        self.command_filter_correction_sum += float(np.linalg.norm(requested - executed))
        return executed

    def _apply_vertical_recovery(self, target_positions: np.ndarray, target_velocities: np.ndarray) -> int:
        """Prioritize an emergency climb when a low defender is descending.

        The frozen policy is trained in a kinematic environment and cannot
        account for transient attitude-induced loss of vertical authority. This
        supervisor only changes the low-level reference while the measured
        state predicts an imminent floor violation: it commands an upward
        position step and holds the horizontal reference to reduce tilt.
        """
        if not self.pybullet_vertical_recovery_enabled:
            self.last_vertical_recovery_mask.fill(False)
            return 0
        active = (self.defender_positions[:, 2] <= self.pybullet_vertical_recovery_altitude) & (
            self.defender_velocities[:, 2] <= -self.pybullet_vertical_recovery_descend_speed
        )
        self.last_vertical_recovery_mask = active.copy()
        for index in np.flatnonzero(active):
            target_positions[index, :2] = self.defender_positions[index, :2]
            target_positions[index, 2] = min(
                self.upper[2] - self.pybullet_boundary_reference_margin[2],
                max(
                    target_positions[index, 2],
                    self.defender_positions[index, 2] + self.pybullet_vertical_recovery_climb_height,
                ),
            )
            target_velocities[index, :2] = 0.0
            target_velocities[index, 2] = max(target_velocities[index, 2], 0.0)
        return int(np.count_nonzero(active))

    def _apply_vertical_emergency(self, target_positions: np.ndarray, target_velocities: np.ndarray) -> int:
        """Override references when calibrated vertical stopping distance is exhausted.

        This supervisor intentionally acts below the policy and response-CBF
        command layers. It is triggered by the measured altitude and downward
        velocity, rather than by an arbitrary altitude threshold. The braking
        model is specific to the pinned PyBullet position-PID backend.
        """
        if not self.pybullet_vertical_emergency_enabled:
            self.last_vertical_emergency_mask.fill(False)
            self.last_vertical_emergency_required_distance.fill(0.0)
            return 0
        downward_speed = np.maximum(-self.defender_velocities[:, 2], 0.0)
        self.last_vertical_emergency_required_distance = (
            downward_speed * self.pybullet_vertical_emergency_reaction_time
            + downward_speed**2 / (2.0 * self.pybullet_vertical_emergency_braking_deceleration)
            + self.pybullet_vertical_emergency_margin
        )
        available_distance = self.defender_positions[:, 2] - self.lower[2] - float(self.agents["drone_radius"])
        active = (downward_speed > 0.0) & (available_distance <= self.last_vertical_emergency_required_distance)
        self.last_vertical_emergency_mask = active.copy()
        for index in np.flatnonzero(active):
            target_positions[index, :2] = self.defender_positions[index, :2]
            target_positions[index, 2] = min(
                self.upper[2] - self.pybullet_boundary_reference_margin[2],
                max(
                    target_positions[index, 2],
                    self.defender_positions[index, 2] + self.pybullet_vertical_emergency_climb_height,
                ),
            )
            target_velocities[index, :2] = 0.0
            target_velocities[index, 2] = max(target_velocities[index, 2], 0.0)
        return int(np.count_nonzero(active))

    def _apply_attitude_recovery(self, target_positions: np.ndarray, target_velocities: np.ndarray) -> int:
        """Preempt aggressive tracking before the position PID leaves its tilt envelope."""
        if not self.pybullet_attitude_recovery_enabled or self.aviary is None:
            self.last_attitude_recovery_mask.fill(False)
            self.last_attitude_tilt.fill(0.0)
            return 0
        roll_pitch = np.asarray(self.aviary.rpy, dtype=np.float64)[:, :2]
        self.last_attitude_tilt = np.linalg.norm(roll_pitch, axis=1)
        active = self.last_attitude_tilt >= self.pybullet_attitude_recovery_max_tilt
        self.last_attitude_recovery_mask = active.copy()
        for index in np.flatnonzero(active):
            target_positions[index, :2] = self.defender_positions[index, :2]
            target_positions[index, 2] = min(
                self.upper[2] - self.pybullet_boundary_reference_margin[2],
                max(
                    target_positions[index, 2],
                    self.defender_positions[index, 2] + self.pybullet_attitude_recovery_climb_height,
                ),
            )
            target_velocities[index, :2] = 0.0
            target_velocities[index, 2] = max(target_velocities[index, 2], 0.0)
        return int(np.count_nonzero(active))

    @staticmethod
    def _boundary_margin(value: Any) -> np.ndarray:
        margin = np.asarray(value, dtype=np.float64)
        if margin.ndim == 0:
            return np.full(3, float(margin), dtype=np.float64)
        if margin.shape != (3,):
            raise ValueError("pybullet_boundary_reference_margin must be a scalar or a three-element vector.")
        return margin.copy()
