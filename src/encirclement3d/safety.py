"""Discrete-time CBF-QP safety filtering for the kinematic benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

from .environment import Encirclement3DEnv, _unit


@dataclass(frozen=True)
class SafetyFilterDiagnostics:
    action_correction_norm: float
    minimum_constraint_value: float
    solver_success: bool
    used_fallback: bool


class DiscreteTimeCBFSafetyFilter:
    """Minimally modify next defender velocities subject to CBF constraints.

    The decision variable is the simultaneous next velocity of all defenders.
    This matches the environment's first-order velocity dynamics and lets
    pairwise separation constraints depend on both agents' actions.
    """

    def __init__(self, env: Encirclement3DEnv):
        self.env = env
        self.gamma = float(env.task["cbf_gamma"])
        self.obstacle_margin = float(env.task["cbf_obstacle_margin"])
        self.inter_agent_margin = float(env.task["cbf_inter_agent_margin"])
        self.target_margin = float(env.task["cbf_target_margin"])
        self.target_activation_distance = float(env.task["cbf_target_activation_distance"])
        self.last_solver_status = 0
        self.last_solver_message = "not run"

    def filter(
        self,
        desired_actions: np.ndarray,
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, SafetyFilterDiagnostics]:
        desired_actions = self.env._clip_rows(
            np.asarray(desired_actions, dtype=np.float64),
            float(self.env.agents["defender_max_speed"]),
        )
        current_velocities = np.asarray(observation["defender_velocities"], dtype=np.float64)
        reference = self._reachable_reference(current_velocities, desired_actions)
        constraint_rows, lower_bounds = self._linear_constraints(observation)

        def objective(flattened: np.ndarray) -> float:
            velocity = flattened.reshape(self.env.n_defenders, 3)
            return 0.5 * float(np.sum((velocity - reference) ** 2))

        constraint_matrix = np.vstack(constraint_rows)
        linear_constraint = LinearConstraint(
            constraint_matrix,
            np.asarray(lower_bounds, dtype=np.float64),
            np.full(len(lower_bounds), np.inf),
        )

        max_speed = float(self.env.agents["defender_max_speed"])
        max_delta = float(self.env.agents["defender_max_acceleration"]) * self.env.dt
        def motion_constraint_values(flattened: np.ndarray) -> np.ndarray:
            velocity = flattened.reshape(self.env.n_defenders, 3)
            speed_slack = max_speed**2 - np.sum(velocity**2, axis=1)
            acceleration_slack = max_delta**2 - np.sum((velocity - current_velocities) ** 2, axis=1)
            return np.concatenate([speed_slack, acceleration_slack])

        motion_constraints = NonlinearConstraint(
            motion_constraint_values,
            np.zeros(2 * self.env.n_defenders),
            np.full(2 * self.env.n_defenders, np.inf),
        )

        result = minimize(
            objective,
            x0=reference.reshape(-1),
            method="SLSQP",
            constraints=[linear_constraint, motion_constraints],
            options={"ftol": 1e-8, "maxiter": 30, "disp": False},
        )
        self.last_solver_status = int(result.status)
        self.last_solver_message = str(result.message)
        # SLSQP can return a constraint-feasible candidate with a non-success
        # status (for example after its iteration budget). Safety therefore
        # depends on the measured residual, not on the optimiser status alone.
        candidate = result.x.reshape(self.env.n_defenders, 3)
        candidate = self._reachable_reference(current_velocities, candidate)
        minimum_constraint = self._minimum_constraint(candidate, constraint_rows, lower_bounds)
        feasible = bool(minimum_constraint >= -1e-5)
        used_repair = False
        if not feasible:
            # Repair a non-converged QP iterate by alternating projections onto
            # the same CBF half-spaces and the exact executable-motion balls.
            candidate = self._project_to_feasible_set(candidate, current_velocities, constraint_rows, lower_bounds)
            minimum_constraint = self._minimum_constraint(candidate, constraint_rows, lower_bounds)
            used_repair = True
        if minimum_constraint < -1e-5:
            candidate = self._fallback(reference, observation)
            minimum_constraint = self._minimum_constraint(candidate, constraint_rows, lower_bounds)

        diagnostics = SafetyFilterDiagnostics(
            action_correction_norm=float(np.mean(np.linalg.norm(candidate - desired_actions, axis=1))),
            minimum_constraint_value=minimum_constraint,
            solver_success=bool(result.success),
            used_fallback=used_repair,
        )
        return candidate, diagnostics

    def _linear_constraints(self, observation: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        target_position = np.asarray(observation["target_position"], dtype=np.float64)
        target_velocity = np.asarray(observation["target_velocity"], dtype=np.float64)
        radius = float(self.env.agents["drone_radius"])
        dt = self.env.dt
        rows: list[np.ndarray] = []
        bounds: list[float] = []

        def add_single(index: int, normal: np.ndarray, bound: float) -> None:
            row = np.zeros(3 * self.env.n_defenders, dtype=np.float64)
            row[3 * index : 3 * index + 3] = normal
            rows.append(row)
            bounds.append(float(bound))

        for index, position in enumerate(positions):
            for obstacle in self.env.obstacles:
                clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                h = clearance - radius - self.obstacle_margin
                add_single(index, normal, self._closing_speed_bound(h, float(self.env.agents["defender_max_acceleration"])))

            target_delta = position - target_position
            target_distance = float(np.linalg.norm(target_delta))
            if target_distance < self.target_activation_distance:
                target_normal = _unit(target_delta, fallback=np.array([1.0, 0.0, 0.0]))
                target_h = target_distance - (2.0 * radius + self.target_margin)
                add_single(
                    index,
                    target_normal,
                    float(target_normal @ target_velocity)
                    + self._closing_speed_bound(
                        target_h,
                        float(self.env.agents["defender_max_acceleration"])
                        + float(self.env.agents["target_max_acceleration"]),
                    ),
                )

            for axis in range(3):
                lower_h = position[axis] - self.env.lower[axis] - radius
                lower_normal = np.zeros(3, dtype=np.float64)
                lower_normal[axis] = 1.0
                add_single(
                    index,
                    lower_normal,
                    self._closing_speed_bound(lower_h, float(self.env.agents["defender_max_acceleration"])),
                )

                upper_h = self.env.upper[axis] - radius - position[axis]
                upper_normal = np.zeros(3, dtype=np.float64)
                upper_normal[axis] = -1.0
                add_single(
                    index,
                    upper_normal,
                    self._closing_speed_bound(upper_h, float(self.env.agents["defender_max_acceleration"])),
                )

        for first in range(self.env.n_defenders):
            for second in range(first + 1, self.env.n_defenders):
                delta = positions[first] - positions[second]
                distance = float(np.linalg.norm(delta))
                normal = _unit(delta, fallback=np.array([1.0, 0.0, 0.0]))
                h = distance - (2.0 * radius + self.inter_agent_margin)
                row = np.zeros(3 * self.env.n_defenders, dtype=np.float64)
                row[3 * first : 3 * first + 3] = normal
                row[3 * second : 3 * second + 3] = -normal
                rows.append(row)
                bounds.append(self._closing_speed_bound(h, 2.0 * float(self.env.agents["defender_max_acceleration"])))
        return rows, bounds

    def _closing_speed_bound(self, clearance: float, maximum_deceleration: float) -> float:
        """Return the least-safe admissible normal closing velocity.

        The first term is the usual discrete CBF condition. The second is a
        braking-distance condition for the environment's rate-limited velocity
        dynamics. Taking the stricter bound prevents a fast agent from entering
        a state from which the next CBF constraint is physically unreachable.
        """
        clearance = max(float(clearance), 0.0)
        first_order_bound = -self.gamma * clearance / self.env.dt
        braking_bound = -float(np.sqrt(2.0 * maximum_deceleration * clearance))
        return max(first_order_bound, braking_bound)

    def _reachable_reference(self, current: np.ndarray, requested: np.ndarray) -> np.ndarray:
        max_delta = float(self.env.agents["defender_max_acceleration"]) * self.env.dt
        next_velocity = self.env._move_toward_velocity(current, requested, max_delta=max_delta)
        return self.env._clip_rows(next_velocity, float(self.env.agents["defender_max_speed"]))

    @staticmethod
    def _minimum_constraint(velocity: np.ndarray, rows: list[np.ndarray], bounds: list[float]) -> float:
        if not rows:
            return float("inf")
        flattened = velocity.reshape(-1)
        return float(min(row @ flattened - bound for row, bound in zip(rows, bounds, strict=True)))

    def _project_to_feasible_set(
        self,
        initial: np.ndarray,
        current: np.ndarray,
        rows: list[np.ndarray],
        bounds: list[float],
    ) -> np.ndarray:
        """Project onto CBF half-spaces and executable velocity sets in turn."""
        candidate = initial.reshape(-1).copy()
        max_speed = float(self.env.agents["defender_max_speed"])
        max_delta = float(self.env.agents["defender_max_acceleration"]) * self.env.dt

        for _ in range(50):
            for row, bound in zip(rows, bounds, strict=True):
                residual = float(row @ candidate - bound)
                if residual < 0.0:
                    candidate += (-residual / float(row @ row)) * row

            velocity = candidate.reshape(self.env.n_defenders, 3)
            velocity = self.env._clip_rows(velocity, max_speed)
            velocity_delta = velocity - current
            delta_norm = np.linalg.norm(velocity_delta, axis=1, keepdims=True)
            velocity = current + velocity_delta * np.minimum(1.0, max_delta / np.maximum(delta_norm, 1e-9))
            candidate = velocity.reshape(-1)
            if self._minimum_constraint(velocity, rows, bounds) >= -1e-5:
                return velocity
        return candidate.reshape(self.env.n_defenders, 3)

    def _fallback(self, reference: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
        """Use conservative outward steering when the full constrained solve fails."""
        safe = reference.copy()
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        radius = float(self.env.agents["drone_radius"])
        for index, position in enumerate(positions):
            correction = np.zeros(3, dtype=np.float64)
            for obstacle in self.env.obstacles:
                clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                h = clearance - radius - self.obstacle_margin
                if h < 1.0:
                    correction += normal * (1.0 - h) * float(self.env.agents["defender_max_speed"])
            for other_index, other_position in enumerate(positions):
                if index == other_index:
                    continue
                delta = position - other_position
                h = float(np.linalg.norm(delta)) - (2.0 * radius + self.inter_agent_margin)
                if h < 1.0:
                    correction += _unit(delta) * (1.0 - h) * float(self.env.agents["defender_max_speed"])
            safe[index] += correction
        return self._reachable_reference(
            np.asarray(observation["defender_velocities"], dtype=np.float64),
            safe,
        )


class PyBulletResponseCBFSafetyFilter(DiscreteTimeCBFSafetyFilter):
    """CBF shield over the PyBullet PID interface's identified response.

    The decision variable is the command that will be executed after the
    command-rate governor, rather than an ideal next velocity. A calibrated
    model predicts interval-average velocity as ``a * velocity + b * command``.
    """

    def __init__(self, env: Encirclement3DEnv):
        super().__init__(env)
        dynamics = env.config.get("dynamics", {})
        self.displacement_velocity_coefficient = float(
            dynamics.get("pybullet_response_displacement_velocity_coefficient", 0.0)
        )
        self.displacement_command_coefficient = float(
            dynamics.get("pybullet_response_displacement_command_coefficient", 0.0)
        )
        self.displacement_error_p95 = float(dynamics.get("pybullet_response_displacement_error_p95", 0.0))
        self.uncertainty_multiplier = float(dynamics.get("pybullet_response_cbf_uncertainty_multiplier", 0.0))
        self.response_gamma = float(dynamics.get("pybullet_response_cbf_gamma", self.gamma))
        self.control_dt = float(getattr(env, "control_dt", env.dt))
        self.command_acceleration = float(
            dynamics.get("pybullet_command_max_acceleration", env.agents["defender_max_acceleration"])
        )
        if self.displacement_velocity_coefficient < 0.0:
            raise ValueError("pybullet_response_displacement_velocity_coefficient must be non-negative.")
        if self.displacement_command_coefficient <= 0.0:
            raise ValueError("pybullet_response_displacement_command_coefficient must be positive.")
        if self.displacement_error_p95 < 0.0 or self.uncertainty_multiplier < 0.0:
            raise ValueError("PyBullet response-model error values must be non-negative.")
        if not 0.0 < self.response_gamma <= 1.0:
            raise ValueError("pybullet_response_cbf_gamma must be in (0, 1].")
        if self.command_acceleration < 0.0:
            raise ValueError("pybullet_command_max_acceleration must be non-negative.")

    @property
    def position_uncertainty_margin(self) -> float:
        return self.uncertainty_multiplier * self.displacement_error_p95 * self.control_dt

    def filter(
        self,
        desired_actions: np.ndarray,
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, SafetyFilterDiagnostics]:
        desired_actions = self.env._clip_rows(
            np.asarray(desired_actions, dtype=np.float64),
            float(self.env.agents["defender_max_speed"]),
        )
        previous_command = self._previous_executed_command()
        reference = self._reachable_command(previous_command, desired_actions)
        constraint_rows, lower_bounds = self._linear_response_constraints(observation)

        def objective(flattened: np.ndarray) -> float:
            command = flattened.reshape(self.env.n_defenders, 3)
            return 0.5 * float(np.sum((command - reference) ** 2))

        linear_constraint = LinearConstraint(
            np.vstack(constraint_rows),
            np.asarray(lower_bounds, dtype=np.float64),
            np.full(len(lower_bounds), np.inf),
        )
        max_speed = float(self.env.agents["defender_max_speed"])
        constraint_count = 2 * self.env.n_defenders if self.command_acceleration > 0.0 else self.env.n_defenders

        def motion_constraint_values(flattened: np.ndarray) -> np.ndarray:
            command = flattened.reshape(self.env.n_defenders, 3)
            values = [max_speed**2 - np.sum(command**2, axis=1)]
            if self.command_acceleration > 0.0:
                max_delta = self.command_acceleration * self.control_dt
                values.append(max_delta**2 - np.sum((command - previous_command) ** 2, axis=1))
            return np.concatenate(values)

        result = minimize(
            objective,
            x0=reference.reshape(-1),
            method="SLSQP",
            constraints=[
                linear_constraint,
                NonlinearConstraint(
                    motion_constraint_values,
                    np.zeros(constraint_count),
                    np.full(constraint_count, np.inf),
                ),
            ],
            options={"ftol": 1e-8, "maxiter": 40, "disp": False},
        )
        self.last_solver_status = int(result.status)
        self.last_solver_message = str(result.message)
        candidate = self._reachable_command(previous_command, result.x.reshape(self.env.n_defenders, 3))
        minimum_constraint = self._minimum_constraint(candidate, constraint_rows, lower_bounds)
        used_repair = False
        if minimum_constraint < -1e-5:
            candidate = self._project_response_feasible_set(
                candidate,
                previous_command,
                constraint_rows,
                lower_bounds,
            )
            minimum_constraint = self._minimum_constraint(candidate, constraint_rows, lower_bounds)
            used_repair = True
        if minimum_constraint < -1e-5:
            candidate = self._response_fallback(reference, previous_command, observation)
            minimum_constraint = self._minimum_constraint(candidate, constraint_rows, lower_bounds)
        return candidate, SafetyFilterDiagnostics(
            action_correction_norm=float(np.mean(np.linalg.norm(candidate - desired_actions, axis=1))),
            minimum_constraint_value=minimum_constraint,
            solver_success=bool(result.success),
            used_fallback=used_repair,
        )

    def _previous_executed_command(self) -> np.ndarray:
        previous = np.asarray(
            getattr(self.env, "filtered_defender_actions", np.zeros((self.env.n_defenders, 3))),
            dtype=np.float64,
        )
        if previous.shape != (self.env.n_defenders, 3):
            raise ValueError("filtered_defender_actions has an unexpected shape.")
        return self.env._clip_rows(previous, float(self.env.agents["defender_max_speed"]))

    def _reachable_command(self, previous: np.ndarray, requested: np.ndarray) -> np.ndarray:
        requested = self.env._clip_rows(requested, float(self.env.agents["defender_max_speed"]))
        if self.command_acceleration <= 0.0:
            return requested
        return self.env._move_toward_velocity(
            previous,
            requested,
            max_delta=self.command_acceleration * self.control_dt,
        )

    def _linear_response_constraints(self, observation: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        velocities = np.asarray(observation["defender_velocities"], dtype=np.float64)
        target_position = np.asarray(observation["target_position"], dtype=np.float64)
        target_velocity = np.asarray(observation["target_velocity"], dtype=np.float64)
        radius = float(self.env.agents["drone_radius"])
        uncertainty = self.position_uncertainty_margin
        rows: list[np.ndarray] = []
        bounds: list[float] = []

        def add_single(index: int, normal: np.ndarray, bound: float) -> None:
            row = np.zeros(3 * self.env.n_defenders, dtype=np.float64)
            row[3 * index : 3 * index + 3] = normal
            rows.append(row)
            bounds.append(float(bound))

        def single_bound(normal: np.ndarray, clearance: float, velocity: np.ndarray, other_normal_speed: float = 0.0) -> float:
            return (
                other_normal_speed
                - self.response_gamma * max(clearance, 0.0) / self.control_dt
                - self.displacement_velocity_coefficient * float(normal @ velocity)
            ) / self.displacement_command_coefficient

        for index, position in enumerate(positions):
            for obstacle in self.env.obstacles:
                clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                h = clearance - radius - self.obstacle_margin - uncertainty
                add_single(index, normal, single_bound(normal, h, velocities[index]))

            target_delta = position - target_position
            target_distance = float(np.linalg.norm(target_delta))
            if target_distance < self.target_activation_distance:
                target_normal = _unit(target_delta, fallback=np.array([1.0, 0.0, 0.0]))
                target_h = target_distance - (2.0 * radius + self.target_margin + uncertainty)
                add_single(
                    index,
                    target_normal,
                    single_bound(
                        target_normal,
                        target_h,
                        velocities[index],
                        other_normal_speed=float(target_normal @ target_velocity),
                    ),
                )

            for axis in range(3):
                lower_normal = np.zeros(3, dtype=np.float64)
                lower_normal[axis] = 1.0
                lower_h = position[axis] - self.env.lower[axis] - radius - uncertainty
                add_single(index, lower_normal, single_bound(lower_normal, lower_h, velocities[index]))

                upper_normal = np.zeros(3, dtype=np.float64)
                upper_normal[axis] = -1.0
                upper_h = self.env.upper[axis] - radius - position[axis] - uncertainty
                add_single(index, upper_normal, single_bound(upper_normal, upper_h, velocities[index]))

        for first in range(self.env.n_defenders):
            for second in range(first + 1, self.env.n_defenders):
                delta = positions[first] - positions[second]
                normal = _unit(delta, fallback=np.array([1.0, 0.0, 0.0]))
                h = float(np.linalg.norm(delta)) - (2.0 * radius + self.inter_agent_margin + 2.0 * uncertainty)
                row = np.zeros(3 * self.env.n_defenders, dtype=np.float64)
                row[3 * first : 3 * first + 3] = normal
                row[3 * second : 3 * second + 3] = -normal
                rows.append(row)
                bounds.append(
                    (
                        -self.response_gamma * max(h, 0.0) / self.control_dt
                        - self.displacement_velocity_coefficient * float(normal @ (velocities[first] - velocities[second]))
                    )
                    / self.displacement_command_coefficient
                )
        return rows, bounds

    def _project_response_feasible_set(
        self,
        initial: np.ndarray,
        previous: np.ndarray,
        rows: list[np.ndarray],
        bounds: list[float],
    ) -> np.ndarray:
        candidate = initial.reshape(-1).copy()
        max_speed = float(self.env.agents["defender_max_speed"])
        max_delta = self.command_acceleration * self.control_dt
        for _ in range(80):
            for row, bound in zip(rows, bounds, strict=True):
                residual = float(row @ candidate - bound)
                if residual < 0.0:
                    candidate += (-residual / float(row @ row)) * row
            command = self.env._clip_rows(candidate.reshape(self.env.n_defenders, 3), max_speed)
            if self.command_acceleration > 0.0:
                delta = command - previous
                delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
                command = previous + delta * np.minimum(1.0, max_delta / np.maximum(delta_norm, 1e-9))
            candidate = command.reshape(-1)
            if self._minimum_constraint(command, rows, bounds) >= -1e-5:
                return command
        return candidate.reshape(self.env.n_defenders, 3)

    def _response_fallback(
        self,
        reference: np.ndarray,
        previous: np.ndarray,
        observation: dict[str, Any],
    ) -> np.ndarray:
        safe = reference.copy()
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        radius = float(self.env.agents["drone_radius"])
        for index, position in enumerate(positions):
            correction = np.zeros(3, dtype=np.float64)
            for obstacle in self.env.obstacles:
                clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                h = clearance - radius - self.obstacle_margin - self.position_uncertainty_margin
                if h < 1.0:
                    correction += normal * (1.0 - h) * float(self.env.agents["defender_max_speed"])
            for other_index, other_position in enumerate(positions):
                if index == other_index:
                    continue
                delta = position - other_position
                h = float(np.linalg.norm(delta)) - (2.0 * radius + self.inter_agent_margin)
                if h < 1.0:
                    correction += _unit(delta) * (1.0 - h) * float(self.env.agents["defender_max_speed"])
            for axis in range(3):
                if position[axis] - self.env.lower[axis] - radius < 1.0:
                    correction[axis] += float(self.env.agents["defender_max_speed"])
                if self.env.upper[axis] - radius - position[axis] < 1.0:
                    correction[axis] -= float(self.env.agents["defender_max_speed"])
            safe[index] += correction
        return self._reachable_command(previous, safe)
