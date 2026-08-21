"""Execution-aware local CBF projection for the E1 action wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .execution_dynamics import DefenderExecutionDynamics
from .pursuit_controllers import PursuitCBFSafetyFilter
from .pursuit_env import CaptureRadiusPursuit3DEnv, _unit


@dataclass(frozen=True)
class ExecutionSafetyDiagnostics:
    action_correction_norm: float
    minimum_barrier_value: float
    mean_execution_margin_m: float
    max_execution_margin_m: float


class ExecutionAwarePursuitCBFSafetyFilter:
    """Local CBF with a predeclared action-execution displacement envelope.

    It sees only the same positions and obstacle geometry as the original
    pursuit CBF.  The added margin is derived from the wrapper's delay,
    acceleration, and bounded-noise assumptions; it never uses future target
    state or an unobserved map.
    """

    def __init__(self, env: CaptureRadiusPursuit3DEnv, execution: DefenderExecutionDynamics) -> None:
        self.env = env
        self.execution = execution
        self.gamma = float(env.task.get("cbf_gamma", 0.25))
        self.margin = float(env.pursuit["safety_margin"])

    def filter(
        self,
        desired_actions: np.ndarray,
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, ExecutionSafetyDiagnostics]:
        desired = self.env._clip_rows(np.asarray(desired_actions, dtype=np.float64), self.execution.max_speed)
        safe = desired.copy()
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        velocities = np.asarray(observation["defender_velocities"], dtype=np.float64)
        margins = self.execution.conservative_displacement_bound(velocities)
        radius = float(self.env.agents["drone_radius"])
        barriers: list[float] = []

        # Match the fixed number of local projections used by the legacy CBF.
        for _ in range(4):
            for index, position in enumerate(positions):
                own_margin = self.margin + float(margins[index])
                for obstacle in self.env.obstacles:
                    clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                    barrier = clearance - radius - own_margin
                    lower_bound = -self.gamma * barrier / self.env.dt
                    projection = float(normal @ safe[index])
                    if projection < lower_bound:
                        safe[index] += (lower_bound - projection) * normal
                    barriers.append(barrier)
                for axis in range(3):
                    lower_barrier = position[axis] - self.env.lower[axis] - radius - own_margin
                    upper_barrier = self.env.upper[axis] - position[axis] - radius - own_margin
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
                    barrier = distance - (2.0 * radius + self.margin + float(margins[first] + margins[second]))
                    lower_bound = -self.gamma * barrier / self.env.dt
                    relative_projection = float(normal @ (safe[first] - safe[second]))
                    if relative_projection < lower_bound:
                        correction = 0.5 * (lower_bound - relative_projection) * normal
                        safe[first] += correction
                        safe[second] -= correction
                    barriers.append(barrier)
            safe = self.env._clip_rows(safe, self.execution.max_speed)

        return safe, ExecutionSafetyDiagnostics(
            action_correction_norm=float(np.mean(np.linalg.norm(safe - desired, axis=1))),
            minimum_barrier_value=float(min(barriers)) if barriers else float("inf"),
            mean_execution_margin_m=float(np.mean(margins)),
            max_execution_margin_m=float(np.max(margins)),
        )


def make_kinematic_cbf(env: CaptureRadiusPursuit3DEnv) -> PursuitCBFSafetyFilter:
    """Name the E1 kinematic baseline without changing its implementation."""
    return PursuitCBFSafetyFilter(env)
