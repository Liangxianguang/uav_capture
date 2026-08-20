"""Controlled mixed-obstacle showcase scenarios for capture-radius replay.

The showcase is deliberately separate from random benchmark sampling. It gives
the renderer a reproducible, solvable central obstacle layout while preserving
the same environment dynamics and policy observation interface.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pursuit_env import CaptureRadiusPursuit3DEnv, CylinderObstacle


@dataclass(frozen=True)
class ShowcaseScenario:
    name: str
    obstacles: tuple[CylinderObstacle, ...]
    defender_positions: np.ndarray
    target_position: np.ndarray
    target_escape_direction: np.ndarray
    obstacle_zone_x: tuple[float, float]
    target_crossing_required: bool = False


def central_mixed_obstacle_scenario(
    initial_side_distance: float = 5.0,
    target_crossing_required: bool = False,
) -> ShowcaseScenario:
    """Return a fixed, solvable S1/S2-style central mixed-obstacle layout."""
    if initial_side_distance < 4.0:
        raise ValueError("initial_side_distance must be at least 4.0 m.")
    obstacles = (
        CylinderObstacle(
            center_xy=np.array([0.0, 0.0], dtype=np.float64),
            radius=1.0,
            height=5.2,
            shape="cylinder",
        ),
        CylinderObstacle(
            center_xy=np.array([0.8, 3.6], dtype=np.float64),
            radius=1.0,
            height=4.8,
            shape="box",
            half_extents_xy=np.array([1.0, 0.8], dtype=np.float64),
        ),
        CylinderObstacle(
            center_xy=np.array([0.5, -3.2], dtype=np.float64),
            radius=0.35,
            height=4.2,
            shape="wall",
            half_extents_xy=np.array([2.1, 0.35], dtype=np.float64),
        ),
    )
    left_x = -float(initial_side_distance)
    right_x = float(initial_side_distance)
    defender_positions = np.array(
        [
            [left_x, -2.4, 2.8],
            [left_x, -0.8, 3.2],
            [left_x, 0.8, 3.6],
            [left_x, 2.4, 4.0],
        ],
        dtype=np.float64,
    )
    target_position = np.array([right_x, 0.0, 4.2], dtype=np.float64)
    return ShowcaseScenario(
        name="central_mixed_obstacles",
        obstacles=obstacles,
        defender_positions=defender_positions,
        target_position=target_position,
        target_escape_direction=np.array([1.0, 0.12, 0.0], dtype=np.float64),
        obstacle_zone_x=(-2.5, 3.0),
        target_crossing_required=bool(target_crossing_required),
    )


def _within_bounds(env: CaptureRadiusPursuit3DEnv, positions: np.ndarray) -> bool:
    return bool(np.all(positions >= env.lower[None, :]) and np.all(positions <= env.upper[None, :]))


def _planar_route_exists(
    env: CaptureRadiusPursuit3DEnv,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: tuple[CylinderObstacle, ...],
    grid_step: float = 0.5,
) -> bool:
    """Check a conservative horizontal route at the goal altitude."""
    half_extent = float(env.world["half_extent_xy"])
    minimum_clearance = float(env.pursuit["safety_margin"]) + float(env.agents["drone_radius"])
    low = -half_extent + minimum_clearance
    high = half_extent - minimum_clearance
    count = int(round((high - low) / grid_step)) + 1

    def to_index(value: float) -> int:
        return int(np.clip(round((value - low) / grid_step), 0, count - 1))

    def to_point(index: tuple[int, int]) -> np.ndarray:
        return np.array([low + index[0] * grid_step, low + index[1] * grid_step, goal[2]], dtype=np.float64)

    start_index = (to_index(float(start[0])), to_index(float(start[1])))
    goal_index = (to_index(float(goal[0])), to_index(float(goal[1])))
    blocked: set[tuple[int, int]] = set()
    for ix in range(count):
        for iy in range(count):
            point = to_point((ix, iy))
            if any(env._obstacle_clearance(point, obstacle) < minimum_clearance for obstacle in obstacles):
                blocked.add((ix, iy))
    if start_index in blocked or goal_index in blocked:
        return False
    queue: deque[tuple[int, int]] = deque([start_index])
    visited = {start_index}
    while queue:
        current = queue.popleft()
        if current == goal_index:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < count and 0 <= neighbor[1] < count):
                continue
            if neighbor in visited or neighbor in blocked:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return False


def validate_showcase_scenario(env: CaptureRadiusPursuit3DEnv, scenario: ShowcaseScenario) -> None:
    """Reject out-of-bounds, overlapping, or unreachable showcase maps."""
    if scenario.defender_positions.shape != (env.n_defenders, 3):
        raise ValueError("Showcase scenario must provide one 3D position per defender.")
    if scenario.target_position.shape != (3,):
        raise ValueError("Showcase scenario target_position must have shape (3,).")
    all_positions = np.vstack([scenario.defender_positions, scenario.target_position[None, :]])
    if not _within_bounds(env, all_positions):
        raise ValueError("Showcase initial positions must stay inside the world bounds.")
    if np.min(np.linalg.norm(scenario.defender_positions[:, None, :] - scenario.defender_positions[None, :, :], axis=2) + np.eye(env.n_defenders) * 1e6) < 2.0 * float(env.agents["drone_radius"]):
        raise ValueError("Showcase defenders overlap at initialization.")
    for obstacle in scenario.obstacles:
        if any(env._obstacle_clearance(position, obstacle) < float(env.pursuit["safety_margin"]) + float(env.agents["drone_radius"]) for position in all_positions):
            raise ValueError("Showcase obstacle is too close to an initial agent position.")
    if not _planar_route_exists(
        env,
        np.mean(scenario.defender_positions, axis=0),
        scenario.target_position,
        scenario.obstacles,
    ):
        raise ValueError("Showcase map has no conservative route from defenders to target.")


def prepare_showcase_episode(
    env: CaptureRadiusPursuit3DEnv,
    scenario: ShowcaseScenario,
    seed: int,
    record_history: bool = True,
) -> dict[str, Any]:
    """Reset an environment and replace its random map with a fixed scenario."""
    env.reset(seed=seed, record_history=False)
    validate_showcase_scenario(env, scenario)
    env.defender_positions = scenario.defender_positions.copy()
    env.defender_velocities.fill(0.0)
    env.target_position = scenario.target_position.copy()
    env.target_velocity.fill(0.0)
    env.target_escape_direction = scenario.target_escape_direction / max(
        np.linalg.norm(scenario.target_escape_direction), 1e-9
    )
    env.obstacles = list(scenario.obstacles)
    env.step_count = 0
    env.history = []
    env.target_belief_positions.fill(0.0)
    env.target_belief_velocities.fill(0.0)
    env.target_observation_confidence.fill(0.0)
    env.target_observation_timestamps.fill(-1)
    env.target_observation_covariance[:] = np.eye(3, dtype=np.float64)[None, :, :] * float(
        env.pursuit["observation_covariance_growth"]
    )
    env.message_age_steps[:] = int(env.pursuit["maximum_message_age_steps"])
    env.detection_loss_burst_remaining.fill(0)
    env._belief_history = []
    env._update_target_beliefs()
    if record_history:
        env._record_history()
    return env.observe()


def crossing_metrics(
    env: CaptureRadiusPursuit3DEnv,
    obstacle_zone_x: tuple[float, float],
) -> dict[str, Any]:
    """Measure whether each trajectory entered the central obstacle zone."""
    if not env.history:
        raise ValueError("Cannot compute crossing metrics without trajectory history.")
    low, high = map(float, obstacle_zone_x)
    defender_trace = np.asarray([frame["defender_positions"] for frame in env.history], dtype=np.float64)
    target_trace = np.asarray([frame["target_position"] for frame in env.history], dtype=np.float64)
    defender_crossed = np.any((defender_trace[:, :, 0] >= low) & (defender_trace[:, :, 0] <= high), axis=0)
    target_crossed = bool(np.any((target_trace[:, 0] >= low) & (target_trace[:, 0] <= high)))
    return {
        "obstacle_zone_x": [low, high],
        "defender_crossed": defender_crossed.astype(bool).tolist(),
        "defender_crossing_rate": float(np.mean(defender_crossed)),
        "target_crossed": target_crossed,
        "target_crossing_rate": float(target_crossed),
    }


def target_min_clearance(env: CaptureRadiusPursuit3DEnv) -> float:
    """Return the minimum target-to-obstacle clearance over the recorded run."""
    if not env.history:
        raise ValueError("Cannot compute target clearance without trajectory history.")
    clearances = []
    for frame in env.history:
        position = np.asarray(frame["target_position"], dtype=np.float64)
        clearances.extend(float(env._obstacle_clearance(position, obstacle)) for obstacle in env.obstacles)
    return float(min(clearances)) if clearances else float("inf")
