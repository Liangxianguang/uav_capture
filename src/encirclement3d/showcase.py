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
    defender_side: str = "left"
    layout_seed: int | None = None


def central_mixed_obstacle_scenario(
    initial_side_distance: float = 5.0,
    target_crossing_required: bool = False,
    layout: str = "mixed",
    defender_side: str = "left",
) -> ShowcaseScenario:
    """Return a fixed, solvable S1/S2-style central obstacle layout."""
    if initial_side_distance < 4.0:
        raise ValueError("initial_side_distance must be at least 4.0 m.")
    if layout not in {"open", "cylinder", "cylinder_box", "mixed"}:
        raise ValueError("layout must be one of: open, cylinder, cylinder_box, mixed.")
    if defender_side not in {"left", "right"}:
        raise ValueError("defender_side must be either 'left' or 'right'.")
    all_obstacles = (
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
    obstacle_count = {"open": 0, "cylinder": 1, "cylinder_box": 2, "mixed": 3}[layout]
    obstacles = all_obstacles[:obstacle_count]
    left_x = -float(initial_side_distance)
    right_x = float(initial_side_distance)
    defender_x = left_x if defender_side == "left" else right_x
    target_x = right_x if defender_side == "left" else left_x
    escape_sign = 1.0 if defender_side == "left" else -1.0
    # In the ordinary showcase the target escapes away from the defenders.  A
    # crossing-required scenario reverses that direction so that the target
    # must also traverse the central obstacle zone and reach the opposite side.
    target_direction_sign = -escape_sign if target_crossing_required else escape_sign
    defender_positions = np.array(
        [
            [defender_x, -2.4, 2.8],
            [defender_x, -0.8, 3.2],
            [defender_x, 0.8, 3.6],
            [defender_x, 2.4, 4.0],
        ],
        dtype=np.float64,
    )
    target_position = np.array([target_x, 0.0, 4.2], dtype=np.float64)
    return ShowcaseScenario(
        name=f"central_{layout}_obstacles_{defender_side}",
        obstacles=obstacles,
        defender_positions=defender_positions,
        target_position=target_position,
        target_escape_direction=np.array([target_direction_sign, 0.12, 0.0], dtype=np.float64),
        obstacle_zone_x=(-2.5, 3.0),
        target_crossing_required=bool(target_crossing_required),
        defender_side=defender_side,
    )


def _opposite_side_positions(initial_side_distance: float, defender_side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if initial_side_distance < 4.0:
        raise ValueError("initial_side_distance must be at least 4.0 m.")
    if defender_side not in {"left", "right"}:
        raise ValueError("defender_side must be either 'left' or 'right'.")
    defender_x = -float(initial_side_distance) if defender_side == "left" else float(initial_side_distance)
    target_x = -defender_x
    escape_sign = 1.0 if defender_side == "left" else -1.0
    return (
        np.array(
            [
                [defender_x, -2.4, 2.8],
                [defender_x, -0.8, 3.2],
                [defender_x, 0.8, 3.6],
                [defender_x, 2.4, 4.0],
            ],
            dtype=np.float64,
        ),
        np.array([target_x, 0.0, 4.2], dtype=np.float64),
        np.array([escape_sign, 0.12, 0.0], dtype=np.float64),
    )


def _random_central_obstacle(
    rng: np.random.Generator,
    shape: str,
    obstacle_zone_x: tuple[float, float],
) -> CylinderObstacle:
    """Sample one axis-aligned central obstacle with a physically recorded orientation.

    The current collision model handles axis-aligned boxes exactly.  Therefore
    a wall orientation means either 0 or 90 degrees in the x-y plane; arbitrary
    yaw is intentionally deferred until collision, clearance, sensing, and
    rendering all support it consistently.
    """
    if shape not in {"cylinder", "box", "wall"}:
        raise ValueError(f"Unsupported S3 obstacle shape: {shape}")
    height = float(rng.uniform(3.8, 6.2))
    half_extents_xy: np.ndarray | None = None
    if shape == "cylinder":
        radius = float(rng.uniform(0.72, 1.08))
        half_x = half_y = radius
    elif shape == "box":
        half_extents_xy = rng.uniform(0.70, 1.25, size=2).astype(np.float64)
        radius = float(np.max(half_extents_xy))
        half_x, half_y = map(float, half_extents_xy)
    else:
        long_extent = float(rng.uniform(1.65, 2.35))
        short_extent = float(rng.uniform(0.28, 0.42))
        half_extents_xy = (
            np.array([long_extent, short_extent], dtype=np.float64)
            if bool(rng.integers(0, 2))
            else np.array([short_extent, long_extent], dtype=np.float64)
        )
        radius = short_extent
        half_x, half_y = map(float, half_extents_xy)
    x_low, x_high = map(float, obstacle_zone_x)
    center_x = float(rng.uniform(x_low + half_x + 0.15, x_high - half_x - 0.15))
    center_y = float(rng.uniform(-6.0 + half_y, 6.0 - half_y))
    return CylinderObstacle(
        center_xy=np.array([center_x, center_y], dtype=np.float64),
        radius=radius,
        height=height,
        shape=shape,
        half_extents_xy=half_extents_xy,
    )


def random_central_mixed_obstacle_scenario(
    env: CaptureRadiusPursuit3DEnv,
    layout_seed: int,
    initial_side_distance: float = 5.0,
    defender_side: str = "left",
    obstacle_count_range: tuple[int, int] = (3, 5),
    max_attempts: int = 500,
) -> ShowcaseScenario:
    """Sample a reproducible, valid S3 map with cylinder/box/wall obstacles.

    Every accepted map contains each required geometry at least once, stays in
    the central zone, clears all spawn positions, and has a conservative route
    for every defender to the target's initial side.  The seed fully determines
    the accepted layout, including any rejected candidate maps.
    """
    if layout_seed < 0:
        raise ValueError("layout_seed must be non-negative.")
    minimum_count, maximum_count = map(int, obstacle_count_range)
    if minimum_count < 3 or maximum_count < minimum_count:
        raise ValueError("S3 obstacle_count_range must satisfy 3 <= minimum <= maximum.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    defender_positions, target_position, target_escape_direction = _opposite_side_positions(
        initial_side_distance, defender_side
    )
    obstacle_zone_x = (-2.5, 3.0)
    protected_points = np.vstack([defender_positions, target_position[None, :]])
    rng = np.random.default_rng(layout_seed)
    for _ in range(max_attempts):
        obstacle_count = int(rng.integers(minimum_count, maximum_count + 1))
        shapes = ["cylinder", "box", "wall"]
        rng.shuffle(shapes)
        shapes.extend(str(rng.choice(["cylinder", "box", "wall"])) for _ in range(obstacle_count - 3))
        obstacles: list[CylinderObstacle] = []
        for shape in shapes:
            for _candidate_attempt in range(100):
                candidate = _random_central_obstacle(rng, shape, obstacle_zone_x)
                if not env._obstacle_clear_of_points(candidate, protected_points):
                    continue
                if any(env._obstacle_horizontal_separation(candidate, current) < 0.70 for current in obstacles):
                    continue
                obstacles.append(candidate)
                break
            else:
                break
        if len(obstacles) != obstacle_count:
            continue
        scenario = ShowcaseScenario(
            name=f"s3_random_central_mixed_{layout_seed}",
            obstacles=tuple(obstacles),
            defender_positions=defender_positions,
            target_position=target_position,
            target_escape_direction=target_escape_direction,
            obstacle_zone_x=obstacle_zone_x,
            defender_side=defender_side,
            layout_seed=int(layout_seed),
        )
        try:
            validate_showcase_scenario(env, scenario)
        except ValueError:
            continue
        return scenario
    raise RuntimeError(f"Unable to sample a valid S3 map after {max_attempts} attempts (seed={layout_seed}).")


def scenario_metadata(scenario: ShowcaseScenario) -> dict[str, Any]:
    """Return JSON-safe geometry and protocol metadata for a showcase scenario."""
    return {
        "name": scenario.name,
        "layout_seed": scenario.layout_seed,
        "defender_side": scenario.defender_side,
        "target_crossing_required": scenario.target_crossing_required,
        "obstacle_zone_x": list(scenario.obstacle_zone_x),
        "defender_positions": scenario.defender_positions.tolist(),
        "target_position": scenario.target_position.tolist(),
        "target_escape_direction": scenario.target_escape_direction.tolist(),
        "obstacles": [
            {
                "shape": obstacle.shape,
                "center_xy": obstacle.center_xy.tolist(),
                "radius": float(obstacle.radius),
                "height": float(obstacle.height),
                "half_extents_xy": None if obstacle.half_extents_xy is None else obstacle.half_extents_xy.tolist(),
                "orientation_degrees": (
                    0.0
                    if obstacle.half_extents_xy is None or obstacle.half_extents_xy[0] >= obstacle.half_extents_xy[1]
                    else 90.0
                ),
            }
            for obstacle in scenario.obstacles
        ],
    }


def sample_training_episode(
    env: CaptureRadiusPursuit3DEnv,
    settings: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
    progress: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample an original random episode or a staged central-layout episode.

    ``progress`` is normalized to ``[0, 1]``.  This permits the same curriculum
    definition to drive behavior cloning (episode progress) and MAPPO
    fine-tuning (environment-step progress).
    """
    if not 0.0 <= progress <= 1.0:
        raise ValueError("training progress must lie in [0, 1].")
    stage: dict[str, Any] = {}
    stages = settings.get("training_showcase_stages", [])
    if stages:
        if not isinstance(stages, list) or not all(isinstance(item, dict) for item in stages):
            raise ValueError("training_showcase_stages must be a list of mappings.")
        selected = next((item for item in stages if progress <= float(item["until_progress"])), None)
        if selected is None:
            selected = stages[-1]
        stage = dict(selected)
    probability = float(stage.get("showcase_probability", settings.get("training_showcase_probability", 0.0)))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("showcase_probability must lie in [0, 1].")
    layouts = stage.get("layouts", settings.get("training_showcase_layouts", ["mixed"]))
    distances = stage.get(
        "initial_side_distances", settings.get("training_showcase_initial_side_distances", [5.0])
    )
    defender_sides = stage.get("defender_sides", settings.get("training_showcase_defender_sides", ["left"]))
    speeds = stage.get("target_speed_scales", settings.get("training_target_speed_scales", [0.55]))
    target_motion_modes = stage.get("target_motion_modes", settings.get("training_target_motion_modes", ["flee_persistence"]))
    randomized_probability = float(
        stage.get("randomized_central_probability", settings.get("training_randomized_central_probability", 0.0))
    )
    randomized_obstacle_count_range = tuple(
        int(value)
        for value in stage.get(
            "randomized_obstacle_count_range",
            settings.get("training_randomized_obstacle_count_range", [3, 5]),
        )
    )
    target_crossing_probability = float(stage.get("target_crossing_probability", 0.0))
    if not isinstance(layouts, list) or not layouts:
        raise ValueError("Showcase layouts must be a non-empty list.")
    if not isinstance(distances, list) or not distances:
        raise ValueError("Showcase initial_side_distances must be a non-empty list.")
    if not isinstance(defender_sides, list) or not defender_sides:
        raise ValueError("Showcase defender_sides must be a non-empty list.")
    if not isinstance(speeds, list) or not speeds:
        raise ValueError("Showcase target_speed_scales must be a non-empty list.")
    if not isinstance(target_motion_modes, list) or not target_motion_modes:
        raise ValueError("Showcase target_motion_modes must be a non-empty list.")
    if not 0.0 <= target_crossing_probability <= 1.0:
        raise ValueError("target_crossing_probability must lie in [0, 1].")
    if not 0.0 <= randomized_probability <= 1.0:
        raise ValueError("randomized_central_probability must lie in [0, 1].")
    if len(randomized_obstacle_count_range) != 2 or not 3 <= randomized_obstacle_count_range[0] <= randomized_obstacle_count_range[1]:
        raise ValueError("randomized_obstacle_count_range must satisfy 3 <= low <= high.")
    if rng.random() < randomized_probability:
        side_distance = float(rng.choice(np.asarray(distances, dtype=np.float64)))
        defender_side = str(rng.choice(defender_sides))
        target_speed_scale = float(rng.choice(np.asarray(speeds, dtype=np.float64)))
        target_motion_mode = str(rng.choice(target_motion_modes))
        layout_seed = int(seed) + 1_000_000 + int(rng.integers(0, 100_000))
        scenario = random_central_mixed_obstacle_scenario(
            env,
            layout_seed=layout_seed,
            initial_side_distance=side_distance,
            defender_side=defender_side,
            obstacle_count_range=randomized_obstacle_count_range,
        )
        env.obstacle_count = len(scenario.obstacles)
        env.target_speed_scale = target_speed_scale
        env.pursuit["target_motion_mode"] = target_motion_mode
        observation = prepare_showcase_episode(env, scenario, seed=seed, record_history=False)
        return observation, {
            "episode_kind": "randomized_showcase",
            "layout": "random_mixed",
            "layout_seed": layout_seed,
            "obstacle_count": len(scenario.obstacles),
            "defender_side": defender_side,
            "target_crossing_required": False,
            "initial_side_distance": side_distance,
            "target_speed_scale": target_speed_scale,
            "target_motion_mode": target_motion_mode,
            "progress": progress,
        }
    if rng.random() < probability:
        layout = str(rng.choice(layouts))
        side_distance = float(rng.choice(np.asarray(distances, dtype=np.float64)))
        target_speed_scale = float(rng.choice(np.asarray(speeds, dtype=np.float64)))
        defender_side = str(rng.choice(defender_sides))
        target_motion_mode = str(rng.choice(target_motion_modes))
        scenario = central_mixed_obstacle_scenario(
            side_distance,
            target_crossing_required=bool(
                defender_side == "right" and rng.random() < target_crossing_probability
            ),
            layout=layout,
            defender_side=defender_side,
        )
        env.obstacle_count = len(scenario.obstacles)
        env.target_speed_scale = target_speed_scale
        env.pursuit["target_motion_mode"] = target_motion_mode
        observation = prepare_showcase_episode(env, scenario, seed=seed, record_history=False)
        return observation, {
            "episode_kind": "showcase",
            "layout": layout,
            "defender_side": defender_side,
            "target_crossing_required": scenario.target_crossing_required,
            "layout_seed": None,
            "obstacle_count": len(scenario.obstacles),
            "initial_side_distance": side_distance,
            "target_speed_scale": target_speed_scale,
            "target_motion_mode": target_motion_mode,
            "progress": progress,
        }
    env.obstacle_count = int(rng.choice(np.asarray(settings["training_obstacle_counts"], dtype=np.int64)))
    env.target_speed_scale = float(rng.choice(np.asarray(settings["training_target_speed_scales"], dtype=np.float64)))
    env.pursuit["target_motion_mode"] = str(rng.choice(target_motion_modes))
    return env.reset(seed=seed), {
        "episode_kind": "random",
        "layout": str(env.pursuit["obstacle_profile"]),
        "defender_side": None,
        "target_crossing_required": None,
        "layout_seed": None,
        "obstacle_count": int(env.obstacle_count),
        "initial_side_distance": None,
        "target_speed_scale": env.target_speed_scale,
        "target_motion_mode": str(env.pursuit["target_motion_mode"]),
        "progress": progress,
    }


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
    for defender_index, defender_position in enumerate(scenario.defender_positions):
        if not _planar_route_exists(env, defender_position, scenario.target_position, scenario.obstacles):
            raise ValueError(f"Showcase map has no conservative route for defender {defender_index}.")


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
    """Measure zone entry and completion of a start-to-opposite-side crossing.

    A trajectory is only marked as ``crossed`` after it reaches beyond the
    opposite edge of the central zone.  This prevents a short incursion into
    the zone from being reported as a completed obstacle traversal.
    """
    if not env.history:
        raise ValueError("Cannot compute crossing metrics without trajectory history.")
    low, high = map(float, obstacle_zone_x)
    defender_trace = np.asarray([frame["defender_positions"] for frame in env.history], dtype=np.float64)
    target_trace = np.asarray([frame["target_position"] for frame in env.history], dtype=np.float64)
    defender_initial_x = defender_trace[0, :, 0]
    defender_zone_entered = np.any(
        (defender_trace[:, :, 0] >= low) & (defender_trace[:, :, 0] <= high), axis=0
    )
    defender_crossed = np.where(
        defender_initial_x <= low,
        np.any(defender_trace[:, :, 0] > high, axis=0),
        np.any(defender_trace[:, :, 0] < low, axis=0),
    )
    target_initial_x = float(target_trace[0, 0])
    target_zone_entered = bool(np.any((target_trace[:, 0] >= low) & (target_trace[:, 0] <= high)))
    target_crossed = bool(
        np.any(target_trace[:, 0] > high)
        if target_initial_x <= low
        else np.any(target_trace[:, 0] < low)
    )
    return {
        "obstacle_zone_x": [low, high],
        "defender_zone_entered": defender_zone_entered.astype(bool).tolist(),
        "defender_crossed": defender_crossed.astype(bool).tolist(),
        "defender_zone_entry_rate": float(np.mean(defender_zone_entered)),
        "defender_crossing_rate": float(np.mean(defender_crossed)),
        "target_zone_entered": target_zone_entered,
        "target_crossed": target_crossed,
        "target_zone_entry_rate": float(target_zone_entered),
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
