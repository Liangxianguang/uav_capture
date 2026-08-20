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
    direction_name = "target_crossing" if target_crossing_required else "target_evading"
    return ShowcaseScenario(
        name=f"central_{layout}_obstacles_{defender_side}_{direction_name}",
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
    target_crossing_required: bool = False,
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
    if target_crossing_required:
        target_escape_direction = -target_escape_direction
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
            target_crossing_required=bool(target_crossing_required),
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
    crossing_speeds = stage.get(
        "target_crossing_speed_scales",
        settings.get("training_target_crossing_speed_scales", speeds),
    )
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
    if not isinstance(crossing_speeds, list) or not crossing_speeds:
        raise ValueError("Showcase target_crossing_speed_scales must be a non-empty list.")
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
        target_crossing_required = bool(rng.random() < target_crossing_probability)
        if target_crossing_required:
            target_speed_scale = float(rng.choice(np.asarray(crossing_speeds, dtype=np.float64)))
        layout_seed = int(seed) + 1_000_000 + int(rng.integers(0, 100_000))
        scenario = random_central_mixed_obstacle_scenario(
            env,
            layout_seed=layout_seed,
            initial_side_distance=side_distance,
            defender_side=defender_side,
            target_crossing_required=target_crossing_required,
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
            "target_crossing_required": scenario.target_crossing_required,
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
        target_crossing_required = bool(rng.random() < target_crossing_probability)
        if target_crossing_required:
            target_speed_scale = float(rng.choice(np.asarray(crossing_speeds, dtype=np.float64)))
        scenario = central_mixed_obstacle_scenario(
            side_distance,
            target_crossing_required=target_crossing_required,
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


def _planar_route(
    env: CaptureRadiusPursuit3DEnv,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: tuple[CylinderObstacle, ...],
    grid_step: float = 0.5,
) -> list[np.ndarray] | None:
    """Plan a conservative horizontal route at the shared route altitude.

    The planner intentionally reasons with the same obstacle-clearance routine
    as the environment and inflates every obstacle by the drone radius plus
    safety margin.  It is used only to reject invalid central showcase maps;
    it is not exposed to an actor or used as a pursuit action.
    """
    half_extent = float(env.world["half_extent_xy"])
    minimum_clearance = float(env.pursuit["safety_margin"]) + float(env.agents["drone_radius"])
    low = -half_extent + minimum_clearance
    high = half_extent - minimum_clearance
    count = int(round((high - low) / grid_step)) + 1

    def to_index(value: float) -> int:
        return int(np.clip(round((value - low) / grid_step), 0, count - 1))

    def to_point(index: tuple[int, int]) -> np.ndarray:
        return np.array([low + index[0] * grid_step, low + index[1] * grid_step, goal[2]], dtype=np.float64)

    def segment_is_clear(first: np.ndarray, second: np.ndarray) -> bool:
        distance = float(np.linalg.norm(second - first))
        samples = max(int(np.ceil(distance / max(grid_step * 0.25, 1e-9))), 1)
        for fraction in np.linspace(0.0, 1.0, samples + 1):
            point = first + float(fraction) * (second - first)
            if any(env._obstacle_clearance(point, obstacle) < minimum_clearance for obstacle in obstacles):
                return False
        return True

    start_index = (to_index(float(start[0])), to_index(float(start[1])))
    goal_index = (to_index(float(goal[0])), to_index(float(goal[1])))
    blocked: set[tuple[int, int]] = set()
    for ix in range(count):
        for iy in range(count):
            point = to_point((ix, iy))
            if any(env._obstacle_clearance(point, obstacle) < minimum_clearance for obstacle in obstacles):
                blocked.add((ix, iy))
    if start_index in blocked or goal_index in blocked:
        return None
    if not segment_is_clear(start, to_point(start_index)) or not segment_is_clear(to_point(goal_index), goal):
        return None
    queue: deque[tuple[int, int]] = deque([start_index])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start_index: None}
    while queue:
        current = queue.popleft()
        if current == goal_index:
            indices: list[tuple[int, int]] = []
            while current is not None:
                indices.append(current)
                current = parents[current]
            points = [start.copy(), *(to_point(index) for index in reversed(indices)), goal.copy()]
            route: list[np.ndarray] = []
            for point in points:
                if not route or not np.allclose(point, route[-1]):
                    route.append(point)
            return route
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < count and 0 <= neighbor[1] < count):
                continue
            if neighbor in parents or neighbor in blocked:
                continue
            if not segment_is_clear(to_point(current), to_point(neighbor)):
                continue
            parents[neighbor] = current
            queue.append(neighbor)
    return None


def _planar_route_exists(
    env: CaptureRadiusPursuit3DEnv,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: tuple[CylinderObstacle, ...],
    grid_step: float = 0.5,
) -> bool:
    """Return whether the conservative grid planner can connect two points."""
    return _planar_route(env, start, goal, obstacles, grid_step=grid_step) is not None


def transit_endpoints(scenario: ShowcaseScenario) -> tuple[np.ndarray, np.ndarray]:
    """Return opposite-side endpoints for independent agent transit checks."""
    defender_goals = scenario.defender_positions.copy()
    defender_goals[:, 0] = float(scenario.target_position[0])
    target_goal = scenario.target_position.copy()
    target_goal[0] = float(np.mean(scenario.defender_positions[:, 0]))
    return defender_goals, target_goal


def transit_route_metrics(
    env: CaptureRadiusPursuit3DEnv,
    scenario: ShowcaseScenario,
    grid_step: float = 0.5,
) -> dict[str, Any]:
    """Verify independent, conservative left-to-right and right-to-left routes.

    This is deliberately separate from a capture rollout.  A capture episode
    terminates as soon as an interceptor reaches the target, so it cannot also
    prove that every participant would later reach the far side.  These route
    checks establish that the generated map has safe traversable paths for the
    target and for each defender without changing the pursuit objective.
    """
    defender_goals, target_goal = transit_endpoints(scenario)
    defender_routes = [
        _planar_route(env, start, goal, scenario.obstacles, grid_step=grid_step)
        for start, goal in zip(scenario.defender_positions, defender_goals, strict=True)
    ]
    target_route = _planar_route(
        env,
        scenario.target_position,
        target_goal,
        scenario.obstacles,
        grid_step=grid_step,
    )

    def route_length(route: list[np.ndarray] | None) -> float | None:
        if route is None:
            return None
        if len(route) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(np.asarray(route), axis=0), axis=1)))

    def route_min_clearance(route: list[np.ndarray] | None) -> float | None:
        if route is None:
            return None
        clearances = [
            float(env._obstacle_clearance(point, obstacle))
            for point in route
            for obstacle in scenario.obstacles
        ]
        return float(min(clearances)) if clearances else float("inf")

    defender_feasible = [route is not None for route in defender_routes]
    return {
        "transit_grid_step_m": float(grid_step),
        "defender_transit_route_feasible": defender_feasible,
        "all_defenders_transit_route_feasible": bool(all(defender_feasible)),
        "target_transit_route_feasible": bool(target_route is not None),
        "transit_route_feasible": bool(all(defender_feasible) and target_route is not None),
        "defender_transit_route_length_m": [route_length(route) for route in defender_routes],
        "target_transit_route_length_m": route_length(target_route),
        "defender_transit_min_clearance_m": [route_min_clearance(route) for route in defender_routes],
        "target_transit_min_clearance_m": route_min_clearance(target_route),
        "defender_transit_goals": defender_goals.tolist(),
        "target_transit_goal": target_goal.tolist(),
    }


def _execute_transit_route(
    env: CaptureRadiusPursuit3DEnv,
    route: list[np.ndarray] | None,
    *,
    max_speed: float,
    max_acceleration: float,
) -> dict[str, Any]:
    """Execute one planned route under the benchmark's velocity kinematics.

    The subject is simulated independently.  This intentionally tests whether
    a single drone/target can traverse the central map, not whether five
    bodies can negotiate the same narrow channel simultaneously.  Multi-agent
    separation remains a separate requirement of the capture rollout and CBF.
    """
    if route is None:
        return {
            "success": False,
            "reason": "no_conservative_route",
            "steps": 0,
            "minimum_clearance_m": None,
            "boundary_violation": False,
        }
    if len(route) < 2:
        return {
            "success": True,
            "reason": "completed",
            "steps": 0,
            "minimum_clearance_m": float("inf"),
            "boundary_violation": False,
        }
    position = route[0].copy()
    velocity = np.zeros(3, dtype=np.float64)
    radius = float(env.agents["drone_radius"])
    safety_margin = float(env.pursuit["safety_margin"])
    clearance_limit = radius + safety_margin
    waypoint_index = 1
    step_limit = max(100, int(np.ceil(sum(np.linalg.norm(second - first) for first, second in zip(route, route[1:])) / max(max_speed * env.dt, 1e-9))) * 8)
    minimum_clearance = float("inf")
    boundary_violation = False

    def segment_clearance(first: np.ndarray, second: np.ndarray) -> float:
        distance = float(np.linalg.norm(second - first))
        samples = max(int(np.ceil(distance / max(0.05, max_speed * env.dt * 0.25))), 1)
        values = [
            float(env._obstacle_clearance(first + float(fraction) * (second - first), obstacle))
            for fraction in np.linspace(0.0, 1.0, samples + 1)
            for obstacle in env.obstacles
        ]
        return float(min(values)) if values else float("inf")

    for step in range(1, step_limit + 1):
        target_waypoint = route[waypoint_index]
        delta = target_waypoint - position
        distance = float(np.linalg.norm(delta))
        desired = np.zeros(3, dtype=np.float64) if distance < 1e-9 else delta / distance * max_speed
        velocity = env._move_toward_velocity(
            velocity[None, :], desired[None, :], max_delta=max_acceleration * env.dt
        )[0]
        candidate = position + velocity * env.dt
        # Clamp only after the finite-acceleration motion would overshoot a
        # waypoint. The segment leading to that waypoint is checked below.
        if float(np.dot(candidate - target_waypoint, position - target_waypoint)) <= 0.0:
            candidate = target_waypoint.copy()
            velocity.fill(0.0)
        minimum_clearance = min(minimum_clearance, segment_clearance(position, candidate))
        if np.any(candidate < env.lower + radius + safety_margin) or np.any(
            candidate > env.upper - radius - safety_margin
        ):
            boundary_violation = True
            return {
                "success": False,
                "reason": "boundary_violation",
                "steps": step,
                "minimum_clearance_m": float(minimum_clearance),
                "boundary_violation": True,
            }
        if minimum_clearance < clearance_limit:
            return {
                "success": False,
                "reason": "obstacle_clearance_violation",
                "steps": step,
                "minimum_clearance_m": float(minimum_clearance),
                "boundary_violation": boundary_violation,
            }
        position = candidate
        if np.allclose(position, target_waypoint, atol=1e-9):
            waypoint_index += 1
            if waypoint_index == len(route):
                return {
                    "success": True,
                    "reason": "completed",
                    "steps": step,
                    "minimum_clearance_m": float(minimum_clearance),
                    "boundary_violation": boundary_violation,
                }
    return {
        "success": False,
        "reason": "timeout",
        "steps": step_limit,
        "minimum_clearance_m": float(minimum_clearance),
        "boundary_violation": boundary_violation,
    }


def transit_execution_metrics(
    env: CaptureRadiusPursuit3DEnv,
    scenario: ShowcaseScenario,
    grid_step: float = 0.5,
) -> dict[str, Any]:
    """Run reproducible independent transit rollouts for all five subjects."""
    defender_goals, target_goal = transit_endpoints(scenario)
    defender_routes = [
        _planar_route(env, start, goal, scenario.obstacles, grid_step=grid_step)
        for start, goal in zip(scenario.defender_positions, defender_goals, strict=True)
    ]
    target_route = _planar_route(
        env,
        scenario.target_position,
        target_goal,
        scenario.obstacles,
        grid_step=grid_step,
    )
    defender_runs = [
        _execute_transit_route(
            env,
            route,
            max_speed=float(env.agents["defender_max_speed"]),
            max_acceleration=float(env.agents["defender_max_acceleration"]),
        )
        for route in defender_routes
    ]
    target_run = _execute_transit_route(
        env,
        target_route,
        max_speed=float(env.agents["target_max_speed"]),
        max_acceleration=float(env.agents["target_max_acceleration"]),
    )
    defender_success = [bool(run["success"]) for run in defender_runs]
    return {
        "defender_transit_success": defender_success,
        "all_defenders_transit_success": bool(all(defender_success)),
        "target_transit_success": bool(target_run["success"]),
        "transit_success": bool(all(defender_success) and target_run["success"]),
        "defender_transit_steps": [int(run["steps"]) for run in defender_runs],
        "target_transit_steps": int(target_run["steps"]),
        "defender_transit_reasons": [str(run["reason"]) for run in defender_runs],
        "target_transit_reason": str(target_run["reason"]),
        "defender_transit_execution_min_clearance_m": [run["minimum_clearance_m"] for run in defender_runs],
        "target_transit_execution_min_clearance_m": target_run["minimum_clearance_m"],
    }


def validate_showcase_scenario(env: CaptureRadiusPursuit3DEnv, scenario: ShowcaseScenario) -> None:
    """Reject out-of-bounds, overlapping, or unreachable showcase maps."""
    if scenario.defender_positions.shape != (env.n_defenders, 3):
        raise ValueError("Showcase scenario must provide one 3D position per defender.")
    if scenario.target_position.shape != (3,):
        raise ValueError("Showcase scenario target_position must have shape (3,).")
    all_positions = np.vstack([scenario.defender_positions, scenario.target_position[None, :]])
    if not _within_bounds(env, all_positions):
        raise ValueError("Showcase initial positions must stay inside the world bounds.")
    boundary_buffer = 1.0
    if np.any(all_positions < env.lower[None, :] + boundary_buffer) or np.any(
        all_positions > env.upper[None, :] - boundary_buffer
    ):
        raise ValueError("Showcase initial positions must preserve the 1.0 m world-boundary buffer.")
    if np.min(np.linalg.norm(scenario.defender_positions[:, None, :] - scenario.defender_positions[None, :, :], axis=2) + np.eye(env.n_defenders) * 1e6) < 2.0 * float(env.agents["drone_radius"]):
        raise ValueError("Showcase defenders overlap at initialization.")
    for obstacle in scenario.obstacles:
        if any(env._obstacle_clearance(position, obstacle) < float(env.pursuit["safety_margin"]) + float(env.agents["drone_radius"]) for position in all_positions):
            raise ValueError("Showcase obstacle is too close to an initial agent position.")
    for defender_index, defender_position in enumerate(scenario.defender_positions):
        if not _planar_route_exists(env, defender_position, scenario.target_position, scenario.obstacles):
            raise ValueError(f"Showcase map has no conservative route for defender {defender_index}.")
    transit = transit_route_metrics(env, scenario)
    if not bool(transit["transit_route_feasible"]):
        raise ValueError("Showcase map does not provide independent conservative transit routes for all participants.")


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
    defender_in_zone = (defender_trace[:, :, 0] >= low) & (defender_trace[:, :, 0] <= high)
    defender_zone_entered = np.any(defender_in_zone, axis=0)
    defender_crossed = np.where(
        defender_initial_x <= low,
        np.any(defender_trace[:, :, 0] > high, axis=0),
        np.any(defender_trace[:, :, 0] < low, axis=0),
    )
    target_initial_x = float(target_trace[0, 0])
    target_in_zone = (target_trace[:, 0] >= low) & (target_trace[:, 0] <= high)
    target_zone_entered = bool(np.any(target_in_zone))
    target_crossed = bool(
        np.any(target_trace[:, 0] > high)
        if target_initial_x <= low
        else np.any(target_trace[:, 0] < low)
    )
    steps = np.asarray([int(frame.get("step", index)) for index, frame in enumerate(env.history)], dtype=np.int64)

    def first_step(mask: np.ndarray) -> int | None:
        indices = np.flatnonzero(mask)
        return int(steps[indices[0]]) if indices.size else None

    return {
        "obstacle_zone_x": [low, high],
        "defender_zone_entered": defender_zone_entered.astype(bool).tolist(),
        "defender_crossed": defender_crossed.astype(bool).tolist(),
        "defender_zone_entry_rate": float(np.mean(defender_zone_entered)),
        "defender_crossing_rate": float(np.mean(defender_crossed)),
        "any_defender_zone_entered": bool(np.any(defender_zone_entered)),
        "all_defenders_zone_entered": bool(np.all(defender_zone_entered)),
        "all_defenders_crossed": bool(np.all(defender_crossed)),
        "defender_first_zone_entry_steps": [first_step(defender_in_zone[:, index]) for index in range(env.n_defenders)],
        "target_zone_entered": target_zone_entered,
        "target_crossed": target_crossed,
        "target_zone_entry_rate": float(target_zone_entered),
        "target_crossing_rate": float(target_crossed),
        "target_first_zone_entry_step": first_step(target_in_zone),
    }


def capture_contract_metrics(
    final_info: dict[str, Any],
    crossing: dict[str, Any],
    *,
    target_collision: bool = False,
    target_crossing_required: bool = False,
) -> dict[str, Any]:
    """Apply the V3 central-obstacle capture contract to one finished rollout."""
    capture_event = bool(final_info.get("capture_event", False))
    safe_capture = bool(final_info.get("safe_capture_success", False)) and not target_collision
    central_encounter = bool(
        crossing["any_defender_zone_entered"]
        and (not target_crossing_required or crossing["target_zone_entered"])
    )
    safe_capture_in_pursuit = bool(safe_capture and central_encounter)
    if target_collision:
        task_termination_reason = "target_safety_failure"
    elif safe_capture_in_pursuit:
        task_termination_reason = "safe_capture_in_pursuit"
    elif capture_event:
        task_termination_reason = "capture_without_zone_entry"
    else:
        task_termination_reason = str(final_info.get("termination_reason", "running"))
    return {
        "central_encounter": central_encounter,
        "safe_capture_in_pursuit": safe_capture_in_pursuit,
        "capture_without_zone_entry": bool(capture_event and not central_encounter),
        "target_crossing_required": bool(target_crossing_required),
        "task_termination_reason": task_termination_reason,
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
