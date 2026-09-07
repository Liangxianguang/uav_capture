"""Obstacle-conditioned route proposals for the safe-capture controller.

The historical :mod:`jepa_safe_capture_candidates` module intentionally keeps
its five- and twelve-candidate contracts unchanged.  This module is the new
development-only route proposal layer for WP1: it consumes only the public
obstacle geometry and target *belief* in an observation, creates physically
interpretable detours, and projects every action chunk through the declared
reachable-dynamics envelope before a downstream JEPA/CBF stack sees it.

The module does not solve a CBF or execute an action.  A route is only a
proposal.  The final Joint CBF filter remains the execution boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Mapping, Sequence

import numpy as np


ROUTE_LABELS = (
    "nominal",
    "left_detour",
    "right_detour",
    "upper_detour",
    "lower_detour",
    "radial_out",
    "formation_split",
    "formation_contract",
    "braking",
    "safe_intercept",
    "visibility_hold",
    "verified_safe_hold",
)

_SHAPES = {"cylinder", "box", "wall"}
_OPTIONAL_ROUTE_LABELS = ("boundary_rescue",)


def _as_finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}.")
    return array.copy()


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > 1e-12 and np.isfinite(norm):
        return value / norm
    return np.asarray(fallback, dtype=np.float64).copy()


def _unit_xy(vector: np.ndarray, fallback: np.ndarray = np.array([1.0, 0.0])) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (2,):
        raise ValueError(f"XY vector must have shape (2,), got {value.shape}.")
    norm = float(np.linalg.norm(value))
    if norm > 1e-12 and np.isfinite(norm):
        return value / norm
    return np.asarray(fallback, dtype=np.float64).copy()


@dataclass(frozen=True)
class ObstacleGeometry:
    """Validated public obstacle record used by route geometry.

    ``center_xy`` and all dimensions are copied at construction, so route
    metadata cannot change if the caller mutates a live simulator observation.
    ``wall`` and ``box`` use the supplied horizontal half extents; cylinders
    use ``radius`` in both horizontal directions.
    """

    obstacle_id: int
    center_xy: np.ndarray
    radius: float
    height: float
    shape: str = "cylinder"
    half_extents_xy: np.ndarray | None = None

    def __post_init__(self) -> None:
        center = _as_finite_vector(self.center_xy, (2,), "center_xy")
        object.__setattr__(self, "center_xy", center)
        if not np.isfinite(self.radius) or float(self.radius) <= 0.0:
            raise ValueError("Obstacle radius must be positive and finite.")
        if not np.isfinite(self.height) or float(self.height) <= 0.0:
            raise ValueError("Obstacle height must be positive and finite.")
        shape = str(self.shape)
        if shape not in _SHAPES:
            raise ValueError(f"Unsupported obstacle shape: {shape!r}.")
        object.__setattr__(self, "shape", shape)
        if shape == "cylinder":
            if self.half_extents_xy is not None:
                half = _as_finite_vector(self.half_extents_xy, (2,), "half_extents_xy")
                if np.any(half <= 0.0):
                    raise ValueError("half_extents_xy must be positive when supplied.")
                object.__setattr__(self, "half_extents_xy", half)
        else:
            if self.half_extents_xy is None:
                raise ValueError(f"{shape} obstacle requires half_extents_xy.")
            half = _as_finite_vector(self.half_extents_xy, (2,), "half_extents_xy")
            if np.any(half <= 0.0):
                raise ValueError("half_extents_xy must be positive and finite.")
            object.__setattr__(self, "half_extents_xy", half)
    @property
    def horizontal_half_extents(self) -> np.ndarray:
        if self.half_extents_xy is not None:
            return self.half_extents_xy.copy()
        return np.full(2, float(self.radius), dtype=np.float64)

    @property
    def center_3d(self) -> np.ndarray:
        return np.array([self.center_xy[0], self.center_xy[1], 0.5 * float(self.height)], dtype=np.float64)

    def signed_distance(self, point: np.ndarray) -> float:
        """Return the signed Euclidean distance to the finite solid.

        Positive values are outside the obstacle, zero is on its surface, and
        negative values are inside.  This is a pure public-geometry function;
        it never accesses an environment or simulator state.
        """

        position = _as_finite_vector(point, (3,), "point")
        if self.shape == "cylinder":
            horizontal_gap = float(np.linalg.norm(position[:2] - self.center_xy) - self.radius)
            if 0.0 <= position[2] <= float(self.height):
                return horizontal_gap
            vertical_gap = float(-position[2] if position[2] < 0.0 else position[2] - self.height)
            if horizontal_gap <= 0.0:
                # Outside the vertical extent but inside the footprint: the
                # nearest point is on the bottom/top cap.
                return vertical_gap
            return float(hypot(horizontal_gap, vertical_gap))
        delta = position - self.center_3d
        half = np.array(
            [self.horizontal_half_extents[0], self.horizontal_half_extents[1], 0.5 * float(self.height)],
            dtype=np.float64,
        )
        signed_components = np.abs(delta) - half
        outside = np.maximum(signed_components, 0.0)
        return float(np.linalg.norm(outside) + min(float(np.max(signed_components)), 0.0))

    def clearance(self, point: np.ndarray, *, vehicle_radius_m: float = 0.0) -> float:
        return float(self.signed_distance(point) - float(vehicle_radius_m))

    def support_radius(self, direction_xy: np.ndarray) -> float:
        """Support distance of the footprint along a unit horizontal vector."""

        direction = _unit_xy(direction_xy)
        if self.shape == "cylinder":
            return float(self.radius)
        return float(np.dot(np.abs(direction), self.horizontal_half_extents))

    def as_dict(self) -> dict[str, Any]:
        return {
            "obstacle_id": int(self.obstacle_id),
            "center_xy": self.center_xy.tolist(),
            "radius": float(self.radius),
            "height": float(self.height),
            "shape": self.shape,
            "half_extents_xy": None if self.half_extents_xy is None else self.half_extents_xy.tolist(),
        }


@dataclass(frozen=True)
class ObstacleRouteConfig:
    """Frozen route-proposal and reachable-dynamics contract."""

    chunk_length_steps: int = 3
    dt_seconds: float = 0.1
    max_speed_mps: float = 5.0
    max_acceleration_mps2: float = 6.0
    max_action_change_mps: float | None = None
    vehicle_radius_m: float = 0.25
    obstacle_margin_m: float = 0.35
    route_buffer_m: float = 0.75
    nominal_speed_mps: float = 2.0
    project_to_reachable_dynamics: bool = True
    world_lower: tuple[float, float, float] | None = None
    world_upper: tuple[float, float, float] | None = None
    corridor_samples: int = 65
    # The historical visibility_hold route is stationary when no target has
    # ever been observed.  This opt-in development contract keeps its route
    # identity index but gives it a bounded, public-observation-only search
    # waypoint.  It is disabled by default for historical replay parity.
    visibility_search_enabled: bool = False
    visibility_search_offset_m: float = 1.5
    visibility_search_mode: str = "lateral_interior_scan_v1"
    # Optional development-only route that points the whole formation toward
    # the public world interior before the boundary CBF becomes infeasible.
    boundary_rescue_enabled: bool = False
    boundary_rescue_trigger_m: float = 3.0
    boundary_rescue_offset_m: float = 2.0

    def __post_init__(self) -> None:
        if self.chunk_length_steps < 3:
            raise ValueError("Obstacle route chunks require at least three control steps.")
        if self.dt_seconds <= 0.0 or self.max_speed_mps <= 0.0 or self.max_acceleration_mps2 <= 0.0:
            raise ValueError("dt_seconds, max_speed_mps, and max_acceleration_mps2 must be positive.")
        if self.max_action_change_mps is not None and self.max_action_change_mps <= 0.0:
            raise ValueError("max_action_change_mps must be positive when supplied.")
        if min(self.vehicle_radius_m, self.obstacle_margin_m, self.route_buffer_m) < 0.0:
            raise ValueError("vehicle_radius_m and route margins must be non-negative.")
        if self.nominal_speed_mps <= 0.0:
            raise ValueError("nominal_speed_mps must be positive.")
        if self.corridor_samples < 3:
            raise ValueError("corridor_samples must be at least three.")
        if not np.isfinite(self.visibility_search_offset_m) or self.visibility_search_offset_m <= 0.0:
            raise ValueError("visibility_search_offset_m must be positive and finite.")
        if not str(self.visibility_search_mode).strip():
            raise ValueError("visibility_search_mode must be non-empty.")
        if not np.isfinite(self.boundary_rescue_trigger_m) or self.boundary_rescue_trigger_m <= 0.0:
            raise ValueError("boundary_rescue_trigger_m must be positive and finite.")
        if not np.isfinite(self.boundary_rescue_offset_m) or self.boundary_rescue_offset_m <= 0.0:
            raise ValueError("boundary_rescue_offset_m must be positive and finite.")
        for name, bounds in (("world_lower", self.world_lower), ("world_upper", self.world_upper)):
            if bounds is not None:
                array = _as_finite_vector(bounds, (3,), name)
                object.__setattr__(self, name, tuple(float(value) for value in array))
        if self.world_lower is not None and self.world_upper is not None:
            if np.any(np.asarray(self.world_upper) <= np.asarray(self.world_lower)):
                raise ValueError("world_upper must be strictly larger than world_lower.")

    @property
    def resolved_max_action_change_mps(self) -> float:
        if self.max_action_change_mps is not None:
            return float(self.max_action_change_mps)
        return float(self.max_acceleration_mps2 * self.dt_seconds)

    @property
    def clearance_margin_m(self) -> float:
        return float(self.vehicle_radius_m + self.obstacle_margin_m)

    def contract(self) -> dict[str, Any]:
        route_labels = list(ROUTE_LABELS)
        if self.boundary_rescue_enabled:
            route_labels.extend(_OPTIONAL_ROUTE_LABELS)
        return {
            "route_profile": "obstacle_route_v1",
            "route_labels": route_labels,
            "candidate_count": len(route_labels),
            "chunk_length_steps": int(self.chunk_length_steps),
            "execute_only_first_step": True,
            "cbf_execution_boundary": "downstream_joint_cbf",
            "dt_seconds": float(self.dt_seconds),
            "max_speed_mps": float(self.max_speed_mps),
            "max_acceleration_mps2": float(self.max_acceleration_mps2),
            "max_action_change_mps": float(self.resolved_max_action_change_mps),
            "vehicle_radius_m": float(self.vehicle_radius_m),
            "obstacle_margin_m": float(self.obstacle_margin_m),
            "route_buffer_m": float(self.route_buffer_m),
            "nominal_speed_mps": float(self.nominal_speed_mps),
            "project_to_reachable_dynamics": bool(self.project_to_reachable_dynamics),
            "world_lower": None if self.world_lower is None else list(self.world_lower),
            "world_upper": None if self.world_upper is None else list(self.world_upper),
            "corridor_samples": int(self.corridor_samples),
            "visibility_search_enabled": bool(self.visibility_search_enabled),
            "visibility_search_offset_m": float(self.visibility_search_offset_m),
            "visibility_search_mode": str(self.visibility_search_mode),
            "boundary_rescue_enabled": bool(self.boundary_rescue_enabled),
            "boundary_rescue_trigger_m": float(self.boundary_rescue_trigger_m),
            "boundary_rescue_offset_m": float(self.boundary_rescue_offset_m),
        }


@dataclass(frozen=True)
class ObstacleRouteCandidate:
    """One interpretable route proposal and its pre-CBF diagnostics."""

    route_id: str
    label: str
    obstacle_id: int | None
    obstacle_shape: str | None
    side: str
    waypoints: np.ndarray
    action_chunk: np.ndarray
    raw_action_chunk: np.ndarray
    projected: bool
    reachable: bool
    geometric_feasible: bool
    minimum_geometric_clearance_m: float
    route_length_m: float
    rejection_reasons: tuple[str, ...]
    fallback_only: bool = False

    def __post_init__(self) -> None:
        if self.action_chunk.ndim != 3 or self.raw_action_chunk.shape != self.action_chunk.shape:
            raise ValueError("Route action chunks must have matching shape [steps, defenders, 3].")
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("Route waypoints must have shape [points, 3].")
        for array_name in ("waypoints", "action_chunk", "raw_action_chunk"):
            array = np.asarray(getattr(self, array_name), dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"Route {array_name} must be finite.")
            object.__setattr__(self, array_name, array.copy())
        if self.label not in ROUTE_LABELS and self.label not in _OPTIONAL_ROUTE_LABELS:
            raise ValueError(f"Unknown route label: {self.label!r}.")
        if self.fallback_only and self.label != "verified_safe_hold":
            raise ValueError("Only verified_safe_hold may be fallback_only.")

    @property
    def valid(self) -> bool:
        return bool(self.reachable and self.geometric_feasible and not self.rejection_reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "label": self.label,
            "obstacle_id": self.obstacle_id,
            "obstacle_shape": self.obstacle_shape,
            "side": self.side,
            "waypoints": self.waypoints.tolist(),
            "action_chunk": self.action_chunk.tolist(),
            "raw_action_chunk": self.raw_action_chunk.tolist(),
            "projected": bool(self.projected),
            "reachable": bool(self.reachable),
            "geometric_feasible": bool(self.geometric_feasible),
            "minimum_geometric_clearance_m": float(self.minimum_geometric_clearance_m),
            "route_length_m": float(self.route_length_m),
            "rejection_reasons": list(self.rejection_reasons),
            "fallback_only": bool(self.fallback_only),
            "valid": bool(self.valid),
        }


@dataclass(frozen=True)
class ObstacleRouteBatch:
    """Stable candidate order plus route metadata for the downstream stack."""

    candidates: tuple[ObstacleRouteCandidate, ...]
    labels: tuple[str, ...]
    chunks: np.ndarray
    valid_mask: np.ndarray
    route_contract: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.candidates) != len(self.labels) or self.chunks.shape[0] != len(self.labels):
            raise ValueError("Route candidate metadata length does not match chunks.")
        if self.valid_mask.shape != (len(self.labels),):
            raise ValueError("valid_mask must contain one entry per route.")
        if self.chunks.ndim != 4 or self.chunks.shape[-1] != 3:
            raise ValueError("chunks must have shape [routes, steps, defenders, 3].")

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "valid_mask": self.valid_mask.tolist(),
            "route_contract": self.route_contract,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def parse_obstacles(observation: Mapping[str, Any]) -> tuple[ObstacleGeometry, ...]:
    """Decode only public ``observation['obstacles']`` records."""

    raw_obstacles = observation.get("obstacles", ())
    if raw_obstacles is None:
        raw_obstacles = ()
    if not isinstance(raw_obstacles, Sequence) or isinstance(raw_obstacles, (str, bytes)):
        raise ValueError("observation['obstacles'] must be a sequence of public records.")
    parsed: list[ObstacleGeometry] = []
    for index, raw in enumerate(raw_obstacles):
        if isinstance(raw, ObstacleGeometry):
            parsed.append(raw)
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"Obstacle record {index} is not a mapping.")
        if "center_xy" not in raw or "radius" not in raw or "height" not in raw:
            raise ValueError(f"Obstacle record {index} is missing center_xy, radius, or height.")
        parsed.append(
            ObstacleGeometry(
                obstacle_id=index,
                center_xy=raw["center_xy"],
                radius=float(raw["radius"]),
                height=float(raw["height"]),
                shape=str(raw.get("shape", "cylinder")),
                half_extents_xy=raw.get("half_extents_xy"),
            )
        )
    return tuple(parsed)


def _observation_bounds(observation: Mapping[str, Any], config: ObstacleRouteConfig) -> tuple[np.ndarray, np.ndarray]:
    lower_value = config.world_lower if config.world_lower is not None else observation.get("world_lower")
    upper_value = config.world_upper if config.world_upper is not None else observation.get("world_upper")
    # Horizontal bounds are not needed to form a route.  A finite altitude
    # interval is required only for upper/lower detour feasibility checks.
    lower = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64) if lower_value is None else _as_finite_vector(lower_value, (3,), "world_lower")
    upper = np.array([np.inf, np.inf, np.inf], dtype=np.float64) if upper_value is None else _as_finite_vector(upper_value, (3,), "world_upper")
    return lower, upper


def _belief_consensus(
    positions: np.ndarray,
    beliefs: np.ndarray,
    belief_velocities: np.ndarray,
    observation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Aggregate only initialized target beliefs from the public observation.

    The environment represents a never-received belief as a finite zero vector
    for compatibility with the frozen actor.  Treating those zeros as target
    measurements pulls the route goal toward the origin whenever only a subset
    of defenders can see the target.  This helper keeps the route layer causal:
    received beliefs are averaged, while an entirely uninitialized belief
    falls back to the defender centroid and zero target velocity.
    """

    defender_count = int(positions.shape[0])
    received_value = observation.get("target_observation_received")
    states_value = observation.get("target_observation_age_state")
    if received_value is None:
        if states_value is None:
            received = np.ones(defender_count, dtype=bool)
        else:
            if not isinstance(states_value, (list, tuple)) or len(states_value) != defender_count:
                raise ValueError("target_observation_age_state must have one entry per defender.")
            received = np.asarray([str(state) != "never_received" for state in states_value], dtype=bool)
    else:
        received = np.asarray(received_value, dtype=bool)
        if received.shape != (defender_count,):
            raise ValueError("target_observation_received must have one entry per defender.")
    if not np.any(received):
        return positions.mean(axis=0), np.zeros(3, dtype=np.float64), False
    return beliefs[received].mean(axis=0), belief_velocities[received].mean(axis=0), True


def _active_search_goal(
    centroid: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    obstacles: Sequence[ObstacleGeometry],
    forward_xy: np.ndarray,
    left_xy: np.ndarray,
    config: ObstacleRouteConfig,
) -> np.ndarray:
    """Choose a short search displacement using only public geometry.

    The first direction points toward the finite world interior.  The two
    lateral alternatives make the scan useful when that direction is blocked
    by a public obstacle; the final opposite direction is a bounded fallback.
    This is a proposal only and remains subject to reachable projection and
    the downstream Joint CBF probes.
    """

    centroid = _as_finite_vector(centroid, (3,), "centroid")
    if np.isfinite(lower[:2]).all() and np.isfinite(upper[:2]).all():
        world_center = 0.5 * (lower[:2] + upper[:2])
        inward = _unit_xy(world_center - centroid[:2], fallback=-np.asarray(forward_xy, dtype=np.float64))
    else:
        inward = _unit_xy(-np.asarray(forward_xy, dtype=np.float64), fallback=np.array([-1.0, 0.0]))
    lateral = _unit_xy(left_xy, fallback=np.array([0.0, 1.0]))
    directions = (inward, lateral, -lateral, -inward)
    radius = float(config.visibility_search_offset_m)
    for direction in directions:
        goal = centroid.copy()
        goal[:2] += radius * direction
        if np.isfinite(lower).all():
            goal = np.maximum(goal, lower + config.vehicle_radius_m + config.obstacle_margin_m)
        if np.isfinite(upper).all():
            goal = np.minimum(goal, upper - config.vehicle_radius_m - config.obstacle_margin_m)
        if np.linalg.norm(goal[:2] - centroid[:2]) <= 1e-9:
            continue
        if all(
            obstacle.clearance(goal, vehicle_radius_m=config.vehicle_radius_m)
            >= config.obstacle_margin_m - 1e-9
            for obstacle in obstacles
        ):
            return goal
    # If every public endpoint is blocked, retain a deterministic bounded
    # proposal.  Geometry validity and CBF verification will reject it.
    goal = centroid.copy()
    goal[:2] += radius * inward
    if np.isfinite(lower).all():
        goal = np.maximum(goal, lower + config.vehicle_radius_m + config.obstacle_margin_m)
    if np.isfinite(upper).all():
        goal = np.minimum(goal, upper - config.vehicle_radius_m - config.obstacle_margin_m)
    return goal


def _segment_points(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, int(count), dtype=np.float64)[:, None]
    return start[None, :] + fractions * (end - start)[None, :]


def segment_min_clearance(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: Sequence[ObstacleGeometry],
    *,
    vehicle_radius_m: float = 0.0,
    samples: int = 65,
) -> float:
    """Conservative sampled corridor clearance to all supplied obstacles."""

    first = _as_finite_vector(start, (3,), "start")
    last = _as_finite_vector(end, (3,), "end")
    if not obstacles:
        return float("inf")
    points = _segment_points(first, last, samples)
    return float(
        min(
            obstacle.clearance(point, vehicle_radius_m=vehicle_radius_m)
            for point in points
            for obstacle in obstacles
        )
    )


def route_min_clearance(
    waypoints: np.ndarray,
    obstacles: Sequence[ObstacleGeometry],
    *,
    vehicle_radius_m: float,
    samples: int,
) -> float:
    points = _as_finite_vector(waypoints, (int(np.asarray(waypoints).shape[0]), 3), "waypoints")
    if len(points) < 1 or not obstacles:
        return float("inf")
    if len(points) == 1:
        return float(min(obstacle.clearance(points[0], vehicle_radius_m=vehicle_radius_m) for obstacle in obstacles))
    values = [
        segment_min_clearance(
            points[index],
            points[index + 1],
            obstacles,
            vehicle_radius_m=vehicle_radius_m,
            samples=samples,
        )
        for index in range(len(points) - 1)
    ]
    return float(min(values))


def _route_length(waypoints: np.ndarray) -> float:
    if len(waypoints) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))


def _principal_obstacle(
    start: np.ndarray,
    target: np.ndarray,
    obstacles: Sequence[ObstacleGeometry],
    config: ObstacleRouteConfig,
) -> ObstacleGeometry | None:
    if not obstacles:
        return None
    # The target belief corridor, rather than simulator target truth, defines
    # relevance.  Min clearance is primary; proximity to the corridor breaks
    # ties deterministically.
    corridor = _segment_points(start, target, config.corridor_samples)
    ranked: list[tuple[float, float, int, ObstacleGeometry]] = []
    segment = target - start
    denominator = float(np.dot(segment, segment))
    for obstacle in obstacles:
        distances = np.asarray(
            [obstacle.clearance(point, vehicle_radius_m=config.vehicle_radius_m) for point in corridor],
            dtype=np.float64,
        )
        minimum = float(np.min(distances))
        if denominator > 1e-12:
            projection = float(np.clip(np.dot(obstacle.center_3d - start, segment) / denominator, 0.0, 1.0))
            nearest_corridor = start + projection * segment
            corridor_distance = float(np.linalg.norm(obstacle.center_3d - nearest_corridor))
        else:
            corridor_distance = float(np.linalg.norm(obstacle.center_3d - start))
        ranked.append((minimum, corridor_distance, obstacle.obstacle_id, obstacle))
    return min(ranked, key=lambda item: (item[0], item[1], item[2]))[-1]


def _safe_action_toward(current: np.ndarray, goal: np.ndarray, speed: float) -> np.ndarray:
    delta = goal - current
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    return delta / np.maximum(norms, 1e-12) * float(speed)


def _project_chunk(
    raw_chunk: np.ndarray,
    previous_action: np.ndarray,
    config: ObstacleRouteConfig,
) -> tuple[np.ndarray, bool, tuple[str, ...]]:
    raw = np.asarray(raw_chunk, dtype=np.float64)
    previous = np.asarray(previous_action, dtype=np.float64)
    reasons: list[str] = []
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError("raw_chunk must have shape [steps, defenders, 3].")
    if previous.shape != raw.shape[1:]:
        raise ValueError("previous_action must match one action frame in raw_chunk.")
    if not np.isfinite(raw).all() or not np.isfinite(previous).all():
        return np.zeros_like(raw), False, ("non_finite",)
    if not config.project_to_reachable_dynamics:
        for step in range(raw.shape[0]):
            if np.any(np.linalg.norm(raw[step], axis=1) > config.max_speed_mps + 1e-9):
                reasons.append("speed_limit")
            reference = previous if step == 0 else raw[step - 1]
            if np.any(np.linalg.norm(raw[step] - reference, axis=1) > config.resolved_max_action_change_mps + 1e-9):
                reasons.append("action_slew_limit")
        return raw.copy(), not reasons, tuple(dict.fromkeys(reasons))
    projected = np.zeros_like(raw)
    reference = previous.copy()
    changed = False
    max_delta = config.resolved_max_action_change_mps
    for step in range(raw.shape[0]):
        command = raw[step].copy()
        norms = np.linalg.norm(command, axis=1, keepdims=True)
        speed_scale = np.minimum(1.0, config.max_speed_mps / np.maximum(norms, 1e-12))
        clipped = command * speed_scale
        if not np.array_equal(clipped, command):
            changed = True
        delta = clipped - reference
        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        slew_scale = np.minimum(1.0, max_delta / np.maximum(delta_norm, 1e-12))
        clipped = reference + delta * slew_scale
        if not np.array_equal(clipped, command * speed_scale):
            changed = True
        final_norm = np.linalg.norm(clipped, axis=1, keepdims=True)
        clipped = clipped * np.minimum(1.0, config.max_speed_mps / np.maximum(final_norm, 1e-12))
        projected[step] = clipped
        reference = clipped
    return projected, changed, ()


def project_route_action_chunk(
    raw_chunk: np.ndarray,
    previous_action: np.ndarray,
    *,
    config: ObstacleRouteConfig | None = None,
) -> tuple[np.ndarray, bool, tuple[str, ...]]:
    """Public reachable-dynamics projection helper used by WP1 tests."""

    return _project_chunk(raw_chunk, previous_action, config or ObstacleRouteConfig())


def _route_waypoints(
    label: str,
    *,
    start: np.ndarray,
    target: np.ndarray,
    centroid: np.ndarray,
    forward_xy: np.ndarray,
    left_xy: np.ndarray,
    obstacle: ObstacleGeometry | None,
    lower: np.ndarray,
    upper: np.ndarray,
    config: ObstacleRouteConfig,
    visibility_search_goal: np.ndarray | None = None,
    boundary_rescue_goal: np.ndarray | None = None,
) -> tuple[np.ndarray, str, tuple[str, ...], int | None, str | None]:
    """Create a route corridor and geometric pre-check reasons."""

    reasons: list[str] = []
    margin = config.clearance_margin_m
    if obstacle is None and label in {"left_detour", "right_detour", "upper_detour", "lower_detour"}:
        return np.stack([centroid, target]), label.removesuffix("_detour"), ("no_obstacle",), None, None

    obstacle_id = None if obstacle is None else obstacle.obstacle_id
    obstacle_shape = None if obstacle is None else obstacle.shape
    if label == "boundary_rescue":
        if boundary_rescue_goal is None:
            return np.stack([centroid]), "boundary_rescue", ("boundary_rescue_not_available",), None, None
        return np.stack([boundary_rescue_goal]), "boundary_rescue", (), None, None
    if label in {"left_detour", "right_detour"} and obstacle is not None:
        side_xy = left_xy if label == "left_detour" else -left_xy
        forward_support = obstacle.support_radius(forward_xy)
        side_support = obstacle.support_radius(side_xy)
        along = np.array(
            [forward_xy[0], forward_xy[1], 0.0],
            dtype=np.float64,
        ) * (forward_support + margin + 0.5 * config.route_buffer_m)
        side = np.array([side_xy[0], side_xy[1], 0.0], dtype=np.float64) * (
            side_support + margin + config.route_buffer_m
        )
        obstacle_center = np.array([obstacle.center_xy[0], obstacle.center_xy[1], centroid[2]], dtype=np.float64)
        corridor = np.stack(
            [
                obstacle_center - along + side,
                obstacle_center + side,
                obstacle_center + along + side,
                target,
            ]
        )
        return corridor, "left" if label == "left_detour" else "right", tuple(reasons), obstacle_id, obstacle_shape
    if label in {"upper_detour", "lower_detour"} and obstacle is not None:
        direction = "upper" if label == "upper_detour" else "lower"
        if direction == "upper":
            route_z = float(obstacle.height + margin + config.route_buffer_m)
            if np.isfinite(upper[2]) and route_z > upper[2] - margin:
                reasons.append("upper_boundary_unreachable")
        else:
            route_z = float(max(lower[2] + margin, 0.5 * obstacle.height - margin - config.route_buffer_m))
            # Obstacles are solid down to z=0 in the simulator.  A lower route
            # is feasible only when its sampled corridor actually clears the
            # bottom face and remains inside the public world bounds.
            if route_z >= 0.5 * float(obstacle.height) - margin:
                reasons.append("lower_face_blocked")
            if np.isfinite(lower[2]) and route_z < lower[2] + margin:
                reasons.append("lower_boundary_unreachable")
        horizontal_offset = np.array(
            [forward_xy[0], forward_xy[1], 0.0],
            dtype=np.float64,
        ) * (obstacle.support_radius(forward_xy) + margin + 0.5 * config.route_buffer_m)
        obstacle_center = np.array([obstacle.center_xy[0], obstacle.center_xy[1], route_z], dtype=np.float64)
        corridor = np.stack(
            [
                np.array([centroid[0], centroid[1], route_z], dtype=np.float64) - np.array([horizontal_offset[0], horizontal_offset[1], 0.0]),
                obstacle_center,
                np.array([obstacle_center[0] + horizontal_offset[0], obstacle_center[1] + horizontal_offset[1], route_z]),
                target,
            ]
        )
        return corridor, direction, tuple(reasons), obstacle_id, obstacle_shape

    if label == "nominal":
        return np.stack([target]), "nominal", (), None, None
    if label == "braking" or label == "verified_safe_hold":
        return np.stack([centroid]), "hold", (), None, None
    if label == "radial_out":
        if obstacle is not None:
            away = _unit(centroid - obstacle.center_3d, np.array([1.0, 0.0, 0.0]))
        else:
            away = _unit(centroid - target, np.array([-1.0, 0.0, 0.0]))
        return np.stack([centroid + away * (config.route_buffer_m + margin), target]), "radial_out", (), obstacle_id, obstacle_shape
    if label == "formation_split":
        split_offset = np.array([left_xy[0], left_xy[1], 0.0], dtype=np.float64) * (
            config.route_buffer_m + margin
        )
        return np.stack([centroid + split_offset, target]), "split", (), obstacle_id, obstacle_shape
    if label == "formation_contract":
        return np.stack([centroid, target]), "contract", (), obstacle_id, obstacle_shape
    if label == "safe_intercept":
        return np.stack([target]), "intercept", (), obstacle_id, obstacle_shape
    if label == "visibility_hold":
        if visibility_search_goal is not None:
            return np.stack([visibility_search_goal]), "visibility_search", (), obstacle_id, obstacle_shape
        return np.stack([centroid + 0.25 * (target - centroid)]), "visibility_hold", (), obstacle_id, obstacle_shape
    raise ValueError(f"Unknown route label: {label!r}.")


def _repair_lateral_detour_waypoints(
    waypoints: np.ndarray,
    side_xy: np.ndarray,
    obstacles: Sequence[ObstacleGeometry],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    config: ObstacleRouteConfig,
) -> np.ndarray:
    """Shift a lateral corridor until all public obstacles have clearance.

    A principal-obstacle bypass can still intersect a second obstacle on the
    same side.  The route layer therefore searches a bounded family of
    parallel corridors.  This changes only the proposal geometry; every
    resulting first action is still projected and independently CBF-checked.
    """

    if not obstacles or waypoints.shape[0] < 2:
        return waypoints
    side = np.array([float(side_xy[0]), float(side_xy[1]), 0.0], dtype=np.float64)
    side_norm = float(np.linalg.norm(side))
    if side_norm <= 1e-12:
        return waypoints
    side /= side_norm
    maximum_extra = max(4.0, 8.0 * float(config.route_buffer_m + config.clearance_margin_m))
    for extra in np.linspace(0.0, maximum_extra, 49, dtype=np.float64):
        candidate = np.asarray(waypoints, dtype=np.float64).copy()
        candidate[:-1] += side[None, :] * float(extra)
        if np.isfinite(lower).all() and np.any(candidate < lower[None, :] + config.vehicle_radius_m - 1e-9):
            continue
        if np.isfinite(upper).all() and np.any(candidate > upper[None, :] - config.vehicle_radius_m + 1e-9):
            continue
        clearance = route_min_clearance(
            candidate,
            obstacles,
            vehicle_radius_m=config.vehicle_radius_m,
            samples=config.corridor_samples,
        )
        if clearance >= config.obstacle_margin_m - 1e-9:
            return candidate
    return waypoints


def _actions_from_waypoints(
    positions: np.ndarray,
    waypoints: np.ndarray,
    *,
    label: str,
    config: ObstacleRouteConfig,
) -> np.ndarray:
    defender_count = positions.shape[0]
    chunk = np.zeros((config.chunk_length_steps, defender_count, 3), dtype=np.float64)
    centroid = positions.mean(axis=0)
    offsets = positions - centroid
    for step in range(config.chunk_length_steps):
        waypoint = waypoints[min(step, len(waypoints) - 1)]
        if label == "formation_contract":
            goals = waypoint[None, :] + 0.5 * offsets
        elif label == "formation_split":
            side = np.array([-1.0, 1.0, 0.0], dtype=np.float64)
            signs = np.where(np.arange(defender_count) % 2 == 0, -1.0, 1.0)[:, None]
            goals = waypoint[None, :] + signs * side[None, :] * config.route_buffer_m
        else:
            goals = waypoint[None, :] + offsets
        if label == "braking" or label == "verified_safe_hold":
            chunk[step] = 0.0
        else:
            chunk[step] = _safe_action_toward(positions, goals, config.nominal_speed_mps)
    return chunk


def _route_geometric_feasibility(
    waypoints: np.ndarray,
    obstacles: Sequence[ObstacleGeometry],
    *,
    config: ObstacleRouteConfig,
) -> tuple[bool, float, tuple[str, ...]]:
    if not obstacles:
        return True, float("inf"), ()
    minimum = route_min_clearance(
        waypoints,
        obstacles,
        vehicle_radius_m=config.vehicle_radius_m,
        samples=config.corridor_samples,
    )
    if minimum < config.obstacle_margin_m - 1e-9:
        return False, minimum, ("route_clearance_below_margin",)
    return True, minimum, ()


def make_obstacle_route_candidates(
    nominal_action: np.ndarray,
    observation: Mapping[str, Any],
    *,
    config: ObstacleRouteConfig | None = None,
    previous_action: np.ndarray | None = None,
) -> ObstacleRouteBatch:
    """Generate geometry-conditioned route proposals from a public observation.

    The returned candidates are ordered by :data:`ROUTE_LABELS`.  Candidate
    ``nominal`` starts from the actor request; detour candidates are generated
    from belief-based waypoints and never inspect simulator target state.  CBF
    verification is intentionally left to the caller.
    """

    settings = config or ObstacleRouteConfig()
    nominal = np.asarray(nominal_action, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1:] != (3,) or not np.isfinite(nominal).all():
        raise ValueError("nominal_action must be finite with shape [defenders, 3].")
    positions = _as_finite_vector(observation.get("defender_positions"), (nominal.shape[0], 3), "defender_positions")
    beliefs = _as_finite_vector(observation.get("target_belief_positions"), positions.shape, "target_belief_positions")
    velocities_raw = observation.get("target_belief_velocities", np.zeros_like(beliefs))
    belief_velocities = _as_finite_vector(velocities_raw, positions.shape, "target_belief_velocities")
    if previous_action is None:
        previous = np.zeros_like(nominal)
    else:
        previous = _as_finite_vector(previous_action, nominal.shape, "previous_action")
    obstacles = parse_obstacles(observation)
    lower, upper = _observation_bounds(observation, settings)
    centroid = positions.mean(axis=0)
    belief_center, belief_velocity, belief_received = _belief_consensus(
        positions,
        beliefs,
        belief_velocities,
        observation,
    )
    target = belief_center + settings.dt_seconds * settings.chunk_length_steps * belief_velocity
    forward = _unit(target - centroid, np.array([1.0, 0.0, 0.0]))
    forward_xy = _unit_xy(forward[:2])
    left_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=np.float64)
    principal = _principal_obstacle(centroid, target, obstacles, settings)
    visibility_search_goal = (
        _active_search_goal(
            centroid,
            lower=lower,
            upper=upper,
            obstacles=obstacles,
            forward_xy=forward_xy,
            left_xy=left_xy,
            config=settings,
        )
        if settings.visibility_search_enabled and not belief_received
        else None
    )
    boundary_rescue_goal: np.ndarray | None = None
    boundary_rescue_active = False
    if settings.boundary_rescue_enabled and np.isfinite(lower).all() and np.isfinite(upper).all():
        boundary_clearance = float(
            min(
                np.min(positions - lower[None, :]),
                np.min(upper[None, :] - positions),
            )
        )
        boundary_rescue_active = boundary_clearance < float(settings.boundary_rescue_trigger_m)
        if boundary_rescue_active:
            world_center = 0.5 * (lower + upper)
            inward = _unit(world_center - centroid, np.array([-forward_xy[0], -forward_xy[1], 0.0]))
            boundary_rescue_goal = centroid + inward * float(settings.boundary_rescue_offset_m)
            boundary_rescue_goal = np.maximum(
                lower + settings.clearance_margin_m,
                np.minimum(upper - settings.clearance_margin_m, boundary_rescue_goal),
            )

    candidates: list[ObstacleRouteCandidate] = []
    labels = list(ROUTE_LABELS)
    if settings.boundary_rescue_enabled:
        labels.extend(_OPTIONAL_ROUTE_LABELS)
    for label in labels:
        route_id = f"{label}:obstacle-{principal.obstacle_id if principal is not None else 'none'}"
        waypoints, side, initial_reasons, obstacle_id, obstacle_shape = _route_waypoints(
            label,
            start=positions[0],
            target=target,
            centroid=centroid,
            forward_xy=forward_xy,
            left_xy=left_xy,
            obstacle=principal,
            lower=lower,
            upper=upper,
            config=settings,
            visibility_search_goal=visibility_search_goal,
            boundary_rescue_goal=boundary_rescue_goal,
        )
        if label in {"left_detour", "right_detour"} and principal is not None:
            side_direction = left_xy if label == "left_detour" else -left_xy
            waypoints = _repair_lateral_detour_waypoints(
                waypoints,
                side_direction,
                obstacles,
                lower=lower,
                upper=upper,
                config=settings,
            )
        if label == "nominal":
            raw_chunk = np.repeat(nominal[None, :, :], settings.chunk_length_steps, axis=0)
        elif label == "braking":
            raw_chunk = np.repeat((0.5 * nominal)[None, :, :], settings.chunk_length_steps, axis=0)
        elif label == "verified_safe_hold":
            raw_chunk = np.zeros((settings.chunk_length_steps, nominal.shape[0], 3), dtype=np.float64)
        else:
            raw_chunk = _actions_from_waypoints(positions, waypoints, label=label, config=settings)
        projected_chunk, projected, projection_reasons = _project_chunk(raw_chunk, previous, settings)
        geometry_path = np.vstack((centroid[None, :], waypoints))
        geometric_feasible, minimum_clearance, geometry_reasons = _route_geometric_feasibility(
            geometry_path,
            obstacles,
            config=settings,
        )
        # A braking/verified-hold route does not traverse the obstacle field;
        # its zero-action safety is established by the downstream CBF.  The
        # geometric corridor check includes the current centroid, so applying
        # it to a hold route would incorrectly reject a recoverable state that
        # is already near an obstacle even when the CBF can safely hold it.
        if label in {"braking", "verified_safe_hold"}:
            geometric_feasible = True
            geometry_reasons = tuple(
                reason for reason in geometry_reasons if reason != "route_clearance_below_margin"
            )
        reasons = tuple(dict.fromkeys(initial_reasons + projection_reasons + geometry_reasons))
        reachable = not bool(projection_reasons)
        if not settings.project_to_reachable_dynamics and reasons:
            reachable = not bool(projection_reasons)
        candidate = ObstacleRouteCandidate(
            route_id=route_id,
            label=label,
            obstacle_id=obstacle_id,
            obstacle_shape=obstacle_shape,
            side=side,
            waypoints=waypoints,
            action_chunk=projected_chunk,
            raw_action_chunk=raw_chunk,
            projected=bool(projected),
            reachable=bool(reachable),
            geometric_feasible=bool(geometric_feasible),
            minimum_geometric_clearance_m=float(minimum_clearance),
            route_length_m=_route_length(waypoints),
            rejection_reasons=reasons,
            fallback_only=label == "verified_safe_hold",
        )
        candidates.append(candidate)
    chunks = np.stack([candidate.action_chunk for candidate in candidates], axis=0)
    return ObstacleRouteBatch(
        candidates=tuple(candidates),
        labels=tuple(labels),
        chunks=chunks,
        valid_mask=np.asarray([candidate.valid for candidate in candidates], dtype=bool),
        route_contract=settings.contract(),
    )


def generate_obstacle_route_candidates(
    nominal_action: np.ndarray,
    observation: Mapping[str, Any],
    *,
    config: ObstacleRouteConfig | None = None,
    previous_action: np.ndarray | None = None,
) -> ObstacleRouteBatch:
    """Alias with an explicit verb for scripts and downstream callers."""

    return make_obstacle_route_candidates(
        nominal_action,
        observation,
        config=config,
        previous_action=previous_action,
    )


__all__ = [
    "ROUTE_LABELS",
    "ObstacleGeometry",
    "ObstacleRouteConfig",
    "ObstacleRouteCandidate",
    "ObstacleRouteBatch",
    "parse_obstacles",
    "segment_min_clearance",
    "route_min_clearance",
    "project_route_action_chunk",
    "make_obstacle_route_candidates",
    "generate_obstacle_route_candidates",
]
