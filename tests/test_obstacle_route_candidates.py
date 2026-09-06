from __future__ import annotations

import numpy as np
import pytest

from encirclement3d.obstacle_route_candidates import (
    ROUTE_LABELS,
    ObstacleGeometry,
    ObstacleRouteConfig,
    generate_obstacle_route_candidates,
    make_obstacle_route_candidates,
    project_route_action_chunk,
    route_min_clearance,
)


def _observation(
    obstacles: list[dict[str, object]],
    *,
    target: tuple[float, float, float] = (6.0, 0.0, 4.0),
) -> dict[str, object]:
    positions = np.array(
        [[-6.0, -1.0, 4.0], [-6.0, 1.0, 4.0], [-6.0, 0.8, 5.0], [-6.0, -0.8, 5.0]],
        dtype=np.float64,
    )
    beliefs = np.repeat(np.asarray(target, dtype=np.float64)[None, :], 4, axis=0)
    return {
        "defender_positions": positions,
        "defender_velocities": np.zeros_like(positions),
        "target_belief_positions": beliefs,
        "target_belief_velocities": np.zeros_like(positions),
        "obstacles": obstacles,
        "world_lower": np.array([-10.0, -10.0, 0.5]),
        "world_upper": np.array([10.0, 10.0, 10.0]),
    }


def _cylinder(center: tuple[float, float], radius: float = 1.0) -> dict[str, object]:
    return {
        "center_xy": np.asarray(center, dtype=np.float64),
        "radius": radius,
        "height": 5.0,
        "shape": "cylinder",
        "half_extents_xy": None,
    }


def _box(center: tuple[float, float], half: tuple[float, float]) -> dict[str, object]:
    return {
        "center_xy": np.asarray(center, dtype=np.float64),
        "radius": max(half),
        "height": 5.0,
        "shape": "box",
        "half_extents_xy": np.asarray(half, dtype=np.float64),
    }


def _wall(center: tuple[float, float], half: tuple[float, float]) -> dict[str, object]:
    return {
        "center_xy": np.asarray(center, dtype=np.float64),
        "radius": min(half),
        "height": 5.0,
        "shape": "wall",
        "half_extents_xy": np.asarray(half, dtype=np.float64),
    }


def test_route_contract_contains_interpretable_three_step_candidates() -> None:
    observation = _observation([_cylinder((0.0, 0.0))])
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        config=ObstacleRouteConfig(),
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )

    assert batch.labels == ROUTE_LABELS
    assert batch.chunks.shape == (len(ROUTE_LABELS), 3, 4, 3)
    assert batch.route_contract["route_profile"] == "obstacle_route_v1"
    assert batch.route_contract["execute_only_first_step"] is True
    assert batch.route_contract["cbf_execution_boundary"] == "downstream_joint_cbf"
    assert all(candidate.action_chunk.shape == (3, 4, 3) for candidate in batch.candidates)
    assert all(np.isfinite(candidate.action_chunk).all() for candidate in batch.candidates)
    assert all(candidate.action_chunk.shape[0] >= 3 for candidate in batch.candidates)


def test_central_obstacle_generates_distinct_left_right_and_upper_routes() -> None:
    observation = _observation([_cylinder((0.0, 0.0))])
    batch = generate_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    left = batch.candidates[batch.labels.index("left_detour")]
    right = batch.candidates[batch.labels.index("right_detour")]
    upper = batch.candidates[batch.labels.index("upper_detour")]

    assert left.obstacle_id == right.obstacle_id == upper.obstacle_id == 0
    assert left.side == "left"
    assert right.side == "right"
    assert upper.side == "upper"
    assert not np.allclose(left.waypoints, right.waypoints)
    assert not np.allclose(left.action_chunk, right.action_chunk)
    assert left.waypoints[1, 1] > 0.0
    assert right.waypoints[1, 1] < 0.0
    assert upper.waypoints[1, 2] > 5.0
    assert left.valid and right.valid and upper.valid
    assert left.minimum_geometric_clearance_m >= 0.35 - 1e-8
    assert right.minimum_geometric_clearance_m >= 0.35 - 1e-8


def test_left_blocked_scene_rejects_left_route_but_keeps_right_route() -> None:
    # The second cylinder is outside the nominal belief corridor, but blocks
    # the left (+y) bypass corridor.  The route generator must use both public
    # obstacle records when checking the proposed route.
    observation = _observation([_cylinder((0.0, 0.0)), _cylinder((0.0, 3.5))])
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    left = batch.candidates[batch.labels.index("left_detour")]
    right = batch.candidates[batch.labels.index("right_detour")]

    assert left.obstacle_id == right.obstacle_id == 0
    assert not left.geometric_feasible
    assert "route_clearance_below_margin" in left.rejection_reasons
    assert right.geometric_feasible
    assert right.valid
    assert not np.allclose(left.waypoints, right.waypoints)


def test_right_blocked_scene_changes_feasible_side() -> None:
    observation = _observation([_cylinder((0.0, 0.0)), _cylinder((0.0, -3.5))])
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    left = batch.candidates[batch.labels.index("left_detour")]
    right = batch.candidates[batch.labels.index("right_detour")]

    assert left.geometric_feasible and left.valid
    assert not right.geometric_feasible
    assert "route_clearance_below_margin" in right.rejection_reasons


def test_obstacle_record_sorting_does_not_change_route_geometry_or_actions() -> None:
    obstacles = [_cylinder((0.0, 0.0)), _cylinder((0.0, 3.5))]
    first = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        _observation(obstacles),
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    second = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        _observation(list(reversed(obstacles))),
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )

    np.testing.assert_allclose(first.chunks, second.chunks, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(first.valid_mask, second.valid_mask)
    for left, right in zip(first.candidates, second.candidates):
        assert left.label == right.label
        assert left.side == right.side
        np.testing.assert_allclose(left.waypoints, right.waypoints, atol=1e-12, rtol=0.0)


def test_single_gap_wall_keeps_the_gap_route_reachable() -> None:
    # Two public wall segments leave exactly one gap around y=0.  The nominal
    # corridor runs through that gap; it must not be rejected merely because
    # the wall records are present.
    observation = _observation([_wall((0.0, -3.0), (0.35, 2.0)), _wall((0.0, 3.0), (0.35, 2.0))])
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    nominal = batch.candidates[batch.labels.index("nominal")]
    left = batch.candidates[batch.labels.index("left_detour")]
    right = batch.candidates[batch.labels.index("right_detour")]

    assert nominal.valid
    assert nominal.minimum_geometric_clearance_m > 0.35
    assert left.action_chunk.shape[0] >= 3
    assert right.action_chunk.shape[0] >= 3
    assert np.isfinite(batch.chunks).all()


def test_shape_aware_signed_distance_has_correct_surface_signs() -> None:
    cylinder = ObstacleGeometry(0, np.array([0.0, 0.0]), 1.0, 5.0)
    box = ObstacleGeometry(1, np.array([0.0, 0.0]), 2.0, 5.0, "box", np.array([1.0, 2.0]))
    wall = ObstacleGeometry(2, np.array([0.0, 0.0]), 0.5, 5.0, "wall", np.array([3.0, 0.5]))

    assert cylinder.signed_distance(np.array([2.0, 0.0, 2.0])) == pytest.approx(1.0)
    assert cylinder.signed_distance(np.array([0.0, 0.0, 6.0])) == pytest.approx(1.0)
    assert cylinder.signed_distance(np.array([0.0, 0.0, 2.0])) < 0.0
    assert box.signed_distance(np.array([1.5, 0.0, 2.5])) == pytest.approx(0.5)
    assert wall.signed_distance(np.array([0.0, 1.0, 2.5])) == pytest.approx(0.5)


def test_reachable_projection_is_causal_and_deterministic() -> None:
    raw = np.repeat(np.array([[[5.0, 0.0, 0.0]] * 4], dtype=np.float64), 3, axis=0)
    previous = np.zeros((4, 3), dtype=np.float64)
    first = project_route_action_chunk(raw, previous)
    second = project_route_action_chunk(raw, previous)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1]
    assert first[2] == ()
    assert np.max(np.linalg.norm(first[0][0] - previous, axis=1)) <= 0.6 + 1e-8
    assert np.max(np.linalg.norm(first[0], axis=2)) <= 5.0 + 1e-8
    for step in range(1, 3):
        assert np.max(np.linalg.norm(first[0][step] - first[0][step - 1], axis=1)) <= 0.6 + 1e-8


def test_route_clearance_rejects_a_corridor_that_enters_obstacle() -> None:
    obstacle = ObstacleGeometry(0, np.array([0.0, 0.0]), 1.0, 5.0)
    waypoints = np.array([[-2.0, 0.0, 2.0], [2.0, 0.0, 2.0]], dtype=np.float64)
    assert route_min_clearance(waypoints, (obstacle,), vehicle_radius_m=0.25, samples=65) < 0.0


def test_malformed_public_obstacle_is_rejected_without_hidden_state_fallback() -> None:
    observation = _observation([{"shape": "cylinder"}])
    with pytest.raises(ValueError, match="missing center_xy"):
        make_obstacle_route_candidates(np.zeros((4, 3)), observation)
