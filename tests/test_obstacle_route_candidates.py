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


def test_open_scene_generates_all_development_route_labels() -> None:
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        _observation([]),
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )

    assert len(batch.candidates) == len(ROUTE_LABELS)
    assert batch.candidates[batch.labels.index("left_detour")].rejection_reasons == ("no_obstacle",)
    assert batch.candidates[batch.labels.index("right_detour")].rejection_reasons == ("no_obstacle",)
    assert batch.candidates[batch.labels.index("radial_out")].side == "radial_out"
    assert batch.candidates[batch.labels.index("formation_split")].side == "split"


def test_boundary_rescue_is_opt_in_and_points_toward_public_world_interior() -> None:
    observation = _observation([])
    positions = np.asarray(observation["defender_positions"], dtype=np.float64).copy()
    positions[:, 0] = 8.2
    observation["defender_positions"] = positions
    disabled = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    assert "boundary_rescue" not in disabled.labels

    enabled = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        config=ObstacleRouteConfig(
            boundary_rescue_enabled=True,
            boundary_rescue_trigger_m=3.0,
            boundary_rescue_offset_m=2.0,
        ),
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    rescue = enabled.candidates[enabled.labels.index("boundary_rescue")]
    assert rescue.valid
    assert rescue.side == "boundary_rescue"
    assert rescue.action_chunk[0, :, 0].mean() < 0.0
    assert enabled.route_contract["boundary_rescue_enabled"] is True


def test_boundary_rescue_is_invalid_when_public_boundary_is_not_close() -> None:
    observation = _observation([])
    enabled = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        config=ObstacleRouteConfig(boundary_rescue_enabled=True),
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    rescue = enabled.candidates[enabled.labels.index("boundary_rescue")]
    assert not rescue.valid
    assert "boundary_rescue_not_available" in rescue.rejection_reasons


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


def test_direct_nominal_corridor_is_rejected_when_start_to_target_crosses_obstacle() -> None:
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        _observation([_cylinder((0.0, 0.0))]),
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )

    nominal = batch.candidates[batch.labels.index("nominal")]
    intercept = batch.candidates[batch.labels.index("safe_intercept")]
    assert not nominal.valid
    assert not intercept.valid
    assert "route_clearance_below_margin" in nominal.rejection_reasons
    assert "route_clearance_below_margin" in intercept.rejection_reasons


def test_verified_safe_hold_is_not_rejected_by_current_centroid_clearance() -> None:
    # The current centroid is inside the geometric margin, but the downstream
    # CBF can still decide whether a zero-action hold is safe.  Route geometry
    # must not discard that fallback before the CBF counterfactual runs.
    observation = _observation([_cylinder((-6.0, 0.0), radius=1.0)])
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    hold = batch.candidates[batch.labels.index("verified_safe_hold")]
    braking = batch.candidates[batch.labels.index("braking")]
    assert hold.minimum_geometric_clearance_m < 0.35
    assert hold.valid
    assert braking.valid
    assert "route_clearance_below_margin" not in hold.rejection_reasons


def test_left_blocked_scene_shifts_left_route_and_keeps_right_route() -> None:
    # The second cylinder is outside the nominal belief corridor, but blocks
    # the first left (+y) bypass corridor.  The route generator must use both
    # public obstacle records and search a farther parallel corridor.
    observation = _observation([_cylinder((0.0, 0.0)), _cylinder((0.0, 3.5))])
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    left = batch.candidates[batch.labels.index("left_detour")]
    right = batch.candidates[batch.labels.index("right_detour")]

    assert left.obstacle_id == right.obstacle_id == 0
    assert left.geometric_feasible and left.valid
    assert left.waypoints[1, 1] > 4.5
    assert right.geometric_feasible
    assert right.valid
    assert not np.allclose(left.waypoints, right.waypoints)


def test_right_blocked_scene_shifts_right_route_and_keeps_left_route() -> None:
    observation = _observation([_cylinder((0.0, 0.0)), _cylinder((0.0, -3.5))])
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    left = batch.candidates[batch.labels.index("left_detour")]
    right = batch.candidates[batch.labels.index("right_detour")]

    assert left.geometric_feasible and left.valid
    assert right.geometric_feasible and right.valid
    assert right.waypoints[1, 1] < -4.5


def test_lateral_route_shifts_away_from_a_second_obstacle_on_the_same_side() -> None:
    # The principal cylinder blocks the direct corridor.  A second cylinder
    # sits near the initial left bypass, so the planner must search a farther
    # parallel corridor instead of giving up on the whole side.
    observation = _observation([_cylinder((0.0, 0.0)), _cylinder((0.0, 2.2))])
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )
    left = batch.candidates[batch.labels.index("left_detour")]
    assert left.valid
    assert left.waypoints[1, 1] > 3.0
    assert left.minimum_geometric_clearance_m >= 0.35 - 1e-8


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


def test_route_goal_ignores_never_received_zero_beliefs() -> None:
    observation = _observation([_cylinder((0.0, 0.0))])
    observation["target_belief_positions"] = np.array(
        [[-6.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    observation["target_observation_received"] = np.array([True, False, False, False], dtype=bool)
    observation["target_observation_age_state"] = ("fresh", "never_received", "never_received", "never_received")
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )

    nominal = batch.candidates[batch.labels.index("nominal")]
    assert nominal.waypoints[-1, 0] == pytest.approx(-6.0)


def test_route_goal_falls_back_to_defender_centroid_without_any_belief() -> None:
    observation = _observation([_cylinder((0.0, 0.0))])
    observation["target_belief_positions"] = np.zeros((4, 3), dtype=np.float64)
    observation["target_observation_received"] = np.zeros(4, dtype=bool)
    observation["target_observation_age_state"] = ("never_received",) * 4
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )

    nominal = batch.candidates[batch.labels.index("nominal")]
    np.testing.assert_allclose(nominal.waypoints[-1], observation["defender_positions"].mean(axis=0))


def test_opt_in_visibility_search_moves_toward_public_world_interior() -> None:
    observation = _observation([_cylinder((0.0, 0.0))])
    observation["target_belief_positions"] = np.zeros((4, 3), dtype=np.float64)
    observation["target_observation_received"] = np.zeros(4, dtype=bool)
    observation["target_observation_age_state"] = ("never_received",) * 4
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        config=ObstacleRouteConfig(
            visibility_search_enabled=True,
            visibility_search_offset_m=1.5,
        ),
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )

    route = batch.candidates[batch.labels.index("visibility_hold")]
    assert route.side == "visibility_search"
    assert route.valid
    assert np.linalg.norm(route.action_chunk[0]) > 1e-9
    assert batch.route_contract["visibility_search_enabled"] is True
    assert batch.route_contract["visibility_search_mode"] == "lateral_interior_scan_v1"


def test_visibility_hold_remains_stationary_without_search_opt_in() -> None:
    observation = _observation([_cylinder((0.0, 0.0))])
    observation["target_belief_positions"] = np.zeros((4, 3), dtype=np.float64)
    observation["target_observation_received"] = np.zeros(4, dtype=bool)
    observation["target_observation_age_state"] = ("never_received",) * 4
    batch = make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )

    route = batch.candidates[batch.labels.index("visibility_hold")]
    np.testing.assert_allclose(route.action_chunk, 0.0, atol=1e-12)
    assert route.side == "visibility_hold"
