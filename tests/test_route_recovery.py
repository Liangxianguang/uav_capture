from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from encirclement3d.route_recovery import (
    compute_barrier_imminence,
    compute_stopping_guard,
    select_verified_progress_route,
    stopping_distance,
)


class _Env:
    n_defenders = 2
    lower = np.array([-10.0, -10.0, 0.5], dtype=np.float64)
    upper = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    dt = 0.1
    agents = {"drone_radius": 0.25, "defender_max_acceleration": 6.0}

    @staticmethod
    def _cylinder_clearance_and_normal(position, obstacle):
        center = np.asarray(obstacle["center_xy"], dtype=np.float64)
        delta = np.asarray(position, dtype=np.float64).copy()
        delta[:2] -= center
        horizontal = float(np.linalg.norm(delta[:2]))
        if horizontal > 1e-12:
            normal = np.array([delta[0] / horizontal, delta[1] / horizontal, 0.0])
        else:
            normal = np.array([1.0, 0.0, 0.0])
        return horizontal - float(obstacle["radius"]), normal


class _Safety:
    env = _Env()
    _obstacle_geometry_margin_m = 0.35
    _boundary_geometry_margin_m = 0.35
    _inter_agent_geometry_margin_m = 0.35

    @staticmethod
    def _obstacles_from_observation(observation):
        return observation.get("obstacles", [])


class _ProbeSafety(_Safety):
    def __init__(self, *, next_slack: float, accepted: bool = True):
        self.env = _Env()
        self.next_slack = float(next_slack)
        self.accepted = bool(accepted)

    def _reachable_reference(self, current, requested):
        return np.asarray(requested, dtype=np.float64)

    def verify_requested_action(self, requested, observation):
        return SimpleNamespace(
            verified_feasible=self.accepted,
            constraint_slacks={"obstacle_0_defender_0": self.next_slack},
        )


def _observation(*, velocity: tuple[float, float, float] = (0.0, 0.0, 0.0), obstacles=None):
    positions = np.array([[8.0, 0.0, 4.0], [8.0, 1.5, 4.0]], dtype=np.float64)
    velocities = np.repeat(np.asarray(velocity, dtype=np.float64)[None, :], 2, axis=0)
    return {
        "defender_positions": positions,
        "defender_velocities": velocities,
        "target_belief_positions": np.repeat(np.array([[9.0, 0.0, 4.0]]), 2, axis=0),
        "obstacles": [] if obstacles is None else obstacles,
    }


def test_stopping_distance_is_constant_deceleration_formula() -> None:
    assert stopping_distance(4.8, 6.0) == pytest.approx(1.92)
    with pytest.raises(ValueError):
        stopping_distance(-1.0, 6.0)
    with pytest.raises(ValueError):
        stopping_distance(1.0, 0.0)


def test_stopping_guard_triggers_only_for_velocity_closing_boundary() -> None:
    closing = compute_stopping_guard(_observation(velocity=(4.8, 0.0, 0.0)), _Env(), _Safety())
    assert closing.triggered
    assert closing.risk_type == "boundary_upper_axis_0"
    assert closing.stopping_distance_m == pytest.approx(1.92)

    tangential = compute_stopping_guard(_observation(velocity=(0.0, 4.8, 0.0)), _Env(), _Safety())
    assert not tangential.triggered
    assert tangential.reason_code == "clear"


def test_stopping_guard_reports_obstacle_and_pairwise_closing_risks() -> None:
    obstacle = {
        "center_xy": np.array([8.8, 0.0]),
        "radius": 0.5,
        "height": 5.0,
        "shape": "cylinder",
        "half_extents_xy": None,
    }
    obstacle_result = compute_stopping_guard(
        _observation(velocity=(4.0, 0.0, 0.0), obstacles=[obstacle]), _Env(), _Safety()
    )
    assert obstacle_result.triggered
    assert obstacle_result.risk_type == "obstacle_0"

    pairwise_observation = _observation(velocity=(1.0, 0.0, 0.0))
    pairwise_observation["defender_positions"] = np.array(
        [[0.0, 0.0, 4.0], [0.7, 0.0, 4.0]], dtype=np.float64
    )
    pairwise_observation["defender_velocities"] = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
    )
    pairwise_result = compute_stopping_guard(pairwise_observation, _Env(), _Safety())
    assert pairwise_result.triggered
    assert pairwise_result.risk_type == "pairwise"
    assert pairwise_result.agent_indices == (0, 1)


def test_barrier_imminence_triggers_for_next_step_slack_without_relaxing_probe() -> None:
    result = compute_barrier_imminence(
        _observation(),
        _ProbeSafety(next_slack=0.02),
        np.zeros((2, 3), dtype=np.float64),
        threshold_m=0.05,
    )
    assert result.triggered
    assert result.risk_type == "obstacle"
    assert result.reason_code == "barrier_imminent"


def test_barrier_imminence_does_not_trigger_when_predicted_slack_is_clear() -> None:
    result = compute_barrier_imminence(
        _observation(),
        _ProbeSafety(next_slack=0.5),
        np.zeros((2, 3), dtype=np.float64),
        threshold_m=0.05,
    )
    assert not result.triggered
    assert result.risk_type == "none"
    assert result.reason_code == "clear"


def _candidate(label: str, action: np.ndarray, *, route_length: float = 1.0, valid: bool = True):
    return SimpleNamespace(
        label=label,
        action_chunk=np.asarray(action, dtype=np.float64)[None, ...],
        route_length_m=float(route_length),
        valid=bool(valid),
    )


def test_select_verified_progress_route_rejects_unverified_and_safe_hold() -> None:
    action = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    route_batch = SimpleNamespace(
        candidates=(
            _candidate("left_detour", action),
            _candidate("right_detour", action * 0.5),
            _candidate("verified_safe_hold", np.zeros_like(action)),
        )
    )
    counterfactuals = (
        SimpleNamespace(accepted=False),
        SimpleNamespace(accepted=True),
        SimpleNamespace(accepted=True),
    )
    index, reason = select_verified_progress_route(
        route_batch,
        counterfactuals,
        _observation(),
    )
    assert index == 1
    assert reason == "verified_progress_route"


def test_select_verified_progress_route_can_require_a_detour_and_falls_back_to_braking() -> None:
    action = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    route_batch = SimpleNamespace(
        candidates=(
            _candidate("nominal", action),
            _candidate("braking", np.zeros_like(action), route_length=0.25),
        )
    )
    counterfactuals = (SimpleNamespace(accepted=True), SimpleNamespace(accepted=True))
    index, reason = select_verified_progress_route(
        route_batch,
        counterfactuals,
        _observation(),
        require_detour=True,
    )
    assert index == 1
    assert reason == "verified_braking_route"
