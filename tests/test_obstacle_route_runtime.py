from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from encirclement3d.obstacle_route_candidates import (
    ObstacleRouteConfig,
    make_obstacle_route_candidates,
)
from encirclement3d.obstacle_route_runtime import (
    CBFRouteCounterfactual,
    probe_independent_cbf_counterfactuals,
    probe_route_batch_with_cbf,
    route_batch_to_candidate_batch,
)


def _observation() -> dict[str, object]:
    positions = np.array(
        [[-6.0, -1.0, 4.0], [-6.0, 1.0, 4.0], [-6.0, 0.8, 5.0], [-6.0, -0.8, 5.0]],
        dtype=np.float64,
    )
    return {
        "defender_positions": positions,
        "defender_velocities": np.zeros_like(positions),
        "target_belief_positions": np.repeat(np.array([[6.0, 0.0, 4.0]]), 4, axis=0),
        "target_belief_velocities": np.zeros_like(positions),
        "obstacles": [
            {
                "center_xy": np.array([0.0, 0.0]),
                "radius": 1.0,
                "height": 5.0,
                "shape": "cylinder",
                "half_extents_xy": None,
            }
        ],
        "world_lower": np.array([-10.0, -10.0, 0.5]),
        "world_upper": np.array([10.0, 10.0, 10.0]),
    }


def _route_batch():
    observation = _observation()
    return make_obstacle_route_candidates(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        config=ObstacleRouteConfig(project_to_reachable_dynamics=True),
        previous_action=np.zeros((4, 3), dtype=np.float64),
    )


def _diagnostics(
    *,
    verified_feasible: bool = True,
    infeasible: bool = False,
    timed_out: bool = False,
    fallback_mode: str = "none",
    solver_status: str = "success",
) -> SimpleNamespace:
    return SimpleNamespace(
        verified_feasible=verified_feasible,
        infeasible=infeasible,
        timed_out=timed_out,
        fallback_mode=fallback_mode,
        solver_status=solver_status,
        minimum_constraint_value=0.25,
        action_correction_norm=0.0,
        active_constraints=(),
    )


def test_route_batch_adapter_preserves_labels_chunks_and_geometry_rejections() -> None:
    route_batch = _route_batch()
    candidate_batch = route_batch_to_candidate_batch(route_batch)

    assert candidate_batch.labels == route_batch.labels
    np.testing.assert_array_equal(candidate_batch.chunks, route_batch.chunks)
    np.testing.assert_array_equal(candidate_batch.valid_mask, route_batch.valid_mask)
    assert candidate_batch.rejection_reasons == tuple(
        tuple(candidate.rejection_reasons) for candidate in route_batch.candidates
    )


def test_route_cbf_prefilter_only_changes_eligibility_and_keeps_actions() -> None:
    route_batch = _route_batch()
    original_chunks = route_batch.chunks.copy()
    valid_indices = np.flatnonzero(route_batch.valid_mask).tolist()
    assert len(valid_indices) >= 4

    class _FakeFilter:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []

        def verify_requested_action(self, action: np.ndarray, observation: dict[str, object]):
            self.calls.append(np.asarray(action).copy())
            call = len(self.calls)
            if call == 2:
                return _diagnostics(verified_feasible=False, infeasible=True, fallback_mode="controlled_abort")
            if call == 3:
                return _diagnostics(verified_feasible=False, infeasible=True, timed_out=True, fallback_mode="none")
            if call == 4:
                return _diagnostics(verified_feasible=True, fallback_mode="nominal_cbf")
            return _diagnostics()

    safety_filter = _FakeFilter()
    runtime = probe_route_batch_with_cbf(route_batch, safety_filter, _observation())

    assert len(safety_filter.calls) == len(valid_indices)
    np.testing.assert_array_equal(runtime.chunks, original_chunks)
    assert runtime.candidate_batch.valid_mask[valid_indices[0]]
    assert not runtime.candidate_batch.valid_mask[valid_indices[1]]
    assert not runtime.candidate_batch.valid_mask[valid_indices[2]]
    assert not runtime.candidate_batch.valid_mask[valid_indices[3]]
    assert "cbf_infeasible" in runtime.candidate_batch.rejection_reasons[valid_indices[1]]
    assert "cbf_timeout" in runtime.candidate_batch.rejection_reasons[valid_indices[2]]
    assert "cbf_infeasible" in runtime.candidate_batch.rejection_reasons[valid_indices[3]]
    assert runtime.cbf_counterfactuals[valid_indices[1]] is not None
    assert runtime.cbf_counterfactuals[valid_indices[1]].accepted is False


@pytest.mark.parametrize(
    ("verified_feasible", "infeasible", "timed_out", "fallback_mode", "expected"),
    [
        (True, False, False, "none", True),
        (True, False, False, "nominal_cbf", False),
        (True, False, True, "none", False),
        (True, True, False, "none", False),
        (False, False, False, "none", False),
    ],
)
def test_counterfactual_acceptance_requires_primary_verified_no_fallback(
    verified_feasible: bool,
    infeasible: bool,
    timed_out: bool,
    fallback_mode: str,
    expected: bool,
) -> None:
    counterfactual = CBFRouteCounterfactual(
        label="left_detour",
        route_id="left_detour:obstacle-0",
        requested_action=np.zeros((4, 3), dtype=np.float64),
        verified_feasible=verified_feasible,
        infeasible=infeasible,
        timed_out=timed_out,
        fallback_mode=fallback_mode,
        solver_status="success",
        minimum_constraint_value=0.0,
        action_correction_norm=0.0,
        active_constraints=(),
    )
    assert counterfactual.accepted is expected


def test_independent_counterfactuals_call_selected_nominal_and_safe_hold_separately() -> None:
    action = np.zeros((4, 3), dtype=np.float64)

    class _FakeFilter:
        def __init__(self) -> None:
            self.calls = 0

        def verify_requested_action(self, requested: np.ndarray, observation: dict[str, object]):
            self.calls += 1
            return _diagnostics()

    safety_filter = _FakeFilter()
    results = probe_independent_cbf_counterfactuals(
        selected_action=action,
        nominal_action=action,
        safe_hold_action=action,
        observation=_observation(),
        safety_filter=safety_filter,
        selected_route_id="left_detour:obstacle-0",
    )

    assert safety_filter.calls == 3
    assert tuple(result.label for result in results) == ("selected", "nominal", "safe_hold")
    assert results[0].route_id == "left_detour:obstacle-0"
    assert results[1].route_id is None
    assert results[2].route_id is None
    assert all(result.accepted for result in results)


def test_independent_counterfactuals_reject_nonfinite_action_before_probe() -> None:
    class _NeverCalled:
        def verify_requested_action(self, requested: np.ndarray, observation: dict[str, object]):
            raise AssertionError("non-finite action must be rejected before CBF probe")

    invalid = np.zeros((4, 3), dtype=np.float64)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="selected action"):
        probe_independent_cbf_counterfactuals(
            selected_action=invalid,
            nominal_action=np.zeros((4, 3), dtype=np.float64),
            safe_hold_action=np.zeros((4, 3), dtype=np.float64),
            observation=_observation(),
            safety_filter=_NeverCalled(),
        )

def test_counterfactual_rejects_nonfinite_requested_action() -> None:
    invalid = np.zeros((4, 3), dtype=np.float64)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        CBFRouteCounterfactual(
            label="nominal",
            route_id=None,
            requested_action=invalid,
            verified_feasible=True,
            infeasible=False,
            timed_out=False,
            fallback_mode="none",
            solver_status="success",
            minimum_constraint_value=0.0,
            action_correction_norm=0.0,
            active_constraints=(),
        )
