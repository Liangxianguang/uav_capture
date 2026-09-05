from __future__ import annotations

from scripts.audit_jepa_safe_capture_v21_s3_divergence import age_episode_summary, candidate_snapshot, earliest_divergence


def _cbf(**overrides):
    value = {
        "solver_status": "success",
        "solver_success": True,
        "infeasible": False,
        "timed_out": False,
        "unverified": False,
        "verified_feasible": True,
        "fallback_mode": "none",
        "requested_action_finite": True,
        "constraint_slacks": {"pairwise_0_1": 1.0, "boundary_upper_defender_0_axis_0": 2.0},
        "active_constraints": [],
    }
    value.update(overrides)
    return value


def _row(step: int, cbf: dict, *, eligible=None, requested=None, nominal=None, observation=None):
    ranking = {
        "selected_index": 0,
        "execution_mode": "trusted",
        "fallback_reason": None,
        "eligible_mask": [True, False, True] if eligible is None else eligible,
        "valid_mask": [True, True, True],
        "candidate_eligibility_reasons": [[], ["insufficient_candidate_separation"], []],
        "minimum_candidate_separation_m": 0.002,
        "predicted_min_clearance_m": [0.3, 0.2, 0.4],
    }
    return {
        "episode_index": 0,
        "step": step,
        "requested_action": requested or [[0.1, 0.0, 0.0]],
        "reachable_nominal_action": nominal or [[0.1, 0.0, 0.0]],
        "candidate_ranking": ranking,
        "observation": observation or {"target_visible": [True], "message_age_steps": [0], "target_observation_age_steps": [0]},
        "cbf": cbf,
    }


def test_earliest_divergence_classifies_negative_pairwise_slack_and_nominal_match() -> None:
    failing_cbf = _cbf(
        solver_status="controlled_abort;primary=solver_failure",
        solver_success=False,
        infeasible=True,
        unverified=True,
        verified_feasible=False,
        fallback_mode="controlled_abort",
        constraint_slacks={"pairwise_0_1": -0.0776, "obstacle_0_defender_0": 0.12},
    )
    result = earliest_divergence([_row(1, _cbf()), _row(2, failing_cbf)])
    assert result["has_failure"] is True
    assert result["first_failure_step"] == 2
    assert result["previous_step"] == 1
    assert result["root_cause"] == "cbf_constraint_infeasible"
    assert result["constraint_category"] == "pairwise"
    assert result["negative_constraint_count"] == 1
    assert result["nominal_action_match"] is True
    assert result["previous_verified_feasible"] is True


def test_earliest_divergence_distinguishes_solver_failure_without_negative_slack() -> None:
    failing_cbf = _cbf(
        solver_status="controlled_abort;primary=solver_failure",
        solver_success=False,
        infeasible=True,
        unverified=True,
        verified_feasible=False,
        fallback_mode="controlled_abort",
    )
    result = earliest_divergence([_row(1, _cbf()), _row(2, failing_cbf)])
    assert result["root_cause"] == "cbf_solver_failure_or_initialization"
    assert result["negative_constraint_count"] == 0
    assert result["constraint_category"] == "pairwise"


def test_candidate_snapshot_preserves_all_ineligible_and_rejection_reasons() -> None:
    snapshot = candidate_snapshot(_row(1, _cbf(), eligible=[False, False, False]))
    assert snapshot["all_ineligible"] is True
    assert snapshot["eligible_count"] == 0
    assert snapshot["valid_count"] == 3
    assert snapshot["candidate_eligibility_reasons"][1] == ["insufficient_candidate_separation"]


def test_age_summary_separates_message_saturation_from_target_staleness() -> None:
    trace = [
        _row(1, _cbf(), observation={"target_visible": [True], "message_age_steps": [60], "target_observation_age_steps": [2]}),
        _row(2, _cbf(), observation={"target_visible": [False], "message_age_steps": [60], "target_observation_age_steps": [46]}),
    ]
    result = age_episode_summary(trace)
    assert result["message_age_saturated"] is True
    assert result["target_observation_stale"] is True
    assert result["message_saturated_rows"] == 2
    assert result["target_stale_rows"] == 1
    assert result["saturation_with_target_visible_rows"] == 1
