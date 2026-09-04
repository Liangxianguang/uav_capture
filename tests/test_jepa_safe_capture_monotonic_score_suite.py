from __future__ import annotations

from scripts.run_jepa_safe_capture_monotonic_score_suite import run_suite


def test_monotonic_score_suite_passes_all_registered_cases() -> None:
    result = run_suite()

    assert result["development_only"] is True
    assert result["locked_test_opened"] is False
    assert result["candidate_contract"]["jepa_role"] == "candidate_trajectory_evaluator_only"
    assert result["all_cases_passed"] is True
    assert len(result["cases"]) == 7
    assert all(case["passed"] for case in result["cases"])


def test_clearance_case_rejects_task_progress_when_prediction_floor_fails() -> None:
    result = run_suite()
    case = next(item for item in result["cases"] if item["name"] == "clearance_gate")

    assert case["selected_index"] == 0
    assert case["eligible_mask"][1] is False
    assert case["predicted_min_clearance_m"][1] < 0.15
