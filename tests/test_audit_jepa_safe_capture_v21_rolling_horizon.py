from __future__ import annotations

from scripts.audit_jepa_safe_capture_v21_rolling_horizon import _compare_runs


def _run(trace: dict) -> dict:
    return {"traces": [trace]}


def test_replay_comparator_ignores_wall_clock_latency_only() -> None:
    left = {
        "requested_action": [[1.0, 0.0, 0.0]],
        "executed_action": [[1.0, 0.0, 0.0]],
        "raw_unverified_executed": False,
        "candidate_ranking": {
            "selected_index": 0,
            "scores": [1.0],
            "rank_total_latency_ms": 11.0,
        },
        "cbf": {
            "verified_feasible": True,
            "solver_status": "success",
            "solve_latency_ms": 3.0,
        },
    }
    right = {
        **left,
        "candidate_ranking": {**left["candidate_ranking"], "rank_total_latency_ms": 2.0},
        "cbf": {**left["cbf"], "solve_latency_ms": 0.5},
    }
    result = _compare_runs(_run(left), _run(right))
    assert result["passed"] is True
    assert result["field_difference_count"] == 0


def test_replay_comparator_keeps_action_and_safety_differences() -> None:
    left = {
        "requested_action": [[1.0, 0.0, 0.0]],
        "executed_action": [[1.0, 0.0, 0.0]],
        "raw_unverified_executed": False,
        "candidate_ranking": {"selected_index": 0},
        "cbf": {"verified_feasible": True},
    }
    right = {**left, "executed_action": [[0.0, 1.0, 0.0]]}
    result = _compare_runs(_run(left), _run(right))
    assert result["passed"] is False
    assert result["field_difference_count"] == 1
