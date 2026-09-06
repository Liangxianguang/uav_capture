from __future__ import annotations

from scripts.index_jepa_safe_capture_v21_failures import (
    _settled_episode_diagnostics,
    classify_failure,
    summarize_trace,
)


def _trace(*, fallback: bool = False, stale: bool = False, message_age: int = 0) -> list[dict]:
    return [{
        "episode_index": 0,
        "candidate_ranking": {
            "selected_index": 0,
            "execution_mode": "fallback_nominal" if fallback else "jepa_ranked",
            "fallback_reason": "no_trusted_candidate" if fallback else None,
            "ledger_states": ["trusted"] * 5,
            "ledger_credits": [0.9] * 5,
            "predicted_min_clearance_m": [0.5] * 5,
        },
        "observation": {
            "target_visible": [True, True, True, True],
            "target_observation_age_steps": [60, 60, 60, 60] if stale else [0, 0, 0, 0],
            "message_age_steps": [message_age, message_age, message_age, message_age],
        },
        "safety_observables": {"minimum_obstacle_clearance_m": 0.5},
        "cbf": {
            "fallback_mode": "none",
            "verified_feasible": True,
            "infeasible": False,
            "timed_out": False,
            "action_correction_norm": 0.0,
            "solve_latency_ms": 1.0,
        },
    }]


def test_summarize_trace_detects_fallback_and_stale_signals() -> None:
    summary = summarize_trace(_trace(fallback=True, stale=True))
    assert summary["ranking_mode_counts"] == {"fallback_nominal": 1}
    assert summary["observation_age_max_steps"] == 60.0


def test_message_age_saturation_is_distinct_from_stale_target_observation() -> None:
    summary = summarize_trace(_trace(message_age=60))
    assert summary["observation_age_max_steps"] == 0.0
    primary, labels = classify_failure(
        {"cooperative_safe_capture": "False", "termination_reason": "timeout"},
        summary,
        None,
    )
    assert primary == "timeout"
    assert "stale_observation" not in labels
    assert "communication_age_saturated" not in labels
    assert "communication_age_unresolved" in labels


def test_explicit_message_age_state_can_prove_saturation() -> None:
    trace = _trace(message_age=60)
    trace[0]["observation"]["message_received"] = [True, True, True, True]
    trace[0]["observation"]["message_age_state"] = ["saturated"] * 4
    summary = summarize_trace(trace)
    assert summary["message_age_semantics"] == "explicit_or_received_inferred"
    assert summary["message_age_saturated_rows"] == 1
    primary, labels = classify_failure(
        {"cooperative_safe_capture": "False", "termination_reason": "timeout"},
        summary,
        None,
    )
    assert primary == "timeout"
    assert "communication_age_saturated" in labels
    assert "communication_age_unresolved" not in labels


def test_classify_failure_prioritizes_cbf_abort() -> None:
    episode = {"cooperative_safe_capture": "False", "cbf_controlled_abort_steps": "1", "termination_reason": "cbf_controlled_abort"}
    summary = summarize_trace(_trace(fallback=True))
    primary, labels = classify_failure(episode, summary, None)
    assert primary == "cbf_controlled_abort"
    assert "low_credit_or_nominal_fallback" in labels


def test_classify_failure_records_settled_rank_and_high_credit() -> None:
    episode = {"cooperative_safe_capture": "False", "termination_reason": "timeout", "cbf_controlled_abort_steps": "0"}
    summary = summarize_trace(_trace())
    settled = {"settled_selected_not_best_count": 1, "settled_high_credit_failure_count": 1}
    primary, labels = classify_failure(episode, summary, settled)
    assert primary == "timeout"
    assert "candidate_capture_regression" in labels
    assert "high_credit_failure" in labels


def test_high_credit_threshold_is_explicit_and_finite() -> None:
    settled = {
        ("m3", 0, 0): {
            "selected_index": 0,
            "ledger_credits": [0.80],
            "selected_settled_safety_ok": False,
            "selected_settled_termination_reason": "timeout",
            "selected_not_best": False,
        }
    }
    summary = _settled_episode_diagnostics(settled, "m3", 0)
    assert summary["settled_high_credit_failure_count"] == 1
