from __future__ import annotations

import csv
from pathlib import Path
from scripts.index_jepa_safe_capture_failures import _read_episode_metadata, classify_episode, summarize_trace


def _episode(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "safe_capture": False,
        "collision": False,
        "boundary_violation": False,
        "pairwise_violation": False,
        "cbf_unverified_steps": 0,
        "cbf_infeasible_steps": 0,
        "cbf_controlled_abort_steps": 0,
        "termination_reason": "timeout",
        "mean_visible_fraction": 1.0,
    }
    value.update(overrides)
    return value


def test_trace_summary_records_switches_fallback_and_cbf_diagnostics() -> None:
    trace = [
        {
            "candidate_ranking": {
                "selected_index": 0,
                "execution_mode": "trusted",
                "ledger_states": ["trusted"],
                "ledger_credits": [0.9],
                "predicted_min_clearance_m": [1.0],
                "predicted_visibility": [0.9],
            },
            "cbf": {"verified_feasible": True, "infeasible": False, "timed_out": False, "action_correction_norm": 0.1, "solve_latency_ms": 3.0},
            "safety_observables": {"minimum_obstacle_clearance_m": 0.8},
            "observation": {"target_visible": [True], "target_observation_age_steps": [0], "message_age_steps": [0]},
        },
        {
            "candidate_ranking": {
                "selected_index": 2,
                "execution_mode": "fallback_nominal",
                "ledger_states": ["fallback_nominal"],
                "ledger_credits": [0.4],
                "predicted_min_clearance_m": [1.0],
                "predicted_visibility": [0.9],
            },
            "cbf": {"verified_feasible": False, "infeasible": True, "timed_out": False, "action_correction_norm": 1.0, "solve_latency_ms": 12.0},
            "safety_observables": {"minimum_obstacle_clearance_m": 0.2},
            "observation": {"target_visible": [False], "target_observation_age_steps": [4], "message_age_steps": [5]},
        },
    ]
    result = summarize_trace(trace)
    assert result["candidate_switch_count"] == 1
    assert result["candidate_switch_rate"] == 1.0
    assert result["cbf_unverified_steps_trace"] == 1
    assert result["cbf_infeasible_steps_trace"] == 1
    assert result["observation_age_max_steps"] == 4.0
    assert result["message_age_max_steps"] == 5.0
    assert result["clearance_prediction_gap_mean_m"] == 0.5
    assert result["clearance_overoptimism_max_m"] == 0.8
    assert result["visibility_prediction_gap_mean"] == 0.4


def test_classification_prioritizes_safety_and_records_regression_labels() -> None:
    trace_summary = {
        "cbf_unverified_steps_trace": 0,
        "ledger_state_counts": {"trusted": 2},
        "observation_age_max_steps": 0.0,
        "message_age_max_steps": 0.0,
        "candidate_switch_rate": 0.0,
        "clearance_overoptimism_max_m": 0.0,
        "visibility_prediction_gap_mean": 0.0,
    }
    primary, labels = classify_episode(
        _episode(collision=True), trace_summary, baseline_safe_capture=True, variant="m3"
    )
    assert primary == "collision"
    assert "candidate_capture_regression" in labels


def test_classification_marks_high_credit_failure_and_stale_observation() -> None:
    trace_summary = {
        "cbf_unverified_steps_trace": 0,
        "ledger_state_counts": {"trusted": 4},
        "observation_age_max_steps": 4.0,
        "message_age_max_steps": 4.0,
        "candidate_switch_rate": 0.0,
        "clearance_overoptimism_max_m": 0.0,
        "visibility_prediction_gap_mean": 0.0,
    }
    primary, labels = classify_episode(
        _episode(termination_reason="timeout"), trace_summary, baseline_safe_capture=False, variant="m3"
    )
    assert primary == "timeout"
    assert "high_credit_failure" in labels
    assert "stale_observation" in labels


def test_episode_metadata_reader_preserves_context_fields(tmp_path: Path) -> None:
    path = tmp_path / "episodes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_index", "layout_signature", "obstacle_count"])
        writer.writeheader()
        writer.writerow({"episode_index": 0, "layout_signature": "cylinder1", "obstacle_count": 3})
    metadata = _read_episode_metadata(path)
    assert metadata[0]["layout_signature"] == "cylinder1"
    assert metadata[0]["obstacle_count"] == "3"
