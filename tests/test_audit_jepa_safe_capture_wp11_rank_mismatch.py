from __future__ import annotations

from scripts.audit_jepa_safe_capture_wp11_rank_mismatch import _episode_metrics


def _record(selected: int, best: int, margin: float, credit: float) -> dict[str, object]:
    return {
        "candidate_ranking": {
            "selected_index": selected,
            "best_score_index": best,
            "top_two_score_margin": margin,
            "ledger_credits": [credit],
        },
        "cbf": {
            "action_correction_norm": 0.2,
            "solve_latency_ms": 2.0,
            "verified_feasible": True,
            "unverified": False,
        },
    }


def test_episode_metrics_exposes_switch_and_selected_not_best() -> None:
    row = {"diagnostic_labels_json": "[\"high_credit_failure\",\"candidate_oscillation\"]"}
    pair = {
        "training_seed": 20260911,
        "episode_index": 3,
        "episode_seed": 649004,
        "pair_label": "degraded",
        "delta": -1,
        "base_safe_capture": True,
        "candidate_safe_capture": False,
    }
    result = _episode_metrics(row, pair, [_record(0, 0, 0.1, 0.8), _record(1, 0, 0.2, 0.8)])
    assert result["selected_switch_rate"] == 1.0
    assert result["selected_not_best_fraction"] == 0.5
    assert result["high_credit_failure"] is True
    assert result["candidate_oscillation"] is True


def test_episode_metrics_counts_unverified_trace_steps() -> None:
    row = {"diagnostic_labels_json": "[]"}
    pair = {
        "training_seed": 20260911,
        "episode_index": 4,
        "episode_seed": 649005,
        "pair_label": "tied",
        "delta": 0,
        "base_safe_capture": False,
        "candidate_safe_capture": False,
    }
    payload = [_record(0, 0, 0.1, 0.8), _record(0, 0, 0.2, 0.8)]
    payload[1]["cbf"] = {"verified_feasible": False, "unverified": True, "action_correction_norm": 0.4, "solve_latency_ms": 3.0}
    result = _episode_metrics(row, pair, payload)
    assert result["cbf_unverified_steps"] == 1
    assert result["selected_switch_rate"] == 0.0
