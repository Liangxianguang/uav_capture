from __future__ import annotations

import numpy as np

from scripts.audit_jepa_safe_capture_v5_p2_ranking import (
    _cbf_abort,
    _confusion_matrix,
    _credit_bucket,
    _deterministic_best,
    _group_stats,
    _nominal_displacement,
    _top_two_margin,
)


def test_deterministic_best_uses_tie_tolerance_and_index_key() -> None:
    scores = [1.0, 1.0002, 1.8, 2.0, 2.1]
    eligible = [True, True, True, False, True]
    assert _deterministic_best(scores, eligible, 5e-4) == 0
    assert np.isclose(_top_two_margin(scores, eligible, 5e-4), 0.0002)


def test_top_two_margin_is_missing_when_only_one_candidate_is_eligible() -> None:
    assert _top_two_margin([1.0, 2.0, 3.0, 4.0, 5.0], [True, False, False, False, False], 5e-4) is None


def test_nominal_displacement_is_mean_per_defender_norm() -> None:
    trace = {
        "reachable_nominal_action": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "requested_action": [[0.0, 3.0, 4.0], [1.0, 0.0, 2.0]],
    }
    assert np.isclose(_nominal_displacement(trace), (5.0 + 2.0) / 2.0)


def test_cbf_abort_requires_unverified_state() -> None:
    assert _cbf_abort({"cbf": {"fallback_mode": "controlled_abort", "verified_feasible": False}}) == (True, "controlled_abort")
    assert _cbf_abort({"cbf": {"infeasible": True, "verified_feasible": True}}) == (False, None)
    assert _cbf_abort({"cbf": {"timed_out": True, "verified_feasible": False}}) == (True, "timeout_unverified")


def test_credit_bucket_uses_frozen_minimum_credit() -> None:
    assert _credit_bucket(0.65, 0.65) == "high"
    assert _credit_bucket(0.649999, 0.65) == "low_or_missing"
    assert _credit_bucket(None, 0.65) == "low_or_missing"


def test_group_stats_exposes_switch_and_safety_rates() -> None:
    rows = [
        {
            "training_seed": 1,
            "variant": "m3",
            "episode_index": 0,
            "selected_not_settled_best": True,
            "selected_matches_predicted_best": False,
            "selected_settled_safety_ok": True,
            "selected_settled_safe_capture": False,
            "cbf_abort": False,
            "candidate_switched": False,
            "top_two_margin_m": 0.1,
            "nominal_displacement_mps": 0.2,
            "selected_ledger_credit": 0.8,
            "hysteresis_applied": False,
            "rank_abstention_reason": None,
        },
        {
            "training_seed": 1,
            "variant": "m3",
            "episode_index": 0,
            "selected_not_settled_best": False,
            "selected_matches_predicted_best": True,
            "selected_settled_safety_ok": False,
            "selected_settled_safe_capture": False,
            "cbf_abort": True,
            "candidate_switched": True,
            "top_two_margin_m": 0.2,
            "nominal_displacement_mps": 0.4,
            "selected_ledger_credit": 0.5,
            "hysteresis_applied": True,
            "rank_abstention_reason": "top_two_margin_abstention",
        },
    ]
    result = _group_stats(rows)
    assert result["decisions"] == 2
    assert result["switch_count"] == 1
    assert result["switch_rate"] == 0.5
    assert result["safety_failure_rate"] == 0.5
    assert result["cbf_abort_count"] == 1


def test_confusion_matrix_has_settled_rows_and_candidate_columns() -> None:
    rows = [
        {"settled_best_index": 1, "predicted_best_index": 0},
        {"settled_best_index": 1, "predicted_best_index": 0},
        {"settled_best_index": 2, "predicted_best_index": 2},
    ]
    matrix = _confusion_matrix(rows, "settled_best_index", "predicted_best_index")
    assert matrix[1][0] == 2
    assert matrix[2][2] == 1
    assert sum(sum(row) for row in matrix) == 3
