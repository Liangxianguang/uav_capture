from __future__ import annotations

import pytest

from scripts.audit_jepa_safe_capture_settled_progress_calibration import (
    _rankdata,
    _spearman,
    _top1_agreement,
)


def test_rankdata_uses_average_ties() -> None:
    assert _rankdata([2.0, 1.0, 1.0]).tolist() == [3.0, 1.5, 1.5]


def test_spearman_is_one_for_matching_order() -> None:
    assert _spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)


def test_spearman_is_negative_for_reverse_order() -> None:
    assert _spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_top1_agreement_uses_settled_after_distance() -> None:
    rows = [
        {"episode_seed": 1, "step": 1, "candidate_index": 0, "score": 2.0, "settled_after_distance_m": 5.0},
        {"episode_seed": 1, "step": 1, "candidate_index": 1, "score": 1.0, "settled_after_distance_m": 4.0},
    ]
    assert _top1_agreement(rows, "score", minimize=True) == 1.0


def test_top1_agreement_keeps_ineligible_candidate_out_of_prediction_pool() -> None:
    rows = [
        {"episode_seed": 1, "step": 1, "candidate_index": 0, "score": 2.0, "online_eligible": True, "settled_after_distance_m": 5.0},
        {"episode_seed": 1, "step": 1, "candidate_index": 1, "score": 1.0, "online_eligible": True, "settled_after_distance_m": 4.0},
        {"episode_seed": 1, "step": 1, "candidate_index": 2, "score": 0.0, "online_eligible": False, "settled_after_distance_m": 3.0},
    ]
    assert _top1_agreement(rows, "score", minimize=True, eligible_only=True) == 0.0
