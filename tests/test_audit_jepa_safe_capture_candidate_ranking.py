from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_jepa_safe_capture_candidate_ranking import (
    CANDIDATE_LABELS,
    _finalize_candidate_stats,
    _spearman,
    _validate_ranking,
)


def _ranking(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "candidate_labels": list(CANDIDATE_LABELS),
        "valid_mask": [True] * 5,
        "eligible_mask": [True] * 5,
        "scores": [0.1, 0.2, 0.3, 0.4, 0.5],
        "selected_index": 1,
        "execution_mode": "trusted",
    }
    value.update(overrides)
    return value


def test_validate_ranking_enforces_fixed_contract_and_margin() -> None:
    result = _validate_ranking(_ranking(), identifier="1:m3:0000", step=1)
    assert result["labels"] == list(CANDIDATE_LABELS)
    assert result["selected_index"] == 1
    assert result["top_two_score_margin"] == pytest.approx(0.1)
    assert result["rejection_reasons_present"] is False


def test_validate_ranking_checks_recorded_rejection_reasons() -> None:
    reasons = [[], [], ["speed_limit"], [], []]
    ranking = _ranking(
        valid_mask=[True, True, False, True, True],
        eligible_mask=[True, True, False, True, True],
        selected_index=0,
        candidate_rejection_reasons=reasons,
    )
    result = _validate_ranking(ranking, identifier="1:m3:0000", step=1)
    assert result["rejection_reasons_present"] is True
    assert result["rejection_reasons"][2] == ["speed_limit"]

    with pytest.raises(ValueError, match="no rejection reason"):
        _validate_ranking(
            _ranking(
                valid_mask=[True, True, False, True, True],
                eligible_mask=[True, True, False, True, True],
                selected_index=0,
                candidate_rejection_reasons=[[], [], [], [], []],
            ),
            identifier="1:m3:0000",
            step=1,
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"candidate_labels": ["nominal"]}, "labels mismatch"),
        ({"eligible_mask": [False, True, True, True, True], "selected_index": 0}, "eligible"),
        ({"valid_mask": [False, True, True, True, True], "selected_index": 0}, "Ineligible"),
        ({"execution_mode": "fallback_nominal", "selected_index": 2}, "Fallback"),
    ],
)
def test_validate_ranking_rejects_contract_violations(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_ranking(_ranking(**overrides), identifier="1:m3:0000", step=1)


def test_spearman_returns_none_for_constant_or_short_samples() -> None:
    assert _spearman([1.0, 1.0, 1.0], [0.0, 1.0, 0.0]) is None
    assert _spearman([1.0, 2.0], [0.0, 1.0]) is None


def test_finalize_candidate_stats_uses_only_selected_episode_outcomes() -> None:
    stats = {
        label: {
            "steps": 2,
            "valid_steps": 2,
            "eligible_steps": 2,
            "selected_steps": 0,
            "trusted_selected_steps": 0,
            "score_sum": 2.0,
            "score_count": 2,
            "episode_score_means": [0.5, 1.5, 2.5],
            "episode_outcomes": [0.0, 1.0, 0.0],
            "selected_episode_outcomes": [1.0],
        }
        for label in CANDIDATE_LABELS
    }
    result = _finalize_candidate_stats(stats)
    assert result["nominal"]["selected_episode_count"] == 1
    assert result["nominal"]["selected_episode_safe_capture_rate"] == 1.0
    assert result["nominal"]["score_vs_episode_safe_capture_spearman"] is not None
