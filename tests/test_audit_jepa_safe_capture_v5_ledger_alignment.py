from __future__ import annotations

from scripts.audit_jepa_safe_capture_v5_ledger_alignment import _credit_stats


def _row(*, credit: float, safe: bool, safety_ok: bool, pair_label: str = "tied") -> dict:
    return {
        "selected_index": 0,
        "ledger_credits": [credit],
        "ledger_states": ["trusted" if credit >= 0.65 else "fallback_nominal"],
        "selected_settled_safe_capture": safe,
        "selected_settled_safety_ok": safety_ok,
        "training_seed": 1,
        "variant": "m3",
        "episode_index": 0,
        "pair_label": pair_label,
    }


def test_credit_stats_uses_safety_failure_for_reliability_gate() -> None:
    stats = _credit_stats(
        [
            _row(credit=0.80, safe=False, safety_ok=True),
            _row(credit=0.80, safe=True, safety_ok=True),
            _row(credit=0.50, safe=False, safety_ok=False),
        ],
        minimum_credit=0.65,
    )
    assert stats["buckets"]["high"]["safe_capture_failure_rate"] == 0.5
    assert stats["buckets"]["high"]["failure_rate"] == 0.0
    assert stats["buckets"]["low_or_missing"]["failure_rate"] == 1.0
    assert stats["high_credit_failure_not_above_low_credit"] is True


def test_credit_stats_reports_missing_low_credit_coverage() -> None:
    stats = _credit_stats([_row(credit=0.90, safe=False, safety_ok=True)], minimum_credit=0.65)
    assert stats["coverage"] == {"high_decisions": 1, "low_decisions": 0, "both_buckets_present": False}
    assert stats["high_credit_failure_not_above_low_credit"] is None


def test_credit_stats_counts_states_and_pair_labels() -> None:
    stats = _credit_stats(
        [_row(credit=0.80, safe=False, safety_ok=True, pair_label="degraded"), _row(credit=0.40, safe=False, safety_ok=False)],
        minimum_credit=0.65,
    )
    assert stats["state_counts"] == {"fallback_nominal": 1, "trusted": 1}
    assert stats["buckets"]["high"]["pair_labels"] == {"degraded": 1}
