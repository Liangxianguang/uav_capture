from __future__ import annotations

import pytest

from encirclement3d.reliability import (
    SafeCaptureReliabilityLedger,
    make_safe_capture_coarse_context_key,
    make_safe_capture_context_key,
    make_safe_capture_global_key,
)


def _context() -> dict[str, object]:
    return {
        "visibility_condition": 1.0,
        "observation_age_steps": 2.0,
        "obstacle_count": 3,
        "layout_signature": "scenario_0",
        "target_motion_mode": "flee_persistence",
        "minimum_clearance_m": 1.0,
        "pairwise_ttc_s": 2.0,
        "uncertainty": 0.05,
        "cbf_risk": 0.1,
        "candidate_separation_m": 0.3,
    }


def _entry(credit: float, sample_count: int) -> dict[str, float | int]:
    return {"credit": credit, "sample_count": sample_count}


def _payload() -> dict[str, object]:
    full = make_safe_capture_context_key(
        3,
        "visible",
        "fresh",
        3,
        "scenario_0",
        "flee_persistence",
        "clear",
        "distant",
        "low",
        "low",
        "high",
    )
    coarse = make_safe_capture_coarse_context_key(
        3,
        "visible",
        "fresh",
        3,
        "flee_persistence",
        "clear",
        "low",
        "low",
    )
    return {
        "ledger_type": SafeCaptureReliabilityLedger.LEDGER_TYPE,
        "ledger_version": 2,
        "not_a_locked_test": True,
        "immutable_after_calibration": True,
        "source": {"checkpoint_sha256": "a" * 64, "calibration_dataset_sha256": "b" * 64},
        "entries": {
            full: _entry(0.90, 200),
            coarse: _entry(0.80, 200),
            make_safe_capture_global_key(3): _entry(0.70, 500),
            make_safe_capture_global_key(4): _entry(0.40, 500),
        },
        "decision_policy": {
            "states": ["trusted", "fallback_nominal", "safe_hold"],
            "minimum_sample_count": 128,
            "minimum_credit": 0.65,
            "maximum_observation_age_steps": 45.0,
            "safe_hold_uncertainty_threshold": 0.40,
            "safe_hold_ttc_seconds": 0.30,
        },
    }


def test_v2_ledger_selects_local_and_fallback_states() -> None:
    ledger = SafeCaptureReliabilityLedger(_payload())
    local = ledger.decision(3, _context())
    assert local.state == "trusted"
    assert local.used_coarse_fallback is False
    assert local.used_global_fallback is False

    altered = _context()
    altered["layout_signature"] = "unseen_layout"
    coarse = ledger.decision(3, altered)
    assert coarse.state == "trusted"
    assert coarse.used_coarse_fallback is True

    low = ledger.decision(4, _context())
    assert low.state == "fallback_nominal"
    assert low.fallback_reason == "low_credit"


def test_v2_ledger_safe_holds_ood_and_hard_contexts() -> None:
    ledger = SafeCaptureReliabilityLedger(_payload())
    ood = _context()
    ood["ood"] = True
    assert ledger.decision(3, ood).state == "safe_hold"
    stale = _context()
    stale["observation_age_steps"] = 46.0
    assert ledger.decision(3, stale).fallback_reason == "stale_observation"
    uncertain = _context()
    uncertain["uncertainty"] = 0.5
    assert ledger.decision(3, uncertain).fallback_reason == "uncertainty_high"


def test_v2_ledger_rejects_missing_context() -> None:
    ledger = SafeCaptureReliabilityLedger(_payload())
    with pytest.raises(ValueError, match="missing fields"):
        ledger.decision(3, {"visibility_condition": 1.0})
