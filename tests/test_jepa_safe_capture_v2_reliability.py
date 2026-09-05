from __future__ import annotations

import pytest

from encirclement3d.reliability import (
    SafeCaptureReliabilityLedger,
    _safe_capture_clearance_bucket,
    _safe_capture_observation_age_bucket,
    _safe_capture_risk_bucket,
    _safe_capture_separation_bucket,
    _safe_capture_ttc_bucket,
    _safe_capture_uncertainty_bucket,
    _safe_capture_visibility_bucket,
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


def test_v3_ledger_accepts_checkpoint_bound_payload_and_nonfinite_safe_hold() -> None:
    payload = _payload()
    payload["ledger_type"] = SafeCaptureReliabilityLedger.LEDGER_TYPE_V3
    payload["ledger_version"] = 3
    ledger = SafeCaptureReliabilityLedger(payload)
    nonfinite = _context()
    nonfinite["uncertainty"] = float("nan")
    decision = ledger.decision(3, nonfinite)
    assert decision.state == "safe_hold"
    assert decision.fallback_reason == "non_finite_context"


def test_v2_ledger_rejects_missing_context() -> None:
    ledger = SafeCaptureReliabilityLedger(_payload())
    with pytest.raises(ValueError, match="missing fields"):
        ledger.decision(3, {"visibility_condition": 1.0})


def test_ledger_routes_never_received_observation_to_explicit_safe_hold() -> None:
    ledger = SafeCaptureReliabilityLedger(_payload())
    context = _context()
    context["observation_age_state"] = "never_received"
    decision = ledger.decision(3, context)
    assert decision.state == "safe_hold"
    assert decision.fallback_reason == "observation_never_received"


def test_ledger_rejects_unknown_observation_age_state() -> None:
    ledger = SafeCaptureReliabilityLedger(_payload())
    context = _context()
    context["observation_age_state"] = "impossible"
    with pytest.raises(ValueError, match="observation_age_state is invalid"):
        ledger.decision(3, context)


def test_conservative_bucket_canonicalization_uses_the_less_safe_side() -> None:
    tolerance = 0.002
    assert _safe_capture_visibility_bucket(0.5009, boundary_tolerance=tolerance) == "occluded"
    assert _safe_capture_observation_age_bucket(0.0999, boundary_tolerance=tolerance) == "delayed"
    assert _safe_capture_clearance_bucket(0.3509, boundary_tolerance=tolerance) == "critical"
    assert _safe_capture_clearance_bucket(0.7509, boundary_tolerance=tolerance) == "near"
    assert _safe_capture_ttc_bucket(0.5009, boundary_tolerance=tolerance) == "imminent"
    assert _safe_capture_uncertainty_bucket(0.0999, boundary_tolerance=tolerance) == "medium"
    assert _safe_capture_risk_bucket(0.5998, boundary_tolerance=tolerance) == "high"
    assert _safe_capture_risk_bucket(0.6009, boundary_tolerance=tolerance) == "high"
    assert _safe_capture_separation_bucket(0.0509, boundary_tolerance=tolerance) == "low"


def test_ledger_canonicalization_routes_joint_risk_boundary_to_safe_hold() -> None:
    payload = _payload()
    payload["decision_policy"]["bucket_boundary_tolerances"] = {
        "visibility_fraction": 0.002,
        "observation_age_steps": 0.05,
        "clearance_m": 0.002,
        "ttc_s": 0.002,
        "uncertainty": 0.002,
        "cbf_risk": 0.002,
        "candidate_separation_m": 0.002,
    }
    ledger = SafeCaptureReliabilityLedger(payload)
    context = _context()
    context["pairwise_ttc_s"] = 0.3009
    context["cbf_risk"] = 0.5998
    decision = ledger.decision(3, context)
    assert decision.state == "safe_hold"
    assert decision.fallback_reason == "joint_ttc_cbf_risk"
