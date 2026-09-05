from __future__ import annotations

from scripts.audit_jepa_safe_capture_v21_hash_bound_ledger import (
    _audit_case,
    _base_context,
    _fault_cases,
    _source_hash_gate,
    _tamper_provenance_gate,
)
from encirclement3d.reliability import SafeCaptureReliabilityLedger


def _ledger() -> SafeCaptureReliabilityLedger:
    return SafeCaptureReliabilityLedger(
        {
            "ledger_type": "jepa_safe_capture_v3_checkpoint_bound_reliability",
            "ledger_version": 3,
            "not_a_locked_test": True,
            "immutable_after_calibration": True,
            "source": {"checkpoint_sha256": "a" * 64, "calibration_dataset_sha256": "b" * 64},
            "entries": {"h2|global": {"credit": 0.9, "sample_count": 128}},
            "decision_policy": {
                "states": ["trusted", "fallback_nominal", "safe_hold"],
                "minimum_sample_count": 128,
                "minimum_credit": 0.65,
                "maximum_observation_age_steps": 45.0,
                "safe_hold_uncertainty_threshold": 0.40,
                "safe_hold_ttc_seconds": 0.30,
                "bucket_boundary_tolerances": {},
            },
        }
    )


def test_source_hash_gate_requires_all_external_bindings() -> None:
    expected = {"protocol_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
    source = {"protocol_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
    assert _source_hash_gate({"source": source}, expected)["passed"]
    assert not _source_hash_gate({"source": {**source, "protocol_sha256": "0" * 64}}, expected)["passed"]


def test_tampered_provenance_is_rejected() -> None:
    expected = {"protocol_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
    payload = {"source": dict(expected)}
    assert _tamper_provenance_gate(payload, expected)


def test_fault_matrix_routes_all_declared_faults_to_safe_hold() -> None:
    ledger = _ledger()
    for horizon, context, expected in _fault_cases(ledger).values():
        result = _audit_case(ledger, horizon, context, expected)
        assert result["passed"]
        assert result["raw_unverified_executed"] is False


def test_base_context_is_finite_and_complete() -> None:
    context = _base_context()
    assert all(key in context for key in ("minimum_clearance_m", "pairwise_ttc_s", "uncertainty", "cbf_risk"))
