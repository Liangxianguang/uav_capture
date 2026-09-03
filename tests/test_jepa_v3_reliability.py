from __future__ import annotations

from encirclement3d.reliability import ReliabilityLedger, make_context_key, make_global_key


def _payload() -> dict:
    return {
        "ledger_type": "jepa_v3_execution_settled_reliability",
        "entries": {
            make_global_key(3): {"credit": 0.8, "sample_count": 500},
            make_context_key(3, 1.0, 0.0, 1.0, 2.0): {"credit": 0.9, "sample_count": 200},
            make_context_key(3, 0.0, 0.9, 0.1, 4.0): {"credit": 0.2, "sample_count": 200},
        },
        "decision_policy": {"minimum_sample_count": 128, "minimum_credit": 0.65},
    }


def test_reliability_ledger_uses_local_global_and_nominal_fallbacks() -> None:
    ledger = ReliabilityLedger(_payload())
    trusted = ledger.decision(3, 1.0, 0.0, 1.0, 2.0)
    assert trusted.fallback_to_nominal is False
    assert trusted.used_global_fallback is False
    unknown = ledger.decision(3, 1.0, 0.9, 0.1, 4.0)
    assert unknown.fallback_to_nominal is False
    assert unknown.used_global_fallback is True
    low_credit = ledger.decision(3, 0.0, 0.9, 0.1, 4.0)
    assert low_credit.fallback_to_nominal is True
    missing_horizon = ledger.decision(99, 1.0, 0.0, 1.0, 2.0)
    assert missing_horizon.fallback_to_nominal is True
    assert missing_horizon.sample_count == 0
