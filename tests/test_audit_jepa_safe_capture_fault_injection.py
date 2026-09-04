from __future__ import annotations

from pathlib import Path

from scripts.audit_jepa_safe_capture_fault_injection import (
    _cbf_case,
    _ledger_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"


def test_ledger_fault_matrix_uses_explicit_states_and_reasons() -> None:
    cases = _ledger_cases()
    assert all(item["passed"] for item in cases)
    by_name = {item["name"]: item for item in cases}
    assert by_name["trusted_baseline"]["state"] == "trusted"
    assert by_name["ood"]["fallback_reason"] == "ood"
    assert by_name["stale_observation"]["fallback_reason"] == "stale_observation"
    assert by_name["high_uncertainty"]["fallback_reason"] == "uncertainty_high"
    assert by_name["nonfinite_context"]["fallback_reason"] == "non_finite_context"


def test_nonfinite_request_never_executes_raw_or_unverified_action() -> None:
    result = _cbf_case("nonfinite_request", CONFIG)
    assert result["solver_status"] == "nonfinite_request"
    assert result["raw_unverified_executed"] is False
    assert result["action_finite"] is True
    assert result["passed"] is True


def test_timeout_and_state_violation_are_explicit_controlled_aborts() -> None:
    timeout = _cbf_case("solver_timeout", CONFIG)
    state = _cbf_case("state_violation", CONFIG)
    assert timeout["timed_out"] is True
    assert timeout["fallback_mode"] == "controlled_abort"
    assert timeout["raw_unverified_executed"] is False
    assert state["infeasible"] is True
    assert state["fallback_mode"] == "controlled_abort"
    assert state["raw_unverified_executed"] is False
    assert state["action_finite"] is True
