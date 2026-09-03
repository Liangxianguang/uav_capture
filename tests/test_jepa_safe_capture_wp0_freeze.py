from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.freeze_jepa_safe_capture_next_phase import validate_parent_summary, validate_protocol


def _protocol() -> dict[str, object]:
    return {
        "phase": "development_only",
        "locked_test_opened": False,
        "objective": {"primary_endpoint": "safe_capture"},
        "immutable_invariants": {
            "online_target_ground_truth": False,
            "jepa_can_generate_final_action": False,
            "execute_only_first_step_then_replan": True,
            "cbf_is_final_execution_boundary": True,
            "candidate_count": 5,
            "chunk_length_steps": 3,
        },
    }


def _parent() -> dict[str, object]:
    return {
        "locked_test_opened": False,
        "decision": {
            "classification": "positive_development_evidence",
            "safety_hard_gate": True,
            "a3_excluded_from_safety_decision": True,
            "m3_mean_paired_delta_rate": 0.0416666667,
            "m3_cross_seed_bootstrap": {"ci95_low": -0.0417},
        },
        "tensorboard": {"required_provenance": True},
    }


def test_freeze_contract_validation_accepts_development_parent() -> None:
    validate_protocol(_protocol(), Path("protocol.yaml"))
    validate_parent_summary(_parent(), Path("summary.json"))


@pytest.mark.parametrize(
    "field,value",
    [("locked_test_opened", True), ("phase", "locked_test"), ("objective", {"primary_endpoint": "capture_time"})],
)
def test_freeze_rejects_protocol_boundary_changes(field: str, value: object) -> None:
    protocol = _protocol()
    protocol[field] = value
    with pytest.raises(ValueError):
        validate_protocol(protocol, Path("protocol.yaml"))


def test_freeze_rejects_parent_without_safety_or_provenance() -> None:
    parent = copy.deepcopy(_parent())
    parent["decision"]["safety_hard_gate"] = False  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_parent_summary(parent, Path("summary.json"))
