from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_e1_e0_feasibility_rejection_preserves_the_gate_and_locked_stop() -> None:
    summary = json.loads((PROJECT_ROOT / "E1_RULE_EXPERT_FEASIBILITY_REJECTION_SUMMARY.json").read_text(encoding="utf-8"))
    completed = summary["completed_profile"]
    assert summary["decision"] == "rejected_early"
    assert completed["profile"] == "E0"
    assert completed["episodes"] == 60
    assert completed["metrics"]["cooperative_safe_capture_rate"] == 56 / 60
    assert completed["gate"]["cooperative_safe_capture_at_least"] == 0.95
    assert completed["gate"]["passed"] is False
    assert summary["protocol_conformant_stop"]["E1_policy_development_opened"] is False
    assert summary["protocol_conformant_stop"]["E1_locked_cases_opened"] is False
    assert summary["preregistration"]["locked_seed_block_unopened"] == 681201
