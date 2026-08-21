from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_p3a_rejection_preserves_stop_rule_and_does_not_open_locked_block() -> None:
    path = PROJECT_ROOT / "P3A_WALL_COVERAGE_REJECTION_SUMMARY.json"
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["status"] == "rejected_stop_v5_tuning"
    assert summary["pilot"]["training_integrity_passed"] is True
    assert summary["pilot"]["wall_coverage"]["passed"] is True
    assert summary["completed_fixed_regression_artifacts"]["s1_cylinder_cbf"]["episodes"] == 20
    assert summary["completed_fixed_regression_artifacts"]["s1_cylinder_cbf"]["cooperative_safe_capture_rate"] < summary["pre_registered_fixed_cbf_threshold"]
    assert summary["locked_seed_block_647201_opened"] is False
    assert "s3_development_raw_cbf" in summary["subsequent_evaluations_not_run"]
