"""Regression tests for the V4/V5 retained-BC contract audit."""

from pathlib import Path

import yaml

from scripts.audit_central_v5_contract import audit


def _write_config(path: Path, *, datasets: list[str], balance: str | None, stages: list[dict]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "environment_config": "env.yaml",
                "imitation": {
                    "seed": 1,
                    "episodes": 2,
                    "expert_datasets": datasets,
                    "expert_dataset_source_balance": balance,
                    "training_required_defender_zone_entries": 2,
                    "action_scale_mode": "full_range",
                    "hidden_dim": 128,
                    "training_showcase_stages": stages,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_audit_detects_fresh_v5_collection_instead_of_fixed_archive(tmp_path: Path) -> None:
    stage = [{"until_progress": 1.0, "layouts": ["mixed"]}]
    v4 = tmp_path / "v4.yaml"
    v5 = tmp_path / "v5.yaml"
    environment = tmp_path / "env.yaml"
    _write_config(v4, datasets=["fixed_shapeaware.npz", "random_s3.npz"], balance="equal_sequences", stages=stage)
    _write_config(v5, datasets=[], balance=None, stages=stage)
    environment.write_text("task: {}\n", encoding="utf-8")

    report = tmp_path / "retention.md"
    report.write_text("Fixed V4 expert archive\nequal-sequence balancing\n", encoding="utf-8")

    result = audit(v4, v5, environment, v4_retention_report=report)

    assert result["findings"]["v4_fixed_archive_declared"] is True
    assert result["findings"]["v5_uses_local_collection"] is True
    assert result["findings"]["high_risk_data_contract_gap"] is True


def test_audit_accepts_equal_balanced_archives_with_matching_stage_contract(tmp_path: Path) -> None:
    stage = [{"until_progress": 1.0, "layouts": ["mixed"]}]
    v4 = tmp_path / "v4.yaml"
    v5 = tmp_path / "v5.yaml"
    environment = tmp_path / "env.yaml"
    datasets = ["fixed_shapeaware.npz", "random_s3.npz"]
    _write_config(v4, datasets=datasets, balance="equal_sequences", stages=stage)
    _write_config(v5, datasets=datasets, balance="equal_sequences", stages=stage)
    environment.write_text("task: {}\n", encoding="utf-8")

    result = audit(v4, v5, environment)

    assert result["findings"]["high_risk_data_contract_gap"] is False
    assert result["findings"]["equal_sequence_balance_preserved"] is True
    assert result["findings"]["stage_contract_changed"] is False
