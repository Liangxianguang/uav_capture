"""Regression tests for the V4/V5 retained-BC contract audit."""

from pathlib import Path

import json
import hashlib
import pytest
import yaml

from scripts.audit_central_v5_contract import _source_integrity, audit


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


def test_recovery_yaml_declares_fixed_and_random_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    document = yaml.safe_load(
        (root / "configs" / "capture_radius_recurrent_behavior_cloning_central_v5_contract_recovery.yaml").read_text(
            encoding="utf-8"
        )
    )
    settings = document["imitation"]
    assert settings["expert_dataset_source_balance"] == "equal_sequences"
    assert len(settings["expert_datasets"]) == 2
    assert "fixed_contract_archive" in settings["expert_datasets"][0]
    assert "bc_baseline_seed661401" in settings["expert_datasets"][1]


def test_source_integrity_validates_recorded_training_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.audit_central_v5_contract as module

    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_dir.joinpath("source_hashes.json").write_text(
        json.dumps({"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}), encoding="utf-8"
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", project)

    result = _source_integrity(run_dir)

    assert result is not None
    assert result["all_recorded_sources_match_workspace"] is True
