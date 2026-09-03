from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.aggregate_jepa_safe_capture_v2_prediction import (
    aggregate,
    load_gate,
    render_markdown,
    write_tensorboard,
)


def _metric(improvement: float, horizon: float) -> dict[str, float | None]:
    return {
        "horizon_seconds": horizon,
        "target_position_mae_m": 0.2,
        "constant_velocity_mae_m": 0.4,
        "target_improvement_over_constant_velocity_fraction": improvement,
        "target_one_std_coverage": 0.8,
        "target_velocity_mae_mps": 0.3,
        "target_acceleration_mae_mps2": 0.2,
        "obstacle_clearance_lower_quantile_mae_m": 0.5,
        "inter_agent_clearance_lower_quantile_mae_m": 0.2,
        "pairwise_ttc_mae_s": 0.4,
        "visibility_brier": 0.1,
        "visibility_auc": 0.7,
        "observation_age_mae_steps": 0.5,
        "cbf_correction_mae_mps": 0.1,
        "cbf_intervention_brier": 0.1,
        "cbf_intervention_auc": 0.8,
        "qp_feasibility_brier": 0.0,
        "qp_feasibility_auc": None,
    }


def _write_gate(root: Path, seed: int, *, dataset_hash: str = "b" * 64) -> Path:
    run = root / f"p2_seed{seed}"
    run.mkdir(parents=True)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint-{seed}".encode("ascii"))
    gate_path = run / "prediction_gate.json"
    gate = {
        "evaluation_type": "jepa_safe_capture_v2_p2_prediction_gate",
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset": str(root / "validation.npz"),
        "dataset_sha256": dataset_hash,
        "metadata_sha256": "c" * 64,
        "samples": 10,
        "metrics_by_horizon": [_metric(0.1 + seed / 10000.0, 0.1), _metric(0.2 + seed / 10000.0, 0.5)],
        "prediction_gate": {
            "all_finite": True,
            "target_better_than_constant_velocity_all_horizons": True,
        },
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return gate_path


def test_three_seed_aggregate_preserves_per_horizon_gate(tmp_path: Path) -> None:
    runs = [load_gate(_write_gate(tmp_path, seed)) for seed in (20260911, 20260912, 20260913)]
    report = aggregate(runs)
    assert report["run_count"] == 3
    assert report["seeds"] == [20260911, 20260912, 20260913]
    assert report["decision"]["all_seed_target_improvements_positive_at_all_horizons"] is True
    assert report["metrics_by_horizon"][0]["seeds_better_than_constant_velocity"] == 3
    assert report["metrics_by_horizon"][0]["qp_feasibility_auc"] is None
    assert "not a closed-loop result" in render_markdown(report)


def test_aggregate_rejects_validation_hash_mismatch(tmp_path: Path) -> None:
    first = load_gate(_write_gate(tmp_path, 20260911))
    second = load_gate(_write_gate(tmp_path, 20260912, dataset_hash="d" * 64))
    third = load_gate(_write_gate(tmp_path, 20260913))
    with pytest.raises(ValueError, match="dataset hashes differ"):
        aggregate([first, second, third])


def test_tensorboard_aggregate_writes_provenance(tmp_path: Path) -> None:
    runs = [load_gate(_write_gate(tmp_path / "inputs", seed)) for seed in (20260911, 20260912, 20260913)]
    report = aggregate(runs)
    audit = write_tensorboard(report, tmp_path / "tb")
    assert audit["required_text_complete"] is True
    assert audit["scalar_tag_count"] > 0
    assert audit["text_tag_count"] >= 4
