"""Regression coverage for the Stage 3C seed-level result aggregation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "aggregate_stage3c_formal.py"


def load_module():
    specification = importlib.util.spec_from_file_location("aggregate_stage3c_formal_test", SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def summary(safe_capture_rate: float, collision_rate: float) -> dict[str, dict[str, float | int]]:
    row = {
        "episodes": 2,
        "safe_capture_rate": safe_capture_rate,
        "capture_rate": min(1.0, safe_capture_rate + 0.02),
        "collision_rate": collision_rate,
        "world_violation_rate": 0.0,
        "mean_capture_time_seconds": 1.5,
        "mean_min_clearance_m": 0.6,
        "mean_visible_fraction": 0.8,
        "mean_message_age_steps": 0.2,
    }
    return {scenario: row.copy() for scenario in ("clutter", "occluded", "open", "overall")}


def test_stage3c_aggregation_uses_training_seed_as_statistical_unit(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    root = tmp_path / "stage3c_formal"
    seeds = (101, 102, 103)
    methods = ("recurrent_no_prediction", "recurrent_gru_prediction")
    root.mkdir(parents=True)
    root.joinpath("protocol.json").write_text(
        json.dumps(
            {
                "stage": "3C_recurrent_formal_multiseed",
                "seeds": list(seeds),
                "methods": list(methods),
                "train_steps": 64,
                "test_seed": 999,
                "episodes_per_scenario": 2,
                "sequence_length": 4,
            }
        ),
        encoding="utf-8",
    )
    for seed_index, seed in enumerate(seeds):
        for method in methods:
            safe_capture = 0.90 + 0.01 * seed_index
            if method == "recurrent_gru_prediction":
                safe_capture += 0.03
            collision = 0.10 - 0.01 * seed_index
            for action in ("raw", "cbf"):
                action_summary = summary(safe_capture if action == "raw" else 1.0, collision if action == "raw" else 0.0)
                output = root / method / f"seed{seed}" / f"evaluation_{action}"
                output.mkdir(parents=True)
                output.joinpath("summary.json").write_text(json.dumps(action_summary), encoding="utf-8")

    output_json = tmp_path / "summary.json"
    output_report = tmp_path / "report.md"
    missing_stage3b = tmp_path / "missing_stage3b.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_stage3c_formal.py",
            "--root",
            str(root),
            "--output-json",
            str(output_json),
            "--output-report",
            str(output_report),
            "--stage3b-summary",
            str(missing_stage3b),
        ],
    )

    module.main()

    aggregate = json.loads(output_json.read_text(encoding="utf-8"))
    paired = aggregate["paired_comparisons"][
        "recurrent_gru_minus_recurrent_no_prediction_raw_safe_capture_percentage_points"
    ]
    metrics = aggregate["results"]["recurrent_gru_prediction"]["raw"]["overall"]["metrics"]
    assert aggregate["statistical_unit"] == "training_seed"
    assert metrics["safe_capture_rate"]["count"] == 3
    assert aggregate["results"]["recurrent_gru_prediction"]["raw"]["overall"]["episodes_total"] == 6
    assert paired["values"] == pytest.approx([3.0, 3.0, 3.0])
    assert aggregate["paired_comparisons"]["positive_seed_count"] == 3
    assert "stage3b_context" not in aggregate
    assert "Recurrent-MAPPO" in output_report.read_text(encoding="utf-8")
