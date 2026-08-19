"""Regression coverage for P1 seed-level stress aggregation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "aggregate_stage3c_p1_stress.py"


def load_module():
    specification = importlib.util.spec_from_file_location("aggregate_stage3c_p1_test", SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def summary(safe_capture_rate: float, collision_rate: float) -> dict[str, dict[str, float | int]]:
    row = {
        "episodes": 4,
        "safe_capture_rate": safe_capture_rate,
        "capture_rate": min(1.0, safe_capture_rate + 0.02),
        "collision_rate": collision_rate,
        "world_violation_rate": 0.0,
        "mean_capture_time_seconds": 1.4,
        "mean_min_clearance_m": 0.5,
        "mean_visible_fraction": 0.7,
        "mean_message_age_steps": 0.3,
        "mean_observation_age_steps": 0.4,
    }
    return {scenario: row.copy() for scenario in ("stress", "overall")}


def test_p1_aggregation_preserves_conditions_and_seed_unit(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    root = tmp_path / "stage3c_p1_stress"
    seeds = (101, 102, 103)
    methods = ("recurrent_no_prediction", "recurrent_gru_prediction")
    actions = ("raw", "cbf")
    conditions = ("delayed_measurements", "burst_occlusion")
    root.mkdir(parents=True)
    root.joinpath("protocol.json").write_text(
        json.dumps(
            {
                "stage": "3C_P1_representative_partial_observation_stress",
                "methods": list(methods),
                "training_seeds": list(seeds),
                "test_seed": 999,
                "episodes_per_condition": 4,
                "actions": list(actions),
                "conditions": {condition: {} for condition in conditions},
            }
        ),
        encoding="utf-8",
    )
    for seed_index, seed in enumerate(seeds):
        for method in methods:
            for condition_index, condition in enumerate(conditions):
                for action in actions:
                    value = 0.80 + 0.01 * seed_index + 0.02 * (method == "recurrent_gru_prediction")
                    value += 0.01 * condition_index
                    if action == "cbf":
                        value = 1.0
                    collision = 0.15 - 0.01 * seed_index if action == "raw" else 0.0
                    output = root / method / f"seed{seed}" / condition
                    output.mkdir(parents=True, exist_ok=True)
                    output.joinpath(f"summary_{action}.json").write_text(
                        json.dumps(summary(value, collision)), encoding="utf-8"
                    )

    output_json = tmp_path / "summary.json"
    output_report = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_stage3c_p1_stress.py",
            "--root",
            str(root),
            "--output-json",
            str(output_json),
            "--output-report",
            str(output_report),
        ],
    )
    module.main()

    aggregate = json.loads(output_json.read_text(encoding="utf-8"))
    key = "gru_minus_no_prediction_raw_safe_capture_percentage_points__delayed_measurements"
    assert aggregate["statistical_unit"] == "training_seed"
    assert aggregate["results"]["recurrent_no_prediction"]["raw"]["delayed_measurements"]["metrics"][
        "safe_capture_rate"
    ]["count"] == 3
    assert aggregate["results"]["recurrent_gru_prediction"]["cbf"]["burst_occlusion"]["episodes_total"] == 12
    assert aggregate["paired_comparisons"][key]["values"] == pytest.approx([2.0, 2.0, 2.0])
    assert aggregate["paired_comparisons"][key]["positive_seed_count"] == 3
    assert "burst_occlusion" in output_report.read_text(encoding="utf-8")
