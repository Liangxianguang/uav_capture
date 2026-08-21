from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wallcoverage_fixed_config_changes_only_wall_distance_sampling() -> None:
    original_path = PROJECT_ROOT / "configs" / "capture_radius_recurrent_behavior_cloning_central_v5_fixed_shapeaware_seed661605.yaml"
    pilot_path = PROJECT_ROOT / "configs" / "capture_radius_recurrent_behavior_cloning_central_v5_fixed_wallcoverage_seed661701.yaml"
    original = yaml.safe_load(original_path.read_text(encoding="utf-8"))["imitation"]
    pilot = yaml.safe_load(pilot_path.read_text(encoding="utf-8"))["imitation"]

    assert pilot["episodes"] == original["episodes"] == 640
    assert pilot["learning_rate"] == original["learning_rate"]
    assert pilot["epochs"] == original["epochs"]
    assert pilot["hidden_dim"] == original["hidden_dim"]
    assert pilot["action_scale_mode"] == original["action_scale_mode"]
    assert pilot["expert_max_rejection_rate"] == original["expert_max_rejection_rate"]
    for pilot_stage, original_stage in zip(pilot["training_showcase_stages"], original["training_showcase_stages"], strict=True):
        for key, value in original_stage.items():
            assert pilot_stage[key] == value
        if "wall" in original_stage["layouts"]:
            assert pilot_stage["layout_overrides"] == {"wall": {"initial_side_distances": [5.0, 5.5]}}
        else:
            assert "layout_overrides" not in pilot_stage


def test_preregistration_hashes_and_seed_exclusions_match_pilot_files() -> None:
    preregistration_path = PROJECT_ROOT / "V5_WALL_COVERAGE_P3A_PREREGISTRATION.json"
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    fixed_path = PROJECT_ROOT / "configs" / "capture_radius_recurrent_behavior_cloning_central_v5_fixed_wallcoverage_seed661701.yaml"
    retained_path = PROJECT_ROOT / "configs" / "capture_radius_recurrent_behavior_cloning_central_v5_wallcoverage_retained_seed661702.yaml"

    assert preregistration["inputs"]["pilot_fixed_config_sha256"] == _sha256(fixed_path)
    assert preregistration["inputs"]["pilot_retained_config_sha256"] == _sha256(retained_path)
    assert preregistration["pilot_seed_pair"] == {"fixed_training_seed": 661701, "retained_training_seed": 661702}
    assert preregistration["excluded_evaluation_seeds"]["diagnosed_seed"] == 660514
    assert preregistration["locked_seed_block_647201_opened"] is False
