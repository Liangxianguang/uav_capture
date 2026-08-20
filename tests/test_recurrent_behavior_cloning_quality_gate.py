"""Tests for accepting only task-valid recurrent BC expert demonstrations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import sample_training_episode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = PROJECT_ROOT / "scripts" / "train_capture_radius_recurrent_behavior_cloning.py"
SPEC = importlib.util.spec_from_file_location("recurrent_bc_trainer", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def test_expert_quality_gate_requires_safe_cooperative_capture() -> None:
    settings = {
        "training_required_defender_zone_entries": 2,
        "expert_require_safe_capture": True,
        "expert_require_cooperative_safe_capture": True,
    }
    metadata = {"required_defender_zone_entries": 2}

    one_entry = TRAINER.expert_episode_quality(
        {"safe_capture_success": True}, np.array([True, False, False, False]), metadata, settings
    )
    assert one_entry["accepted"] is False
    assert one_entry["cooperative_requirement_met"] is False

    accepted = TRAINER.expert_episode_quality(
        {"safe_capture_success": True}, np.array([True, True, False, False]), metadata, settings
    )
    assert accepted["accepted"] is True
    assert accepted["defender_zone_entry_count"] == 2

    unsafe = TRAINER.expert_episode_quality(
        {"safe_capture_success": False}, np.array([True, True, False, False]), metadata, settings
    )
    assert unsafe["accepted"] is False


def test_expert_quality_gate_uses_the_default_entry_requirement_for_random_episodes() -> None:
    result = TRAINER.expert_episode_quality(
        {"safe_capture_success": True},
        np.array([True, True, False, False]),
        {"required_defender_zone_entries": None},
        {
            "training_required_defender_zone_entries": 2,
            "expert_require_safe_capture": True,
            "expert_require_cooperative_safe_capture": True,
        },
    )
    assert result["accepted"] is True
    assert result["required_defender_zone_entries"] == 2


def _load_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_time_aligned_uncertainty_dev.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_curriculum_pursuit_overrides_are_stage_local() -> None:
    config = _load_config()
    base_dropout = config["task"]["pursuit"]["detection_dropout_probability"]
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    settings = {
        "training_obstacle_counts": [0],
        "training_target_speed_scales": [0.55],
        "training_showcase_stages": [
            {
                "until_progress": 0.5,
                "showcase_probability": 1.0,
                "layouts": ["cylinder"],
                "initial_side_distances": [5.0],
                "target_speed_scales": [0.55],
                "pursuit_overrides": {"detection_dropout_probability": 0.22},
            },
            {
                "until_progress": 1.0,
                "showcase_probability": 1.0,
                "layouts": ["cylinder"],
                "initial_side_distances": [5.0],
                "target_speed_scales": [0.55],
            },
        ],
    }

    _observation, first = sample_training_episode(env, settings, np.random.default_rng(1), seed=100, progress=0.1)
    assert first["pursuit_overrides"] == {"detection_dropout_probability": 0.22}
    assert env.pursuit["detection_dropout_probability"] == 0.22

    _observation, second = sample_training_episode(env, settings, np.random.default_rng(2), seed=101, progress=0.9)
    assert second["pursuit_overrides"] == {}
    assert env.pursuit["detection_dropout_probability"] == base_dropout


def test_curriculum_rejects_unknown_pursuit_override() -> None:
    env = CaptureRadiusPursuit3DEnv(_load_config(), obstacle_count=0, target_speed_scale=0.55)
    settings = {
        "training_obstacle_counts": [0],
        "training_target_speed_scales": [0.55],
        "training_showcase_stages": [
            {
                "until_progress": 1.0,
                "showcase_probability": 1.0,
                "layouts": ["cylinder"],
                "initial_side_distances": [5.0],
                "target_speed_scales": [0.55],
                "pursuit_overrides": {"not_a_pursuit_setting": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown pursuit"):
        sample_training_episode(env, settings, np.random.default_rng(3), seed=102, progress=0.1)
