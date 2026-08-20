"""Tests for accepting only task-valid recurrent BC expert demonstrations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from encirclement3d.learning import RecurrentCentralizedSharedActorCritic
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
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").read_text(
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


def test_reused_expert_archives_can_be_sequence_balanced(tmp_path: Path) -> None:
    config = _load_config()
    settings = {"training_obstacle_counts": [3], "training_target_speed_scales": [0.45], "seed": 55}
    datasets: list[Path] = []
    for index, sequence_count in enumerate((2, 3)):
        directory = tmp_path / f"source_{index}"
        directory.mkdir()
        dataset = directory / "expert_sequence_dataset.npz"
        np.savez_compressed(
            dataset,
            local_observations=np.full((sequence_count, 32, 4, 63), index, dtype=np.float32),
            actions=np.zeros((sequence_count, 32, 4, 3), dtype=np.float32),
            reset_masks=np.zeros((sequence_count, 32), dtype=np.float32),
        )
        dataset.with_name("expert_dataset_manifest.json").write_text("{}", encoding="utf-8")
        datasets.append(dataset)

    local, actions, resets, manifest, state_dim = TRAINER.load_reused_expert_datasets(
        datasets,
        config,
        settings,
        source_balance="equal_sequences",
        seed=55,
    )

    assert local.shape == (6, 32, 4, 63)
    assert actions.shape == (6, 32, 4, 3)
    assert resets.shape == (6, 32)
    assert state_dim == 46
    assert [source["selected_sequences"] for source in manifest["reused_expert_datasets"]] == [3, 3]
    assert manifest["source_balance"] == "equal_sequences"


def test_recurrent_bc_warm_start_requires_a_compatible_checkpoint(tmp_path: Path) -> None:
    source = RecurrentCentralizedSharedActorCritic(63, 46, hidden_dim=16)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "local_observation_dim": 63,
            "centralized_state_dim": 46,
            "action_scale": 5.0,
            "actor_recurrent": True,
            "algorithm": "behavior_cloning_recurrent_local_rule_expert",
            "seed": 42,
        },
        checkpoint,
    )
    target = RecurrentCentralizedSharedActorCritic(63, 46, hidden_dim=16)

    metadata = TRAINER.initialize_recurrent_actor(
        target,
        checkpoint,
        local_observation_dim=63,
        centralized_state_dim=46,
        action_scale=5.0,
        device=torch.device("cpu"),
    )

    assert metadata is not None
    assert metadata["source_seed"] == 42
    assert torch.equal(target.actor_base_body[0].weight, source.actor_base_body[0].weight)
