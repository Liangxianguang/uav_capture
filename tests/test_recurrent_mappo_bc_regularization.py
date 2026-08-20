"""Unit tests for the optional BC retention term used by recurrent MAPPO."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = PROJECT_ROOT / "scripts" / "train_capture_radius_recurrent_mappo.py"
SPEC = importlib.util.spec_from_file_location("recurrent_mappo_trainer", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def test_bc_regularizer_dataset_accepts_a_compatible_recurrent_archive(tmp_path: Path) -> None:
    dataset = tmp_path / "expert_sequence_dataset.npz"
    local = np.zeros((2, 32, 4, 63), dtype=np.float32)
    actions = np.full((2, 32, 4, 3), 1.5, dtype=np.float32)
    resets = np.zeros((2, 32), dtype=np.float32)
    resets[:, 0] = 1.0
    np.savez_compressed(dataset, local_observations=local, actions=actions, reset_masks=resets)

    loaded_local, loaded_actions, loaded_resets, metadata = TRAINER.load_behavior_cloning_regularizer_dataset(
        dataset,
        sequence_length=32,
        local_observation_dim=63,
        defender_count=4,
        action_scale=5.0,
    )

    assert loaded_local.shape == local.shape
    assert loaded_actions.shape == actions.shape
    assert loaded_resets.shape == resets.shape
    assert metadata["sequences"] == 2


def test_bc_regularizer_dataset_rejects_actions_outside_policy_scale(tmp_path: Path) -> None:
    dataset = tmp_path / "expert_sequence_dataset.npz"
    np.savez_compressed(
        dataset,
        local_observations=np.zeros((1, 32, 4, 63), dtype=np.float32),
        actions=np.full((1, 32, 4, 3), 5.1, dtype=np.float32),
        reset_masks=np.zeros((1, 32), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="exceed"):
        TRAINER.load_behavior_cloning_regularizer_dataset(
            dataset,
            sequence_length=32,
            local_observation_dim=63,
            defender_count=4,
            action_scale=5.0,
        )
