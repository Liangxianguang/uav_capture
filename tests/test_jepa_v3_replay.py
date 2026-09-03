from __future__ import annotations

import numpy as np
import torch

from scripts.build_jepa_v3_hard_replay_weights import build_weights, hard_example_masks
from scripts.evaluate_jepa_v3_replay_subsets import group_ranking_metrics, paired_block_bootstrap_interval
from scripts.train_interaction_aware_jepa_multitask import MixedReplaySampler


def test_hard_replay_weights_mark_clearance_and_cbf_examples_without_dropping_samples() -> None:
    arrays = {
        "labels_obstacle_clearance": np.array([[0.10, 0.20], [0.90, 0.90], [0.90, 0.90], [0.90, 0.90]], dtype=np.float32),
        "labels_inter_agent_clearance": np.full((4, 2), 0.90, dtype=np.float32),
        "labels_cbf_correction": np.array([[0.01, 0.01], [0.30, 0.01], [0.01, 0.01], [0.01, 0.01]], dtype=np.float32),
        "labels_collision": np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        "labels_boundary": np.zeros((4, 2), dtype=np.float32),
    }
    policy = {
        "low_clearance_threshold_m": 0.30,
        "high_cbf_correction_threshold_mps": 0.25,
        "include_collision_or_boundary_labels": True,
        "hard_sample_weight": 3.0,
    }
    weights, diagnostics = build_weights(arrays, extent=1.0, policy=policy)
    np.testing.assert_allclose(weights, [3.0, 3.0, 3.0, 1.0])
    assert diagnostics["samples"] == 4
    assert diagnostics["hard_samples"] == 3
    masks = hard_example_masks(arrays, extent=1.0, policy=policy)
    np.testing.assert_array_equal(masks["hard"], [True, True, True, False])
    np.testing.assert_array_equal(masks["low_clearance"], [True, False, False, False])
    np.testing.assert_array_equal(masks["high_cbf_correction"], [False, True, False, False])


def test_mixed_replay_sampler_is_deterministic_and_preserves_epoch_size() -> None:
    weights = torch.tensor([1.0, 1.0, 1.0, 8.0, 8.0, 8.0])
    first = list(MixedReplaySampler(weights, uniform_fraction=0.50, seed=17))
    second = list(MixedReplaySampler(weights, uniform_fraction=0.50, seed=17))
    assert first == second
    assert len(first) == len(weights)
    assert set(first).issubset(set(range(len(weights))))


def test_group_ranking_keeps_all_candidates_when_marking_hard_groups() -> None:
    arrays = {
        "episode_seed": np.array([1, 1, 1, 2, 2, 2]),
        "time_index": np.array([4, 4, 4, 5, 5, 5]),
        "agent_id": np.array([0, 0, 0, 0, 0, 0]),
        "candidate_index": np.array([0, 1, 2, 0, 1, 2]),
        "action_history": np.zeros((6, 8, 3), dtype=np.float32),
        "labels_relative": np.array([[[3.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=np.float32),
    }
    predictions = {
        "target_relative": np.array([[[3.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=np.float32)
    }
    metadata = {"candidate_count": 3, "horizon_seconds": [1.0], "action_scale": 1.0}
    result = group_ranking_metrics(arrays, predictions, np.array([False, True, False, False, False, False]), metadata, extent=1.0)
    assert result[0]["hard_group"]["groups"] == 1
    assert result[0]["hard_group"]["candidate_ranking_win_rate"] == 1.0
    assert result[0]["non_hard_group"]["groups"] == 1
    assert result[0]["non_hard_group"]["candidate_ranking_win_rate"] == 0.0


def test_paired_block_bootstrap_reports_a_reproducible_negative_error_reduction() -> None:
    inverse = np.array([0, 0, 1, 1], dtype=np.int64)
    off = np.array([1.0, 1.0, 1.0, 1.0])
    on = np.array([1.5, 1.5, 1.5, 1.5])
    first = paired_block_bootstrap_interval(off, on, inverse, group_count=2, replicates=100, rng=np.random.default_rng(7))
    second = paired_block_bootstrap_interval(off, on, inverse, group_count=2, replicates=100, rng=np.random.default_rng(7))
    assert first == second
    assert first["point_estimate_replay_on_error_reduction"] == -0.5
    assert first["ci95_low"] == -0.5
    assert first["ci95_high"] == -0.5
