from __future__ import annotations

import numpy as np

from scripts.audit_jepa_safe_capture_v4_calibration import _action_conditioning_stats, _binary_auc, _ece


def test_binary_auc_is_one_for_ordered_scores() -> None:
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert _binary_auc(labels, scores) == 1.0


def test_binary_auc_returns_none_for_single_class() -> None:
    assert _binary_auc(np.ones(4), np.linspace(0.1, 0.9, 4)) is None


def test_ece_is_zero_for_perfect_binary_probabilities() -> None:
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    probabilities = np.asarray([0.01, 0.02, 0.98, 0.99])
    assert _ece(labels, probabilities) < 0.03


def test_action_conditioning_requires_five_candidates_per_group() -> None:
    arrays = {
        "episode_seed": np.repeat([11, 12], 5),
        "time_index": np.zeros(10, dtype=np.int64),
        "agent_id": np.zeros(10, dtype=np.int64),
    }
    predictions = np.asarray(
        [[float(index), 0.0, 0.0] for index in range(5)]
        + [[0.0, float(index), 0.0] for index in range(5)],
        dtype=np.float32,
    )
    result = _action_conditioning_stats(arrays, predictions)
    assert result["group_count"] == 2
    assert result["nonzero_spread_fraction"] == 1.0
