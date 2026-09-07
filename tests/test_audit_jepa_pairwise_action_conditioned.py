from __future__ import annotations

import numpy as np

from scripts.audit_jepa_pairwise_action_conditioned import _features, _metrics, _pairwise_ttc


def test_pairwise_ttc_marks_approaching_projected_pair() -> None:
    positions = np.array([[[1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 5.0, 0.0]]])
    velocities = np.array([[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    values = _pairwise_ttc(positions, velocities)
    assert values.shape == (1, 3)
    assert values[0, 0] < 10.0


def test_action_projection_changes_focal_relative_velocity() -> None:
    inputs = np.zeros((1, 8, 63), dtype=np.float32)
    inputs[:, -1, 0:3] = [0.0, 0.0, 0.0]
    inputs[:, -1, 15:18] = [0.1, 0.0, 0.0]
    inputs[:, -1, 24:27] = [-0.2, 0.0, 0.0]
    actions = np.zeros((1, 3, 3), dtype=np.float32)
    actions[:, 0, 0] = 1.0
    features = _features(inputs, actions)
    assert features["candidate_max_closing_speed_mps"][0] > 0.0
    assert features["candidate_min_ttc_s"][0] < 10.0


def test_metrics_returns_finite_auc_and_best_threshold() -> None:
    labels = np.array([0.2, 0.4, 2.0, 3.0])
    values = np.array([0.1, 0.2, 2.0, 4.0])
    result = _metrics(labels, values, "low")
    assert result["auc_le_1s"] is not None
    assert result["f1"] is not None
