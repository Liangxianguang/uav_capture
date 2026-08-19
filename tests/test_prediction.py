from __future__ import annotations

import torch
import numpy as np
import yaml
from pathlib import Path

from encirclement3d.prediction import (
    HistoryTargetPredictor,
    LearnedPredictionObserver,
    deterministic_mse,
    gaussian_nll,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_history_target_predictor_shapes_and_finite_losses() -> None:
    model = HistoryTargetPredictor(input_dim=52, horizon_count=4, hidden_dim=16)
    inputs = torch.randn(5, 8, 52)
    target = torch.randn(5, 4, 3)
    mean, log_variance = model(inputs)
    assert mean.shape == (5, 4, 3)
    assert log_variance.shape == (5, 4, 3)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_variance).all()
    assert torch.isfinite(gaussian_nll(mean, log_variance, target))
    assert torch.isfinite(deterministic_mse(mean, target))


def test_history_target_predictor_rejects_wrong_feature_dimension() -> None:
    model = HistoryTargetPredictor(input_dim=52, horizon_count=4, hidden_dim=16)
    try:
        model(torch.randn(2, 8, 48))
    except ValueError as error:
        assert "Expected" in str(error)
    else:
        raise AssertionError("Expected wrong feature dimension to raise ValueError.")


def test_learned_prediction_observer_preserves_actor_width_and_replaces_block() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_prediction_dev.yaml").read_text(encoding="utf-8")
    )
    config["task"]["pursuit"]["include_uncertainty_features"] = True
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    predictor = HistoryTargetPredictor(input_dim=52, horizon_count=4, hidden_dim=16)
    adapter = LearnedPredictionObserver(env, predictor, torch.device("cpu"), history_length=8, horizon_index=2)
    observation = env.reset(seed=520201)
    augmented = adapter.reset(observation)
    assert augmented.shape == (4, 52)
    feature_slice = env.prediction_feature_slice()
    assert feature_slice == slice(15, 19)
    assert np.isfinite(augmented).all()
    next_observation, *_ = env.step(np.zeros((4, 3)))
    repeated = adapter.observe(next_observation)
    assert repeated.shape == (4, 52)
    assert adapter.last_prediction_mean is not None
    assert adapter.last_prediction_std is not None
