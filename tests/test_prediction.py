from __future__ import annotations

import torch

from encirclement3d.prediction import HistoryTargetPredictor, deterministic_mse, gaussian_nll


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
