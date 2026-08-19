"""Supervised local-history target trajectory predictors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class HistoryTargetPredictor(nn.Module):
    """GRU predictor that maps local observation history to target means/uncertainty."""

    def __init__(
        self,
        input_dim: int,
        horizon_count: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or horizon_count <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("Predictor dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_dim = int(input_dim)
        self.horizon_count = int(horizon_count)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.horizon_count * 6),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, history, {self.input_dim}] inputs, got {tuple(inputs.shape)}."
            )
        encoded, _hidden = self.encoder(inputs)
        output = self.head(encoded[:, -1])
        output = output.view(inputs.shape[0], self.horizon_count, 6)
        mean = output[..., :3]
        log_variance = torch.clamp(output[..., 3:], min=-8.0, max=5.0)
        return mean, log_variance


def gaussian_nll(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != log_variance.shape or mean.shape != target.shape:
        raise ValueError("mean, log_variance, and target must have identical shapes.")
    variance = torch.exp(log_variance)
    return 0.5 * (log_variance + (target - mean).square() / variance).mean()


def deterministic_mse(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mean.shape != target.shape:
        raise ValueError("mean and target must have identical shapes.")
    return (mean - target).square().mean()


class LearnedPredictionObserver:
    """Replace an environment's constant-velocity feature block with GRU output.

    The adapter keeps the actor input width unchanged. It maintains a short
    history separately for each episode, runs the frozen predictor without
    gradients, and replaces only the four prediction columns (relative xyz
    plus one uncertainty scalar). All other local observation fields and the
    centralized critic state remain unchanged.
    """

    def __init__(
        self,
        env: Any,
        predictor: HistoryTargetPredictor,
        device: torch.device,
        history_length: int,
        horizon_index: int,
    ) -> None:
        if history_length <= 0:
            raise ValueError("history_length must be positive.")
        if horizon_index < 0 or horizon_index >= predictor.horizon_count:
            raise ValueError("horizon_index is outside the predictor output range.")
        self.env = env
        self.predictor = predictor.to(device).eval()
        self.device = device
        self.history_length = int(history_length)
        self.horizon_index = int(horizon_index)
        self._history: list[np.ndarray] = []
        self.last_prediction_mean: np.ndarray | None = None
        self.last_prediction_std: np.ndarray | None = None

    @classmethod
    def from_checkpoint(
        cls,
        env: Any,
        checkpoint_path: Path,
        device: torch.device,
        history_length: int,
        horizon_index: int,
    ) -> "LearnedPredictionObserver":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_config = checkpoint.get("model")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
            raise ValueError("Prediction checkpoint must contain model and model_state_dict.")
        predictor = HistoryTargetPredictor(**model_config)
        predictor.load_state_dict(state_dict, strict=True)
        return cls(env, predictor, device, history_length, horizon_index)

    def reset(self, observation: dict[str, Any]) -> np.ndarray:
        self._history = []
        self.last_prediction_mean = None
        self.last_prediction_std = None
        return self.observe(observation)

    def observe(self, observation: dict[str, Any]) -> np.ndarray:
        base = np.asarray(self.env.policy_observations(observation), dtype=np.float32)
        feature_slice = self.env.prediction_feature_slice()
        if base.shape[-1] < feature_slice.stop:
            raise ValueError("Environment observation is shorter than its prediction feature block.")
        if base.shape[-1] != self.predictor.input_dim:
            raise ValueError(
                "Prediction checkpoint input dimension "
                f"{self.predictor.input_dim} does not match environment base dimension {base.shape[-1]}."
            )
        self._history.append(base.copy())
        if len(self._history) > self.history_length:
            self._history.pop(0)
        padded = [self._history[0]] * (self.history_length - len(self._history)) + self._history
        # [history, defenders, features] -> [defenders, history, features].
        window = np.stack(padded, axis=0)
        window = np.transpose(window, (1, 0, 2)).copy()
        with torch.no_grad():
            mean, log_variance = self.predictor(torch.as_tensor(window, device=self.device))
        selected_mean = mean[:, self.horizon_index].detach().cpu().numpy().astype(np.float32)
        selected_std = torch.exp(0.5 * log_variance[:, self.horizon_index]).detach().cpu().numpy().astype(np.float32)
        uncertainty = np.mean(selected_std, axis=1, keepdims=True)
        augmented = base.copy()
        augmented[:, feature_slice] = np.concatenate([selected_mean, uncertainty], axis=1)
        self.last_prediction_mean = selected_mean
        self.last_prediction_std = selected_std
        if not np.isfinite(augmented).all():
            raise RuntimeError("Learned prediction adapter emitted a non-finite observation.")
        return augmented
