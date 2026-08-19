"""Supervised local-history target trajectory predictors."""

from __future__ import annotations

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
