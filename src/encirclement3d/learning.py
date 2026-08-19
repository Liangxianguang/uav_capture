"""Parameter-sharing PPO components for the fixed four-defender benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def defender_observations(
    observation: dict[str, Any],
    n_defenders: int,
    position_scale: float = 1.0,
    defender_speed_scale: float = 1.0,
    target_speed_scale: float = 1.0,
    include_agent_id: bool = False,
    obstacle_feature_count: int = 0,
) -> np.ndarray:
    """Build per-defender full-truth local observations for Phase 3 only.

    ``obstacle_feature_count`` adds a fixed-size, nearest-first description of
    vertical cylinders. Each obstacle contributes relative xy position,
    relative vertical midpoint, radius, and height. Zero padding makes the
    encoder compatible with scenes containing fewer cylinders, while the zero
    default preserves all existing checkpoint input dimensions.
    """
    if obstacle_feature_count < 0:
        raise ValueError("obstacle_feature_count must be non-negative.")
    positions = np.asarray(observation["defender_positions"], dtype=np.float32)
    velocities = np.asarray(observation["defender_velocities"], dtype=np.float32)
    target_position = np.asarray(observation["target_position"], dtype=np.float32)
    target_velocity = np.asarray(observation["target_velocity"], dtype=np.float32)
    slots = np.asarray(observation["slot_positions"], dtype=np.float32)
    obstacles = list(observation.get("obstacles", []))
    features: list[np.ndarray] = []
    for index in range(n_defenders):
        teammates = [positions[other] - positions[index] for other in range(n_defenders) if other != index]
        obstacle_features = np.zeros((obstacle_feature_count, 5), dtype=np.float32)
        nearest_obstacles = sorted(
            obstacles,
            key=lambda obstacle: float(
                np.linalg.norm(np.asarray(obstacle["center_xy"], dtype=np.float32) - positions[index, :2])
                - float(obstacle["radius"])
            ),
        )[:obstacle_feature_count]
        for obstacle_index, obstacle in enumerate(nearest_obstacles):
            center_xy = np.asarray(obstacle["center_xy"], dtype=np.float32)
            height = float(obstacle["height"])
            obstacle_features[obstacle_index] = np.array(
                [
                    center_xy[0] - positions[index, 0],
                    center_xy[1] - positions[index, 1],
                    0.5 * height - positions[index, 2],
                    float(obstacle["radius"]),
                    height,
                ],
                dtype=np.float32,
            )
        features.append(
            np.concatenate(
                [
                    slots[index] - positions[index],
                    target_position - positions[index],
                    velocities[index],
                    target_velocity,
                    *teammates,
                    obstacle_features.reshape(-1),
                ]
            )
        )
    values = np.stack(features).astype(np.float32)
    values[:, 0:6] /= position_scale
    values[:, 6:9] /= defender_speed_scale
    values[:, 9:12] /= target_speed_scale
    teammate_end = 12 + 3 * (n_defenders - 1)
    values[:, 12:teammate_end] /= position_scale
    if obstacle_feature_count:
        obstacle_start = teammate_end
        obstacle_values = values[:, obstacle_start:].reshape(n_defenders, obstacle_feature_count, 5)
        obstacle_values[:, :, 0:3] /= position_scale
        obstacle_values[:, :, 3] /= position_scale
        obstacle_values[:, :, 4] /= position_scale
    if include_agent_id:
        values = np.concatenate([values, np.eye(n_defenders, dtype=np.float32)], axis=1)
    return values


class SharedActorCritic(nn.Module):
    """One policy/value network shared across homogeneous defenders."""

    def __init__(self, observation_dim: int, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.2))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def distribution_and_value(self, observations: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        latent = self.body(observations)
        mean = self.actor_mean(latent)
        distribution = Normal(mean, self.log_std.exp().expand_as(mean))
        return distribution, self.critic(latent).squeeze(-1)

    def sample_actions(self, observations: torch.Tensor, action_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_and_value(observations)
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions) * action_scale
        log_probabilities = self._squashed_log_probability(distribution, raw_actions).sum(dim=-1)
        return actions, log_probabilities, value

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        scaled_actions: torch.Tensor,
        action_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_and_value(observations)
        normalized_actions = torch.clamp(scaled_actions / action_scale, -0.999999, 0.999999)
        raw_actions = torch.atanh(normalized_actions)
        log_probabilities = self._squashed_log_probability(distribution, raw_actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probabilities, entropy, value

    @staticmethod
    def _squashed_log_probability(distribution: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        correction = torch.log(torch.clamp(1.0 - torch.tanh(raw_actions).pow(2), min=1e-6))
        return distribution.log_prob(raw_actions) - correction


class CentralizedSharedActorCritic(nn.Module):
    """MAPPO network with a decentralized shared actor and centralized critic.

    The actor consumes one defender's partial observation. The critic consumes
    the simulator-only centralized state during training, never at execution.
    """

    def __init__(
        self,
        local_observation_dim: int,
        centralized_state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(local_observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic_body = nn.Sequential(
            nn.Linear(centralized_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.2))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def distribution(self, local_observations: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor_body(local_observations))
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def actor_parameters(self):
        """Return only decentralized actor parameters for imitation pretraining."""
        return list(self.actor_body.parameters()) + list(self.actor_mean.parameters()) + [self.log_std]

    def value(self, centralized_states: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_body(centralized_states)).squeeze(-1)

    def sample_actions(
        self,
        local_observations: torch.Tensor,
        action_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(local_observations)
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions) * action_scale
        log_probabilities = self._squashed_log_probability(distribution, raw_actions).sum(dim=-1)
        return actions, log_probabilities

    def evaluate_actions(
        self,
        local_observations: torch.Tensor,
        scaled_actions: torch.Tensor,
        action_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(local_observations)
        normalized_actions = torch.clamp(scaled_actions / action_scale, -0.999999, 0.999999)
        raw_actions = torch.atanh(normalized_actions)
        log_probabilities = self._squashed_log_probability(distribution, raw_actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probabilities, entropy

    @staticmethod
    def _squashed_log_probability(distribution: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        correction = torch.log(torch.clamp(1.0 - torch.tanh(raw_actions).pow(2), min=1e-6))
        return distribution.log_prob(raw_actions) - correction


class CapturePolicy(nn.Module):
    """Multi-task behavior-cloning policy for motion and binary cage closure.

    This model is intentionally separate from ``SharedActorCritic`` so all
    historical three-dimensional action checkpoints remain loadable.  The
    closure head is a global binary decision copied to each defender's local
    observation during training; deployment accepts closure only when the
    geometric environment guard also agrees.
    """

    def __init__(self, observation_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_mean = nn.Linear(hidden_dim, 3)
        self.close_logit = nn.Linear(hidden_dim, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.close_logit.weight, gain=0.01)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.body(observations)
        return self.action_mean(latent), self.close_logit(latent).squeeze(-1)


class ClosurePolicy(nn.Module):
    """Predict one global cage-closure decision from frozen-policy features.

    The input may contain one local feature vector per defender with shape
    ``[batch, defenders, features]``.  Logits are pooled across defenders so
    the learned command cannot close from only one agent's private opinion.
    A two-dimensional input remains supported for small unit tests.
    """

    def __init__(self, observation_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.close_logit = nn.Linear(hidden_dim, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.close_logit.weight, gain=0.01)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim not in (2, 3):
            raise ValueError("ClosurePolicy expects [batch, features] or [batch, defenders, features].")
        latent = self.body(observations)
        logits = self.close_logit(latent).squeeze(-1)
        return logits.mean(dim=1) if observations.ndim == 3 else logits


@dataclass
class PPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
