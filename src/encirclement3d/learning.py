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


class RecurrentCentralizedSharedActorCritic(nn.Module):
    """MAPPO with a decentralized parameter-sharing GRU actor.

    The critic is intentionally identical in information scope to
    :class:`CentralizedSharedActorCritic`: it receives centralized simulator
    state only during training. The actor receives one defender's local
    observation and maintains a separate hidden state for each defender at
    execution. Sequence evaluation accepts reset masks so PPO updates do not
    carry memory across episode boundaries.
    """

    def __init__(
        self,
        local_observation_dim: int,
        centralized_state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if local_observation_dim <= 0 or centralized_state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Recurrent actor-critic dimensions must be positive.")
        self.local_observation_dim = int(local_observation_dim)
        self.centralized_state_dim = int(centralized_state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor_base_body = nn.Sequential(
            nn.Linear(self.local_observation_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.actor_base_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.actor_gru_encoder = nn.Sequential(nn.Linear(self.local_observation_dim, self.hidden_dim), nn.Tanh())
        self.actor_gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.actor_residual_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.critic_body = nn.Sequential(
            nn.Linear(self.centralized_state_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -1.2))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        for name, parameter in self.actor_gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.actor_base_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_residual_mean.weight)
        nn.init.zeros_(self.actor_residual_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def initial_actor_hidden(
        self,
        defender_count: int,
        *,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return zero hidden state shaped ``[defenders, hidden]`` or batched."""
        if defender_count <= 0:
            raise ValueError("defender_count must be positive.")
        resolved_device = device if device is not None else self.log_std.device
        shape = (
            (int(defender_count), self.hidden_dim)
            if batch_size is None
            else (int(batch_size), int(defender_count), self.hidden_dim)
        )
        return torch.zeros(shape, device=resolved_device, dtype=self.log_std.dtype)

    def distribution_step(
        self,
        local_observations: torch.Tensor,
        hidden_state: torch.Tensor | None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[Normal, torch.Tensor]:
        """Advance each defender's local GRU state by one observation frame."""
        if local_observations.ndim != 2 or local_observations.shape[-1] != self.local_observation_dim:
            raise ValueError(
                "Expected [defenders, "
                f"{self.local_observation_dim}] local observations, got {tuple(local_observations.shape)}."
            )
        defender_count = int(local_observations.shape[0])
        if hidden_state is None:
            hidden_state = self.initial_actor_hidden(defender_count, device=local_observations.device)
        if hidden_state.shape != (defender_count, self.hidden_dim):
            raise ValueError("Single-step hidden state has incompatible shape.")
        if reset_mask is not None:
            mask = torch.as_tensor(reset_mask, dtype=hidden_state.dtype, device=hidden_state.device)
            if mask.ndim == 0:
                hidden_state = hidden_state * (1.0 - mask)
            elif mask.shape == (defender_count,):
                hidden_state = hidden_state * (1.0 - mask[:, None])
            else:
                raise ValueError("Single-step reset mask must be scalar or one value per defender.")
        base_mean = self.actor_base_mean(self.actor_base_body(local_observations))
        encoded = self.actor_gru_encoder(local_observations)
        next_hidden = self.actor_gru(encoded, hidden_state)
        mean = base_mean + self.actor_residual_mean(next_hidden)
        return Normal(mean, self.log_std.exp().expand_as(mean)), next_hidden

    def sample_actions_step(
        self,
        local_observations: torch.Tensor,
        hidden_state: torch.Tensor | None,
        action_scale: float,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, next_hidden = self.distribution_step(local_observations, hidden_state, reset_mask)
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions) * action_scale
        log_probabilities = self._squashed_log_probability(distribution, raw_actions).sum(dim=-1)
        return actions, log_probabilities, next_hidden

    def evaluate_actions_sequence(
        self,
        local_observations: torch.Tensor,
        initial_hidden: torch.Tensor,
        reset_masks: torch.Tensor,
        scaled_actions: torch.Tensor,
        action_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate fixed action sequences with truncated back-propagation.

        Args:
            local_observations: ``[batch, time, defenders, features]``.
            initial_hidden: hidden state immediately before the first frame.
            reset_masks: one mask per batch/time frame; a one zeros hidden
                state immediately before that frame.
            scaled_actions: ``[batch, time, defenders, action_dim]``.
        """
        if local_observations.ndim != 4 or local_observations.shape[-1] != self.local_observation_dim:
            raise ValueError("Sequence local observations have incompatible shape.")
        batch_size, sequence_length, defender_count, _ = local_observations.shape
        expected_hidden = (batch_size, defender_count, self.hidden_dim)
        if initial_hidden.shape != expected_hidden:
            raise ValueError(f"Expected initial hidden shape {expected_hidden}, got {tuple(initial_hidden.shape)}.")
        if reset_masks.shape != (batch_size, sequence_length):
            raise ValueError("Sequence reset masks must have shape [batch, time].")
        expected_actions = (batch_size, sequence_length, defender_count, self.action_dim)
        if scaled_actions.shape != expected_actions:
            raise ValueError(f"Expected action shape {expected_actions}, got {tuple(scaled_actions.shape)}.")
        hidden = initial_hidden
        log_probability_frames: list[torch.Tensor] = []
        entropy_frames: list[torch.Tensor] = []
        mean_frames: list[torch.Tensor] = []
        for index in range(sequence_length):
            hidden = hidden * (1.0 - reset_masks[:, index, None, None].to(hidden.dtype))
            flattened_local = local_observations[:, index].reshape(-1, self.local_observation_dim)
            base_mean = self.actor_base_mean(self.actor_base_body(flattened_local)).reshape(
                batch_size, defender_count, self.action_dim
            )
            encoded = self.actor_gru_encoder(flattened_local)
            hidden = self.actor_gru(encoded, hidden.reshape(-1, self.hidden_dim)).reshape(expected_hidden)
            mean = base_mean + self.actor_residual_mean(hidden)
            distribution = Normal(mean, self.log_std.exp().view(1, 1, -1).expand_as(mean))
            normalized_actions = torch.clamp(scaled_actions[:, index] / action_scale, -0.999999, 0.999999)
            raw_actions = torch.atanh(normalized_actions)
            log_probability_frames.append(self._squashed_log_probability(distribution, raw_actions).sum(dim=-1))
            entropy_frames.append(distribution.entropy().sum(dim=-1))
            mean_frames.append(mean)
        return (
            torch.stack(log_probability_frames, dim=1),
            torch.stack(entropy_frames, dim=1),
            torch.stack(mean_frames, dim=1),
        )

    def actor_parameters(self):
        return (
            list(self.actor_base_body.parameters())
            + list(self.actor_base_mean.parameters())
            + list(self.actor_gru_encoder.parameters())
            + list(self.actor_gru.parameters())
            + list(self.actor_residual_mean.parameters())
            + [self.log_std]
        )

    def value(self, centralized_states: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_body(centralized_states)).squeeze(-1)

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
