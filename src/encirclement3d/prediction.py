"""Supervised local-history target trajectory predictors."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass

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


class ActionConditionedJEPAPredictor(nn.Module):
    """Structured action-conditioned latent predictor for the JEPA pilot.

    The model predicts a stop-gradient target representation of future target
    displacement instead of reconstructing observations.  Actions are the
    already executed action leading to each history frame, so deployment never
    needs simulator-only future actions.  A small position decoder keeps the
    model usable by the existing prediction-feature adapter.
    """

    def __init__(
        self,
        input_dim: int,
        horizon_count: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if min(input_dim, horizon_count, action_dim, hidden_dim, latent_dim, num_layers) <= 0:
            raise ValueError("Predictor dimensions must be positive.")
        self.input_dim = int(input_dim)
        self.horizon_count = int(horizon_count)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.observation_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.context_encoder = nn.GRU(
            input_size=2 * self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.latent_predictor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.horizon_count * self.latent_dim),
        )
        self.position_decoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        self.uncertainty_decoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        # A fixed target projector prevents representation collapse in this
        # small offline pilot while keeping the target branch stop-gradient.
        self.target_projector = nn.Linear(3, self.latent_dim, bias=False)
        nn.init.orthogonal_(self.target_projector.weight)
        for parameter in self.target_projector.parameters():
            parameter.requires_grad_(False)

    def _encode_observations(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode observations before action/history fusion.

        This hook keeps the original checkpoint behavior unchanged while
        allowing an interaction-aware variant to reweight semantic groups.
        """
        return self.observation_encoder(inputs)

    def forward(
        self,
        inputs: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, history, {self.input_dim}] inputs, got {tuple(inputs.shape)}."
            )
        if actions is None:
            actions = torch.zeros(
                inputs.shape[0], inputs.shape[1], self.action_dim, dtype=inputs.dtype, device=inputs.device
            )
        if actions.shape != (inputs.shape[0], inputs.shape[1], self.action_dim):
            raise ValueError(
                "Expected action history shape "
                f"{(inputs.shape[0], inputs.shape[1], self.action_dim)}, got {tuple(actions.shape)}."
            )
        observation_features = self._encode_observations(inputs)
        action_features = self.action_encoder(actions)
        context, _hidden = self.context_encoder(torch.cat([observation_features, action_features], dim=-1))
        predicted_latent = self.latent_predictor(context[:, -1]).view(
            inputs.shape[0], self.horizon_count, self.latent_dim
        )
        mean = self.position_decoder(predicted_latent)
        log_variance = torch.clamp(self.uncertainty_decoder(predicted_latent), min=-8.0, max=5.0)
        return mean, log_variance, predicted_latent

    def target_latent(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 3 or target.shape[-1] != 3 or target.shape[1] != self.horizon_count:
            raise ValueError(
                f"Expected [batch, {self.horizon_count}, 3] target labels, got {tuple(target.shape)}."
            )
        with torch.no_grad():
            return torch.tanh(self.target_projector(target))


class InteractionAwareActionConditionedJEPAPredictor(ActionConditionedJEPAPredictor):
    """Action-conditioned JEPA with an IMPACT-inspired interaction gate.

    The input remains the policy-safe observation contract.  For the current
    shape-aware V5 observation, the default groups are target/belief fields,
    teammate relative state, legacy obstacle geometry, and shape metadata.
    Group embeddings are mixed with a learned, context-dependent gate before
    the recurrent action/history encoder.  This is deliberately a small
    structured-state model rather than a visual world model, so it can be
    trained and audited on the RTX 5050 without changing the frozen actor.
    """

    def __init__(
        self,
        input_dim: int,
        horizon_count: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        interaction_group_slices: tuple[tuple[int, int], ...] | list[list[int]] | None = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            horizon_count=horizon_count,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        if interaction_group_slices is None:
            # 63-D shape-aware V5 observation: 15 target/belief, 18 teammate,
            # 15 legacy obstacle, and 15 shape/type metadata dimensions.
            interaction_group_slices = ((0, 15), (15, 33), (33, 48), (48, 63))
        normalized = tuple((int(start), int(stop)) for start, stop in interaction_group_slices)
        if not normalized or any(start < 0 or stop <= start or stop > input_dim for start, stop in normalized):
            raise ValueError("interaction_group_slices must contain valid non-empty input ranges.")
        self.interaction_group_slices = normalized
        self.interaction_group_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(stop - start, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.SiLU(),
                )
                for start, stop in normalized
            ]
        )
        self.interaction_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, len(normalized)),
        )

    def _encode_observations(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.observation_encoder(inputs)
        group_features = torch.stack(
            [encoder(inputs[..., start:stop]) for encoder, (start, stop) in zip(self.interaction_group_encoders, self.interaction_group_slices)],
            dim=-2,
        )
        gate = torch.softmax(self.interaction_gate(base), dim=-1).unsqueeze(-1)
        return base + torch.sum(gate * group_features, dim=-2)


class InteractionAwareActionConditionedMultitaskJEPAPredictor(InteractionAwareActionConditionedJEPAPredictor):
    """Interaction-aware JEPA with auditable counterfactual auxiliary heads.

    The inherited :meth:`forward` deliberately retains the target-prediction
    tuple used by the frozen V5 reranker.  Auxiliary predictions are exposed
    through :meth:`forward_multitask`, so new clearance/visibility/risk
    training cannot silently alter the actor observation contract.
    """

    def __init__(
        self,
        input_dim: int,
        horizon_count: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        interaction_group_slices: tuple[tuple[int, int], ...] | list[list[int]] | None = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            horizon_count=horizon_count,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            interaction_group_slices=interaction_group_slices,
        )
        self.clearance_decoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2),
        )
        self.visibility_decoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.cbf_correction_decoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.cbf_intervention_decoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def auxiliary_predictions(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode per-horizon auxiliary quantities from predicted latents."""
        if latent.ndim != 3 or latent.shape[1:] != (self.horizon_count, self.latent_dim):
            raise ValueError(
                "Expected predicted latent shaped "
                f"[batch, {self.horizon_count}, {self.latent_dim}], got {tuple(latent.shape)}."
            )
        clearance = self.clearance_decoder(latent)
        return {
            "obstacle_clearance": clearance[..., 0],
            "inter_agent_clearance": clearance[..., 1],
            "target_visibility_logit": self.visibility_decoder(latent).squeeze(-1),
            "cbf_correction": torch.nn.functional.softplus(self.cbf_correction_decoder(latent).squeeze(-1)),
            "cbf_intervention_logit": self.cbf_intervention_decoder(latent).squeeze(-1),
        }

    def forward_multitask(
        self,
        inputs: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        mean, log_variance, latent = self.forward(inputs, actions)
        return mean, log_variance, latent, self.auxiliary_predictions(latent)


def build_action_conditioned_predictor(
    model_type: str,
    model_config: dict[str, Any],
) -> ActionConditionedJEPAPredictor:
    """Instantiate a serialized action-conditioned predictor variant."""
    normalized = str(model_type)
    if normalized == "action_conditioned_jepa":
        predictor_class = ActionConditionedJEPAPredictor
    elif normalized == "interaction_aware_action_conditioned_jepa":
        predictor_class = InteractionAwareActionConditionedJEPAPredictor
    elif normalized == "interaction_aware_action_conditioned_jepa_multitask":
        predictor_class = InteractionAwareActionConditionedMultitaskJEPAPredictor
    else:
        raise ValueError(f"Unsupported action-conditioned predictor model_type: {normalized!r}.")
    return predictor_class(**model_config)


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


class ActionConditionedLearnedPredictionObserver(LearnedPredictionObserver):
    """Prediction-feature adapter for :class:`ActionConditionedJEPAPredictor`.

    The action history is maintained separately from environment state.  At
    each frame it contains the action that produced the previous transition;
    the current requested action is appended only after the next observation
    is received.  This preserves the information boundary required for
    deployment and makes the adapter compatible with the existing actor-width
    replacement used by :class:`LearnedPredictionObserver`.
    """

    def __init__(
        self,
        env: Any,
        predictor: ActionConditionedJEPAPredictor,
        device: torch.device,
        history_length: int,
        horizon_index: int,
        action_scale: float = 1.0,
    ) -> None:
        super().__init__(env, predictor, device, history_length, horizon_index)
        if action_scale <= 0.0:
            raise ValueError("action_scale must be positive.")
        self.action_scale = float(action_scale)
        self._action_history: list[np.ndarray] = []
        self._last_action = np.zeros((env.n_defenders, predictor.action_dim), dtype=np.float32)

    def reset(self, observation: dict[str, Any]) -> np.ndarray:
        self._action_history = []
        self._last_action = np.zeros((self.env.n_defenders, self.predictor.action_dim), dtype=np.float32)
        return self.observe(observation)

    def observe(self, observation: dict[str, Any]) -> np.ndarray:
        base = np.asarray(self.env.policy_observations(observation), dtype=np.float32)
        feature_slice = self.env.prediction_feature_slice()
        if base.shape[-1] != self.predictor.input_dim:
            raise ValueError(
                "Prediction checkpoint input dimension "
                f"{self.predictor.input_dim} does not match environment base dimension {base.shape[-1]}."
            )
        self._history.append(base.copy())
        self._action_history.append((self._last_action / self.action_scale).copy())
        if len(self._history) > self.history_length:
            self._history.pop(0)
            self._action_history.pop(0)
        padded_observations = [self._history[0]] * (self.history_length - len(self._history)) + self._history
        padded_actions = [self._action_history[0]] * (self.history_length - len(self._action_history)) + self._action_history
        observation_window = np.transpose(np.stack(padded_observations, axis=0), (1, 0, 2)).copy()
        action_window = np.transpose(np.stack(padded_actions, axis=0), (1, 0, 2)).copy()
        with torch.no_grad():
            mean, log_variance, _latent = self.predictor(
                torch.as_tensor(observation_window, device=self.device),
                torch.as_tensor(action_window, device=self.device),
            )
        selected_mean = mean[:, self.horizon_index].detach().cpu().numpy().astype(np.float32)
        selected_std = torch.exp(0.5 * log_variance[:, self.horizon_index]).detach().cpu().numpy().astype(np.float32)
        augmented = base.copy()
        augmented[:, feature_slice] = np.concatenate(
            [selected_mean, np.mean(selected_std, axis=1, keepdims=True)], axis=1
        )
        self.last_prediction_mean = selected_mean
        self.last_prediction_std = selected_std
        if not np.isfinite(augmented).all():
            raise RuntimeError("Action-conditioned prediction adapter emitted a non-finite observation.")
        return augmented

    def observe_after_action(self, observation: dict[str, Any], action: np.ndarray) -> np.ndarray:
        """Record the action that produced ``observation`` before observing it."""
        action_array = np.asarray(action, dtype=np.float32)
        expected = (self.env.n_defenders, self.predictor.action_dim)
        if action_array.shape != expected:
            raise ValueError(f"Expected action shape {expected}, got {action_array.shape}.")
        self._last_action = action_array.copy()
        return self.observe(observation)

    def predict_candidates(self, candidate_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict target displacement for counterfactual current actions.

        The candidate replaces the final action-history token while all prior
        tokens remain the executed history.  This is the inference-time
        action-conditioned reranking interface; no target truth is consulted.
        """
        candidates = np.asarray(candidate_actions, dtype=np.float32)
        expected_prefix = (candidates.shape[0], self.env.n_defenders)
        if candidates.ndim != 3 or candidates.shape[1:] != (self.env.n_defenders, self.predictor.action_dim):
            raise ValueError(
                "Expected candidate action shape "
                f"[K, {self.env.n_defenders}, {self.predictor.action_dim}], got {candidates.shape}."
            )
        if not self._history or not self._action_history:
            raise RuntimeError("reset or observe must be called before predict_candidates.")
        padded_observations = [self._history[0]] * (self.history_length - len(self._history)) + self._history
        padded_actions = [self._action_history[0]] * (self.history_length - len(self._action_history)) + self._action_history
        observation_window = np.transpose(np.stack(padded_observations, axis=0), (1, 0, 2)).copy()
        action_window = np.transpose(np.stack(padded_actions, axis=0), (1, 0, 2)).copy()
        observation_batch = np.repeat(observation_window[None, ...], candidates.shape[0], axis=0)
        action_batch = np.repeat(action_window[None, ...], candidates.shape[0], axis=0)
        action_batch[:, :, -1, :] = candidates / self.action_scale
        with torch.no_grad():
            mean, log_variance, _latent = self.predictor(
                torch.as_tensor(observation_batch.reshape(-1, observation_batch.shape[2], observation_batch.shape[3]), device=self.device),
                torch.as_tensor(action_batch.reshape(-1, action_batch.shape[2], action_batch.shape[3]), device=self.device),
            )
        means = mean[:, self.horizon_index].detach().cpu().numpy().reshape(candidates.shape[0], self.env.n_defenders, 3)
        stds = torch.exp(0.5 * log_variance[:, self.horizon_index]).detach().cpu().numpy().reshape(
            candidates.shape[0], self.env.n_defenders, 3
        )
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise RuntimeError("Action-conditioned candidate prediction emitted non-finite values.")
        return means, stds


class ActionConditionedCandidateHistory:
    """Keep policy-safe observation/action history for counterfactual scoring.

    Unlike :class:`ActionConditionedLearnedPredictionObserver`, this adapter
    does not rewrite actor features.  That makes it suitable for frozen
    shape-aware V5 checkpoints whose input width has no prediction block: the
    JEPA model is used only as a candidate-action world model and the actor
    contract remains byte-for-byte unchanged.
    """

    def __init__(
        self,
        env: Any,
        predictor: ActionConditionedJEPAPredictor,
        device: torch.device,
        history_length: int,
        action_scale: float,
    ) -> None:
        if history_length <= 0 or action_scale <= 0.0:
            raise ValueError("history_length and action_scale must be positive.")
        self.env = env
        self.predictor = predictor.to(device).eval()
        self.device = device
        self.history_length = int(history_length)
        self.action_scale = float(action_scale)
        self._history: list[np.ndarray] = []
        # Each item is the action *outgoing from* the observation with the
        # same index.  It therefore has one fewer entry than `_history` until
        # a candidate is appended virtually during prediction.
        self._outgoing_action_history: list[np.ndarray] = []

    def reset(self, base_observation: np.ndarray) -> None:
        base = np.asarray(base_observation, dtype=np.float32)
        if base.ndim != 2 or base.shape[0] != self.env.n_defenders or base.shape[1] != self.predictor.input_dim:
            raise ValueError(
                "Candidate-history observation shape must be "
                f"({self.env.n_defenders}, {self.predictor.input_dim}), got {base.shape}."
            )
        self._history = [base.copy()]
        self._outgoing_action_history = []

    def observe_after_action(self, base_observation: np.ndarray, action: np.ndarray) -> None:
        base = np.asarray(base_observation, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        if base.shape != (self.env.n_defenders, self.predictor.input_dim):
            raise ValueError(f"Candidate-history observation shape mismatch: {base.shape}.")
        if action_array.shape != (self.env.n_defenders, self.predictor.action_dim):
            raise ValueError(f"Candidate-history action shape mismatch: {action_array.shape}.")
        # `action_array` was executed from the last stored observation to
        # obtain `base`, so append it before the new observation.
        self._outgoing_action_history.append((action_array / self.action_scale).copy())
        self._history.append(base.copy())
        if len(self._history) > self.history_length:
            self._history.pop(0)
            self._outgoing_action_history.pop(0)

    def _windows(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current observation window and its H-1 outgoing actions.

        The final H-th action is deliberately absent: it is supplied by each
        counterfactual candidate in :meth:`predict_candidates`.
        """
        if not self._history:
            raise RuntimeError("reset must be called before candidate prediction.")
        padded_observations = [self._history[0]] * (self.history_length - len(self._history)) + self._history
        zero = np.zeros((self.env.n_defenders, self.predictor.action_dim), dtype=np.float32)
        padded_actions = [zero] * (self.history_length - len(self._history)) + self._outgoing_action_history
        return (
            np.transpose(np.stack(padded_observations, axis=0), (1, 0, 2)).copy(),
            np.transpose(np.stack(padded_actions, axis=0), (1, 0, 2)).copy(),
        )

    def predict_candidates(self, candidate_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidate_actions, dtype=np.float32)
        expected = (candidates.shape[0], self.env.n_defenders, self.predictor.action_dim)
        if candidates.ndim != 3 or candidates.shape != expected:
            raise ValueError(f"Expected candidate action shape {expected}, got {candidates.shape}.")
        observation_window, past_action_window = self._windows()
        batch_size = int(candidates.shape[0])
        observation_batch = np.repeat(observation_window[None, ...], batch_size, axis=0)
        past_batch = np.repeat(past_action_window[None, ...], batch_size, axis=0)
        action_batch = np.concatenate(
            [past_batch, (candidates / self.action_scale)[:, :, None, :]], axis=2
        )
        with torch.no_grad():
            mean, log_variance, _latent = self.predictor(
                torch.as_tensor(
                    observation_batch.reshape(-1, observation_batch.shape[2], observation_batch.shape[3]),
                    device=self.device,
                ),
                torch.as_tensor(
                    action_batch.reshape(-1, action_batch.shape[2], action_batch.shape[3]),
                    device=self.device,
                ),
            )
        means = mean[:, -1].detach().cpu().numpy().reshape(batch_size, self.env.n_defenders, 3)
        stds = torch.exp(0.5 * log_variance[:, -1]).detach().cpu().numpy().reshape(
            batch_size, self.env.n_defenders, 3
        )
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise RuntimeError("Candidate-history prediction emitted non-finite values.")
        return means, stds

    def predict_candidates_multitask(
        self,
        candidate_actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Decode final-horizon target and auxiliary values for each candidate.

        This is intentionally separate from :meth:`predict_candidates` so
        legacy target-only rerankers retain their tuple contract. The returned
        auxiliary values are model predictions, not safety decisions.
        """
        if not isinstance(self.predictor, InteractionAwareActionConditionedMultitaskJEPAPredictor):
            raise ValueError("Reliability-aware reranking requires a JEPA-v3 multitask predictor.")
        candidates = np.asarray(candidate_actions, dtype=np.float32)
        expected = (candidates.shape[0], self.env.n_defenders, self.predictor.action_dim)
        if candidates.ndim != 3 or candidates.shape != expected:
            raise ValueError(f"Expected candidate actions shaped {expected}, got {candidates.shape}.")
        observation_window, past_action_window = self._windows()
        batch_size = int(candidates.shape[0])
        observation_batch = np.repeat(observation_window[None, ...], batch_size, axis=0)
        past_batch = np.repeat(past_action_window[None, ...], batch_size, axis=0)
        action_batch = np.concatenate(
            [past_batch, (candidates / self.action_scale)[:, :, None, :]], axis=2
        )
        with torch.no_grad():
            mean, log_variance, _latent, auxiliary = self.predictor.forward_multitask(
                torch.as_tensor(
                    observation_batch.reshape(-1, observation_batch.shape[2], observation_batch.shape[3]),
                    device=self.device,
                ),
                torch.as_tensor(
                    action_batch.reshape(-1, action_batch.shape[2], action_batch.shape[3]),
                    device=self.device,
                ),
            )
        means = mean[:, -1].detach().cpu().numpy().reshape(batch_size, self.env.n_defenders, 3)
        stds = torch.exp(0.5 * log_variance[:, -1]).detach().cpu().numpy().reshape(
            batch_size, self.env.n_defenders, 3
        )
        values = {
            key: value[:, -1].detach().cpu().numpy().reshape(batch_size, self.env.n_defenders)
            for key, value in auxiliary.items()
        }
        if not np.isfinite(means).all() or not np.isfinite(stds).all() or not all(
            np.isfinite(value).all() for value in values.values()
        ):
            raise RuntimeError("Multitask candidate prediction emitted non-finite values.")
        return means, stds, values


@dataclass(frozen=True)
class JEPACandidateSelection:
    """Auditable diagnostics for action-conditioned candidate reranking."""

    selected_index: int
    scores: tuple[float, ...]
    predicted_mean_distance_m: tuple[float, ...]
    uncertainty_penalty_m: tuple[float, ...]
    ledger_credit: float | None = None
    ledger_sample_count: int | None = None
    ledger_fallback_to_nominal: bool = False
    ledger_used_global_fallback: bool = False
    ledger_key: str | None = None
    predicted_min_clearance_m: float | None = None


class ActionConditionedCandidateReranker:
    """Select a short-horizon action candidate before deterministic CBF."""

    def __init__(
        self,
        observer: ActionConditionedLearnedPredictionObserver,
        horizon_seconds: float,
        position_extent: float,
        uncertainty_weight: float = 0.10,
        action_change_weight: float = 0.02,
        reliability_ledger: Any | None = None,
        reliability_horizon_index: int = -1,
    ) -> None:
        if horizon_seconds <= 0.0 or position_extent <= 0.0:
            raise ValueError("horizon_seconds and position_extent must be positive.")
        if uncertainty_weight < 0.0 or action_change_weight < 0.0:
            raise ValueError("Candidate penalty weights must be non-negative.")
        self.observer = observer
        self.horizon_seconds = float(horizon_seconds)
        self.position_extent = float(position_extent)
        self.uncertainty_weight = float(uncertainty_weight)
        self.action_change_weight = float(action_change_weight)
        self.reliability_ledger = reliability_ledger
        horizon_count = int(getattr(observer.predictor, "horizon_count", 0))
        resolved_horizon = int(reliability_horizon_index)
        if resolved_horizon < 0:
            resolved_horizon += horizon_count
        if reliability_ledger is not None and not 0 <= resolved_horizon < horizon_count:
            raise ValueError("reliability_horizon_index must refer to a JEPA prediction horizon.")
        self.reliability_horizon_index = resolved_horizon

    def select(
        self,
        observation: dict[str, Any],
        candidate_actions: np.ndarray,
    ) -> tuple[np.ndarray, JEPACandidateSelection]:
        candidates = np.asarray(candidate_actions, dtype=np.float32)
        if candidates.ndim != 3 or candidates.shape[1:] != (self.observer.env.n_defenders, 3):
            raise ValueError(
                "Expected candidate actions shaped "
                f"[K, {self.observer.env.n_defenders}, 3], got {candidates.shape}."
            )
        auxiliary: dict[str, np.ndarray] | None = None
        if self.reliability_ledger is None:
            means, stds = self.observer.predict_candidates(candidates)
        else:
            means, stds, auxiliary = self.observer.predict_candidates_multitask(candidates)
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        predicted_target_positions = positions[None, :, None, :] + means[:, :, None, :] * self.position_extent
        future_defender_positions = positions[None, :, :] + candidates * self.horizon_seconds
        distances = np.linalg.norm(predicted_target_positions[:, :, 0, :] - future_defender_positions, axis=2)
        mean_distance = np.mean(distances, axis=1)
        uncertainty_penalty = self.uncertainty_weight * self.position_extent * np.mean(stds, axis=(1, 2))
        baseline = candidates[0]
        action_penalty = self.action_change_weight * np.mean(
            np.linalg.norm(candidates - baseline[None, :, :], axis=2), axis=1
        )
        scores = mean_distance + uncertainty_penalty + action_penalty
        selected_index = int(np.argmin(scores))
        ledger_credit: float | None = None
        ledger_sample_count: int | None = None
        ledger_fallback_to_nominal = False
        ledger_used_global_fallback = False
        ledger_key: str | None = None
        predicted_min_clearance_m: float | None = None
        if self.reliability_ledger is not None:
            assert auxiliary is not None
            predicted_min_clearance_m = float(
                np.min(
                    np.minimum(
                        auxiliary["obstacle_clearance"][selected_index],
                        auxiliary["inter_agent_clearance"][selected_index],
                    )
                )
                * self.position_extent
            )
            visible_fraction = float(np.mean(np.asarray(observation["target_visible"], dtype=np.float64)))
            maximum_age = float(getattr(self.observer.env, "pursuit", {}).get("maximum_message_age_steps", 1.0))
            normalized_message_age = float(
                np.clip(np.mean(np.asarray(observation["message_age_steps"], dtype=np.float64)) / maximum_age, 0.0, 1.0)
            )
            action_magnitude = float(np.mean(np.linalg.norm(candidates[selected_index], axis=1)))
            decision = self.reliability_ledger.decision(
                self.reliability_horizon_index,
                visible_fraction,
                normalized_message_age,
                predicted_min_clearance_m,
                action_magnitude,
            )
            ledger_credit = decision.credit
            ledger_sample_count = decision.sample_count
            ledger_fallback_to_nominal = decision.fallback_to_nominal
            ledger_used_global_fallback = decision.used_global_fallback
            ledger_key = decision.key
            if ledger_fallback_to_nominal:
                selected_index = 0
        diagnostics = JEPACandidateSelection(
            selected_index=selected_index,
            scores=tuple(float(value) for value in scores),
            predicted_mean_distance_m=tuple(float(value) for value in mean_distance),
            uncertainty_penalty_m=tuple(float(value) for value in uncertainty_penalty),
            ledger_credit=ledger_credit,
            ledger_sample_count=ledger_sample_count,
            ledger_fallback_to_nominal=ledger_fallback_to_nominal,
            ledger_used_global_fallback=ledger_used_global_fallback,
            ledger_key=ledger_key,
            predicted_min_clearance_m=predicted_min_clearance_m,
        )
        return candidates[selected_index].copy(), diagnostics


def make_action_candidates(
    desired_actions: np.ndarray,
    perturbation_mps: float = 0.60,
    candidate_count: int = 5,
) -> np.ndarray:
    """Create deterministic local action alternatives for model reranking."""
    desired = np.asarray(desired_actions, dtype=np.float32)
    if desired.ndim != 2 or desired.shape[-1] != 3:
        raise ValueError(f"Expected desired action shape [defenders, 3], got {desired.shape}.")
    if perturbation_mps < 0.0 or candidate_count <= 0:
        raise ValueError("perturbation_mps must be non-negative and candidate_count must be positive.")
    offsets = np.zeros((candidate_count, desired.shape[0], 3), dtype=np.float32)
    axes = np.eye(3, dtype=np.float32)
    for candidate in range(1, candidate_count):
        axis = axes[(candidate - 1) % 3]
        sign = 1.0 if ((candidate - 1) // 3) % 2 == 0 else -1.0
        offsets[candidate] = sign * float(perturbation_mps) * axis[None, :]
    return desired[None, :, :] + offsets
