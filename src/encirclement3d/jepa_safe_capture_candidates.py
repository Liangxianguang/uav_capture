"""Safe-capture v2 candidate chunks and action-conditioned history windows.

This module owns the v2 deployment contract between a frozen nominal actor and
the JEPA evaluator.  It deliberately stops before CBF: candidates are checked
for basic kinematic feasibility, scored by JEPA, and returned with an explicit
execution mode.  A downstream CBF/QP must still filter the returned first
control step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .prediction import InteractionAwareActionConditionedSafeCaptureJEPAPredictor


CANDIDATE_LABELS = (
    "nominal",
    "intercept",
    "lateral_clearance",
    "formation_clearance",
    "visibility_hold",
)


def _normalize_rows(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    result = array / np.maximum(norms, 1e-12)
    zero = norms[:, 0] <= 1e-12
    if np.any(zero):
        replacement = np.zeros_like(array) if fallback is None else np.asarray(fallback, dtype=np.float64)
        if replacement.ndim == 1:
            replacement = np.repeat(replacement[None, :], array.shape[0], axis=0)
        replacement_norm = np.linalg.norm(replacement, axis=1, keepdims=True)
        result[zero] = replacement[zero] / np.maximum(replacement_norm[zero], 1e-12)
    return result


@dataclass(frozen=True)
class SafeCaptureCandidateConfig:
    """Frozen first-version candidate contract."""

    candidate_count: int = 5
    chunk_length_steps: int = 3
    perturbation_mps: float = 0.10
    max_speed_mps: float = 5.0
    max_acceleration_mps2: float = 6.0
    dt_seconds: float = 0.1
    max_action_change_mps: float | None = None

    def __post_init__(self) -> None:
        if self.candidate_count != len(CANDIDATE_LABELS):
            raise ValueError(f"Safe-capture v2 requires exactly {len(CANDIDATE_LABELS)} candidates.")
        if self.chunk_length_steps != 3:
            raise ValueError("Safe-capture v2 first chunk contract requires exactly 3 control steps.")
        if self.perturbation_mps < 0.0 or self.max_speed_mps <= 0.0:
            raise ValueError("perturbation_mps must be non-negative and max_speed_mps positive.")
        if self.max_acceleration_mps2 <= 0.0 or self.dt_seconds <= 0.0:
            raise ValueError("max_acceleration_mps2 and dt_seconds must be positive.")
        if self.max_action_change_mps is not None and self.max_action_change_mps <= 0.0:
            raise ValueError("max_action_change_mps must be positive when provided.")

    @property
    def resolved_max_action_change_mps(self) -> float:
        return float(
            self.max_action_change_mps
            if self.max_action_change_mps is not None
            else self.max_acceleration_mps2 * self.dt_seconds
        )


@dataclass(frozen=True)
class SafeCaptureCandidateBatch:
    """Candidate chunks plus deterministic pre-check results."""

    chunks: np.ndarray
    labels: tuple[str, ...]
    valid_mask: np.ndarray
    rejection_reasons: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if self.chunks.ndim != 4:
            raise ValueError("Candidate chunks must have shape [K, steps, defenders, 3].")
        if self.chunks.shape[0] != len(self.labels) or self.chunks.shape[0] != len(self.rejection_reasons):
            raise ValueError("Candidate metadata length does not match chunk count.")
        if self.valid_mask.shape != (self.chunks.shape[0],):
            raise ValueError("valid_mask must have one entry per candidate.")


def _observation_vectors(observation: Mapping[str, Any], defender_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(observation.get("defender_positions"), dtype=np.float64)
    if positions.shape != (defender_count, 3):
        raise ValueError(f"defender_positions must have shape {(defender_count, 3)}, got {positions.shape}.")
    beliefs = np.asarray(observation.get("target_belief_positions", positions), dtype=np.float64)
    if beliefs.shape != positions.shape or not np.isfinite(beliefs).all():
        raise ValueError("target_belief_positions must be finite and match defender_positions.")
    velocities = np.asarray(observation.get("target_belief_velocities", np.zeros_like(positions)), dtype=np.float64)
    if velocities.shape != positions.shape or not np.isfinite(velocities).all():
        raise ValueError("target_belief_velocities must be finite and match defender_positions.")
    if not np.isfinite(positions).all():
        raise ValueError("defender_positions must be finite.")
    return positions, beliefs, velocities


def precheck_action_chunk(
    chunk: np.ndarray,
    *,
    previous_action: np.ndarray,
    config: SafeCaptureCandidateConfig,
) -> tuple[bool, tuple[str, ...]]:
    """Check finite, constant, speed, acceleration, and slew constraints."""

    candidate = np.asarray(chunk, dtype=np.float64)
    previous = np.asarray(previous_action, dtype=np.float64)
    reasons: list[str] = []
    if candidate.ndim != 3 or candidate.shape[0] != config.chunk_length_steps or candidate.shape[-1] != 3:
        return False, ("shape",)
    if previous.shape != candidate.shape[1:]:
        return False, ("previous_action_shape",)
    if not np.isfinite(candidate).all() or not np.isfinite(previous).all():
        reasons.append("non_finite")
    if not np.allclose(candidate, candidate[:1], rtol=0.0, atol=1e-9):
        reasons.append("non_constant_chunk")
    if np.any(np.linalg.norm(candidate, axis=-1) > config.max_speed_mps + 1e-8):
        reasons.append("speed_limit")
    delta = candidate[0] - previous
    if np.any(np.linalg.norm(delta, axis=-1) / config.dt_seconds > config.max_acceleration_mps2 + 1e-8):
        reasons.append("acceleration_limit")
    if np.any(np.linalg.norm(delta, axis=-1) > config.resolved_max_action_change_mps + 1e-8):
        reasons.append("action_slew_limit")
    return not reasons, tuple(dict.fromkeys(reasons))


def make_safe_capture_candidate_chunks(
    nominal_action: np.ndarray,
    observation: Mapping[str, Any],
    *,
    config: SafeCaptureCandidateConfig | None = None,
    previous_action: np.ndarray | None = None,
) -> SafeCaptureCandidateBatch:
    """Generate the fixed K=5 constant desired-action chunks.

    Candidate 0 is an exact copy of the nominal action.  The remaining four
    candidates are small, deterministic geometric perturbations based only on
    defender positions and target beliefs available in the online observation.
    No target ground truth or CBF output is read here.
    """

    settings = config or SafeCaptureCandidateConfig()
    nominal = np.asarray(nominal_action)
    if nominal.ndim != 2 or nominal.shape[-1] != 3:
        raise ValueError(f"nominal_action must have shape [defenders, 3], got {nominal.shape}.")
    if not np.issubdtype(nominal.dtype, np.floating):
        nominal = nominal.astype(np.float64)
    if not np.isfinite(nominal).all():
        raise ValueError("nominal_action must be finite.")
    defender_count = int(nominal.shape[0])
    positions, beliefs, belief_velocities = _observation_vectors(observation, defender_count)
    target_direction = _normalize_rows(
        beliefs + belief_velocities * settings.dt_seconds * settings.chunk_length_steps - positions,
        fallback=np.array([1.0, 0.0, 0.0]),
    )
    lateral_direction = np.column_stack((-target_direction[:, 1], target_direction[:, 0], np.zeros(defender_count)))
    lateral_direction = _normalize_rows(lateral_direction, fallback=np.array([0.0, 1.0, 0.0]))
    formation_direction = _normalize_rows(positions - positions.mean(axis=0, keepdims=True), fallback=lateral_direction)
    visibility_direction = _normalize_rows(
        target_direction + 0.25 * _normalize_rows(belief_velocities, fallback=target_direction),
        fallback=target_direction,
    )
    directions = (target_direction, lateral_direction, formation_direction, visibility_direction)
    chunks: list[np.ndarray] = [np.repeat(nominal[None, :, :], settings.chunk_length_steps, axis=0)]
    for direction in directions:
        chunks.append(np.repeat((nominal + settings.perturbation_mps * direction)[None, :, :], settings.chunk_length_steps, axis=0))
    all_chunks = np.stack(chunks, axis=0)
    reference_action = nominal if previous_action is None else np.asarray(previous_action)
    if reference_action.shape != nominal.shape:
        raise ValueError(f"previous_action must have shape {nominal.shape}, got {reference_action.shape}.")
    valid: list[bool] = []
    reasons: list[tuple[str, ...]] = []
    for candidate in all_chunks:
        candidate_valid, candidate_reasons = precheck_action_chunk(
            candidate,
            previous_action=reference_action,
            config=settings,
        )
        valid.append(candidate_valid)
        reasons.append(candidate_reasons)
    if not valid[0]:
        # A nominal action that violates the declared contract cannot be
        # silently repaired by the evaluator.  Downstream safety code must
        # enter its explicit hold/abort path.
        reasons[0] = tuple(sorted(set(reasons[0] + ("nominal_infeasible",))))
    return SafeCaptureCandidateBatch(
        chunks=all_chunks,
        labels=CANDIDATE_LABELS,
        valid_mask=np.asarray(valid, dtype=bool),
        rejection_reasons=tuple(reasons),
    )


class SafeCaptureCandidateHistory:
    """Causal observation/action history for v2 counterfactual prediction."""

    def __init__(
        self,
        predictor: InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
        *,
        defender_count: int,
        device: torch.device,
        history_length: int = 8,
        action_scale: float = 5.0,
    ) -> None:
        if not isinstance(predictor, InteractionAwareActionConditionedSafeCaptureJEPAPredictor):
            raise TypeError("P4 requires the safe-capture v2 JEPA predictor.")
        if defender_count <= 0 or history_length <= 0 or action_scale <= 0.0:
            raise ValueError("defender_count, history_length, and action_scale must be positive.")
        self.predictor = predictor.to(device).eval()
        self.defender_count = int(defender_count)
        self.device = device
        self.history_length = int(history_length)
        self.action_scale = float(action_scale)
        self._history: list[np.ndarray] = []
        self._outgoing_actions: list[np.ndarray] = []

    def reset(self, base_observation: np.ndarray) -> None:
        base = np.asarray(base_observation, dtype=np.float32)
        expected = (self.defender_count, self.predictor.input_dim)
        if base.shape != expected or not np.isfinite(base).all():
            raise ValueError(f"base_observation must be finite with shape {expected}, got {base.shape}.")
        self._history = [base.copy()]
        self._outgoing_actions = []

    def observe_after_action(self, base_observation: np.ndarray, action: np.ndarray) -> None:
        base = np.asarray(base_observation, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        expected_observation = (self.defender_count, self.predictor.input_dim)
        expected_action = (self.defender_count, self.predictor.action_dim)
        if base.shape != expected_observation or not np.isfinite(base).all():
            raise ValueError(f"base_observation must be finite with shape {expected_observation}, got {base.shape}.")
        if action_array.shape != expected_action or not np.isfinite(action_array).all():
            raise ValueError(f"action must be finite with shape {expected_action}, got {action_array.shape}.")
        if not self._history:
            raise RuntimeError("reset must be called before observe_after_action.")
        self._outgoing_actions.append((action_array / self.action_scale).copy())
        self._history.append(base.copy())
        if len(self._history) > self.history_length:
            self._history.pop(0)
            self._outgoing_actions.pop(0)

    def _windows(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._history:
            raise RuntimeError("reset must be called before prediction.")
        zero = np.zeros((self.defender_count, self.predictor.action_dim), dtype=np.float32)
        observations = [self._history[0]] * (self.history_length - len(self._history)) + self._history
        actions = [zero] * (self.history_length - len(self._history)) + self._outgoing_actions
        return (
            np.transpose(np.stack(observations, axis=0), (1, 0, 2)).copy(),
            np.transpose(np.stack(actions, axis=0), (1, 0, 2)).copy(),
        )

    def predict_candidates_multitask(
        self,
        candidate_actions: np.ndarray,
        *,
        horizon_index: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        candidates = np.asarray(candidate_actions, dtype=np.float32)
        expected = (candidates.shape[0], self.defender_count, self.predictor.action_dim)
        if candidates.ndim != 3 or candidates.shape != expected or not np.isfinite(candidates).all():
            raise ValueError(f"candidate_actions must be finite with shape {expected}, got {candidates.shape}.")
        if horizon_index < 0 or horizon_index >= self.predictor.horizon_count:
            raise ValueError("horizon_index is outside the predictor output range.")
        observation_window, past_action_window = self._windows()
        count = int(candidates.shape[0])
        observation_batch = np.repeat(observation_window[None, ...], count, axis=0)
        past_batch = np.repeat(past_action_window[None, ...], count, axis=0)
        action_batch = np.concatenate(
            [past_batch, (candidates / self.action_scale)[:, :, None, :]],
            axis=2,
        )
        with torch.no_grad():
            mean, log_variance, _latent, auxiliary = self.predictor.forward_multitask(
                torch.as_tensor(observation_batch.reshape(-1, observation_batch.shape[2], observation_batch.shape[3]), device=self.device),
                torch.as_tensor(action_batch.reshape(-1, action_batch.shape[2], action_batch.shape[3]), device=self.device),
            )
        std = torch.exp(0.5 * log_variance)
        mean_np = mean.detach().cpu().numpy().reshape(count, self.defender_count, self.predictor.horizon_count, 3)[:, :, horizon_index]
        std_np = std.detach().cpu().numpy().reshape(count, self.defender_count, self.predictor.horizon_count, 3)[:, :, horizon_index]
        values: dict[str, np.ndarray] = {}
        for key, value in auxiliary.items():
            array = value.detach().cpu().numpy()
            if key == "action_consistency":
                values[key] = array.reshape(count, self.defender_count, self.predictor.action_dim)
            elif array.ndim == 3:
                values[key] = array.reshape(count, self.defender_count, self.predictor.horizon_count, array.shape[-1])[:, :, horizon_index]
            else:
                values[key] = array.reshape(count, self.defender_count, self.predictor.horizon_count)[:, :, horizon_index]
        if not np.isfinite(mean_np).all() or not np.isfinite(std_np).all() or not all(np.isfinite(value).all() for value in values.values()):
            raise RuntimeError("Safe-capture v2 candidate prediction emitted non-finite values.")
        return mean_np.astype(np.float32), std_np.astype(np.float32), values
