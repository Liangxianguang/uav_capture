"""Shared policy residual and obstacle-clearance gating utilities.

The residual is a deployment-time action correction used during the hold phase
to reduce slot drift. Training and evaluation must use the same implementation
so that the learned policy sees the same action protocol as deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicyResidualResult:
    """Result of applying the configured residual to a policy action."""

    action: np.ndarray
    residual: np.ndarray
    norm: float
    minimum_clearance: float
    mean_clearance_scale: float
    active: bool


def _obstacle_clearances(observation: dict[str, Any], env: Any) -> np.ndarray:
    del observation  # The simulator state is authoritative for clearance.
    if not env.obstacles:
        return np.full(env.n_defenders, float("inf"), dtype=np.float64)
    positions = np.asarray(env.defender_positions, dtype=np.float64)
    return np.asarray(
        [
            min(
                float(env._cylinder_clearance_and_normal(position, obstacle)[0])
                - float(env.agents["drone_radius"])
                for obstacle in env.obstacles
            )
            for position in positions
        ],
        dtype=np.float64,
    )


def compute_policy_residual(
    action: np.ndarray,
    observation: dict[str, Any],
    env: Any,
    document: dict[str, Any],
) -> PolicyResidualResult:
    """Apply a bounded slot residual and clearance gate to ``action``."""

    requested = np.asarray(action, dtype=np.float64)
    residual_document = dict(document or {})
    clearances = _obstacle_clearances(observation, env)
    threshold = float(residual_document.get("clearance_gate_threshold", 0.0))
    if threshold < 0.0:
        raise ValueError("clearance_gate_threshold must be non-negative.")
    scales = np.clip(clearances / threshold, 0.0, 1.0) if threshold > 0.0 else np.ones(env.n_defenders)

    if not bool(residual_document.get("enabled", False)):
        return PolicyResidualResult(
            action=requested,
            residual=np.zeros_like(requested),
            norm=0.0,
            minimum_clearance=float(np.min(clearances)),
            mean_clearance_scale=float(np.mean(scales)),
            active=False,
        )

    slot_gain = float(residual_document.get("slot_gain", 0.0))
    target_feedforward = float(residual_document.get("target_velocity_feedforward", 0.0))
    activation_error = float(residual_document.get("activation_error", 1.0))
    hold_only = bool(residual_document.get("hold_only", True))
    if slot_gain < 0.0 or target_feedforward < 0.0 or activation_error <= 0.0:
        raise ValueError("policy_residual parameters must be non-negative with activation_error > 0.")

    slot_delta = np.asarray(observation["slot_positions"], dtype=np.float64) - np.asarray(
        observation["defender_positions"], dtype=np.float64
    )
    active = not hold_only or float(np.max(np.linalg.norm(slot_delta, axis=1))) <= activation_error
    if not active:
        return PolicyResidualResult(
            action=requested,
            residual=np.zeros_like(requested),
            norm=0.0,
            minimum_clearance=float(np.min(clearances)),
            mean_clearance_scale=float(np.mean(scales)),
            active=False,
        )

    residual = slot_gain * slot_delta
    residual += target_feedforward * np.asarray(observation["target_velocity"], dtype=np.float64)[None, :]
    residual *= scales[:, None]
    applied = env._clip_rows(requested + residual, float(env.agents["defender_max_speed"]))
    applied_residual = applied - requested
    return PolicyResidualResult(
        action=applied,
        residual=applied_residual,
        norm=float(np.mean(np.linalg.norm(applied_residual, axis=1))),
        minimum_clearance=float(np.min(clearances)),
        mean_clearance_scale=float(np.mean(scales)),
        active=True,
    )


def apply_policy_residual(
    action: np.ndarray,
    observation: dict[str, Any],
    env: Any,
    document: dict[str, Any],
) -> tuple[np.ndarray, float, float, float]:
    """Compatibility tuple used by evaluation and existing tests."""

    result = compute_policy_residual(action, observation, env, document)
    return result.action, result.norm, result.minimum_clearance, result.mean_clearance_scale
