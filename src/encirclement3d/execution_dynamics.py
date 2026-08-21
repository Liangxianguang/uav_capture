"""Auditable defender command-execution dynamics for the E1 benchmark.

The main pursuit environment intentionally remains an ideal velocity-level
benchmark.  This module is a wrapper-side execution model: it turns a policy
or CBF velocity command into the velocity actually supplied to that unchanged
environment.  Keeping this logic out of :mod:`pursuit_env` makes the E1
assumption explicit and permits an exact disabled-mode regression check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


def _clip_rows(values: np.ndarray, maximum_norm: float) -> np.ndarray:
    """Project every row of ``values`` onto a three-dimensional speed ball."""
    if maximum_norm <= 0.0 or not np.isfinite(maximum_norm):
        raise ValueError("maximum_norm must be finite and positive.")
    result = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    factors = np.minimum(1.0, maximum_norm / np.maximum(norms, 1e-12))
    return result * factors


def _move_toward(current: np.ndarray, desired: np.ndarray, maximum_delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Limit the row-wise velocity change and return its saturation flags."""
    if maximum_delta < 0.0 or not np.isfinite(maximum_delta):
        raise ValueError("maximum_delta must be finite and non-negative.")
    delta = np.asarray(desired, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    factors = np.minimum(1.0, maximum_delta / np.maximum(norms, 1e-12))
    saturated = (norms[:, 0] > maximum_delta + 1e-12)
    return np.asarray(current, dtype=np.float64) + delta * factors, saturated


@dataclass(frozen=True)
class ExecutionDynamicsConfig:
    """Fully specified defender-side command execution assumptions.

    ``noise_std_mps`` is a per-axis Gaussian standard deviation.  A sampled
    vector is row-clipped to ``noise_clip_sigma * noise_std_mps``, making the
    injected error bounded and therefore reportable to the execution-aware
    safety filter.
    """

    enabled: bool = False
    action_delay_steps: int = 0
    max_speed_scale: float = 1.0
    max_acceleration_scale: float = 1.0
    noise_std_mps: float = 0.0
    noise_clip_sigma: float = 3.0

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any] | None) -> "ExecutionDynamicsConfig":
        values = dict(document or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"Unknown E1 execution settings: {', '.join(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if isinstance(self.action_delay_steps, bool) or int(self.action_delay_steps) != self.action_delay_steps:
            raise ValueError("action_delay_steps must be a non-negative integer.")
        if int(self.action_delay_steps) < 0:
            raise ValueError("action_delay_steps must be non-negative.")
        for name in ("max_speed_scale", "max_acceleration_scale", "noise_std_mps", "noise_clip_sigma"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if float(self.max_speed_scale) <= 0.0 or float(self.max_acceleration_scale) <= 0.0:
            raise ValueError("max_speed_scale and max_acceleration_scale must be positive.")
        if float(self.noise_std_mps) < 0.0 or float(self.noise_clip_sigma) <= 0.0:
            raise ValueError("noise_std_mps must be non-negative and noise_clip_sigma positive.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionStep:
    """One defender command transition, retained for per-step audit trails."""

    requested_velocity: np.ndarray
    delayed_velocity: np.ndarray
    executed_velocity: np.ndarray
    noise_velocity: np.ndarray
    command_age_steps: int
    acceleration_saturated: np.ndarray
    speed_saturated: np.ndarray

    @property
    def error_norms(self) -> np.ndarray:
        return np.linalg.norm(self.executed_velocity - self.requested_velocity, axis=1)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "command_age_steps": int(self.command_age_steps),
            "mean_command_execution_error_mps": float(np.mean(self.error_norms)),
            "max_command_execution_error_mps": float(np.max(self.error_norms)),
            "mean_noise_norm_mps": float(np.mean(np.linalg.norm(self.noise_velocity, axis=1))),
            "acceleration_saturated_defenders": int(np.count_nonzero(self.acceleration_saturated)),
            "speed_saturated_defenders": int(np.count_nonzero(self.speed_saturated)),
        }


class DefenderExecutionDynamics:
    """FIFO delay, acceleration/speed limit, and bounded execution noise.

    The class owns an RNG distinct from the environment RNG.  Its reset seed
    is deterministically derived by :class:`ExecutionDynamicsPursuitWrapper`,
    so enabling execution noise never changes obstacle or target sampling.
    """

    def __init__(
        self,
        config: ExecutionDynamicsConfig,
        *,
        defender_count: int,
        dt: float,
        nominal_max_speed: float,
        nominal_max_acceleration: float,
    ) -> None:
        config.validate()
        if defender_count <= 0:
            raise ValueError("defender_count must be positive.")
        if dt <= 0.0 or nominal_max_speed <= 0.0 or nominal_max_acceleration <= 0.0:
            raise ValueError("dt and nominal motion limits must be positive.")
        self.config = config
        self.defender_count = int(defender_count)
        self.dt = float(dt)
        self.nominal_max_speed = float(nominal_max_speed)
        self.nominal_max_acceleration = float(nominal_max_acceleration)
        self.rng = np.random.default_rng()
        self.executed_velocity = np.zeros((self.defender_count, 3), dtype=np.float64)
        self._queue: list[np.ndarray] = []
        self.last_step: ExecutionStep | None = None

    @property
    def max_speed(self) -> float:
        return self.nominal_max_speed * float(self.config.max_speed_scale)

    @property
    def max_acceleration(self) -> float:
        return self.nominal_max_acceleration * float(self.config.max_acceleration_scale)

    @property
    def noise_bound(self) -> float:
        return float(self.config.noise_std_mps) * float(self.config.noise_clip_sigma)

    def reset(self, *, seed: int, initial_velocity: np.ndarray) -> None:
        initial = self._validate_velocity(initial_velocity, "initial_velocity")
        self.rng = np.random.default_rng(int(seed))
        self.executed_velocity = initial.copy()
        self._queue = [np.zeros_like(initial) for _ in range(int(self.config.action_delay_steps))]
        self.last_step = None

    def execute(self, requested_velocity: np.ndarray) -> ExecutionStep:
        requested = self._validate_velocity(requested_velocity, "requested_velocity")
        if not bool(self.config.enabled):
            record = ExecutionStep(
                requested_velocity=requested.copy(),
                delayed_velocity=requested.copy(),
                executed_velocity=requested.copy(),
                noise_velocity=np.zeros_like(requested),
                command_age_steps=0,
                acceleration_saturated=np.zeros(self.defender_count, dtype=bool),
                speed_saturated=np.zeros(self.defender_count, dtype=bool),
            )
            self.executed_velocity = requested.copy()
            self.last_step = record
            return record

        if self._queue:
            delayed = self._queue.pop(0)
            self._queue.append(requested.copy())
        else:
            delayed = requested.copy()
        pre_noise, acceleration_saturated = _move_toward(
            self.executed_velocity,
            delayed,
            self.max_acceleration * self.dt,
        )
        raw_noise = self.rng.normal(0.0, float(self.config.noise_std_mps), size=requested.shape)
        noise = _clip_rows(raw_noise, self.noise_bound) if self.noise_bound > 0.0 else np.zeros_like(requested)
        before_speed_clip = pre_noise + noise
        executed = _clip_rows(before_speed_clip, self.max_speed)
        speed_saturated = np.linalg.norm(before_speed_clip, axis=1) > self.max_speed + 1e-12
        record = ExecutionStep(
            requested_velocity=requested.copy(),
            delayed_velocity=delayed.copy(),
            executed_velocity=executed.copy(),
            noise_velocity=noise.copy(),
            command_age_steps=int(self.config.action_delay_steps),
            acceleration_saturated=acceleration_saturated.copy(),
            speed_saturated=speed_saturated.copy(),
        )
        self.executed_velocity = executed.copy()
        self.last_step = record
        return record

    def conservative_displacement_bound(self, current_velocity: np.ndarray) -> np.ndarray:
        """Return a per-defender nonnegative E-CBF motion-margin increment.

        The increment is zero for E0.  With a delayed command, a defender can
        keep moving for ``d * dt`` before a newly filtered command arrives.
        A reduced acceleration contributes only the *additional* braking
        distance relative to the original environment; bounded noise adds one
        interval of worst-case displacement.  This is a transparent envelope,
        not a real-flight calibration claim.
        """
        velocities = self._validate_velocity(current_velocity, "current_velocity")
        if not bool(self.config.enabled):
            return np.zeros(self.defender_count, dtype=np.float64)
        speeds = np.linalg.norm(velocities, axis=1)
        delay = int(self.config.action_delay_steps) * self.dt * self.max_speed
        braking_extra = 0.5 * speeds**2 * max(
            0.0,
            1.0 / self.max_acceleration - 1.0 / self.nominal_max_acceleration,
        )
        noise = self.dt * self.noise_bound
        return np.full(self.defender_count, delay + noise, dtype=np.float64) + braking_extra

    def runtime_dict(self) -> dict[str, Any]:
        return {
            **self.config.as_dict(),
            "nominal_max_speed_mps": self.nominal_max_speed,
            "nominal_max_acceleration_mps2": self.nominal_max_acceleration,
            "effective_max_speed_mps": self.max_speed,
            "effective_max_acceleration_mps2": self.max_acceleration,
            "noise_bound_mps": self.noise_bound,
        }

    def _validate_velocity(self, values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.defender_count, 3):
            raise ValueError(f"{name} must have shape {(self.defender_count, 3)}, got {result.shape}.")
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite.")
        return result
