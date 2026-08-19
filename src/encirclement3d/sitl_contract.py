"""Simulator-independent contract for future PX4 capture integration.

This module deliberately contains no MAVLink or PX4 transport.  It defines
the timestamped information a future adapter must exchange and blocks unsafe
high-level commands before they reach an Offboard transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class NetMode(str, Enum):
    STOWED = "stowed"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    REELING = "reeling"
    FAULT = "fault"


class NetCommand(str, Enum):
    HOLD = "hold"
    DEPLOY = "deploy"
    REEL = "reel"
    ABORT = "abort"


def _finite_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _finite_scalar(value: float, name: str, *, non_negative: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


@dataclass(frozen=True)
class StampedTargetEstimate:
    """Target state and uncertainty expressed in a common world frame."""

    timestamp_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    position_covariance_m2: np.ndarray

    def __post_init__(self) -> None:
        _finite_scalar(self.timestamp_s, "target timestamp_s", non_negative=True)
        _finite_array(self.position_m, (3,), "target position_m")
        _finite_array(self.velocity_m_s, (3,), "target velocity_m_s")
        covariance = _finite_array(self.position_covariance_m2, (3, 3), "target position_covariance_m2")
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("target position_covariance_m2 must be symmetric.")
        a, b, c = covariance[0]
        _a, d, e = covariance[1]
        _b, _e, f = covariance[2]
        principal_minors = np.array(
            [
                a,
                d,
                f,
                a * d - b * b,
                a * f - c * c,
                d * f - e * e,
                a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d),
            ],
            dtype=np.float64,
        )
        if np.any(principal_minors < -1e-12):
            raise ValueError("target position_covariance_m2 must be positive semidefinite.")


@dataclass(frozen=True)
class NetTelemetry:
    """Measured or estimated net state supplied by the capture mechanism."""

    timestamp_s: float
    mode: NetMode
    deployment_progress: float
    min_mesh_margin_m: float
    max_tension_n: float
    max_strain: float

    def __post_init__(self) -> None:
        _finite_scalar(self.timestamp_s, "net timestamp_s", non_negative=True)
        progress = _finite_scalar(self.deployment_progress, "deployment_progress", non_negative=True)
        if progress > 1.0:
            raise ValueError("deployment_progress must not exceed 1.")
        _finite_scalar(self.min_mesh_margin_m, "min_mesh_margin_m")
        _finite_scalar(self.max_tension_n, "max_tension_n", non_negative=True)
        _finite_scalar(self.max_strain, "max_strain", non_negative=True)


@dataclass(frozen=True)
class CaptureSITLObservation:
    """Timestamped inputs required by the high-level capture supervisor."""

    timestamp_s: float
    defender_positions_m: np.ndarray
    defender_velocities_m_s: np.ndarray
    target: StampedTargetEstimate
    obstacles_xy_radius_height: np.ndarray
    net: NetTelemetry
    communication_age_s: float

    def __post_init__(self) -> None:
        _finite_scalar(self.timestamp_s, "observation timestamp_s", non_negative=True)
        _finite_array(self.defender_positions_m, (4, 3), "defender_positions_m")
        _finite_array(self.defender_velocities_m_s, (4, 3), "defender_velocities_m_s")
        obstacles = np.asarray(self.obstacles_xy_radius_height, dtype=np.float64)
        if obstacles.ndim != 2 or obstacles.shape[1] != 4 or not np.all(np.isfinite(obstacles)):
            raise ValueError("obstacles_xy_radius_height must have finite shape (N, 4).")
        _finite_scalar(self.communication_age_s, "communication_age_s", non_negative=True)


@dataclass(frozen=True)
class CaptureSITLCommand:
    """Bounded velocity setpoints plus an explicit capture-mechanism command."""

    issued_at_s: float
    valid_until_s: float
    defender_velocity_setpoints_m_s: np.ndarray
    net_command: NetCommand

    def __post_init__(self) -> None:
        issued = _finite_scalar(self.issued_at_s, "issued_at_s", non_negative=True)
        valid_until = _finite_scalar(self.valid_until_s, "valid_until_s", non_negative=True)
        if valid_until <= issued:
            raise ValueError("valid_until_s must be after issued_at_s.")
        _finite_array(self.defender_velocity_setpoints_m_s, (4, 3), "defender_velocity_setpoints_m_s")


@dataclass(frozen=True)
class SITLCommandGateConfig:
    """Preflight limits shared by a SITL transport and future hardware adapter."""

    max_defender_speed_m_s: float
    max_target_age_s: float
    max_net_age_s: float
    max_communication_age_s: float
    max_position_variance_m2: float
    max_net_tension_n: float
    max_net_strain: float
    max_command_horizon_s: float

    def __post_init__(self) -> None:
        for name in (
            "max_defender_speed_m_s",
            "max_target_age_s",
            "max_net_age_s",
            "max_communication_age_s",
            "max_position_variance_m2",
            "max_net_tension_n",
            "max_net_strain",
            "max_command_horizon_s",
        ):
            if _finite_scalar(getattr(self, name), name, non_negative=True) <= 0.0:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class SITLCommandDecision:
    allowed: bool
    reason: str


class CaptureSITLCommandGate:
    """Reject stale, uncertain, overloaded, or malformed capture commands."""

    def __init__(self, config: SITLCommandGateConfig) -> None:
        self.config = config

    def evaluate(self, observation: CaptureSITLObservation, command: CaptureSITLCommand) -> SITLCommandDecision:
        speeds = np.linalg.norm(command.defender_velocity_setpoints_m_s, axis=1)
        if np.any(speeds > self.config.max_defender_speed_m_s + 1e-12):
            return SITLCommandDecision(False, "defender_speed_limit")
        if command.issued_at_s < observation.timestamp_s:
            return SITLCommandDecision(False, "command_predates_observation")
        if command.valid_until_s - command.issued_at_s > self.config.max_command_horizon_s:
            return SITLCommandDecision(False, "command_horizon_limit")

        if command.net_command == NetCommand.ABORT:
            return SITLCommandDecision(True, "abort_allowed")

        target_age = observation.timestamp_s - observation.target.timestamp_s
        if target_age < 0.0:
            return SITLCommandDecision(False, "target_clock_skew")
        if target_age > self.config.max_target_age_s:
            return SITLCommandDecision(False, "target_estimate_stale")
        net_age = observation.timestamp_s - observation.net.timestamp_s
        if net_age < 0.0:
            return SITLCommandDecision(False, "net_clock_skew")
        if net_age > self.config.max_net_age_s:
            return SITLCommandDecision(False, "net_telemetry_stale")
        if observation.communication_age_s > self.config.max_communication_age_s:
            return SITLCommandDecision(False, "communication_stale")
        if float(np.max(np.diag(observation.target.position_covariance_m2))) > self.config.max_position_variance_m2:
            return SITLCommandDecision(False, "target_uncertainty_limit")
        if observation.net.mode == NetMode.FAULT:
            return SITLCommandDecision(False, "net_fault")
        if observation.net.max_tension_n > self.config.max_net_tension_n:
            return SITLCommandDecision(False, "net_tension_limit")
        if observation.net.max_strain > self.config.max_net_strain:
            return SITLCommandDecision(False, "net_strain_limit")
        if command.net_command == NetCommand.DEPLOY and observation.net.mode != NetMode.STOWED:
            return SITLCommandDecision(False, "deploy_state_invalid")
        if command.net_command == NetCommand.REEL and observation.net.mode not in {
            NetMode.DEPLOYING,
            NetMode.DEPLOYED,
        }:
            return SITLCommandDecision(False, "reel_state_invalid")
        return SITLCommandDecision(True, "accepted")
