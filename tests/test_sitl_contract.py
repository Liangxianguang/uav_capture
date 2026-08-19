from __future__ import annotations

import numpy as np

from encirclement3d.sitl_contract import (
    CaptureSITLCommand,
    CaptureSITLCommandGate,
    CaptureSITLObservation,
    NetCommand,
    NetMode,
    NetTelemetry,
    SITLCommandGateConfig,
    StampedTargetEstimate,
)


def gate() -> CaptureSITLCommandGate:
    return CaptureSITLCommandGate(
        SITLCommandGateConfig(
            max_defender_speed_m_s=5.0,
            max_target_age_s=0.15,
            max_net_age_s=0.10,
            max_communication_age_s=0.10,
            max_position_variance_m2=0.04,
            max_net_tension_n=10.0,
            max_net_strain=0.15,
            max_command_horizon_s=0.20,
        )
    )


def observation(
    *,
    target_timestamp_s: float = 1.0,
    net_timestamp_s: float = 1.05,
    communication_age_s: float = 0.02,
    tension_n: float = 0.0,
) -> CaptureSITLObservation:
    return CaptureSITLObservation(
        timestamp_s=1.05,
        defender_positions_m=np.zeros((4, 3)),
        defender_velocities_m_s=np.zeros((4, 3)),
        target=StampedTargetEstimate(
            timestamp_s=target_timestamp_s,
            position_m=np.zeros(3),
            velocity_m_s=np.zeros(3),
            position_covariance_m2=np.eye(3) * 0.01,
        ),
        obstacles_xy_radius_height=np.empty((0, 4)),
        net=NetTelemetry(
            timestamp_s=net_timestamp_s,
            mode=NetMode.STOWED,
            deployment_progress=0.0,
            min_mesh_margin_m=0.0,
            max_tension_n=tension_n,
            max_strain=0.0,
        ),
        communication_age_s=communication_age_s,
    )


def command(net_command: NetCommand = NetCommand.DEPLOY, speed: float = 1.0) -> CaptureSITLCommand:
    return CaptureSITLCommand(
        issued_at_s=1.05,
        valid_until_s=1.15,
        defender_velocity_setpoints_m_s=np.full((4, 3), [speed, 0.0, 0.0]),
        net_command=net_command,
    )


def test_sitl_contract_accepts_fresh_bounded_deploy_command() -> None:
    decision = gate().evaluate(observation(), command())
    assert decision.allowed
    assert decision.reason == "accepted"


def test_sitl_contract_rejects_stale_target_and_communication() -> None:
    assert gate().evaluate(observation(target_timestamp_s=0.80), command()).reason == "target_estimate_stale"
    assert gate().evaluate(observation(net_timestamp_s=0.80), command()).reason == "net_telemetry_stale"
    assert gate().evaluate(observation(communication_age_s=0.20), command()).reason == "communication_stale"


def test_sitl_contract_rejects_overload_but_allows_abort() -> None:
    overloaded = observation(tension_n=12.0)
    assert gate().evaluate(overloaded, command(NetCommand.HOLD)).reason == "net_tension_limit"
    abort = command(NetCommand.ABORT)
    decision = gate().evaluate(overloaded, abort)
    assert decision.allowed
    assert decision.reason == "abort_allowed"


def test_sitl_contract_rejects_excessive_velocity_command() -> None:
    decision = gate().evaluate(observation(), command(speed=6.0))
    assert not decision.allowed
    assert decision.reason == "defender_speed_limit"
