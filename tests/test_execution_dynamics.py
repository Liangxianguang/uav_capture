from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from encirclement3d.execution_dynamics import DefenderExecutionDynamics, ExecutionDynamicsConfig
from encirclement3d.execution_env import ExecutionDynamicsPursuitWrapper
from encirclement3d.execution_safety import ExecutionAwarePursuitCBFSafetyFilter, make_kinematic_cbf
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "e1_execution_dynamics_protocol.yaml"


def base_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def dynamics(**overrides: object) -> DefenderExecutionDynamics:
    return DefenderExecutionDynamics(
        ExecutionDynamicsConfig.from_mapping({"enabled": True, **overrides}),
        defender_count=4,
        dt=0.1,
        nominal_max_speed=5.0,
        nominal_max_acceleration=6.0,
    )


def assert_observations_equal(left: object, right: object) -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=0.0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_observations_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for item_left, item_right in zip(left, right):
            assert_observations_equal(item_left, item_right)
    else:
        assert left == right


def test_disabled_wrapper_matches_the_original_environment_step_by_step() -> None:
    config = base_config()
    direct = CaptureRadiusPursuit3DEnv(copy.deepcopy(config), obstacle_count=3, target_speed_scale=0.45)
    wrapped_env = CaptureRadiusPursuit3DEnv(copy.deepcopy(config), obstacle_count=3, target_speed_scale=0.45)
    wrapped = ExecutionDynamicsPursuitWrapper(wrapped_env, ExecutionDynamicsConfig())
    direct_observation = direct.reset(seed=681001)
    wrapped_observation = wrapped.reset(seed=681001)
    assert_observations_equal(direct_observation, wrapped_observation)
    rng = np.random.default_rng(44)
    for _ in range(12):
        actions = rng.uniform(-1.0, 1.0, size=(4, 3))
        direct_result = direct.step(actions)
        wrapped_result = wrapped.step(actions)
        assert_observations_equal(direct_result[0], wrapped_result[0])
        assert direct_result[1:4] == wrapped_result[1:4]
        for key, value in direct_result[4].items():
            assert wrapped_result[4][key] == value
        np.testing.assert_allclose(direct.defender_positions, wrapped.defender_positions, atol=1e-12, rtol=0.0)
        if direct_result[2] or direct_result[3]:
            break


def test_fifo_delay_and_reset_are_deterministic() -> None:
    model = dynamics(action_delay_steps=2)
    zero = np.zeros((4, 3), dtype=np.float64)
    command = np.tile(np.array([[1.0, 0.0, 0.0]]), (4, 1))
    model.reset(seed=7, initial_velocity=zero)
    first = model.execute(command)
    second = model.execute(command)
    third = model.execute(command)
    np.testing.assert_allclose(first.executed_velocity, zero)
    np.testing.assert_allclose(second.executed_velocity, zero)
    np.testing.assert_allclose(third.delayed_velocity, command)
    np.testing.assert_allclose(third.executed_velocity, np.tile(np.array([[0.6, 0.0, 0.0]]), (4, 1)))
    assert third.command_age_steps == 2
    model.reset(seed=7, initial_velocity=zero)
    np.testing.assert_allclose(model.execute(command).executed_velocity, zero)


def test_acceleration_speed_and_noise_bounds_are_enforced() -> None:
    model = dynamics(max_speed_scale=0.5, max_acceleration_scale=0.5, noise_std_mps=0.25, noise_clip_sigma=3.0)
    zero = np.zeros((4, 3), dtype=np.float64)
    command = np.tile(np.array([[10.0, 0.0, 0.0]]), (4, 1))
    model.reset(seed=8, initial_velocity=zero)
    transition = model.execute(command)
    assert np.all(np.linalg.norm(transition.executed_velocity, axis=1) <= 2.5 + 1e-12)
    assert np.all(np.linalg.norm(transition.noise_velocity, axis=1) <= 0.75 + 1e-12)
    assert np.all(transition.acceleration_saturated)
    assert np.all(np.linalg.norm(transition.executed_velocity - transition.noise_velocity, axis=1) <= 0.3 + 1e-12)


def test_noise_is_reproducible_without_consuming_environment_rng() -> None:
    command = np.tile(np.array([[0.8, -0.2, 0.1]]), (4, 1))
    first = dynamics(noise_std_mps=0.25)
    second = dynamics(noise_std_mps=0.25)
    third = dynamics(noise_std_mps=0.25)
    for model, seed in ((first, 11), (second, 11), (third, 12)):
        model.reset(seed=seed, initial_velocity=np.zeros((4, 3)))
    first_values = first.execute(command).executed_velocity
    second_values = second.execute(command).executed_velocity
    third_values = third.execute(command).executed_velocity
    np.testing.assert_allclose(first_values, second_values)
    assert not np.allclose(first_values, third_values)


def test_execution_aware_cbf_matches_kinematic_cbf_in_e0_and_inflates_under_delay() -> None:
    config = base_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.45)
    observation = env.reset(seed=681002)
    requested = np.tile(np.array([[2.0, 0.5, 0.0]]), (4, 1))
    e0 = DefenderExecutionDynamics(
        ExecutionDynamicsConfig(),
        defender_count=4,
        dt=env.dt,
        nominal_max_speed=float(env.agents["defender_max_speed"]),
        nominal_max_acceleration=float(env.agents["defender_max_acceleration"]),
    )
    e0.reset(seed=1, initial_velocity=env.defender_velocities)
    kinematic_actions, kinematic_diagnostics = make_kinematic_cbf(env).filter(requested, observation)
    aware_actions, aware_diagnostics = ExecutionAwarePursuitCBFSafetyFilter(env, e0).filter(requested, observation)
    np.testing.assert_allclose(kinematic_actions, aware_actions, atol=1e-12, rtol=0.0)
    assert aware_diagnostics.mean_execution_margin_m == pytest.approx(0.0)
    assert aware_diagnostics.action_correction_norm == pytest.approx(kinematic_diagnostics.action_correction_norm)

    delayed = dynamics(action_delay_steps=1, max_acceleration_scale=0.75, noise_std_mps=0.25)
    delayed.reset(seed=1, initial_velocity=env.defender_velocities)
    _actions, delayed_diagnostics = ExecutionAwarePursuitCBFSafetyFilter(env, delayed).filter(requested, observation)
    assert delayed_diagnostics.mean_execution_margin_m > 0.0
    assert delayed_diagnostics.max_execution_margin_m >= delayed_diagnostics.mean_execution_margin_m


def test_execution_protocol_is_complete_and_uses_disjoint_seed_blocks() -> None:
    document = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    profiles = document["execution_profiles"]
    assert set(profiles) == {f"E{index}" for index in range(7)}
    assert document["primary_profile"] == "E6"
    assert len(set(document["seed_blocks"].values())) == 3
    for profile in profiles.values():
        ExecutionDynamicsConfig.from_mapping(profile)
