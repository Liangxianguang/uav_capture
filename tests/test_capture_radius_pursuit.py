from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from encirclement3d.pursuit_controllers import DynamicEncirclementController
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_dev.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_capture_radius_is_the_terminal_success_event() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    observation = env.reset(seed=520101)
    env.defender_positions[0] = env.target_position + np.array([0.70, 0.0, 0.0])
    env.defender_velocities.fill(0.0)

    _observation, _reward, terminated, truncated, info = env.step(np.zeros((4, 3)))

    assert terminated
    assert not truncated
    assert info["capture_event"]
    assert info["safe_capture_success"]
    assert info["termination_reason"] == "safe_capture"
    assert info["capturing_defender_id"] == 0


def test_safety_failure_has_priority_over_capture_event() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=520102)
    env.defender_positions[0] = env.target_position.copy()
    env.defender_positions[1] = env.target_position.copy()
    env.defender_velocities.fill(0.0)

    _observation, _reward, terminated, _truncated, info = env.step(np.zeros((4, 3)))

    assert terminated
    assert info["capture_event"]
    assert not info["safe_capture_success"]
    assert info["termination_reason"] == "safety_failure"
    assert info["collision"]


def test_boundary_violation_has_priority_over_later_capture() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=520110)
    env.defender_positions[0] = env.target_position.copy()
    env.defender_positions[0, 0] = env.upper[0] + 1.0
    env.defender_velocities.fill(0.0)
    env._enforce_world_bounds(env.defender_positions, env.defender_velocities)
    env.defender_positions[0] = env.target_position.copy()

    _observation, _reward, terminated, _truncated, info = env.step(np.zeros((4, 3)))

    assert terminated
    assert info["capture_event"]
    assert not info["safe_capture_success"]
    assert info["world_violation_steps"] > 0
    assert info["termination_reason"] == "safety_failure"


def test_policy_observation_does_not_expose_target_truth() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.55)
    observation = env.reset(seed=520103)
    assert "target_position" not in observation
    assert "target_velocity" not in observation
    encoded = env.policy_observations(observation)
    assert encoded.shape == (4, 44)
    assert np.isfinite(encoded).all()


def test_partial_observation_is_deterministic_and_has_dropout() -> None:
    config = load_config()
    first = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.55)
    second = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.55)
    first_observation = first.reset(seed=520104)
    second_observation = second.reset(seed=520104)
    np.testing.assert_allclose(
        first_observation["target_belief_positions"],
        second_observation["target_belief_positions"],
    )
    np.testing.assert_array_equal(first_observation["target_visible"], second_observation["target_visible"])
    for _ in range(20):
        first_observation, *_ = first.step(np.zeros((4, 3)))
        second_observation, *_ = second.step(np.zeros((4, 3)))
    np.testing.assert_allclose(
        first_observation["target_belief_positions"],
        second_observation["target_belief_positions"],
    )
    assert float(np.mean(first_observation["target_visible"])) < 1.0


def test_dynamic_controller_uses_only_policy_safe_observation() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.55)
    observation = env.reset(seed=520105)
    controller = DynamicEncirclementController(env)
    action = controller.act(observation)
    assert action.shape == (4, 3)
    assert np.isfinite(action).all()
    assert float(np.linalg.norm(action, axis=1).max()) <= float(config["agents"]["defender_max_speed"]) + 1e-8


def test_environment_steps_after_torch_import() -> None:
    """The training process must not trigger a second OpenMP runtime."""
    assert torch.__version__
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.55)
    env.reset(seed=520106)
    _observation, reward, terminated, truncated, _info = env.step(np.zeros((4, 3)))
    assert np.isfinite(reward)
    assert not (terminated and truncated)


def test_prediction_features_are_partial_observation_only() -> None:
    config = load_config()
    config["task"]["pursuit"]["include_prediction_features"] = True
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.55)
    observation = env.reset(seed=520107)
    encoded = env.policy_observations(observation)
    assert encoded.shape == (4, 48)
    assert np.isfinite(encoded).all()

    # Prediction is derived from belief state. Altering hidden target truth
    # cannot change an already published decentralized observation.
    baseline_prediction = observation["target_prediction_positions"].copy()
    env.target_position += np.array([5.0, -4.0, 2.0])
    repeated = env.observe()
    np.testing.assert_allclose(repeated["target_prediction_positions"], baseline_prediction)


def test_uncertainty_fields_are_partial_observation_only_and_optional() -> None:
    config = load_config()
    config["task"]["pursuit"].update(
        {
            "include_prediction_features": True,
            "include_uncertainty_features": True,
            "observation_delay_steps": 2,
            "detection_dropout_probability": 0.0,
            "message_dropout_probability": 1.0 - 1e-6,
        }
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    observation = env.reset(seed=520111)
    assert observation["target_observation_confidence"].shape == (4,)
    assert observation["target_observation_timestamps"].shape == (4,)
    assert observation["target_observation_covariance"].shape == (4, 3, 3)
    encoded = env.policy_observations(observation)
    assert encoded.shape == (4, 52)
    published = {
        key: np.asarray(observation[key]).copy()
        for key in (
            "target_belief_positions",
            "target_belief_velocities",
            "target_observation_confidence",
            "target_observation_timestamps",
            "target_observation_covariance",
        )
    }
    env.target_position += np.array([4.0, -3.0, 1.0])
    env.target_velocity += np.array([-1.0, 0.5, 0.2])
    repeated = env.observe()
    for key, value in published.items():
        np.testing.assert_allclose(repeated[key], value)


def test_delayed_observation_updates_belief_only_at_configured_timestamp() -> None:
    config = load_config()
    config["task"]["pursuit"].update(
        {
            "observation_delay_steps": 3,
            "detection_dropout_probability": 0.0,
            "message_dropout_probability": 1.0 - 1e-6,
            "communication_link_dropout_probability": 1.0 - 1e-6,
        }
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    observation = env.reset(seed=520112)
    assert np.all(observation["target_observation_timestamps"] == -1)
    for _ in range(2):
        observation, *_ = env.step(np.zeros((4, 3)))
        assert np.all(observation["target_observation_timestamps"] == -1)
    observation, *_ = env.step(np.zeros((4, 3)))
    assert np.all(observation["target_observation_timestamps"] == 0)
    assert np.all(observation["target_observation_age_steps"] == 3)


def test_continuous_detection_loss_increases_age_and_covariance() -> None:
    config = load_config()
    config["task"]["pursuit"].update(
        {
            "detection_dropout_probability": 0.0,
            "detection_loss_burst_probability": 0.0,
            "observation_covariance_growth": 0.4,
        }
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=520113)
    env.detection_loss_burst_remaining[:] = 3
    initial_trace = np.trace(env.target_observation_covariance, axis1=1, axis2=2).copy()
    for _ in range(3):
        observation, *_ = env.step(np.zeros((4, 3)))
    final_trace = np.trace(observation["target_observation_covariance"], axis1=1, axis2=2)
    assert np.all(observation["target_observation_age_steps"] >= 3)
    assert np.all(final_trace > initial_trace)


def test_hard_benchmark_motion_modes_are_deterministic() -> None:
    config = load_config()
    config["task"]["pursuit"].update(
        {
            "target_motion_mode": "random_turn",
            "obstacle_profile": "mixed",
            "map_seed_offset": 100000,
        }
    )
    first = CaptureRadiusPursuit3DEnv(config, obstacle_count=4, target_speed_scale=0.75)
    second = CaptureRadiusPursuit3DEnv(config, obstacle_count=4, target_speed_scale=0.75)
    first.reset(seed=520108)
    second.reset(seed=520108)
    assert [item.shape for item in first.obstacles] == [item.shape for item in second.obstacles]
    np.testing.assert_allclose(
        np.asarray([item.center_xy for item in first.obstacles]),
        np.asarray([item.center_xy for item in second.obstacles]),
    )
    for _ in range(8):
        first.step(np.zeros((4, 3)))
        second.step(np.zeros((4, 3)))
    np.testing.assert_allclose(first.target_position, second.target_position)
    np.testing.assert_allclose(first.target_velocity, second.target_velocity)


def test_box_and_wall_obstacles_have_finite_local_observations() -> None:
    config = load_config()
    for profile in ("boxes", "walls", "narrow_channels"):
        config["task"]["pursuit"]["obstacle_profile"] = profile
        env = CaptureRadiusPursuit3DEnv(config, obstacle_count=4, target_speed_scale=0.75)
        observation = env.reset(seed=520109)
        assert all(item.shape in {"box", "wall"} for item in env.obstacles)
        encoded = env.policy_observations(observation)
        assert np.isfinite(encoded).all()
        _observation, reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
        assert np.isfinite(reward)
        assert np.isfinite(info["min_clearance_so_far"])
        assert not (terminated and truncated)


def test_hard_benchmark_preserves_frozen_actor_observation_dimension() -> None:
    benchmark = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_hard_benchmark.yaml").read_text(encoding="utf-8")
    )
    for experiment in benchmark["experiments"]:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        config["task"]["pursuit"].update(benchmark["task"]["pursuit"])
        config["task"]["pursuit"].update(experiment.get("pursuit_overrides", {}))
        env = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=int(experiment["obstacle_count"]),
            target_speed_scale=float(experiment["target_speed_scale"]),
        )
        observation = env.reset(seed=620001)
        assert env.policy_observations(observation).shape == (4, 44)
