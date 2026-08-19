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
