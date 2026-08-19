from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from encirclement3d.controllers import HoldAwareTetrahedralSlotController, TetrahedralSlotController
from encirclement3d.dynamics import InertialEncirclement3DEnv
from encirclement3d.environment import CylinderObstacle, Encirclement3DEnv
from encirclement3d.safety import DiscreteTimeCBFSafetyFilter, PyBulletResponseCBFSafetyFilter
from encirclement3d.learning import ClosurePolicy, SharedActorCritic, defender_observations


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_reset_is_deterministic() -> None:
    config = load_config()
    env = Encirclement3DEnv(config, obstacle_count=3)
    first = env.reset(seed=42)
    second = env.reset(seed=42)
    assert (first["defender_positions"] == second["defender_positions"]).all()
    assert (first["target_position"] == second["target_position"]).all()
    assert len(first["obstacles"]) == 3


def test_rule_controller_runs_to_completion() -> None:
    config = load_config()
    env = Encirclement3DEnv(config, obstacle_count=3)
    observation = env.reset(seed=7)
    controller = TetrahedralSlotController(env)
    for _ in range(config["world"]["max_steps"]):
        observation, _reward, terminated, truncated, info = env.step(controller.act(observation))
        if terminated or truncated:
            break
    assert env.step_count > 0
    assert "min_clearance_so_far" in info
    assert terminated or truncated


def test_hold_aware_controller_gates_target_feedforward() -> None:
    config = load_config()
    config["dynamics"].update(
        {
            "backend": "pybullet",
            "hold_activation_error": 0.95,
            "hold_exit_error": 1.10,
            "hold_feedforward_scale": 0.50,
        }
    )
    env = Encirclement3DEnv(config, obstacle_count=0)
    observation = env.reset(seed=8)
    controller = HoldAwareTetrahedralSlotController(env)
    env.target_velocity = np.array([1.0, 0.0, 0.0])

    # The reset state is outside the activation radius, so no target-motion
    # term is present during approach.
    approach_action = controller.act(observation)
    assert not controller.hold_mode
    np.testing.assert_allclose(approach_action, TetrahedralSlotController(env).act(observation))

    # Once all slots are reached, the same controller enters hold mode and
    # contributes the configured target-velocity compensation.
    env.defender_positions = env.slot_positions.copy()
    hold_observation = env.observe()
    hold_action = controller.act(hold_observation)
    assert controller.hold_mode
    assert np.all(hold_action[:, 0] > 0.0)


def test_slot_geometry_is_three_dimensional() -> None:
    config = load_config()
    env = Encirclement3DEnv(config, obstacle_count=0)
    observation = env.reset(seed=5)
    slots = observation["slot_positions"]
    assert len({round(float(slot[2]), 5) for slot in slots}) > 1


def test_collision_cannot_be_reported_as_success() -> None:
    config = load_config()
    env = Encirclement3DEnv(config, obstacle_count=0)
    observation = env.reset(seed=13)
    env.defender_positions = observation["slot_positions"].copy()
    env.hold_steps = int(round(config["task"]["hold_seconds"] / config["world"]["dt"])) - 1

    # A past collision must make a later slot hold ineligible for success.
    env.collision_steps = 1
    _observation, _reward, terminated, truncated, info = env.step(
        np.zeros((4, 3)),
    )
    assert not info["success"]
    assert not terminated
    assert not truncated


def test_cbf_filter_prevents_a_near_obstacle_inward_velocity() -> None:
    config = load_config()
    env = Encirclement3DEnv(config, obstacle_count=0)
    env.reset(seed=3)
    env.target_position = np.array([-5.0, 0.0, 8.0])
    env.target_velocity = np.zeros(3)
    env.defender_positions = np.array(
        [
            [1.7, 0.0, 2.0],
            [0.0, 4.0, 7.0],
            [-4.0, -4.0, 7.0],
            [-4.0, 4.0, 2.0],
        ]
    )
    env.defender_velocities = np.zeros((4, 3))
    env.obstacles = [CylinderObstacle(center_xy=np.array([0.0, 0.0]), radius=1.0, height=5.0)]

    requested = np.zeros((4, 3))
    requested[0] = np.array([-5.0, 0.0, 0.0])
    safe_action, diagnostics = DiscreteTimeCBFSafetyFilter(env).filter(requested, env.observe())

    # h = 1.7 - 1.0 - 0.25 - 0.35 = 0.10, so the discrete CBF constraint is
    # v_x >= -gamma*h/dt = -0.25 m/s; the requested action would violate it.
    assert diagnostics.solver_success
    assert safe_action[0, 0] >= -0.2501
    assert diagnostics.action_correction_norm > 0.0


def test_response_cbf_uses_identified_displacement_response() -> None:
    config = load_config()
    config["dynamics"].update(
        {
            "pybullet_response_displacement_velocity_coefficient": 0.853,
            "pybullet_response_displacement_command_coefficient": 0.0946,
            "pybullet_response_displacement_error_p95": 0.0,
            "pybullet_response_cbf_uncertainty_multiplier": 0.0,
            "pybullet_response_cbf_gamma": 0.25,
            "pybullet_command_max_acceleration": 6.0,
        }
    )
    env = Encirclement3DEnv(config, obstacle_count=0)
    env.reset(seed=3)
    env.target_position = np.array([-5.0, 0.0, 8.0])
    env.target_velocity = np.zeros(3)
    env.defender_positions = np.array(
        [
            [1.61, 0.0, 2.0],
            [0.0, 4.0, 7.0],
            [-4.0, -4.0, 7.0],
            [-4.0, 4.0, 2.0],
        ]
    )
    env.defender_velocities = np.zeros((4, 3))
    env.filtered_defender_actions = np.zeros((4, 3))
    env.obstacles = [CylinderObstacle(center_xy=np.array([0.0, 0.0]), radius=1.0, height=5.0)]

    requested = np.zeros((4, 3))
    requested[0] = np.array([-5.0, 0.0, 0.0])
    safe_action, diagnostics = PyBulletResponseCBFSafetyFilter(env).filter(requested, env.observe())

    # h = 1.61 - 1.0 - 0.25 - 0.35 = 0.01. With a 0.1 s interval
    # and b=0.0946, h_next >= 0.75*h requires u_x >= -0.264 m/s.
    assert safe_action[0, 0] >= -0.265
    assert diagnostics.action_correction_norm > 0.0


def test_inertial_backend_is_deterministic_and_delayed() -> None:
    config = load_config()
    env = InertialEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.8)
    observation = env.reset(seed=23)
    initial_positions = observation["defender_positions"].copy()
    initial_masses = observation["defender_masses"].copy()
    initial_drag = observation["defender_drag_coefficients"].copy()
    action = np.full((4, 3), 4.0)

    for _ in range(config["dynamics"]["action_delay_steps"]):
        observation, _reward, terminated, truncated, _info = env.step(action)
        assert not terminated
        assert not truncated
        np.testing.assert_allclose(observation["defender_positions"], initial_positions)

    observation, _reward, _terminated, _truncated, _info = env.step(action)
    assert np.linalg.norm(observation["defender_positions"] - initial_positions) > 0.0

    repeated = InertialEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.8)
    repeated_observation = repeated.reset(seed=23)
    np.testing.assert_allclose(repeated_observation["defender_masses"], initial_masses)
    np.testing.assert_allclose(repeated_observation["defender_drag_coefficients"], initial_drag)


def test_shared_ppo_policy_has_four_bounded_actions() -> None:
    config = load_config()
    env = Encirclement3DEnv(config, obstacle_count=0)
    observations = defender_observations(env.reset(seed=31), env.n_defenders)
    policy = SharedActorCritic(observations.shape[-1])
    actions, log_probabilities, values = policy.sample_actions(
        torch.as_tensor(observations),
        action_scale=float(env.agents["defender_max_speed"]) / np.sqrt(3.0),
    )
    assert actions.shape == (4, 3)
    assert log_probabilities.shape == (4,)
    assert values.shape == (4,)
    assert float(actions.norm(dim=1).max().detach()) <= float(env.agents["defender_max_speed"]) + 1e-5


def test_shared_ppo_observations_are_finite_and_normalized() -> None:
    config = load_config()
    env = Encirclement3DEnv(config, obstacle_count=0)
    observations = defender_observations(
        env.reset(seed=37),
        env.n_defenders,
        position_scale=float(config["world"]["half_extent_xy"]),
        defender_speed_scale=float(config["agents"]["defender_max_speed"]),
        target_speed_scale=float(config["agents"]["target_max_speed"]),
    )
    assert observations.shape == (4, 21)
    assert np.isfinite(observations).all()
    assert float(np.abs(observations).max()) < 3.0


def test_shared_ppo_role_encoding_adds_one_hot_slot_identity() -> None:
    config = load_config()
    env = Encirclement3DEnv(config, obstacle_count=0)
    observations = defender_observations(env.reset(seed=38), env.n_defenders, include_agent_id=True)
    assert observations.shape == (4, 25)
    np.testing.assert_array_equal(observations[:, -4:], np.eye(4, dtype=np.float32))


def test_obstacle_encoder_has_fixed_nearest_first_features() -> None:
    config = load_config()
    env = Encirclement3DEnv(config, obstacle_count=3)
    observation = env.reset(seed=39)
    encoded = defender_observations(
        observation,
        env.n_defenders,
        position_scale=float(config["world"]["half_extent_xy"]),
        defender_speed_scale=float(config["agents"]["defender_max_speed"]),
        target_speed_scale=float(config["agents"]["target_max_speed"]),
        obstacle_feature_count=3,
    )

    assert encoded.shape == (4, 36)
    assert np.isfinite(encoded).all()
    # Three non-empty obstacles create three non-zero radius entries per agent.
    assert np.all(encoded[:, [24, 29, 34]] > 0.0)


def test_closure_policy_pools_one_global_command_from_all_defenders() -> None:
    policy = ClosurePolicy(observation_dim=7, hidden_dim=16)
    batched_logits = policy(torch.randn(3, 4, 7))
    local_logits = policy(torch.randn(3, 7))
    assert batched_logits.shape == (3,)
    assert local_logits.shape == (3,)
    with np.testing.assert_raises(ValueError):
        policy(torch.randn(3, 4, 2, 7))
