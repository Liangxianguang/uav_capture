from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from encirclement3d.pybullet_env import PyBulletEncirclement3DEnv
from encirclement3d.pybullet_softbody_net import (
    PyBulletSoftBodyConfig,
    attach_tetrahedral_softbody_anchors,
    load_tetrahedral_softbody,
    softbody_vertices,
    write_tetrahedral_softbody_obj,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def _softbody_capture_config(mesh_directory: Path) -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dynamics"].update(
        {
            "backend": "pybullet",
            "pybullet_physics": "pyb_drag",
            "pybullet_control_mode": "position_pid",
            "pybullet_frequency": 240,
            "pybullet_aggregate_steps": 24,
            "pybullet_speed_limit": 5.0,
            "pybullet_position_horizon": 0.15,
            "pybullet_deformable_world": True,
            "pybullet_softbody_net_enabled": True,
            "pybullet_softbody_net_face_subdivisions": 3,
            "pybullet_softbody_net_mass_kg": 0.08,
            "pybullet_softbody_net_elastic_stiffness": 40.0,
            "pybullet_softbody_net_damping_stiffness": 0.40,
            "pybullet_softbody_net_friction": 0.5,
            "pybullet_softbody_net_mesh_directory": str(mesh_directory),
            "pybullet_gui": False,
        }
    )
    config["task"]["capture"] = {
        "enabled": True,
        "model": "flexible_net",
        "target_radius": 0.25,
        "contact_tolerance": 0.005,
        "contact_projection_iterations": 96,
        "compression_tolerance": 0.02,
        "net_node_mass": 0.02,
        "net_spring_stiffness": 40.0,
        "net_spring_damping": 0.40,
        "net_drag_coefficient": 0.03,
        "net_gravity": 9.81,
        "net_substeps": 8,
        "net_face_subdivisions": 3,
        "net_spring_pretension": 0.0,
        "net_max_tension": 50.0,
        "net_max_strain": 0.40,
        "target_mass": 0.03,
        "closure_slot_tolerance": 0.70,
        "minimum_face_clearance": 0.15,
        "minimum_edge_length": 3.60,
        "maximum_edge_length": 5.40,
        "maximum_relative_speed": 1.25,
        "hold_seconds": 2.0,
        "escape_tolerance": 0.05,
        "encirclement_reward_weight": 1.0,
        "feasibility_reward_weight": 0.75,
        "closure_bonus": 4.0,
        "success_bonus": 20.0,
        "escape_penalty": 10.0,
        "collision_penalty": 5.0,
    }
    return config


def test_pybullet_environment_uses_velocity_mapping_and_returns_finite_state() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dynamics"].update(
        {
            "backend": "pybullet",
            "pybullet_physics": "pyb_drag",
            "pybullet_control_mode": "position_pid",
            "pybullet_frequency": 240,
            "pybullet_aggregate_steps": 24,
            "pybullet_speed_limit": 5.0,
            "pybullet_position_horizon": 0.5,
            "pybullet_position_velocity_feedforward": 1.0,
            "pybullet_boundary_reference_margin": [0.75, 0.75, 0.75],
            "pybullet_gui": False,
        }
    )
    env = PyBulletEncirclement3DEnv(config, obstacle_count=1, target_speed_scale=0.8)
    try:
        observation = env.reset(seed=47)
        mapped = env.velocity_actions_to_aviary(np.array([[3.0, 4.0, 0.0]] * 4))
        np.testing.assert_allclose(mapped["0"], np.array([0.6, 0.8, 0.0, 1.0]), atol=1e-6)
        observation, _reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
        assert not terminated
        assert not truncated
        assert len(env.obstacle_body_ids) == 1
        assert np.isfinite(observation["defender_positions"]).all()
        assert np.isfinite(observation["defender_velocities"]).all()
        assert info["pybullet_control_dt"] == config["world"]["dt"]
        assert info["pybullet_position_velocity_feedforward"] == 1.0
        assert info["boundary_governor_active_steps"] >= 0
    finally:
        env.close()


def test_pybullet_deformable_world_can_attach_a_softbody_net_to_aviary_drones(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dynamics"].update(
        {
            "backend": "pybullet",
            "pybullet_physics": "pyb_drag",
            "pybullet_control_mode": "position_pid",
            "pybullet_frequency": 240,
            "pybullet_aggregate_steps": 24,
            "pybullet_speed_limit": 5.0,
            "pybullet_position_horizon": 0.15,
            "pybullet_deformable_world": True,
            "pybullet_gui": False,
        }
    )
    env = PyBulletEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.0)
    try:
        env.reset(seed=48)
        assert env.aviary is not None
        assert env.pybullet is not None
        assert env.aviary.DEFORMABLE_WORLD
        mesh = write_tetrahedral_softbody_obj(
            tmp_path / "aviary_tetrahedral_net.obj", env.defender_positions, face_subdivisions=3
        )
        softbody = load_tetrahedral_softbody(
            env.pybullet,
            mesh,
            PyBulletSoftBodyConfig(
                mass_kg=0.08,
                spring_elastic_stiffness=40.0,
                spring_damping_stiffness=0.40,
            ),
            physics_client_id=env.aviary.CLIENT,
        )
        constraints = attach_tetrahedral_softbody_anchors(
            env.pybullet,
            softbody,
            env.aviary.DRONE_IDS,
            physics_client_id=env.aviary.CLIENT,
        )
        assert len(constraints) == 4
        np.testing.assert_allclose(
            softbody_vertices(env.pybullet, softbody, physics_client_id=env.aviary.CLIENT)[:4],
            env.defender_positions,
            atol=1e-8,
        )
    finally:
        env.close()


def test_pybullet_softbody_net_deploys_only_after_an_accepted_capture_closure(tmp_path: Path) -> None:
    config = _softbody_capture_config(tmp_path)
    env = PyBulletEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.0)
    try:
        env.reset(seed=49)
        assert env.aviary is not None
        assert env.pybullet is not None
        assert env.pybullet_softbody_net_id is None
        for drone_id, position in zip(env.aviary.DRONE_IDS, env.slot_positions, strict=True):
            env.pybullet.resetBasePositionAndOrientation(
                drone_id,
                position.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=env.aviary.CLIENT,
            )
            env.pybullet.resetBaseVelocity(
                drone_id,
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=env.aviary.CLIENT,
            )
        env._sync_defender_state()
        _observation, _reward, _terminated, _truncated, info = env.step(
            np.zeros((4, 3)), close_cage=True
        )
        assert info["capture_close_accepted"]
        assert info["pybullet_softbody_net_deployed"]
        assert info["pybullet_softbody_deployment_events"] == 1
        assert info["pybullet_softbody_anchor_constraints"] == 4
        assert info["pybullet_softbody_mesh_vertices"] == 20
        assert info["pybullet_softbody_mesh_triangles"] == 36
        assert Path(info["pybullet_softbody_mesh_path"]).is_file()
        assert info["pybullet_softbody_max_anchor_error"] < 1e-5
    finally:
        env.close()


def test_pybullet_softbody_net_requires_a_deformable_world(tmp_path: Path) -> None:
    config = _softbody_capture_config(tmp_path)
    config["dynamics"]["pybullet_deformable_world"] = False
    with pytest.raises(ValueError, match="requires pybullet_deformable_world"):
        PyBulletEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.0)


def test_vertical_recovery_holds_horizontal_reference_and_commands_climb() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dynamics"].update(
        {
            "pybullet_vertical_recovery_enabled": True,
            "pybullet_vertical_recovery_altitude": 6.0,
            "pybullet_vertical_recovery_descend_speed": 1.0,
            "pybullet_vertical_recovery_climb_height": 2.0,
        }
    )
    env = PyBulletEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.3)
    env.defender_positions = np.array(
        [[1.0, 2.0, 4.0], [2.0, 3.0, 7.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.5]]
    )
    env.defender_velocities = np.array(
        [[0.4, -0.3, -1.2], [0.0, 0.0, -2.0], [0.0, 0.0, -0.5], [0.0, 0.0, 0.0]]
    )
    references = env.defender_positions + 0.4
    target_velocities = np.ones((4, 3))

    active_agents = env._apply_vertical_recovery(references, target_velocities)

    assert active_agents == 1
    np.testing.assert_allclose(references[0, :2], env.defender_positions[0, :2])
    assert references[0, 2] >= env.defender_positions[0, 2] + 2.0
    np.testing.assert_allclose(target_velocities[0, :2], 0.0)
    assert target_velocities[0, 2] >= 0.0


def test_vertical_emergency_uses_braking_distance_and_overrides_reference() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dynamics"].update(
        {
            "pybullet_vertical_emergency_enabled": True,
            "pybullet_vertical_emergency_braking_deceleration": 12.0,
            "pybullet_vertical_emergency_reaction_time": 0.1,
            "pybullet_vertical_emergency_margin": 0.5,
            "pybullet_vertical_emergency_climb_height": 1.0,
        }
    )
    env = PyBulletEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.3)
    env.defender_positions = np.array(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 7.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.5]]
    )
    env.defender_velocities = np.array(
        [[0.4, -0.3, -7.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]
    )
    references = env.defender_positions + 0.4
    target_velocities = np.ones((4, 3))

    active_agents = env._apply_vertical_emergency(references, target_velocities)

    assert active_agents == 1
    np.testing.assert_allclose(references[0, :2], env.defender_positions[0, :2])
    assert references[0, 2] >= env.defender_positions[0, 2] + 1.0
    np.testing.assert_allclose(target_velocities[0, :2], 0.0)
    assert target_velocities[0, 2] >= 0.0
    assert env.last_vertical_emergency_required_distance[0] > 2.25
    assert not env.last_vertical_emergency_mask[1]


def test_attitude_recovery_preempts_horizontal_reference() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dynamics"].update(
        {
            "pybullet_attitude_recovery_enabled": True,
            "pybullet_attitude_recovery_max_tilt": 0.35,
            "pybullet_attitude_recovery_climb_height": 1.0,
        }
    )
    env = PyBulletEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.3)
    env.defender_positions = np.array(
        [[1.0, 2.0, 4.0], [2.0, 3.0, 7.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.5]]
    )
    env.aviary = SimpleNamespace(
        rpy=np.array([[0.4, 0.1, 0.0], [0.1, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    references = env.defender_positions + 0.4
    target_velocities = np.ones((4, 3))

    active_agents = env._apply_attitude_recovery(references, target_velocities)

    assert active_agents == 1
    np.testing.assert_allclose(references[0, :2], env.defender_positions[0, :2])
    assert references[0, 2] >= env.defender_positions[0, 2] + 1.0
    np.testing.assert_allclose(target_velocities[0, :2], 0.0)
    assert env.last_attitude_tilt[0] > 0.35


def test_command_filter_limits_the_first_velocity_increment() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dynamics"]["pybullet_command_max_acceleration"] = 2.0
    env = PyBulletEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.3)
    requested = np.array([[5.0, 0.0, 0.0]] * 4)

    executed = env._filter_defender_actions(requested)

    np.testing.assert_allclose(executed, np.array([[0.2, 0.0, 0.0]] * 4))
    np.testing.assert_allclose(env.last_requested_defender_actions, requested)
