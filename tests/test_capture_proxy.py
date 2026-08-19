from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from encirclement3d.capture import resolve_sphere_tetrahedral_contact, tetrahedral_cage_metrics
from encirclement3d.controllers import CaptureAwareTetrahedralSlotController
from encirclement3d.environment import Encirclement3DEnv, TETRAHEDRON_DIRECTIONS
from encirclement3d.flexible_net import FlexibleTetrahedralNet, resolve_sphere_flexible_net_contact


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def capture_config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["task"]["capture"] = {
        "enabled": True,
        "closure_slot_tolerance": 0.70,
        "minimum_face_clearance": 0.20,
        "minimum_edge_length": 3.60,
        "maximum_edge_length": 5.40,
        "maximum_relative_speed": 2.00,
        "hold_seconds": 0.20,
        "escape_tolerance": 0.05,
        "encirclement_reward_weight": 1.00,
        "feasibility_reward_weight": 0.75,
        "closure_bonus": 4.00,
        "success_bonus": 20.00,
        "escape_penalty": 10.00,
        "collision_penalty": 5.00,
    }
    return config


def rigid_contact_config() -> dict:
    config = capture_config()
    config["task"]["capture"].update(
        {
            "model": "rigid_contact",
            "target_radius": 0.25,
            "contact_tolerance": 0.001,
            "hold_seconds": 5.0,
        }
    )
    return config


def flexible_net_config() -> dict:
    config = rigid_contact_config()
    config["task"]["capture"].update(
        {
            "model": "flexible_net",
            # Keep geometry tests deterministic; gravity is tested separately
            # through the structural-limit case below.
            "net_gravity": 0.0,
            "net_max_tension": 50.0,
            "net_max_strain": 0.40,
            "target_mass": 0.03,
        }
    )
    return config


def position_valid_cage(env: Encirclement3DEnv) -> None:
    env.target_position = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    env.target_velocity.fill(0.0)
    env.defender_positions = env.target_position + TETRAHEDRON_DIRECTIONS * float(env.task["encirclement_radius"])
    env.defender_velocities.fill(0.0)


def test_tetrahedral_cage_geometry_distinguishes_inside_and_escape() -> None:
    corners = TETRAHEDRON_DIRECTIONS * 2.8
    centered = tetrahedral_cage_metrics(corners, np.zeros(3))
    assert centered.target_inside
    np.testing.assert_allclose(centered.min_face_clearance, 2.8 / 3.0, atol=1e-12)
    np.testing.assert_allclose(centered.min_edge_length, np.sqrt(8.0 / 3.0) * 2.8, atol=1e-12)
    assert not tetrahedral_cage_metrics(corners, TETRAHEDRON_DIRECTIONS[0] * 4.0).target_inside


def test_rigid_sphere_contact_stops_a_swept_escape_at_the_net_face() -> None:
    corners = TETRAHEDRON_DIRECTIONS * 2.8
    result = resolve_sphere_tetrahedral_contact(
        corners,
        corners,
        np.zeros(3),
        TETRAHEDRON_DIRECTIONS[0] * 4.0,
        radius=0.25,
        tolerance=1e-6,
    )
    resolved = tetrahedral_cage_metrics(corners, result.position)
    assert result.contact
    assert result.contact_face_index is not None
    assert result.contact_fraction is not None
    assert 0.0 < result.contact_fraction < 1.0
    assert result.contained
    assert resolved.min_face_clearance >= 0.25 - 2e-6


def test_rigid_contact_close_gate_includes_the_target_radius() -> None:
    env = Encirclement3DEnv(rigid_contact_config(), obstacle_count=0, target_speed_scale=0.0)
    env.reset(seed=15)
    env.target_position = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    env.target_velocity.fill(0.0)
    # The point target has 0.30 m face clearance, but the 0.25 m sphere
    # leaves only 0.05 m net margin and must not be closable with a 0.20 m
    # required margin.
    env.defender_positions = env.target_position + TETRAHEDRON_DIRECTIONS * 0.9
    env.defender_velocities.fill(0.0)
    np.testing.assert_allclose(
        tetrahedral_cage_metrics(env.defender_positions, env.target_position).min_face_clearance,
        0.3,
        atol=1e-12,
    )
    assert not env.capture_close_feasible()


def test_rigid_contact_environment_reports_net_contact_and_margin() -> None:
    env = Encirclement3DEnv(rigid_contact_config(), obstacle_count=0, target_speed_scale=0.0)
    env.reset(seed=16)
    position_valid_cage(env)
    _observation, _reward, _terminated, _truncated, info = env.step(np.zeros((4, 3)), close_cage=True)
    assert info["capture_closed"]

    # A high initial speed produces an outward swept crossing in the next
    # interval. The rigid contact proxy must retain the full target sphere.
    env.target_velocity = TETRAHEDRON_DIRECTIONS[0] * 40.0
    _observation, _reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert not terminated
    assert not truncated
    assert info["capture_net_contact"]
    assert info["capture_net_contact_face"] is not None
    assert info["cage_sphere_contained"]
    assert info["cage_net_margin"] >= -float(env.capture["contact_tolerance"])


def test_flexible_net_resolves_swept_escape_and_receives_contact_impulse() -> None:
    anchors = TETRAHEDRON_DIRECTIONS * 2.8
    net = FlexibleTetrahedralNet(
        anchors,
        node_mass=0.02,
        spring_stiffness=40.0,
        spring_damping=0.40,
        drag_coefficient=0.03,
        gravity=0.0,
        substeps=8,
    )
    previous = net.advance(anchors, 0.1)
    result = resolve_sphere_flexible_net_contact(
        previous,
        net.snapshot(),
        np.zeros(3),
        TETRAHEDRON_DIRECTIONS[0] * 4.0,
        radius=0.25,
        tolerance=1e-6,
    )
    assert result.contact
    assert result.contained
    assert result.panel_index is not None
    assert result.inward_normal is not None
    net.apply_contact_impulse(result.panel_index, result.inward_normal, impulse=0.25)
    metrics = net.metrics(result.position)
    assert metrics.min_face_clearance >= 0.25 - 2e-6
    assert metrics.peak_contact_impulse == 0.25
    assert np.isfinite(metrics.max_tension)
    assert np.isfinite(metrics.max_strain)


def test_refined_flexible_net_shares_edge_nodes_and_resolves_contact() -> None:
    anchors = TETRAHEDRON_DIRECTIONS * 2.8
    net = FlexibleTetrahedralNet(
        anchors,
        node_mass=0.02,
        spring_stiffness=40.0,
        spring_damping=0.40,
        drag_coefficient=0.03,
        gravity=0.0,
        substeps=8,
        face_subdivisions=3,
    )
    initial = net.snapshot()
    assert initial.vertices is not None
    assert initial.triangle_nodes is not None
    assert initial.triangle_panels is not None
    # Four anchors, twelve shared edge nodes, and four face-interior nodes.
    assert initial.vertices.shape == (20, 3)
    assert initial.triangle_nodes.shape == (36, 3)
    assert np.unique(initial.triangle_nodes).size == 20

    previous = net.advance(anchors, 0.1)
    result = resolve_sphere_flexible_net_contact(
        previous,
        net.snapshot(),
        np.zeros(3),
        TETRAHEDRON_DIRECTIONS[0] * 4.0,
        radius=0.25,
        tolerance=1e-6,
    )
    assert result.contact
    assert result.contained
    assert result.triangle_index is not None
    assert result.panel_index is not None
    assert result.inward_normal is not None
    velocity_before = net.mesh_velocities.copy()
    net.apply_contact_impulse(
        result.panel_index,
        result.inward_normal,
        impulse=0.25,
        triangle_index=result.triangle_index,
    )
    assert not np.allclose(net.mesh_velocities, velocity_before)
    assert net.metrics(result.position).min_face_clearance >= 0.25 - 2e-6


def test_refined_flexible_net_records_initial_pretension_as_structural_load() -> None:
    anchors = TETRAHEDRON_DIRECTIONS * 2.8
    net = FlexibleTetrahedralNet(
        anchors,
        node_mass=0.02,
        spring_stiffness=40.0,
        spring_damping=0.40,
        drag_coefficient=0.03,
        gravity=0.0,
        substeps=8,
        face_subdivisions=3,
        spring_pretension=0.35,
    )
    assert np.allclose(net.last_tensions, 0.35)
    net.advance(anchors, 0.1)
    assert net.metrics(np.zeros(3)).max_tension >= 0.35 - 1e-12


def test_flexible_net_environment_reports_contact_and_peak_impulse() -> None:
    env = Encirclement3DEnv(flexible_net_config(), obstacle_count=0, target_speed_scale=0.0)
    env.reset(seed=17)
    position_valid_cage(env)
    _observation, _reward, _terminated, _truncated, info = env.step(np.zeros((4, 3)), close_cage=True)
    assert info["capture_closed"]
    assert env.flexible_net is not None

    env.target_velocity = TETRAHEDRON_DIRECTIONS[0] * 40.0
    _observation, _reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert not terminated
    assert not truncated
    assert info["capture_net_contact"]
    assert info["capture_net_contact_face"] is not None
    assert info["capture_net_contact_panel"] is not None
    assert info["capture_peak_contact_impulse"] > 0.0
    assert info["cage_sphere_contained"]


def test_refined_flexible_net_environment_reports_contact_and_peak_impulse() -> None:
    config = flexible_net_config()
    config["task"]["capture"]["net_face_subdivisions"] = 3
    env = Encirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.0)
    env.reset(seed=19)
    position_valid_cage(env)
    _observation, _reward, _terminated, _truncated, info = env.step(np.zeros((4, 3)), close_cage=True)
    assert info["capture_closed"]
    assert env.flexible_net is not None
    assert env.flexible_net.face_subdivisions == 3

    env.target_velocity = TETRAHEDRON_DIRECTIONS[0] * 40.0
    _observation, _reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert not terminated
    assert not truncated
    assert info["capture_net_contact"]
    assert info["capture_net_contact_panel"] is not None
    assert info["capture_peak_contact_impulse"] > 0.0
    assert info["cage_sphere_contained"]


def test_flexible_net_tension_limit_is_an_irrevocable_structural_failure() -> None:
    config = flexible_net_config()
    config["task"]["capture"].update({"net_gravity": 9.81, "net_max_tension": 0.0})
    env = Encirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.0)
    env.reset(seed=18)
    position_valid_cage(env)
    env.step(np.zeros((4, 3)), close_cage=True)
    _observation, _reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert terminated
    assert not truncated
    assert info["capture_structural_failure"]
    assert info["capture_compression_event"]
    assert info["capture_peak_net_tension"] > 0.0


def test_capture_proxy_requires_close_then_hold_before_success() -> None:
    env = Encirclement3DEnv(capture_config(), obstacle_count=0, target_speed_scale=0.0)
    env.reset(seed=11)
    position_valid_cage(env)
    assert env.capture_close_feasible()

    _observation, reward, terminated, truncated, info = env.step(np.zeros((4, 3)), close_cage=True)
    assert not terminated
    assert not truncated
    assert info["capture_close_accepted"]
    assert info["capture_closed"]
    assert info["capture_closure_events"] == 1
    assert info["capture_hold_steps"] == 1
    assert not info["capture_success"]
    assert info["reward_components"]["capture_closure"] > 0.0

    _observation, _reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert terminated
    assert not truncated
    assert info["success"]
    assert info["capture_success"]
    assert info["capture_time_seconds"] == 0.2


def test_capture_proxy_escape_is_an_irrevocable_failure() -> None:
    env = Encirclement3DEnv(capture_config(), obstacle_count=0, target_speed_scale=0.0)
    env.reset(seed=12)
    position_valid_cage(env)
    _observation, _reward, _terminated, _truncated, info = env.step(np.zeros((4, 3)), close_cage=True)
    assert info["capture_closed"]

    env.target_position = np.array([0.0, 0.0, 5.0]) + TETRAHEDRON_DIRECTIONS[0] * 4.0
    env.target_velocity.fill(0.0)
    _observation, _reward, terminated, truncated, info = env.step(np.zeros((4, 3)))
    assert terminated
    assert not truncated
    assert info["failure"]
    assert info["capture_escaped"]
    assert info["capture_escape_event"]
    assert info["capture_escape_events"] == 1
    assert info["capture_closure_events"] == 1
    assert not info["capture_success"]


def test_capture_proxy_rejects_invalid_close_without_changing_legacy_rewards() -> None:
    config = capture_config()
    env = Encirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.0)
    env.reset(seed=13)
    # Reset positions are deliberately outside the closure-slot tolerance.
    _observation, _reward, terminated, truncated, info = env.step(np.zeros((4, 3)), close_cage=True)
    assert not terminated
    assert not truncated
    assert info["capture_close_requested"]
    assert not info["capture_close_accepted"]
    assert info["capture_close_rejected_steps"] == 1

    legacy = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    legacy_env = Encirclement3DEnv(legacy, obstacle_count=0, target_speed_scale=0.0)
    legacy_env.reset(seed=13)
    _observation, reward, _terminated, _truncated, legacy_info = legacy_env.step(np.zeros((4, 3)))
    np.testing.assert_allclose(reward, -legacy_info["mean_slot_error"])
    assert not legacy_info["capture_enabled"]


def test_capture_rule_controller_tracks_target_after_accepted_closure() -> None:
    env = Encirclement3DEnv(capture_config(), obstacle_count=0, target_speed_scale=0.0)
    env.reset(seed=14)
    position_valid_cage(env)
    controller = CaptureAwareTetrahedralSlotController(env)
    env.step(np.zeros((4, 3)), close_cage=True)
    assert env.capture_closed
    env.target_velocity = np.array([1.0, 0.0, 0.0])
    action = controller.act(env.observe())
    assert np.all(action[:, 0] > 0.0)
