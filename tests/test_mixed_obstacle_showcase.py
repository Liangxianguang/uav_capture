from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import (
    capture_contract_metrics,
    central_capture_v4_scenario,
    central_mixed_obstacle_scenario,
    crossing_metrics,
    load_central_capture_protocol,
    prepare_showcase_episode,
    sample_training_episode,
    transit_execution_metrics,
    transit_route_metrics,
    validate_showcase_scenario,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_v4_protocol():
    return load_central_capture_protocol(PROJECT_ROOT / "configs" / "central_bidirectional_v4.yaml")


def load_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_time_aligned_uncertainty_dev.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_v4_flee_curricula_match_the_frozen_sensor_and_motion_contract() -> None:
    protocol = load_v4_protocol()
    for path, section in (
        ("capture_radius_recurrent_behavior_cloning_central_v4_flee.yaml", "imitation"),
        ("capture_radius_recurrent_mappo_central_v4_flee.yaml", "training"),
    ):
        document = yaml.safe_load((PROJECT_ROOT / "configs" / path).read_text(encoding="utf-8"))
        environment_path = PROJECT_ROOT / "configs" / document["environment_config"]
        environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
        settings = document[section]
        assert settings["action_scale_mode"] == "full_range"
        assert environment["task"]["pursuit"]["detection_range"] == protocol.detection_range
        assert environment["task"]["pursuit"]["obstacle_profile"] == "mixed"
        for stage in settings["training_showcase_stages"]:
            assert stage["target_crossing_probability"] == 0.0
            assert stage["target_motion_modes"] == ["flee_persistence"]
            assert stage["defender_sides"] == ["left"]
            assert stage["initial_side_distances"] == [protocol.initial_side_distance]
            assert stage["required_defender_zone_entries"] == protocol.required_defender_zone_entries


def test_central_showcase_contains_all_required_obstacle_shapes() -> None:
    scenario = central_mixed_obstacle_scenario()
    assert {obstacle.shape for obstacle in scenario.obstacles} == {"cylinder", "box", "wall"}
    assert scenario.defender_positions[:, 0].max() < scenario.obstacle_zone_x[0]
    assert scenario.target_position[0] > scenario.obstacle_zone_x[1]


def test_single_obstacle_layouts_keep_the_geometry_and_routes_explicit() -> None:
    config = load_config()
    for layout, shape in (("cylinder", "cylinder"), ("box", "box"), ("wall", "wall")):
        scenario = central_mixed_obstacle_scenario(layout=layout)
        env = CaptureRadiusPursuit3DEnv(config, obstacle_count=len(scenario.obstacles), target_speed_scale=0.55)
        validate_showcase_scenario(env, scenario)
        assert len(scenario.obstacles) == 1
        assert scenario.obstacles[0].shape == shape


def test_central_showcase_is_inside_bounds_and_reachable() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.55)
    scenario = central_mixed_obstacle_scenario()
    validate_showcase_scenario(env, scenario)
    observation = prepare_showcase_episode(env, scenario, seed=642002)
    assert observation["defender_positions"].shape == (4, 3)
    assert len(observation["obstacles"]) == 3
    assert all(item["shape"] in {"cylinder", "box", "wall"} for item in observation["obstacles"])
    assert np.all(env.defender_positions >= env.lower)
    assert np.all(env.defender_positions <= env.upper)
    transit = transit_route_metrics(env, scenario)
    assert transit["transit_route_feasible"] is True
    assert transit["all_defenders_transit_route_feasible"] is True
    assert transit["target_transit_route_feasible"] is True
    assert len(transit["defender_transit_route_length_m"]) == env.n_defenders
    executed = transit_execution_metrics(env, scenario)
    assert executed["transit_success"] is True
    assert executed["all_defenders_transit_success"] is True
    assert executed["target_transit_success"] is True


def test_v4_fixed_scene_freezes_opposite_sides_central_obstacles_and_separation() -> None:
    protocol = load_v4_protocol()
    scenario = central_capture_v4_scenario(protocol)
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=protocol.target_speed_scale)
    validate_showcase_scenario(env, scenario)
    assert scenario.required_defender_zone_entries == 2
    assert scenario.target_crossing_required is False
    assert scenario.require_target_zone_entry is False
    assert scenario.defender_positions[:, 0].max() < protocol.obstacle_zone_x[0]
    assert scenario.target_position[0] > protocol.obstacle_zone_x[1]
    assert np.min(np.linalg.norm(scenario.defender_positions - scenario.target_position[None, :], axis=1)) >= (
        protocol.minimum_initial_target_defender_distance
    )
    for obstacle in scenario.obstacles:
        half_x = obstacle.radius if obstacle.half_extents_xy is None else obstacle.half_extents_xy[0]
        assert obstacle.center_xy[0] - half_x >= protocol.obstacle_zone_x[0]
        assert obstacle.center_xy[0] + half_x <= protocol.obstacle_zone_x[1]


def test_s2_reverses_sides_and_requires_target_crossing() -> None:
    scenario = central_mixed_obstacle_scenario(
        target_crossing_required=True,
        defender_side="right",
    )
    assert scenario.target_crossing_required is True
    assert scenario.defender_positions[:, 0].min() > scenario.obstacle_zone_x[1]
    assert scenario.target_position[0] < scenario.obstacle_zone_x[0]
    assert scenario.target_escape_direction[0] > 0.0


def test_reversed_defender_side_keeps_an_evading_target_by_default() -> None:
    scenario = central_mixed_obstacle_scenario(defender_side="right")
    assert scenario.target_crossing_required is False
    assert scenario.target_escape_direction[0] < 0.0


def test_crossing_metrics_distinguish_zone_entry_from_opposite_side_completion() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    scenario = central_mixed_obstacle_scenario()
    prepare_showcase_episode(env, scenario, seed=642003)
    env.history = [
        {
            "defender_positions": scenario.defender_positions.copy(),
            "target_position": scenario.target_position.copy(),
        },
        {
            "defender_positions": scenario.defender_positions.copy() + np.array([4.0, 0.0, 0.0]),
            "target_position": scenario.target_position.copy(),
        },
    ]
    metrics = crossing_metrics(env, scenario.obstacle_zone_x)
    assert metrics["defender_zone_entry_rate"] == 1.0
    assert metrics["defender_crossing_rate"] == 0.0
    assert metrics["target_crossed"] is False

    env.history.append(
        {
            "defender_positions": scenario.defender_positions.copy() + np.array([9.0, 0.0, 0.0]),
            "target_position": scenario.target_position.copy(),
        }
    )
    completed = crossing_metrics(env, scenario.obstacle_zone_x)
    assert completed["defender_crossing_rate"] == 1.0
    assert completed["all_defenders_crossed"] is True
    assert completed["any_defender_zone_entered"] is True


def test_capture_contract_requires_a_central_encounter() -> None:
    crossing = {
        "target_zone_entered": True,
        "any_defender_zone_entered": True,
    }
    captured = capture_contract_metrics(
        {"capture_event": True, "safe_capture_success": True, "termination_reason": "safe_capture"},
        crossing,
    )
    assert captured["safe_capture_in_pursuit"] is True
    assert captured["task_termination_reason"] == "safe_capture_in_pursuit"

    outside_capture = capture_contract_metrics(
        {"capture_event": True, "safe_capture_success": True, "termination_reason": "safe_capture"},
        {"target_zone_entered": False, "any_defender_zone_entered": True},
        target_crossing_required=True,
    )
    assert outside_capture["safe_capture_in_pursuit"] is False
    assert outside_capture["capture_without_zone_entry"] is True
    assert outside_capture["task_termination_reason"] == "capture_without_zone_entry"

    ordinary_pursuit = capture_contract_metrics(
        {"capture_event": True, "safe_capture_success": True, "termination_reason": "safe_capture"},
        {"target_zone_entered": False, "any_defender_zone_entered": True},
        target_crossing_required=False,
    )
    assert ordinary_pursuit["safe_capture_in_pursuit"] is True


def test_capture_contract_can_require_target_entry_and_v4_requires_two_defenders_only() -> None:
    final_info = {"capture_event": True, "safe_capture_success": True, "termination_reason": "safe_capture"}
    one_defender = capture_contract_metrics(
        final_info,
        {"target_zone_entered": True, "defender_zone_entered": [True, False, False, False]},
        required_defender_zone_entries=2,
        require_target_zone_entry=True,
    )
    assert one_defender["cooperative_safe_capture"] is False
    assert one_defender["capture_without_zone_entry"] is True
    assert one_defender["defender_zone_entry_count"] == 1

    target_absent = capture_contract_metrics(
        final_info,
        {"target_zone_entered": False, "defender_zone_entered": [True, True, False, False]},
        required_defender_zone_entries=2,
        require_target_zone_entry=True,
    )
    assert target_absent["cooperative_safe_capture"] is False
    assert target_absent["capture_without_zone_entry"] is True

    cooperative = capture_contract_metrics(
        final_info,
        {"target_zone_entered": True, "defender_zone_entered": [True, True, False, False]},
        required_defender_zone_entries=2,
        require_target_zone_entry=True,
    )
    assert cooperative["cooperative_safe_capture"] is True
    assert cooperative["required_defender_zone_entries"] == 2

    v4_cooperative = capture_contract_metrics(
        final_info,
        {"target_zone_entered": False, "defender_zone_entered": [True, True, False, False]},
        required_defender_zone_entries=2,
        require_target_zone_entry=False,
    )
    assert v4_cooperative["cooperative_safe_capture"] is True
    assert v4_cooperative["target_zone_entry_required"] is False


def test_curriculum_sampler_can_select_a_central_mixed_episode() -> None:
    config = load_config()
    config["task"]["pursuit"]["detection_range"] = 14.0
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    settings = {
        "training_obstacle_counts": [0, 3],
        "training_target_speed_scales": [0.55],
        "training_showcase_stages": [
            {
                "until_progress": 1.0,
                "showcase_probability": 1.0,
                "layouts": ["mixed"],
                "initial_side_distances": [5.0],
                "target_speed_scales": [0.55],
            }
        ],
    }
    observation, metadata = sample_training_episode(env, settings, np.random.default_rng(123), seed=642004, progress=0.7)
    assert metadata["episode_kind"] == "showcase"
    assert metadata["layout"] == "mixed"
    assert metadata["defender_side"] == "left"
    assert metadata["target_crossing_required"] is False
    assert metadata["obstacle_count"] == 3
    assert env.obstacle_count == 3
    assert observation["defender_positions"][:, 0].max() < -2.5


def test_curriculum_random_episode_has_showcase_compatible_metadata() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    settings = {
        "training_obstacle_counts": [0],
        "training_target_speed_scales": [0.55],
        "training_showcase_stages": [
            {
                "until_progress": 1.0,
                "showcase_probability": 0.0,
            }
        ],
    }
    _observation, metadata = sample_training_episode(
        env, settings, np.random.default_rng(321), seed=642005, progress=0.7
    )
    assert metadata["episode_kind"] == "random"
    assert metadata["defender_side"] is None
    assert metadata["target_crossing_required"] is None
    assert metadata["obstacle_count"] == 0


def test_curriculum_sampler_can_select_randomized_central_episode() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    settings = {
        "training_obstacle_counts": [0],
        "training_target_speed_scales": [0.55],
        "training_showcase_stages": [
            {
                "until_progress": 1.0,
                "showcase_probability": 0.0,
                "randomized_central_probability": 1.0,
                "initial_side_distances": [6.0],
                "defender_sides": ["right"],
                "target_speed_scales": [0.55],
                "target_motion_modes": ["s_curve"],
                "randomized_obstacle_count_range": [3, 5],
            }
        ],
    }
    _observation, metadata = sample_training_episode(
        env, settings, np.random.default_rng(99), seed=642006, progress=0.7
    )
    assert metadata["episode_kind"] == "randomized_showcase"
    assert metadata["layout"] == "random_mixed"
    assert metadata["layout_seed"] is not None
    assert metadata["defender_side"] == "right"
    assert 3 <= metadata["obstacle_count"] <= 5
    assert metadata["target_motion_mode"] == "s_curve"


def test_crossing_curriculum_uses_the_frozen_target_motion_profile() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    settings = {
        "training_obstacle_counts": [0],
        "training_target_speed_scales": [0.55],
        "training_showcase_stages": [
            {
                "until_progress": 1.0,
                "showcase_probability": 1.0,
                "layouts": ["mixed"],
                "initial_side_distances": [5.0],
                "defender_sides": ["left"],
                "target_speed_scales": [0.55],
                "target_crossing_probability": 1.0,
                "target_crossing_speed_scales": [0.90],
                "target_motion_modes": ["flee_persistence"],
            }
        ],
    }
    _observation, metadata = sample_training_episode(
        env, settings, np.random.default_rng(777), seed=642007, progress=0.75
    )
    assert metadata["target_crossing_required"] is True
    assert metadata["target_speed_scale"] == 0.90
    assert env.pursuit["target_heading_persistence"] == 4.0
    assert env.pursuit["target_flee_gain"] == 0.05
