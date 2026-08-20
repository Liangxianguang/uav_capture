from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import (
    capture_contract_metrics,
    central_mixed_obstacle_scenario,
    crossing_metrics,
    prepare_showcase_episode,
    sample_training_episode,
    transit_execution_metrics,
    transit_route_metrics,
    validate_showcase_scenario,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_time_aligned_uncertainty_dev.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_central_showcase_contains_all_required_obstacle_shapes() -> None:
    scenario = central_mixed_obstacle_scenario()
    assert {obstacle.shape for obstacle in scenario.obstacles} == {"cylinder", "box", "wall"}
    assert scenario.defender_positions[:, 0].max() < scenario.obstacle_zone_x[0]
    assert scenario.target_position[0] > scenario.obstacle_zone_x[1]


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
