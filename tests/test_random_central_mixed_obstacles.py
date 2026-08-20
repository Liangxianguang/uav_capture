from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import random_central_mixed_obstacle_scenario, scenario_metadata
from scripts.evaluate_random_central_mixed_obstacles import (
    config_for_spec,
    episode_spec,
    load_protocol,
    resolved_episode_count,
    summarize_rows,
)
from encirclement3d.observation_encoding import policy_observations


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def showcase_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_showcase_mixed_curriculum.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_s3_random_layout_is_reproducible_and_contains_all_geometry() -> None:
    config = showcase_config()
    first_env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    second_env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    first = random_central_mixed_obstacle_scenario(first_env, layout_seed=1_645_001, defender_side="left")
    second = random_central_mixed_obstacle_scenario(second_env, layout_seed=1_645_001, defender_side="left")
    first_metadata = scenario_metadata(first)
    assert first_metadata == scenario_metadata(second)
    assert 3 <= len(first.obstacles) <= 5
    assert {obstacle.shape for obstacle in first.obstacles} == {"cylinder", "box", "wall"}
    low, high = first.obstacle_zone_x
    for obstacle in first.obstacles:
        half_x = obstacle.radius if obstacle.half_extents_xy is None else float(obstacle.half_extents_xy[0])
        assert low <= float(obstacle.center_xy[0]) - half_x
        assert float(obstacle.center_xy[0]) + half_x <= high


def test_s3_random_layout_supports_both_start_sides_and_axis_aligned_walls() -> None:
    config = showcase_config()
    for defender_side, seed in (("left", 1_645_011), ("right", 1_645_012)):
        env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
        scenario = random_central_mixed_obstacle_scenario(env, layout_seed=seed, defender_side=defender_side)
        if defender_side == "left":
            assert scenario.defender_positions[:, 0].max() < scenario.obstacle_zone_x[0]
        else:
            assert scenario.defender_positions[:, 0].min() > scenario.obstacle_zone_x[1]
        metadata = scenario_metadata(scenario)
        for obstacle in metadata["obstacles"]:
            if obstacle["shape"] == "wall":
                assert obstacle["orientation_degrees"] in {0.0, 90.0}


def test_s3_protocol_has_disjoint_reproducible_motion_and_layout_seed_blocks() -> None:
    protocol = load_protocol(PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_protocol.yaml")
    assert protocol["s3"]["target_crossing_required"] is False
    assert protocol["s3"]["required_defender_zone_entries"] == 2
    specs = {
        split: [episode_spec(protocol, split, index) for index in range(3)]
        for split in ("train", "validation", "locked_test")
    }
    assert specs["train"] == [episode_spec(protocol, "train", index) for index in range(3)]
    assert not any(spec["target_crossing_required"] for values in specs.values() for spec in values)
    episode_seeds = {spec["episode_seed"] for values in specs.values() for spec in values}
    layout_seeds = {spec["layout_seed"] for values in specs.values() for spec in values}
    assert len(episode_seeds) == 9
    assert len(layout_seeds) == 9
    assert episode_seeds.isdisjoint(layout_seeds)
    validation_specs = [episode_spec(protocol, "validation", index) for index in range(192)]
    assert {spec["defender_side"] for spec in validation_specs} == {"left", "right"}
    assert {spec["obstacle_count"] for spec in validation_specs} == {3, 4, 5}
    assert {spec["observation_condition"] for spec in validation_specs} == {"nominal", "delayed_noisy"}
    assert {spec["target_speed_scale"] for spec in validation_specs} == {0.45, 0.55}
    assert {spec["target_motion_mode"] for spec in validation_specs} == {"flee_persistence", "s_curve"}
    side_observation_counts = {}
    for spec in validation_specs:
        key = (spec["defender_side"], spec["observation_condition"])
        side_observation_counts[key] = side_observation_counts.get(key, 0) + 1
    assert side_observation_counts == {
        ("left", "nominal"): 48,
        ("left", "delayed_noisy"): 48,
        ("right", "nominal"): 48,
        ("right", "delayed_noisy"): 48,
    }
    assert resolved_episode_count(protocol, "locked_test", None) == 100
    assert resolved_episode_count(protocol, "locked_test", 100) == 100
    with pytest.raises(ValueError, match="exactly 100 episodes"):
        resolved_episode_count(protocol, "locked_test", 40)


def test_s3_v4_environment_override_preserves_shape_aware_actor_contract() -> None:
    protocol = load_protocol(PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_protocol.yaml")
    spec = episode_spec(protocol, "validation", 0)
    config = config_for_spec(
        "f2",
        spec,
        PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml",
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=float(spec["target_speed_scale"]))
    observation = env.reset(seed=int(spec["episode_seed"]))
    assert config["task"]["policy_obstacle_geometry"] == "shape_extents_and_type"
    assert policy_observations(env, observation).shape == (4, 63)


def test_s3_summary_groups_layout_and_observation_conditions() -> None:
    rows = [
        {
            "safe_capture_success": True,
            "safe_capture_in_pursuit": True,
            "capture_event": True,
            "showcase_success": True,
            "target_zone_entry_rate": 1.0,
            "defender_zone_entry_rate": 1.0,
            "defender_crossing_rate": 1.0,
            "all_defenders_crossed": True,
            "target_crossing_rate": 0.0,
            "transit_route_feasible": True,
            "transit_success": True,
            "collision": False,
            "world_violation_steps": 0,
            "min_clearance_m": 0.42,
            "capture_time_seconds": 4.0,
            "mean_visible_fraction": 0.8,
            "mean_observation_age_steps": 1.0,
            "mean_defender_path_length_m": 8.0,
            "total_defender_path_length_m": 32.0,
            "mean_cbf_action_correction_norm": 0.2,
            "max_cbf_action_correction_norm": 0.5,
            "termination_reason": "safe_capture",
            "defender_side": "left",
            "obstacle_count": 3,
            "layout_signature": "cylinder1+box1+wall1",
            "target_speed_scale": 0.55,
            "observation_condition": "nominal",
            "target_motion_mode": "flee_persistence",
        },
        {
            "safe_capture_success": False,
            "safe_capture_in_pursuit": False,
            "capture_event": False,
            "showcase_success": False,
            "target_zone_entry_rate": 0.0,
            "defender_zone_entry_rate": 0.5,
            "defender_crossing_rate": 0.5,
            "all_defenders_crossed": False,
            "target_crossing_rate": 0.0,
            "transit_route_feasible": True,
            "transit_success": True,
            "collision": False,
            "world_violation_steps": 0,
            "min_clearance_m": 0.31,
            "capture_time_seconds": None,
            "mean_visible_fraction": 0.5,
            "mean_observation_age_steps": 3.0,
            "mean_defender_path_length_m": 12.0,
            "total_defender_path_length_m": 48.0,
            "mean_cbf_action_correction_norm": 0.4,
            "max_cbf_action_correction_norm": 0.8,
            "termination_reason": "timeout",
            "defender_side": "right",
            "obstacle_count": 5,
            "layout_signature": "cylinder2+box1+wall2",
            "target_speed_scale": 0.45,
            "observation_condition": "delayed_noisy",
            "target_motion_mode": "s_curve",
        },
    ]
    summary = summarize_rows(rows)
    assert summary["overall"]["safe_capture_rate"] == 0.5
    assert summary["by_defender_side"]["left"]["episodes"] == 1
    assert summary["by_obstacle_count"]["5"]["episodes"] == 1
    assert summary["by_observation_condition"]["delayed_noisy"]["termination_reasons"] == {"timeout": 1}
    assert summary["overall"]["mean_defender_path_length_m"] == 10.0
    assert summary["overall"]["mean_cbf_action_correction_norm"] == pytest.approx(0.3)
    assert summary["overall"]["max_cbf_action_correction_norm"] == 0.8
