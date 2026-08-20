from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import random_central_mixed_obstacle_scenario, scenario_metadata
from scripts.evaluate_random_central_mixed_obstacles import episode_spec, load_protocol, summarize_rows


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
    specs = {
        split: [episode_spec(protocol, split, index) for index in range(3)]
        for split in ("train", "validation", "locked_test")
    }
    assert specs["train"] == [episode_spec(protocol, "train", index) for index in range(3)]
    episode_seeds = {spec["episode_seed"] for values in specs.values() for spec in values}
    layout_seeds = {spec["layout_seed"] for values in specs.values() for spec in values}
    assert len(episode_seeds) == 9
    assert len(layout_seeds) == 9
    assert episode_seeds.isdisjoint(layout_seeds)
    validation_specs = [episode_spec(protocol, "validation", index) for index in range(6)]
    assert {spec["defender_side"] for spec in validation_specs} == {"left", "right"}
    assert {spec["obstacle_count"] for spec in validation_specs} == {3, 4, 5}
    assert {spec["observation_condition"] for spec in validation_specs} == {"nominal", "delayed_noisy"}


def test_s3_summary_groups_layout_and_observation_conditions() -> None:
    rows = [
        {
            "safe_capture_success": True,
            "capture_event": True,
            "showcase_success": True,
            "defender_crossing_rate": 1.0,
            "obstacle_crossing_success": True,
            "collision": False,
            "world_violation_steps": 0,
            "min_clearance_m": 0.42,
            "capture_time_seconds": 4.0,
            "mean_visible_fraction": 0.8,
            "mean_observation_age_steps": 1.0,
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
            "capture_event": False,
            "showcase_success": False,
            "defender_crossing_rate": 0.5,
            "obstacle_crossing_success": False,
            "collision": False,
            "world_violation_steps": 0,
            "min_clearance_m": 0.31,
            "capture_time_seconds": None,
            "mean_visible_fraction": 0.5,
            "mean_observation_age_steps": 3.0,
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
