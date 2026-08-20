from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import (
    central_mixed_obstacle_scenario,
    crossing_metrics,
    prepare_showcase_episode,
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


def test_crossing_metrics_require_trajectory_entry_into_obstacle_zone() -> None:
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
    assert metrics["defender_crossing_rate"] == 1.0
    assert metrics["target_crossed"] is False
