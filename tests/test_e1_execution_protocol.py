from __future__ import annotations

from pathlib import Path

import pytest

from encirclement3d.e1_protocol import case_sha256, environment_config, episode_count, episode_spec, execution_config, load_e1_protocol
from encirclement3d.execution_evaluation import rollout_execution_expert
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import random_central_mixed_obstacle_scenario, scenario_metadata
from scripts.evaluate_e1_execution_dynamics import summarize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "e1_execution_dynamics_protocol.yaml"


def test_e1_protocol_generates_reproducible_and_disjoint_cases() -> None:
    protocol = load_e1_protocol(PROTOCOL_PATH)
    first = episode_spec(protocol, "development", 4)
    repeated = episode_spec(protocol, "development", 4)
    locked = episode_spec(protocol, "locked_test", 4)
    assert first == repeated
    assert first["episode_seed"] != locked["episode_seed"]
    assert first["layout_seed"] != locked["layout_seed"]
    assert first["execution_noise_seed"] != locked["execution_noise_seed"]
    assert episode_count(protocol, "smoke") == 10
    assert episode_count(protocol, "development") == 60
    assert episode_count(protocol, "locked_test") == 100
    with pytest.raises(ValueError, match="requires exactly"):
        episode_count(protocol, "locked_test", 99)


def test_e1_case_signature_and_execution_modes_are_pairable() -> None:
    protocol = load_e1_protocol(PROTOCOL_PATH)
    spec = episode_spec(protocol, "smoke", 0)
    config = environment_config(protocol, PROTOCOL_PATH, spec)
    prototype = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=float(spec["target_speed_scale"]))
    scenario = random_central_mixed_obstacle_scenario(
        prototype,
        layout_seed=int(spec["layout_seed"]),
        initial_side_distance=float(spec["initial_side_distance"]),
        defender_side=str(spec["defender_side"]),
        obstacle_count_range=(int(spec["obstacle_count"]), int(spec["obstacle_count"])),
        required_defender_zone_entries=int(spec["required_defender_zone_entries"]),
    )
    signature = case_sha256(spec, scenario_metadata(scenario))
    assert signature == case_sha256(spec, scenario_metadata(scenario))
    raw, _env, _wrapper = rollout_execution_expert(
        config,
        scenario,
        seed=int(spec["episode_seed"]),
        execution_config=execution_config(protocol, "E0"),
        execution_mode="raw",
    )
    aware, _env, _wrapper = rollout_execution_expert(
        config,
        scenario,
        seed=int(spec["episode_seed"]),
        execution_config=execution_config(protocol, "E6"),
        execution_mode="execution_aware_cbf",
    )
    assert raw["mean_command_execution_error_mps"] == pytest.approx(0.0)
    assert aware["mean_command_age_steps"] == pytest.approx(1.0)
    assert raw["required_defender_zone_entries"] == aware["required_defender_zone_entries"] == 2
    assert raw["capture_radius"] if "capture_radius" in raw else True


def test_e1_summary_keeps_execution_metrics_and_termination_counts() -> None:
    rows = [
        {
            "capture_event": True,
            "safe_capture_success": True,
            "cooperative_safe_capture": True,
            "collision": False,
            "world_violation_steps": 0,
            "transit_success": True,
            "capture_time_seconds": 2.0,
            "min_clearance_m": 0.4,
            "mean_defender_path_length_m": 3.0,
            "mean_command_execution_error_mps": 0.2,
            "p95_command_execution_error_mps": 0.3,
            "max_command_execution_error_mps": 0.4,
            "mean_command_age_steps": 1.0,
            "acceleration_saturation_rate": 0.1,
            "speed_saturation_rate": 0.0,
            "mean_cbf_action_correction_norm": 0.2,
            "max_cbf_action_correction_norm": 0.4,
            "termination_reason": "safe_capture",
        },
        {
            "capture_event": False,
            "safe_capture_success": False,
            "cooperative_safe_capture": False,
            "collision": True,
            "world_violation_steps": 1,
            "transit_success": True,
            "capture_time_seconds": None,
            "min_clearance_m": -0.1,
            "mean_defender_path_length_m": 2.0,
            "mean_command_execution_error_mps": 0.4,
            "p95_command_execution_error_mps": 0.5,
            "max_command_execution_error_mps": 0.6,
            "mean_command_age_steps": 1.0,
            "acceleration_saturation_rate": 0.3,
            "speed_saturation_rate": 0.1,
            "mean_cbf_action_correction_norm": 0.5,
            "max_cbf_action_correction_norm": 0.8,
            "termination_reason": "safety_failure",
        },
    ]
    result = summarize(rows)
    assert result["cooperative_safe_capture_rate"] == pytest.approx(0.5)
    assert result["boundary_violation_rate"] == pytest.approx(0.5)
    assert result["termination_reasons"] == {"safe_capture": 1, "safety_failure": 1}
    assert result["max_command_execution_error_mps"] == pytest.approx(0.6)

