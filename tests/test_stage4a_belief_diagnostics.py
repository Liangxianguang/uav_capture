"""Regression coverage for deterministic Stage 4A failure diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_stage4a_belief_failures.py"


def load_module():
    specification = importlib.util.spec_from_file_location("stage4a_belief_diagnostic_test", SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def failed_row(method: str, training_seed: int, episode_seed: int) -> dict[str, str]:
    return {
        "method": method,
        "training_seed": str(training_seed),
        "seed": str(episode_seed),
        "safe_capture_success": "False",
    }


def test_round_robin_failure_selection_covers_groups_before_repeating() -> None:
    module = load_module()
    rows = [
        failed_row("d", 1, 12),
        failed_row("d", 1, 10),
        failed_row("e", 1, 20),
        failed_row("e", 2, 30),
        failed_row("e", 2, 31),
    ]
    selected = module.select_stratified_failures(rows, maximum=5)
    assert [(row["method"], int(row["training_seed"]), int(row["seed"])) for row in selected] == [
        ("d", 1, 10),
        ("e", 1, 20),
        ("e", 2, 30),
        ("d", 1, 12),
        ("e", 2, 31),
    ]


def test_diagnostic_metrics_bins_error_by_observation_age() -> None:
    module = load_module()
    frames = {
        "target_positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "target_velocities": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "belief_positions": [
            [[1.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
        ],
        "belief_velocities": [
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ],
        "observation_ages": [[0], [5]],
        "message_ages": [[0], [5]],
        "visible": [[False], [True]],
        "observation_timestamps": [[0], [1]],
        "covariance_traces": [[0.1], [0.2]],
    }
    arrays = {key: module.np.asarray(value) for key, value in frames.items()}
    metrics = module.diagnostic_metrics(arrays)
    assert metrics["mean_belief_position_error_m"] == pytest.approx(2.0)
    assert metrics["uninitialized_belief_frames"] == 0
    assert metrics["fresh_0_1_frames"] == 1
    assert metrics["fresh_0_1_position_error_m"] == pytest.approx(1.0)
    assert metrics["stale_5_plus_frames"] == 1
    assert metrics["stale_5_plus_position_error_m"] == pytest.approx(3.0)
    assert metrics["visibility_reacquisition_events"] == 1
    assert metrics["delivered_belief_updates"] == 1


def test_summary_uses_frame_counts_for_age_binned_error() -> None:
    module = load_module()
    rows = [
        {
            "condition": "delayed_measurements",
            "collision": True,
            "termination_reason": "safety_failure",
            "mean_belief_position_error_m": 1.0,
            "p95_belief_position_error_m": 1.2,
            "mean_belief_velocity_error_mps": 1.0,
            "mean_observation_age_steps_from_frames": 3.0,
            "max_observation_age_steps": 4,
            "visibility_reacquisition_events": 0,
            "delivered_belief_updates": 2,
            "uninitialized_belief_frames": 0,
            "uninitialized_belief_position_error_m": None,
            "fresh_0_1_frames": 1,
            "fresh_0_1_position_error_m": 1.0,
            "moderate_2_4_frames": 0,
            "moderate_2_4_position_error_m": None,
            "stale_5_plus_frames": 0,
            "stale_5_plus_position_error_m": None,
        },
        {
            "condition": "delayed_measurements",
            "collision": True,
            "termination_reason": "safety_failure",
            "mean_belief_position_error_m": 3.0,
            "p95_belief_position_error_m": 4.0,
            "mean_belief_velocity_error_mps": 2.0,
            "mean_observation_age_steps_from_frames": 5.0,
            "max_observation_age_steps": 6,
            "visibility_reacquisition_events": 1,
            "delivered_belief_updates": 3,
            "uninitialized_belief_frames": 0,
            "uninitialized_belief_position_error_m": None,
            "fresh_0_1_frames": 3,
            "fresh_0_1_position_error_m": 5.0,
            "moderate_2_4_frames": 0,
            "moderate_2_4_position_error_m": None,
            "stale_5_plus_frames": 0,
            "stale_5_plus_position_error_m": None,
        },
    ]
    summary = module.summarize_diagnostics(rows)["delayed_measurements"]
    assert summary["trajectories"] == 2
    assert summary["collision_failures"] == 2
    assert summary["fresh_0_1_position_error_m"] == pytest.approx(4.0)
    assert summary["fresh_0_1_frames"] == 4
    assert summary["uninitialized_belief_frames"] == 0
    assert summary["uninitialized_belief_position_error_m"] is None
