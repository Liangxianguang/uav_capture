"""Regression coverage for Stage 4B estimator-only aggregation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_stage4b_belief_baselines.py"


def load_module():
    specification = importlib.util.spec_from_file_location("stage4b_belief_baseline_test", SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_age_binned_error_excludes_uninitialized_beliefs() -> None:
    module = load_module()
    result = module.age_binned_error(
        errors=np.asarray([[9.0, 1.0], [3.0, 5.0]]),
        ages=np.asarray([[1, 1], [5, 3]]),
        timestamps=np.asarray([[-1, 0], [0, 0]]),
    )
    assert result["uninitialized_frames"] == 1
    assert result["initialized_frames"] == 3
    assert result["fresh_0_1_position_error_m"] == pytest.approx(1.0)
    assert result["moderate_2_4_position_error_m"] == pytest.approx(5.0)
    assert result["stale_5_plus_position_error_m"] == pytest.approx(3.0)


def test_summary_aggregates_episodes_by_condition_and_mode() -> None:
    module = load_module()
    base = {
        "condition": "delayed_measurements",
        "mode": "time_aligned",
        "mean_position_error_m_all": 2.0,
        "mean_velocity_error_mps_all": 1.0,
        "mean_position_error_m_initialized": 1.0,
        "mean_velocity_error_mps_initialized": 0.5,
        "mean_observation_age_steps": 3.0,
        "mean_new_timestamp_position_error_m": 0.4,
        "mean_reacquisition_to_update_steps": 3.0,
        "reacquisition_events_recovered": 2,
        "initialized_frames": 10,
        "uninitialized_frames": 2,
        "fresh_0_1_frames": 0,
        "fresh_0_1_position_error_m": None,
        "moderate_2_4_frames": 10,
        "moderate_2_4_position_error_m": 1.0,
        "stale_5_plus_frames": 0,
        "stale_5_plus_position_error_m": None,
    }
    alternate = {**base, "mean_position_error_m_initialized": 3.0, "reacquisition_events_recovered": 4}
    result = module.summarize([base, alternate])["delayed_measurements"]["time_aligned"]
    assert result["episodes"] == 2
    assert result["mean_position_error_m_initialized"] == pytest.approx(2.0)
    assert result["reacquisition_events_recovered"] == 6
    assert result["fresh_0_1_position_error_m"] is None
