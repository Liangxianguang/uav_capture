"""Regression coverage for validation-only velocity-gate selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "select_stage4b_velocity_gate.py"


def load_module():
    specification = importlib.util.spec_from_file_location("stage4b_velocity_gate_selection_test", SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_selection_requires_packet_arrival_constraint_before_minimizing_error() -> None:
    module = load_module()
    summary = {}
    for condition in ("delayed_measurements", "burst_occlusion"):
        summary[condition] = {
            "aligned_decay_0_00": {
                "mean_position_error_m_initialized": 0.1,
                "mean_new_timestamp_position_error_m": 2.0,
            },
            "aligned_decay_0_50": {
                "mean_position_error_m_initialized": 0.5,
                "mean_new_timestamp_position_error_m": 1.0,
            },
            "aligned_decay_0_80": {
                "mean_position_error_m_initialized": 0.6,
                "mean_new_timestamp_position_error_m": 1.05,
            },
            "aligned_decay_1_00": {
                "mean_position_error_m_initialized": 0.7,
                "mean_new_timestamp_position_error_m": 1.0,
            },
        }
    selected, details = module.select_candidate(summary, list(summary), maximum_update_error_regression=0.10)
    assert selected == "aligned_decay_0_50"
    zero = next(row for row in details["candidates"] if row["candidate"] == "aligned_decay_0_00")
    assert not zero["eligible"]
