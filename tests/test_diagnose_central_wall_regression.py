from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "diagnose_central_wall_regression.py"
SPEC = importlib.util.spec_from_file_location("diagnose_central_wall_regression", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


class _Env:
    def __init__(self) -> None:
        self.agents = {"drone_radius": 0.2}
        self.n_defenders = 2
        self.defender_positions = np.asarray([[0.0, 0.0, 1.0], [0.3, 0.0, 1.0]])
        self.lower = np.asarray([-2.0, -2.0, 0.0])
        self.upper = np.asarray([2.0, 2.0, 3.0])
        self.obstacles = [SimpleNamespace()]

    def _obstacle_clearance(self, _position: np.ndarray, _obstacle: object) -> float:
        return 0.5


def test_safety_components_separate_obstacle_agent_and_boundary_clearance() -> None:
    values = DIAGNOSTIC.safety_components(_Env())

    assert values["obstacle_clearance_m"] == pytest.approx(0.3)
    assert values["inter_agent_clearance_m"] == pytest.approx(-0.1)
    assert values["boundary_clearance_m"] == pytest.approx(0.8)
    assert DIAGNOSTIC.collision_category(values, 0) == "defender_defender"


def test_boundary_violation_labels_identify_target_upper_x() -> None:
    labels = DIAGNOSTIC.boundary_violation_labels(
        np.asarray([9.95, 0.0, 4.0]),
        np.asarray([10.0, 0.0, 4.0]),
        np.asarray([-0.2, 0.0, 0.0]),
        np.asarray([-10.0, -10.0, 0.0]),
        np.asarray([10.0, 10.0, 8.0]),
        "target",
    )

    assert labels == ["target_x_upper"]


def test_render_report_preserves_diagnostic_only_boundary() -> None:
    summary = {
        "checkpoint_sha256": "a" * 64,
        "seed": 660514,
        "steps": 35,
        "termination_reason": "safety_failure",
        "collision_category": "defender_obstacle",
        "world_boundary_violation_events": [],
        "capture_event": False,
        "safe_capture_success": False,
        "minimum_obstacle_clearance_m": -0.1,
        "minimum_inter_agent_clearance_m": 0.2,
        "minimum_boundary_clearance_m": 0.5,
        "minimum_target_boundary_clearance_m": 0.0,
        "minimum_target_distance_m": 0.85,
        "minimum_target_distance_step": 31,
        "minimum_target_distance_defender_id": 2,
        "maximum_cbf_action_correction_norm": 0.4,
        "minimum_cbf_barrier_value": -0.2,
        "maximum_message_age_steps": 2,
        "maximum_observation_age_steps": 3,
    }

    report = DIAGNOSTIC.render_report(summary, {"steps": True}, "b" * 64)

    assert "not a new evaluation" in report
    assert "must not be used to add this evaluation seed" in report
