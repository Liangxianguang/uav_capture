from __future__ import annotations

import pytest

from scripts.aggregate_central_v4_locked_test import (
    build_failure_analysis,
    build_transit_audit,
    summarize_rows,
)


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "cooperative_safe_capture": "True",
        "capture_event": "True",
        "target_zone_entered": "False",
        "transit_success": "True",
        "collision": "False",
        "world_violation_steps": "0",
        "defender_zone_entry_count": "4",
        "capture_time_seconds": "4.0",
        "min_clearance_m": "0.5",
        "mean_defender_path_length_m": "12.0",
        "total_defender_path_length_m": "48.0",
        "mean_cbf_action_correction_norm": "0.2",
        "max_cbf_action_correction_norm": "0.6",
        "termination_reason": "safe_capture",
        "task_termination_reason": "safe_capture_in_pursuit",
        "episode_index": "0",
        "episode_seed": "647001",
        "layout_seed": "1647001",
        "training_seed": "661201",
        "obstacle_count": "3",
        "layout_signature": "cylinder1+box1+wall1",
        "defender_side": "left",
        "observation_condition": "nominal",
        "target_speed_scale": "0.45",
        "target_motion_mode": "s_curve",
        "defender_transit_min_clearance_m": "[0.7, 0.8, 0.9, 1.0]",
        "target_transit_min_clearance_m": "0.75",
        "target_transit_success": "True",
        "target_transit_reason": "completed",
        "target_transit_execution_min_clearance_m": "0.75",
    }
    row.update(overrides)
    return row


def test_summarize_rows_reports_all_d3_metrics() -> None:
    summary = summarize_rows(
        [
            _row(),
            _row(
                cooperative_safe_capture="False",
                capture_event="False",
                target_zone_entered="True",
                transit_success="False",
                collision="True",
                world_violation_steps="1",
                defender_zone_entry_count="2",
                capture_time_seconds="",
                min_clearance_m="0.1",
                mean_defender_path_length_m="14.0",
                total_defender_path_length_m="56.0",
                mean_cbf_action_correction_norm="0.4",
                max_cbf_action_correction_norm="0.9",
                termination_reason="safety_failure",
            ),
        ]
    )
    assert summary["cooperative_safe_capture_rate"] == 0.5
    assert summary["capture_rate"] == 0.5
    assert summary["target_zone_entry_rate"] == 0.5
    assert summary["mean_pursuer_zone_entry_count"] == 3.0
    assert summary["transit_success_rate"] == 0.5
    assert summary["collision_rate"] == 0.5
    assert summary["boundary_violation_rate"] == 0.5
    assert summary["mean_time_to_capture_seconds"] == 4.0
    assert summary["mean_min_clearance_m"] == 0.3
    assert summary["worst_min_clearance_m"] == 0.1
    assert summary["mean_defender_path_length_m"] == 13.0
    assert summary["mean_cbf_action_correction_norm"] == pytest.approx(0.3)
    assert summary["max_cbf_action_correction_norm"] == 0.9


def test_failure_groups_and_transit_audit_keep_shared_scene_evidence() -> None:
    failed = _row(
        cooperative_safe_capture="False",
        capture_event="False",
        transit_success="False",
        target_transit_success="False",
        target_transit_reason="obstacle_clearance_violation",
        target_transit_execution_min_clearance_m="0.599",
        defender_transit_min_clearance_m="[0.6, 0.8, 0.9, 1.0]",
        termination_reason="timeout",
    )
    analysis = build_failure_analysis({"cbf": [failed]})
    narrow = analysis["cbf"]["planned_route_clearance_band"]["narrow: planned clearance <0.65 m"]
    assert narrow["cooperative_capture_failures"] == 1
    assert narrow["transit_failures"] == 1

    audit = build_transit_audit({"cbf": [failed], "raw": [dict(failed)]})
    assert audit["failed_scenario_evaluations"] == 2
    assert audit["unique_failed_locked_scenarios"] == 1
    assert audit["unique_scenarios"][0]["target_transit_reason"] == "obstacle_clearance_violation"
