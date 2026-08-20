from __future__ import annotations

from scripts.evaluate_mixed_obstacle_showcase import summarize


def test_showcase_summary_reports_crossing_and_capture_separately() -> None:
    rows = [
        {
            "safe_capture_success": True,
            "safe_capture_in_pursuit": True,
            "showcase_success": True,
            "target_zone_entry_rate": 1.0,
            "defender_zone_entry_rate": 1.0,
            "defender_crossing_rate": 1.0,
            "target_crossing_rate": 0.0,
            "transit_route_feasible": True,
            "transit_success": True,
            "collision": False,
            "world_violation_steps": 0,
            "min_clearance_m": 0.4,
            "target_min_obstacle_clearance_m": 1.1,
            "capture_time_seconds": 7.0,
            "mean_visible_fraction": 0.8,
            "mean_observation_age_steps": 1.0,
        },
        {
            "safe_capture_success": False,
            "safe_capture_in_pursuit": False,
            "showcase_success": False,
            "target_zone_entry_rate": 1.0,
            "defender_zone_entry_rate": 1.0,
            "defender_crossing_rate": 1.0,
            "target_crossing_rate": 0.0,
            "transit_route_feasible": True,
            "transit_success": True,
            "collision": False,
            "world_violation_steps": 0,
            "min_clearance_m": 0.35,
            "target_min_obstacle_clearance_m": 1.3,
            "capture_time_seconds": None,
            "mean_visible_fraction": 0.9,
            "mean_observation_age_steps": 2.0,
        },
    ]
    result = summarize(rows)
    assert result["safe_capture_rate"] == 0.5
    assert result["safe_capture_in_pursuit_rate"] == 0.5
    assert result["showcase_success_rate"] == 0.5
    assert result["target_zone_entry_rate"] == 1.0
    assert result["transit_route_feasible_rate"] == 1.0
    assert result["transit_success_rate"] == 1.0
    assert result["defender_obstacle_crossing_rate"] == 1.0
    assert result["boundary_violation_rate"] == 0.0
