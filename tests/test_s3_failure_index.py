from __future__ import annotations

import csv

from scripts.build_s3_failure_index import build_index, classify_failure, planned_route_clearance_band, render_report


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "episode_index": "0",
        "episode_seed": "646101",
        "layout_seed": "1646101",
        "method": "dynamic_encirclement",
        "use_cbf": "True",
        "observation_condition": "nominal",
        "obstacle_count": "3",
        "layout_signature": "cylinder1+box1+wall1",
        "defender_side": "left",
        "target_speed_scale": "0.45",
        "target_motion_mode": "flee_persistence",
        "cooperative_safe_capture": "True",
        "capture_event": "True",
        "safe_capture_success": "True",
        "collision": "False",
        "target_obstacle_collision": "False",
        "world_violation_steps": "0",
        "termination_reason": "safe_capture",
        "task_termination_reason": "safe_capture_in_pursuit",
        "min_clearance_m": "0.35",
        "capture_time_seconds": "4.0",
        "max_cbf_action_correction_norm": "0.10",
        "transit_success": "True",
        "defender_zone_entry_count": "4",
        "defender_transit_min_clearance_m": "[0.7, 0.8, 0.9, 1.0]",
        "target_transit_min_clearance_m": "0.75",
    }
    row.update(overrides)
    return row


def test_failure_stage_prioritizes_safety_over_timeout() -> None:
    assert classify_failure(_row()) == "cooperative_safe_capture"
    assert classify_failure(
        _row(
            cooperative_safe_capture="False",
            safe_capture_success="False",
            capture_event="False",
            termination_reason="timeout",
            task_termination_reason="timeout",
        )
    ) == "timeout"
    assert classify_failure(
        _row(
            cooperative_safe_capture="False",
            safe_capture_success="False",
            collision="True",
            termination_reason="timeout",
        )
    ) == "safety_failure"
    assert classify_failure(_row(capture_event="False", cooperative_safe_capture="False")) == "no_capture"


def test_build_index_keeps_episode_identity_and_groups(tmp_path) -> None:
    path = tmp_path / "episodes.csv"
    rows = [
        _row(),
        _row(
            episode_index="1",
            episode_seed="646102",
            layout_seed="1646102",
            cooperative_safe_capture="False",
            capture_event="False",
            safe_capture_success="False",
            termination_reason="timeout",
            task_termination_reason="timeout",
            min_clearance_m="0.20",
            max_cbf_action_correction_norm="0.40",
        ),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    index = build_index(path)

    assert index["episode_level"] is True
    assert index["summary"]["episodes"] == 2
    assert index["summary"]["cooperative_failures"] == 1
    assert index["summary"]["methods"] == ["dynamic_encirclement"]
    assert index["summary"]["cbf_modes"] == [True]
    assert index["summary"]["failure_stages"] == {"timeout": 1}
    assert index["groups"]["observation_condition"]["nominal"]["episodes"] == 2
    assert index["groups"]["planned_route_clearance_band"]["medium: planned clearance 0.65-0.80 m"]["episodes"] == 2
    assert index["failures"][0]["episode_seed"] == 646102
    assert set(index["failures"][0]["hard_example_flags"]) == {"task_failure", "low_clearance", "large_cbf_correction"}


def test_render_report_distinguishes_policy_artifacts(tmp_path) -> None:
    path = tmp_path / "episodes.csv"
    rows = [_row(method="f2", use_cbf="False"), _row(episode_index="1", method="f2", use_cbf="False")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = render_report(build_index(path))

    assert "Policy Artifact" in report
    assert "deployable learned policy" not in report
    assert "method(s) `f2`" in report


def test_planned_route_clearance_band_uses_the_tightest_recorded_route() -> None:
    assert planned_route_clearance_band(_row(defender_transit_min_clearance_m="[0.50, 0.70]")) == (
        "narrow: planned clearance <0.65 m"
    )
    assert planned_route_clearance_band(_row(defender_transit_min_clearance_m="[0.90, 0.85]", target_transit_min_clearance_m="0.82")) == (
        "wide: planned clearance >=0.80 m"
    )
