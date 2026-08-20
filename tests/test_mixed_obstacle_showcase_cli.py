from __future__ import annotations

from pathlib import Path

from encirclement3d.showcase import load_central_capture_protocol
from scripts.evaluate_mixed_obstacle_showcase import load_locked_test_contract
from scripts.run_mixed_obstacle_showcase import build_showcase_scenario, transit_metrics_from_episode_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_s1_cross_is_the_main_opposite_side_encounter_task() -> None:
    scenario = build_showcase_scenario("s1_cross", 5.0)
    assert scenario.defender_positions[:, 0].max() < scenario.obstacle_zone_x[0]
    assert scenario.target_position[0] > scenario.obstacle_zone_x[1]
    assert scenario.target_crossing_required is True
    assert scenario.target_escape_direction[0] < 0.0


def test_s2_reverses_the_defender_side_without_forcing_target_crossing() -> None:
    scenario = build_showcase_scenario("s2", 5.0)
    assert scenario.defender_positions[:, 0].min() > scenario.obstacle_zone_x[1]
    assert scenario.target_position[0] < scenario.obstacle_zone_x[0]
    assert scenario.target_crossing_required is False
    assert scenario.target_escape_direction[0] < 0.0


def test_s2_cross_is_the_explicit_target_crossing_task() -> None:
    scenario = build_showcase_scenario("s2_cross", 5.0)
    assert scenario.target_crossing_required is True
    assert scenario.target_escape_direction[0] > 0.0


def test_single_obstacle_layout_can_be_selected_from_the_showcase_cli_helper() -> None:
    scenario = build_showcase_scenario("s1", 5.0, layout="wall")
    assert len(scenario.obstacles) == 1
    assert scenario.obstacles[0].shape == "wall"


def test_v4_s2_requires_and_uses_the_frozen_protocol() -> None:
    protocol = load_central_capture_protocol(PROJECT_ROOT / "configs" / "central_bidirectional_v4.yaml")
    scenario = build_showcase_scenario("v4_s2", 5.0, protocol=protocol)
    assert scenario.required_defender_zone_entries == 2
    assert scenario.target_crossing_required is False
    assert scenario.defender_positions[:, 0].max() == -protocol.initial_side_distance
    assert scenario.target_position[0] == protocol.initial_side_distance


def test_v4_locked_test_contract_is_explicit() -> None:
    seed, episodes = load_locked_test_contract(PROJECT_ROOT / "configs" / "central_bidirectional_v4.yaml")
    assert seed == 660501
    assert episodes == 100


def test_locked_metric_replay_restores_typed_transit_evidence() -> None:
    row = {
        "transit_grid_step_m": "0.5",
        "defender_transit_route_feasible": "[True, True, True, True]",
        "all_defenders_transit_route_feasible": "True",
        "target_transit_route_feasible": "True",
        "transit_route_feasible": "True",
        "defender_transit_route_length_m": "[10.0, 11.0, 12.0, 13.0]",
        "target_transit_route_length_m": "14.0",
        "defender_transit_min_clearance_m": "[0.6, 0.7, 0.8, 0.9]",
        "target_transit_min_clearance_m": "0.65",
        "defender_transit_goals": "[[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]",
        "target_transit_goal": "[-1.0, 0.0, 1.0]",
        "defender_transit_success": "[True, True, True, True]",
        "all_defenders_transit_success": "True",
        "target_transit_success": "False",
        "transit_success": "False",
        "defender_transit_steps": "[10, 11, 12, 13]",
        "target_transit_steps": "14",
        "defender_transit_reasons": "['completed', 'completed', 'completed', 'completed']",
        "target_transit_reason": "obstacle_clearance_violation",
        "defender_transit_execution_min_clearance_m": "[0.6, 0.7, 0.8, 0.9]",
        "target_transit_execution_min_clearance_m": "0.599",
    }
    restored = transit_metrics_from_episode_row(row)
    assert restored["defender_transit_route_feasible"] == [True, True, True, True]
    assert restored["target_transit_success"] is False
    assert restored["transit_success"] is False
    assert restored["target_transit_steps"] == 14
    assert restored["target_transit_execution_min_clearance_m"] == 0.599
