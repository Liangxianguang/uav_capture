from __future__ import annotations

from scripts.run_mixed_obstacle_showcase import build_showcase_scenario


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
