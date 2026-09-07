from types import SimpleNamespace

import numpy as np
import pytest

from encirclement3d.dn_mpc import DNMPCConfig, DistributedMinimaxMPC


def _observation() -> dict[str, object]:
    return {
        "defender_positions": np.array([[0.0, -1.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64),
        "target_belief_positions": np.array([[5.0, 0.0, 1.0], [5.0, 0.0, 1.0]], dtype=np.float64),
        "target_belief_velocities": np.zeros((2, 3), dtype=np.float64),
        "target_observation_received": np.array([True, True]),
    }


def _candidate(route_id: str, action: np.ndarray, *, valid: bool = True, side: str = "nominal"):
    return SimpleNamespace(
        route_id=route_id,
        side=side,
        valid=valid,
        action_chunk=np.asarray(action, dtype=np.float64),
    )


def test_dn_mpc_selects_a_valid_minimax_route_and_exposes_local_costs() -> None:
    action = np.array(
        [[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    batch = SimpleNamespace(candidates=(_candidate("nominal:0", action),))
    decision = DistributedMinimaxMPC().plan(batch, _observation(), previous_action=np.zeros((2, 3)))
    assert decision.selected_index == 0
    assert decision.reason == "minimax_best_route"
    assert len(decision.local_agent_costs) == 1
    assert len(decision.local_agent_costs[0]) == 2
    assert len(decision.escape_hypotheses) == 6
    assert np.isfinite(decision.scores[0])


def test_dn_mpc_holds_current_route_until_switch_improvement_is_large_enough() -> None:
    old = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    new = np.array([[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float64)
    batch = SimpleNamespace(candidates=(_candidate("old:0", old), _candidate("new:0", new, side="right")))
    planner = DistributedMinimaxMPC(DNMPCConfig(minimum_hold_steps=2, switch_improvement_m=100.0))
    first = planner.plan(batch, _observation())
    assert first.selected_route_id == "new:0" or first.selected_route_id == "old:0"
    planner._route_id = "old:0"
    planner._route_age = 1
    held = planner.plan(batch, _observation())
    assert held.selected_route_id == "old:0"
    assert held.reason == "route_hold_minimum_age"


def test_dn_mpc_skips_invalid_candidates_and_reports_empty_batch() -> None:
    action = np.zeros((1, 2, 3), dtype=np.float64)
    batch = SimpleNamespace(candidates=(_candidate("bad:0", action, valid=False),))
    decision = DistributedMinimaxMPC().plan(batch, _observation())
    assert decision.selected_index is None
    assert decision.reason == "no_valid_candidate"
    with pytest.raises(ValueError):
        DistributedMinimaxMPC().plan(SimpleNamespace(candidates=()), _observation())


def test_dn_mpc_persists_tangent_side_during_hold_window() -> None:
    left = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    right = np.array([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=np.float64)
    left_candidate = _candidate("left:0", left, side="left")
    right_candidate = _candidate("right:0", right, side="right")
    left_candidate.label = "left_detour"
    right_candidate.label = "right_detour"
    candidates = (left_candidate, right_candidate)
    planner = DistributedMinimaxMPC(DNMPCConfig(tangent_route_hold_steps=3, minimum_hold_steps=0))
    first = planner.plan(SimpleNamespace(candidates=candidates), _observation())
    assert first.selected_side in {"left", "right"}
    planner._phase = "tangent_left" if first.selected_side == "left" else "tangent_right"
    planner._route_side = "left" if first.selected_side == "left" else "right"
    planner._route_id = f"{planner._route_side}:0"
    planner._tangent_age = 1
    held = planner.plan(SimpleNamespace(candidates=candidates), _observation())
    assert held.selected_side == planner._route_side
    assert held.reason == "tangent_side_hold"


def test_dn_mpc_honors_external_cbf_eligibility_mask() -> None:
    action = np.zeros((1, 2, 3), dtype=np.float64)
    candidates = SimpleNamespace(candidates=(_candidate("a:0", action), _candidate("b:0", action)))
    decision = DistributedMinimaxMPC().plan(candidates, _observation(), eligible_mask=[False, True])
    assert decision.selected_index == 1
    with pytest.raises(ValueError):
        DistributedMinimaxMPC().plan(candidates, _observation(), eligible_mask=[True])


def test_dn_mpc_prioritizes_boundary_rescue_over_route_hysteresis() -> None:
    observation = _observation()
    observation["defender_positions"] = np.array(
        [[8.2, -1.0, 1.0], [8.2, 1.0, 1.0]], dtype=np.float64
    )
    observation["world_lower"] = np.array([-10.0, -10.0, 0.5], dtype=np.float64)
    observation["world_upper"] = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    nominal_action = np.array([[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float64)
    rescue_action = np.array([[[-2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]], dtype=np.float64)
    nominal = _candidate("nominal:0", nominal_action)
    nominal.label = "nominal"
    rescue = _candidate("boundary_rescue:none", rescue_action, side="boundary_rescue")
    rescue.label = "boundary_rescue"
    planner = DistributedMinimaxMPC(
        DNMPCConfig(minimum_hold_steps=5, boundary_rescue_trigger_m=3.0)
    )
    planner._route_id = "nominal:0"
    planner._route_age = 1
    decision = planner.plan(
        SimpleNamespace(candidates=(nominal, rescue)),
        observation,
        previous_action=np.zeros((2, 3), dtype=np.float64),
    )
    assert decision.selected_route_id == "boundary_rescue:none"
    assert decision.reason == "boundary_rescue_priority"
