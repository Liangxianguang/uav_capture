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
