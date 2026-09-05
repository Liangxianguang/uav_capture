from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_jepa_safe_capture_v21_candidate_separation import (
    _candidate_separation,
    _device_compare,
    _fixed_point_argmin,
    _sequence_stats,
    _v21_eligible,
)


def _ranking(*, eligible: list[bool], target_cost: list[float]) -> dict[str, object]:
    scores = target_cost[:]
    return {
        "valid_mask": [True] * 5,
        "eligible_mask": eligible,
        "scores": scores,
        "target_cost_m": target_cost,
    }


def test_candidate_separation_is_reconstructed_from_target_cost() -> None:
    ranking = _ranking(eligible=[True] * 5, target_cost=[0.0, 0.20, 1.0, 1.30, 9.0])
    np.testing.assert_allclose(_candidate_separation(ranking), [0.20, 0.20, 0.30, 0.30, 7.70])


def test_v21_gate_preserves_nominal_and_rejects_close_alternatives() -> None:
    ranking = _ranking(eligible=[True] * 5, target_cost=[0.0, 0.001, 0.10, 0.30, 0.50])
    eligible, separation = _v21_eligible(ranking, 0.002)
    assert bool(eligible[0])
    assert not bool(eligible[1])
    assert bool(eligible[2])
    assert separation[1] == pytest.approx(0.001)


def test_fixed_point_argmin_uses_index_as_deterministic_tie_break() -> None:
    ranking = _ranking(eligible=[True] * 5, target_cost=[1.0, 0.0, 0.004, 0.5, 0.6])
    assert _fixed_point_argmin(ranking, [1, 2], 0.004) == 1


def test_sequence_stats_reports_switches_and_longest_alternation() -> None:
    stats = _sequence_stats([0, 1, 0, 0, 2, 2])
    assert stats["count"] == 6
    assert stats["switch_count"] == 3
    assert stats["max_alternating_run"] == 3


def test_device_compare_includes_execution_and_cbf_fields() -> None:
    trace = {
        "episode_index": 0,
        "step": 1,
        "candidate_ranking": {
            "selected_index": 0,
            "eligible_mask": [True, False, False, False, False],
            "candidate_order": [0],
            "score_comparison_keys": [1, None, None, None, None],
            "scores": [0.0, 1.0, 2.0, 3.0, 4.0],
            "valid_mask": [True] * 5,
        },
        "cbf": {"verified_feasible": True, "solver_status": "success", "fallback_mode": "none"},
        "executed_action": [[0.0, 0.0, 0.0]],
        "raw_unverified_executed": False,
    }
    assert _device_compare({(0, 1): trace}, {(0, 1): json.loads(json.dumps(trace))})["decision_equal"]


def test_device_compare_rejects_cbf_or_action_mismatch() -> None:
    left = {
        "episode_index": 0,
        "step": 1,
        "candidate_ranking": {
            "selected_index": 0,
            "eligible_mask": [True] * 5,
            "candidate_order": [0],
            "score_comparison_keys": [1, 2, 3, 4, 5],
            "scores": [0.0] * 5,
            "valid_mask": [True] * 5,
        },
        "cbf": {"verified_feasible": True, "solver_status": "success", "fallback_mode": "none"},
        "executed_action": [[0.0, 0.0, 0.0]],
        "raw_unverified_executed": False,
    }
    right = json.loads(json.dumps(left))
    right["cbf"]["fallback_mode"] = "nominal"
    assert _device_compare({(0, 1): left}, {(0, 1): right})["decision_difference_count"] == 1
