from __future__ import annotations

import pytest
import numpy as np

from scripts.audit_dn_mpc_p36_route_utility import (
    _at_grid_boundary,
    _parse_nonnegative_grid,
    _parse_switch_grid,
)
from scripts.diagnose_dn_mpc_p44_utility_scale import _summarize
from scripts.audit_dn_mpc_p45_safety_first_hierarchy import _select_candidate
from scripts.audit_dn_mpc_p46_planner_distillation import _group_audit


def test_switch_grid_parser_sorts_and_deduplicates_values() -> None:
    assert _parse_switch_grid("0.5, 0, 1, 0.5") == (0.0, 0.5, 1.0)


def test_switch_grid_parser_rejects_negative_or_empty_values() -> None:
    with pytest.raises(ValueError):
        _parse_switch_grid("0.0,-0.1")
    with pytest.raises(ValueError):
        _parse_switch_grid("")


def test_generic_weight_grid_parser_has_named_errors() -> None:
    assert _parse_nonnegative_grid("2, 0, 1, 1", name="--length-grid") == (0.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="--length-grid"):
        _parse_nonnegative_grid("-1", name="--length-grid")
    with pytest.raises(ValueError, match="--cbf-grid"):
        _parse_nonnegative_grid("nan", name="--cbf-grid")


def test_grid_boundary_is_computed_against_the_active_grid() -> None:
    grid = {"length": (0.0, 1.0), "switch": (0.0, 0.5, 1.0)}
    assert _at_grid_boundary({"length": 1.0, "switch": 0.5}, grid) == {
        "length": True,
        "switch": False,
    }


def test_p44_summary_excludes_unknown_switch_rows_from_switch_statistics() -> None:
    records = [
        {"progress": 0.1, "route_length": 0.0, "escape": 0.2, "cbf": 1.0, "switch": 0.0},
        {"progress": 0.2, "route_length": 0.1, "escape": 0.3, "cbf": 1.0, "switch": 1.0},
        {"progress": 0.3, "route_length": 0.2, "escape": 0.4, "cbf": 1.0, "switch": float("nan")},
    ]
    report = _summarize(records)
    assert report["terms"]["switch"]["count"] == 2
    assert report["terms"]["switch_known_fraction"] == 2 / 3
    assert report["terms"]["switch_rate"] == 0.5


def test_p45_hierarchy_uses_route_length_only_inside_primary_tie_band() -> None:
    candidates = [
        {
            "candidate": 0,
            "eligible": True,
            "truth_progress": 1.0,
            "predicted_progress": 1.0,
            "truth_escape": 0.0,
            "predicted_escape": 0.0,
            "route_length": 2.0,
            "switch_penalty": 0.0,
        },
        {
            "candidate": 1,
            "eligible": True,
            "truth_progress": 0.95,
            "predicted_progress": 0.95,
            "truth_escape": 0.0,
            "predicted_escape": 0.0,
            "route_length": 0.1,
            "switch_penalty": 1.0,
        },
    ]
    scales = {"progress": 1.0, "escape": 1.0, "route_length": 1.0}
    assert _select_candidate(candidates, scales, predicted=True, escape_weight=0.0, tie_band=0.01) == 0
    assert _select_candidate(candidates, scales, predicted=True, escape_weight=0.0, tie_band=0.1) == 1


def test_p46_teacher_labels_preserve_ineligible_and_abstention_as_unknown() -> None:
    arrays = {
        "sample_type": np.zeros(6, dtype=np.float32),
        "scenario_index": np.zeros(6, dtype=np.int64),
        "time_index": np.array([0, 0, 0, 0, 1, 1], dtype=np.int64),
        "route_candidate_index": np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        "route_geometry_valid": np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
        "labels_cbf_feasible": np.ones((6, 5), dtype=np.float32),
        "planner_selected_candidate_index": np.zeros(6, dtype=np.int64),
        "previous_selected_candidate_index": np.full(6, -1, dtype=np.int64),
        "planner_route_switch_outcome": np.full(6, -1, dtype=np.int64),
    }
    arrays["planner_selected_candidate_index"][4:] = -1
    report, labels = _group_audit(arrays)
    assert report["invalid_planner_selection_groups"] == 0
    assert report["abstention_groups"] == 1
    assert labels["planner_teacher_label"].tolist() == [1, 1, -1, -1, -1, -1]
    assert labels["planner_eligible"].tolist() == [1, 1, 0, 0, 0, 0]
