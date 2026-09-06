from __future__ import annotations

import pytest

from scripts.audit_jepa_safe_capture_v21_independent_cbf import (
    COUNTERFACTUAL_LABELS,
    _probe_map,
    classify_counterfactuals,
    constraint_category,
)


def _probe(label: str, accepted: bool) -> dict[str, object]:
    return {
        "label": label,
        "route_id": None,
        "accepted": accepted,
        "verified_feasible": accepted,
        "infeasible": not accepted,
        "timed_out": False,
        "fallback_mode": "none",
        "solver_status": "success" if accepted else "failure",
        "minimum_constraint_value": 0.1 if accepted else -0.1,
        "action_correction_norm": 0.0,
        "active_constraints": [],
        "requested_action": [[0.0, 0.0, 0.0]],
    }


def test_probe_map_requires_all_three_labels_but_accepts_reordered_input() -> None:
    probes = [_probe("safe_hold", False), _probe("selected", False), _probe("nominal", False)]
    mapped = _probe_map(probes)
    assert tuple(mapped) == ("safe_hold", "selected", "nominal")
    assert set(mapped) == set(COUNTERFACTUAL_LABELS)
    assert classify_counterfactuals(mapped) == "all_three_infeasible"


def test_classify_selected_only_infeasible() -> None:
    probes = {label: _probe(label, label != "selected") for label in COUNTERFACTUAL_LABELS}
    assert classify_counterfactuals(probes) == "selected_only_infeasible"


def test_classify_all_three_feasible_abort_is_inconsistent() -> None:
    probes = {label: _probe(label, True) for label in COUNTERFACTUAL_LABELS}
    assert classify_counterfactuals(probes) == "all_three_feasible_abort_inconsistent"


def test_probe_map_rejects_missing_or_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="exactly"):
        _probe_map([_probe("selected", False), _probe("nominal", False)])
    with pytest.raises(ValueError, match="label set"):
        _probe_map([_probe("selected", False), _probe("selected", False), _probe("safe_hold", False)])


def test_constraint_category_uses_first_negative_constraint_family() -> None:
    assert constraint_category("obstacle_0_defender_1") == "obstacle"
    assert constraint_category("pairwise_0_1") == "pairwise"
    assert constraint_category("boundary_lower_defender_0_axis_0") == "boundary"
    assert constraint_category(None) == "none"
