from __future__ import annotations

from scripts.audit_dn_mpc_p31_route_utility_label import _evaluate, _utility


def _candidate(candidate: int, *, eligible: bool = True, progress: float = 0.1) -> dict[str, float | int | bool]:
    return {
        "candidate": candidate,
        "eligible": eligible,
        "truth_progress": progress,
        "predicted_progress": progress,
        "truth_escape": 0.0,
        "predicted_escape": 0.0,
        "truth_cbf": 1.0,
        "predicted_cbf": 1.0,
        "route_length": float(candidate),
    }


def test_public_route_length_penalizes_longer_equal_progress_route() -> None:
    short = _candidate(0, progress=0.1)
    long = _candidate(1, progress=0.1)
    weights = {"length": 0.3, "escape": 0.0, "cbf": 0.0}
    assert _utility(short, weights, predicted=False) > _utility(long, weights, predicted=False)


def test_evaluation_excludes_ineligible_candidates_and_keeps_selected_trace() -> None:
    groups = [{"scenario_index": 0, "time_index": 1, "selected": 0, "candidates": [_candidate(0), _candidate(1, eligible=False)]}]
    result = _evaluate(groups, {"length": 0.0, "escape": 0.0, "cbf": 0.0})
    assert result["group_count"] == 0
    assert result["selected_agreement"] is None
