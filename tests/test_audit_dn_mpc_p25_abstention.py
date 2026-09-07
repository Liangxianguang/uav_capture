import torch

from scripts.audit_dn_mpc_p25_abstention import _group_report


def test_abstention_audit_reports_zero_eligible_group() -> None:
    tensors = {
        "sample_type": torch.zeros(3),
        "route_candidate_index": torch.tensor([0, 1, 11]),
        "scenario_index": torch.tensor([2, 2, 2]),
        "time_index": torch.tensor([9, 9, 9]),
        "route_geometry_valid": torch.tensor([1.0, 0.0, 1.0]),
        "labels_cbf_feasible": torch.zeros(3, 5),
        "independent_cbf_trace_present": torch.zeros(3),
    }
    report = _group_report(tensors)
    assert report["runtime_group_count"] == 1
    assert report["zero_eligible_group_count"] == 1
    assert report["trace_missing_group_count"] == 1
    assert report["zero_eligible_groups"][0]["scenario_index"] == 2


def test_abstention_audit_keeps_two_eligible_candidates() -> None:
    tensors = {
        "sample_type": torch.zeros(3),
        "route_candidate_index": torch.tensor([0, 1, 11]),
        "scenario_index": torch.tensor([2, 2, 2]),
        "time_index": torch.tensor([9, 9, 9]),
        "route_geometry_valid": torch.ones(3),
        "labels_cbf_feasible": torch.ones(3, 5),
        "independent_cbf_trace_present": torch.ones(3),
    }
    report = _group_report(tensors)
    assert report["zero_eligible_group_count"] == 0
    assert report["min_eligible_candidates"] == 3
    assert report["trace_missing_group_count"] == 0
