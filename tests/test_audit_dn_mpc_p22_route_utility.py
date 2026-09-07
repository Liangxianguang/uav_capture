from scripts.audit_dn_mpc_p22_route_utility import _report, _select


def test_route_length_utility_is_public_and_selectable_from_calibration() -> None:
    rows = [
        {"scenario_index": "0", "time_index": "7", "candidate_index": "0", "eligible": "1", "truth_progress": "0.20", "predicted_progress": "0.10"},
        {"scenario_index": "0", "time_index": "7", "candidate_index": "1", "eligible": "1", "truth_progress": "0.18", "predicted_progress": "0.19"},
    ]
    lengths = {(0, 7, 0): 1.0, (0, 7, 1): 4.0}
    zero = _report(rows, lengths, 0.0, 10.0)
    penalized = _report(rows, lengths, 0.5, 10.0)
    assert zero["exact_top1"] == 0.0
    assert penalized["exact_top1"] == 1.0
    assert _select([zero, penalized])["weight"] == 0.5
