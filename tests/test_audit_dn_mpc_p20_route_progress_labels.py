import csv
import io

from scripts.audit_dn_mpc_p20_route_progress_labels import _group_report


def test_group_report_distinguishes_near_ties_from_informative_groups() -> None:
    rows = []
    for time_index, values, predictions in (
        (7, (0.100, 0.099), (0.099, 0.100)),
        (15, (0.200, 0.100), (0.150, 0.120)),
    ):
        for candidate, (truth, prediction) in enumerate(zip(values, predictions)):
            rows.append(
                {
                    "split": "validation",
                    "scenario_index": "0",
                    "time_index": str(time_index),
                    "candidate_index": str(candidate),
                    "eligible": "1",
                    "truth_progress": str(truth),
                    "predicted_progress": str(prediction),
                }
            )
    report = _group_report(rows, tie_margin=0.005, informative_margin=0.005)
    assert report["group_count"] == 2
    assert report["top_gap_le_tie_margin_fraction"] == 0.5
    assert report["tie_aware_top1"] == 1.0
    assert report["informative_group_count"] == 1
    assert report["informative_exact_top1"] == 1.0
    assert report["pairwise_agreement"] == 1.0
