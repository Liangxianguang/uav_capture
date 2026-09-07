import numpy as np

from scripts.analyze_jepa_pairwise_calibration import _mode_report, _threshold_metrics


def test_threshold_metrics_reports_counts_and_rates() -> None:
    metrics = _threshold_metrics(
        np.asarray([0.9, 0.8, 0.2, 0.1]),
        np.asarray([True, False, True, False]),
        0.5,
    )
    assert metrics["positive_count"] == 2
    assert metrics["predicted_positive_count"] == 2
    assert metrics["false_positive_count"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5


def test_mode_report_contains_all_hazard_bands_and_cutoffs() -> None:
    report = _mode_report(
        np.asarray([[0.9, 0.2], [0.1, 0.8]], dtype=np.float64),
        np.asarray([[0.4, 2.0], [1.5, 0.7]], dtype=np.float64),
    )
    assert report["count"] == 4
    assert set(report["bands"]) == {"lt_0.5", "lt_1", "lt_2"}
    assert set(report["bands"]["lt_1"]["threshold_sweep"]) == {
        "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"
    }
