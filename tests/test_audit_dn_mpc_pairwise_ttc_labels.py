import json

import numpy as np

from scripts.audit_dn_mpc_pairwise_ttc_labels import _candidate_metrics, _cell_metrics, summarize_archive


def test_pairwise_ttc_cell_metrics_expose_safe_and_cbf_intersections() -> None:
    ttc = np.asarray([[0.5, 2.0], [0.8, 0.9]], dtype=np.float64)
    clearance = np.asarray([[0.5, 0.2], [0.4, 0.2]], dtype=np.float64)
    cbf = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    failure = np.asarray([[False, True], [False, False]])
    result = _cell_metrics(ttc, clearance, cbf, failure, pairwise_margin_m=0.35, ttc_threshold_s=1.0)
    assert result["pairwise_ttc_hazard"]["positive_count"] == 3
    assert result["hazard_and_clearance_safe"]["positive_count"] == 2
    assert result["hazard_and_cbf_feasible"]["positive_count"] == 3
    assert result["hazard_and_branch_failure"]["positive_count"] == 0
    assert result["conditional_rates"]["clearance_safe_given_hazard"] == 2 / 3


def test_candidate_metrics_do_not_count_nonhazard_horizons() -> None:
    ttc = np.asarray([[5.0, 5.0], [0.8, 5.0]], dtype=np.float64)
    clearance = np.asarray([[0.1, 0.1], [0.5, 0.5]], dtype=np.float64)
    cbf = np.ones_like(ttc)
    failure = np.zeros_like(ttc, dtype=bool)
    result = _candidate_metrics(ttc, clearance, cbf, failure, pairwise_margin_m=0.35, ttc_threshold_s=1.0)
    assert result["pairwise_ttc_hazard"]["positive_count"] == 1
    assert result["hazard_and_clearance_safe"]["positive_count"] == 1
    assert result["strict_pairwise_margin_violation"]["positive_count"] == 1


def test_summarize_archive_preserves_sample_type_contract(tmp_path) -> None:
    archive = tmp_path / "archive.npz"
    metadata = tmp_path / "metadata.json"
    np.savez(
        archive,
        labels_pairwise_ttc=np.asarray([[0.5, 5.0], [5.0, 5.0]], dtype=np.float32),
        labels_inter_agent_clearance=np.asarray([[0.5, 0.5], [0.1, 0.1]], dtype=np.float32),
        labels_cbf_feasible=np.ones((2, 2), dtype=np.float32),
        earliest_failure_step=np.asarray([3, 3], dtype=np.int64),
        branch_terminated=np.asarray([False, True]),
        sample_type=np.asarray([0, 2], dtype=np.float32),
    )
    metadata.write_text(
        json.dumps({"split": "calibration", "development_only": True, "locked_test_opened": False}),
        encoding="utf-8",
    )
    result = summarize_archive(archive, metadata)
    assert result["rows"] == 2
    assert result["reports"]["runtime"]["rows"] == 1
    assert result["reports"]["near_pass"]["rows"] == 1
    assert result["reports"]["split_merge"]["rows"] == 0
