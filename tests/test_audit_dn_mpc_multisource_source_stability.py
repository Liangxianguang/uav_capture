import numpy as np

from scripts.audit_dn_mpc_multisource_source_stability import compare_bundles


def _bundle(virtual_strict: float, route_branch: float) -> dict[str, np.ndarray]:
    rows = 2
    source_names = np.asarray(["virtual_probe", "route_outcomes"])
    return {
        "calibration_source_name": source_names,
        "labels_strict_margin_violation": np.asarray(
            [[virtual_strict, virtual_strict], [0.0, 0.0]], dtype=np.float32
        ),
        "labels_branch_failure": np.asarray(
            [[0.0, 0.0], [route_branch, route_branch]], dtype=np.float32
        ),
        "labels_cbf_infeasible": np.zeros((rows, 2), dtype=np.float32),
    }


def test_source_stability_gate_requires_both_roles_and_thresholds() -> None:
    comparison = compare_bundles(
        _bundle(1.0, 1.0),
        _bundle(1.0, 1.0),
        thresholds={
            "strict_margin_cell_rate": 0.05,
            "strict_margin_row_rate": 0.05,
            "branch_failure_row_rate": 0.05,
            "cbf_infeasible_cell_rate": 0.05,
        },
        minimum_rows=1,
    )
    assert comparison["metric_type"] == "model_independent_source_stability"
    assert comparison["gate_passed"] is True
    assert comparison["online_training_authorized"] is False


def test_source_stability_gate_rejects_large_shift() -> None:
    comparison = compare_bundles(
        _bundle(1.0, 1.0),
        _bundle(0.0, 1.0),
        minimum_rows=1,
    )
    assert comparison["gate_passed"] is False
    assert comparison["sources"]["virtual_probe"]["passed"] is False
