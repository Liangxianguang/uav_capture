import numpy as np

from scripts.build_dn_mpc_pairwise_multisource_calibration import build_bundle


def _write_source(
    tmp_path,
    name: str,
    sample_type: int,
    rows: int,
    *,
    virtual: bool,
    split: str = "calibration",
) -> tuple:
    dataset = tmp_path / f"{name}.npz"
    metadata = tmp_path / f"{name}.json"
    horizon = 2
    arrays = {
        "inputs": np.zeros((rows, 8, 63), dtype=np.float32),
        "action_history": np.zeros((rows, 8, 3), dtype=np.float32),
        "route_action_chunk": np.zeros((rows, 5, 3), dtype=np.float32),
        "route_relative_action_chunk": np.zeros((rows, 5, 3), dtype=np.float32),
        "route_pairwise_relative_action_chunk": np.zeros((rows, 5, 9), dtype=np.float32),
        "labels_predicted_ttc_hazard": np.zeros((rows, horizon), dtype=np.float32),
        "labels_strict_margin_violation": np.zeros((rows, horizon), dtype=np.float32),
        "labels_cbf_infeasible": np.zeros((rows, horizon), dtype=np.float32),
        "labels_branch_failure": np.zeros((rows, horizon), dtype=np.float32),
        "labels_inter_agent_clearance": np.ones((rows, horizon), dtype=np.float32),
        "labels_cbf_feasible": np.ones((rows, horizon), dtype=np.float32),
        "sample_type": np.full(rows, sample_type, dtype=np.float32),
        "earliest_failure_step": np.full(rows, 6, dtype=np.int64),
        "branch_terminated": np.zeros(rows, dtype=np.float32),
    }
    if virtual:
        arrays["virtual_probe_mode"] = np.zeros(rows, dtype=np.int64)
        arrays["virtual_probe_pair_index"] = np.zeros(rows, dtype=np.int64)
    np.savez_compressed(dataset, **arrays)
    metadata.write_text(
        f'{{"split":"{split}","development_only":true,"locked_test_opened":false}}\n',
        encoding="utf-8",
    )
    return dataset, metadata


def test_build_bundle_preserves_source_semantics(tmp_path) -> None:
    p14, p14_meta = _write_source(tmp_path, "p14", 5, 2, virtual=True)
    p15, p15_meta = _write_source(tmp_path, "p15", 0, 3, virtual=False)
    arrays, metadata = build_bundle(p14, p14_meta, p15, p15_meta)
    assert arrays["inputs"].shape[0] == 5
    assert arrays["calibration_source_id"].tolist() == [0, 0, 1, 1, 1]
    assert arrays["virtual_probe_mode"].tolist() == [0, 0, -1, -1, -1]
    assert metadata["label_semantics_preserved"] is True
    assert metadata["source_rows"] == {"p14_virtual_probe": 2, "p15_route_outcomes": 3}


def test_build_bundle_accepts_disjoint_validation_sources(tmp_path) -> None:
    p17_virtual, p17_virtual_meta = _write_source(
        tmp_path, "p17_virtual", 5, 1, virtual=True, split="validation"
    )
    p17_route, p17_route_meta = _write_source(
        tmp_path, "p17_route", 0, 1, virtual=False, split="validation"
    )
    arrays, metadata = build_bundle(
        p17_virtual,
        p17_virtual_meta,
        p17_route,
        p17_route_meta,
        expected_split="validation",
        source_names=("p17_virtual_probe", "p17_route_outcomes"),
    )
    assert metadata["split"] == "validation"
    assert metadata["expected_split"] == "validation"
    assert arrays["calibration_source_name"].tolist() == ["p17_virtual_probe", "p17_route_outcomes"]
