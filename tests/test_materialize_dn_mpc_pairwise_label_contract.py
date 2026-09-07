import numpy as np
import pytest

from scripts.materialize_dn_mpc_pairwise_label_contract import (
    CONTRACT_VERSION,
    _write_tensorboard,
    materialize_labels,
)


def test_materialized_labels_separate_warning_from_outcome() -> None:
    arrays = {
        "labels_pairwise_ttc": np.asarray([[0.5, 5.0], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        "labels_inter_agent_clearance": np.asarray([[0.5, 0.5], [0.4, 0.2], [-0.1, -0.1]], dtype=np.float32),
        "labels_cbf_feasible": np.asarray([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "earliest_failure_step": np.asarray([3, 1, 1], dtype=np.int64),
        "branch_terminated": np.asarray([False, True, False]),
        "sample_type": np.asarray([0, 0, 1], dtype=np.float32),
    }
    result, contract = materialize_labels(arrays)
    assert contract["version"] == CONTRACT_VERSION
    assert result["labels_predicted_ttc_hazard"].tolist() == [[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    assert result["labels_strict_margin_violation"].tolist() == [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert result["labels_cbf_infeasible"].tolist() == [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    # The boundary-shadow row is an offline negative, not a real branch failure.
    assert result["labels_branch_failure"].tolist() == [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]


def test_materialized_labels_preserve_source_arrays() -> None:
    arrays = {
        "labels_pairwise_ttc": np.ones((1, 2), dtype=np.float32),
        "labels_inter_agent_clearance": np.ones((1, 2), dtype=np.float32),
        "labels_cbf_feasible": np.ones((1, 2), dtype=np.float32),
        "earliest_failure_step": np.asarray([3], dtype=np.int64),
        "branch_terminated": np.asarray([False]),
        "sample_type": np.asarray([0], dtype=np.float32),
        "route_action_chunk": np.zeros((1, 2, 3), dtype=np.float32),
    }
    result, _contract = materialize_labels(arrays)
    assert np.array_equal(result["route_action_chunk"], arrays["route_action_chunk"])
    assert set(arrays).issubset(result)


def test_tensorboard_namespace_is_explicit_and_path_safe(tmp_path) -> None:
    arrays = {
        "labels_predicted_ttc_hazard": np.zeros((1, 2), dtype=np.float32),
        "labels_strict_margin_violation": np.zeros((1, 2), dtype=np.float32),
        "labels_cbf_infeasible": np.zeros((1, 2), dtype=np.float32),
        "labels_branch_failure": np.zeros((1, 2), dtype=np.float32),
        "sample_type": np.asarray([0], dtype=np.int64),
    }
    _write_tensorboard(
        tmp_path / "tb",
        arrays,
        {"version": CONTRACT_VERSION},
        source_dataset=tmp_path / "source.npz",
        source_metadata=tmp_path / "source.json",
        source_dataset_sha256="a" * 64,
        source_metadata_sha256="b" * 64,
        tensorboard_namespace="P15",
    )
    event_files = list((tmp_path / "tb").glob("events.*"))
    assert event_files
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tags = EventAccumulator(str(tmp_path / "tb")).Reload().Tags()
    assert "P15/runtime/rows" in tags["scalars"]
    assert "P15/contract/text_summary" in tags["tensors"]
    with pytest.raises(ValueError, match="path-safe"):
        _write_tensorboard(
            tmp_path / "invalid",
            arrays,
            {},
            source_dataset=tmp_path / "source.npz",
            source_metadata=tmp_path / "source.json",
            source_dataset_sha256="a" * 64,
            source_metadata_sha256="b" * 64,
            tensorboard_namespace="P15/bad",
        )
