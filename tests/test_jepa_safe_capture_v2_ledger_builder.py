from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_jepa_safe_capture_v2_reliability_ledger import _group_ranking, _load_metadata


def test_ledger_builder_preserves_candidate_specific_separation() -> None:
    candidate_actions = np.array([0.0, 0.20, 1.00, 1.30, 2.50], dtype=np.float32)
    arrays = {
        "episode_seed": np.zeros(5, dtype=np.int64),
        "time_index": np.zeros(5, dtype=np.int64),
        "agent_id": np.zeros(5, dtype=np.int64),
        "action_history": candidate_actions[:, None, None],
        "labels_relative": np.zeros((5, 2, 1), dtype=np.float32),
    }

    _credit, _win, separation = _group_ranking(
        arrays,
        np.zeros((5, 1), dtype=np.float32),
        horizon_index=1,
        horizon_seconds=1.0,
        extent=1.0,
        action_scale=1.0,
    )

    np.testing.assert_allclose(separation, [0.20, 0.20, 0.30, 0.30, 1.20], atol=1e-6)


def _write_metadata(tmp_path: Path, *, dataset_version: str, target_frame: object = None, frame_revision: object = None) -> tuple[Path, Path]:
    dataset = tmp_path / "calibration.npz"
    np.savez(dataset, inputs=np.zeros((1, 8, 63), dtype=np.float32))
    metadata: dict[str, object] = {
        "dataset_version": dataset_version,
        "split": "calibration",
        "information_boundary": {
            "target_truth_used_only_for_offline_labels": True,
            "development_or_locked_data_used_for_training": False,
            "locked_test_opened": False,
        },
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    if target_frame is not None:
        metadata["target_relative_frame"] = target_frame
    if frame_revision is not None:
        metadata["label_frame_correction_version"] = frame_revision
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return dataset, metadata_path


def test_ledger_metadata_accepts_corrected_frame(tmp_path: Path) -> None:
    dataset, metadata = _write_metadata(
        tmp_path,
        dataset_version="jepa_safe_capture_v2_p1_corrected_frame",
        target_frame="post_action_defender_position",
        frame_revision=1,
    )

    loaded = _load_metadata(metadata, dataset)

    assert loaded["target_relative_frame"] == "post_action_defender_position"


@pytest.mark.parametrize(
    ("target_frame", "frame_revision"),
    [("pre_action_defender_position", 1), ("post_action_defender_position", 0), (None, None)],
)
def test_ledger_metadata_rejects_incomplete_corrected_frame(
    tmp_path: Path, target_frame: object, frame_revision: object
) -> None:
    dataset, metadata = _write_metadata(
        tmp_path,
        dataset_version="jepa_safe_capture_v2_p1_corrected_frame",
        target_frame=target_frame,
        frame_revision=frame_revision,
    )

    with pytest.raises(ValueError, match="Corrected-frame calibration metadata"):
        _load_metadata(metadata, dataset)


def test_ledger_metadata_preserves_legacy_calibration_support(tmp_path: Path) -> None:
    dataset, metadata = _write_metadata(tmp_path, dataset_version="jepa_safe_capture_v2_p1")

    loaded = _load_metadata(metadata, dataset)

    assert loaded["dataset_version"] == "jepa_safe_capture_v2_p1"
