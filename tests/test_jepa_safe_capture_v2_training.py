from __future__ import annotations

import json

import torch

from scripts.evaluate_jepa_safe_capture_v2 import load_metadata
from scripts.train_jepa_safe_capture_v2 import _validate_metadata, pinball_loss


def test_pinball_loss_penalizes_underprediction_more_at_lower_quantile() -> None:
    target = torch.tensor([[1.0, 1.0]])
    under = pinball_loss(torch.tensor([[0.0, 0.0]]), target, 0.1)
    over = pinball_loss(torch.tensor([[2.0, 2.0]]), target, 0.1)
    assert under.item() < over.item()


def test_corrected_frame_metadata_is_accepted_only_with_frame_revision() -> None:
    base = {
        "dataset_version": "jepa_safe_capture_v2_p1_corrected_frame",
        "split": "train",
        "information_boundary": {
            "target_truth_used_only_for_offline_labels": True,
            "locked_test_opened": False,
        },
        "history_length": 8,
        "candidate_count": 5,
        "candidate_action_semantics": "constant_desired_action_chunk_execute_first_step_then_replan",
        "target_relative_frame": "post_action_defender_position",
        "label_frame_correction_version": 1,
    }
    _validate_metadata(base, "train")
    invalid = dict(base)
    invalid["label_frame_correction_version"] = 0
    try:
        _validate_metadata(invalid, "train")
    except ValueError as error:
        assert "label_frame_correction_version" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("Corrected-frame metadata without a revision must be rejected.")


def test_prediction_evaluator_accepts_corrected_frame_validation_metadata(tmp_path) -> None:
    metadata = {
        "dataset_version": "jepa_safe_capture_v2_p1_corrected_frame",
        "split": "validation",
        "target_relative_frame": "post_action_defender_position",
        "label_frame_correction_version": 1,
        "information_boundary": {"locked_test_opened": False},
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert load_metadata(path)["target_relative_frame"] == "post_action_defender_position"
