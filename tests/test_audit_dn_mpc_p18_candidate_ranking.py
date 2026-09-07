"""Unit tests for the development-only P19 P18 ranking audit."""

import csv
import io

import numpy as np
import torch

from scripts.audit_dn_mpc_p18_candidate_ranking import (
    _counterfactual_contract,
    _paired_contract,
    _rank_split,
)


def _metadata(split: str, version: str) -> dict:
    return {
        "split": split,
        "dataset_version": version,
        "candidate_profile": "obstacle_route_v1",
        "candidate_count": 12,
        "history_length": 8,
        "chunk_length_steps": 5,
        "horizon_steps": [1, 2, 3, 5],
        "action_scale": 5.0,
        "interaction_action_conditioned_route_chunk": True,
        "pairwise_action_conditioned_route_chunk": True,
    }


def test_paired_contract_allows_independent_calibration_version() -> None:
    contract = _paired_contract(
        {
            "train": _metadata("train", "train-v1"),
            "validation": _metadata("validation", "train-v1"),
            "calibration": _metadata("calibration", "calibration-v2"),
        }
    )
    assert contract["dataset_versions"]["calibration"] == "calibration-v2"
    assert contract["structural_fields"]["chunk_length_steps"] == 5


def test_rank_split_uses_higher_progress_as_better() -> None:
    rows = []
    for scenario, time_index in ((0, 7), (0, 15)):
        for candidate in (0, 1):
            rows.append((scenario, time_index, candidate))
    n = len(rows)
    tensors = {
        "sample_type": torch.zeros(n),
        "route_candidate_index": torch.tensor([row[2] for row in rows]),
        "route_side_index": torch.tensor([row[2] for row in rows]),
        "scenario_index": torch.tensor([row[0] for row in rows]),
        "time_index": torch.tensor([row[1] for row in rows]),
        "route_geometry_valid": torch.ones(n),
        "labels_cbf_feasible": torch.ones(n, 5),
        "labels_route_progress": torch.tensor([[0.8, 0.8, 0.8, 0.8, 0.8], [0.2] * 5] * 2),
    }
    outputs = {
        "route_progress": tensors["labels_route_progress"].numpy().copy(),
        "route_identity_logits": np.zeros((n, 12), dtype=np.float32),
        "route_side_logits": np.zeros((n, 12), dtype=np.float32),
        "route_geometry_logit": np.ones(n, dtype=np.float32),
    }
    outputs["route_identity_logits"][np.arange(n), tensors["route_candidate_index"].numpy()] = 1.0
    outputs["route_side_logits"][np.arange(n), tensors["route_side_index"].numpy()] = 1.0
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "split", "scenario_index", "time_index", "candidate_index", "candidate_label",
            "eligible", "truth_progress", "predicted_progress", "truth_rank", "predicted_rank",
        ),
    )
    writer.writeheader()
    report = _rank_split("validation", tensors, outputs, 2, 0.005, writer)
    assert report["top1_agreement"] == 1.0
    assert report["pairwise_agreement"] == 1.0
    assert report["runtime_group_count"] == 2


def test_counterfactual_contract_requires_selected_trace() -> None:
    tensors = {
        "sample_type": torch.zeros(2),
        "route_candidate_index": torch.tensor([0, 11]),
        "labels_cbf_feasible": torch.ones(2, 5),
        "labels_cbf_min_slack": torch.ones(2, 5),
        "labels_cbf_correction": torch.zeros(2, 5),
        "labels_cbf_intervention": torch.zeros(2, 5),
    }
    report = _counterfactual_contract({"validation": tensors}, {"validation": {}})
    assert report["nominal_counterfactual_present"] is True
    assert report["verified_safe_hold_counterfactual_present"] is True
    assert report["selected_counterfactual_present"] is False
    assert report["independent_selected_nominal_safe_hold_contract"] is False
