"""Audit JEPA-v3 counterfactual dataset schema, coverage, and split separation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ARRAYS = {
    "inputs",
    "action_history",
    "labels_relative",
    "labels_obstacle_clearance",
    "labels_inter_agent_clearance",
    "labels_target_visible",
    "labels_cbf_correction",
    "labels_cbf_intervention",
    "labels_collision",
    "labels_boundary",
    "agent_id",
    "time_index",
    "episode_seed",
    "scenario_index",
    "candidate_index",
    "candidate_is_nominal",
    "candidate_action_norm_mps",
    "chunk_length_steps",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    dataset_path = directory / "counterfactual_multitask_dataset.npz"
    metadata_path = directory / "metadata.json"
    if not dataset_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Expected counterfactual_multitask_dataset.npz and metadata.json.")
    with np.load(dataset_path) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return arrays, metadata, dataset_path


def audit(directory: Path) -> dict[str, Any]:
    arrays, metadata, dataset_path = load(directory)
    samples = int(arrays["inputs"].shape[0])
    if samples <= 0 or arrays["inputs"].shape[1:] != (8, 63) or arrays["action_history"].shape[1:] != (8, 3):
        raise ValueError("Dataset does not match the frozen 8x63 observation and 8x3 action contract.")
    for name, array in arrays.items():
        if array.shape[0] != samples:
            raise ValueError(f"Array {name} has inconsistent sample count.")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"Array {name} contains non-finite values.")
    if metadata.get("split") not in {"train", "validation"}:
        raise ValueError("Counterfactual data must be train or validation only.")
    boundary = metadata.get("information_boundary", {})
    if boundary.get("development_s3_or_locked_data_used_for_training") is not False:
        raise ValueError("Counterfactual metadata does not prove development/locked exclusion.")
    action_scale = float(metadata.get("action_scale", 0.0))
    frozen_action_scale = float(metadata.get("frozen_actor_action_scale", 0.0))
    if metadata.get("action_history_normalization") != "actions_divided_by_frozen_actor_action_scale":
        raise ValueError("Counterfactual action history is not normalized by the frozen actor scale.")
    if action_scale <= 0.0 or not np.isclose(action_scale, frozen_action_scale, rtol=0.0, atol=1e-7):
        raise ValueError("Counterfactual action scale does not match the frozen actor contract.")
    maximum_normalized_action = float(np.max(np.abs(arrays["action_history"])))
    if maximum_normalized_action > 1.05:
        raise ValueError(
            "Counterfactual action history exceeds the expected normalized action range: "
            f"{maximum_normalized_action:.6f} > 1.05."
        )
    candidate_count = int(metadata["candidate_count"])
    groups = np.stack(
        [arrays["episode_seed"], arrays["time_index"], arrays["agent_id"]], axis=1
    )
    unique_groups, group_counts = np.unique(groups, axis=0, return_counts=True)
    expected_nominal_fraction = 1.0 / candidate_count
    candidate_coverage = {
        "state_agent_groups": int(unique_groups.shape[0]),
        "minimum_candidates_per_state_agent": int(np.min(group_counts)),
        "maximum_candidates_per_state_agent": int(np.max(group_counts)),
        "all_groups_have_expected_candidate_count": bool(np.all(group_counts == candidate_count)),
        "nominal_fraction": float(np.mean(arrays["candidate_is_nominal"])),
        "expected_nominal_fraction": expected_nominal_fraction,
        "nominal_fraction_matches": bool(np.isclose(np.mean(arrays["candidate_is_nominal"]), expected_nominal_fraction)),
    }
    scenario_ids, scenario_counts = np.unique(arrays["scenario_index"], return_counts=True)
    candidate_ids, candidate_counts = np.unique(arrays["candidate_index"], return_counts=True)
    labels = {
        "obstacle_clearance_min_normalized": float(np.min(arrays["labels_obstacle_clearance"])),
        "inter_agent_clearance_min_normalized": float(np.min(arrays["labels_inter_agent_clearance"])),
        "visible_fraction": float(np.mean(arrays["labels_target_visible"])),
        "cbf_intervention_fraction": float(np.mean(arrays["labels_cbf_intervention"])),
        "collision_fraction": float(np.mean(arrays["labels_collision"])),
        "boundary_fraction": float(np.mean(arrays["labels_boundary"])),
    }
    return {
        "directory": str(directory.resolve()),
        "dataset_sha256": sha256(dataset_path),
        "metadata_sha256": sha256(directory / "metadata.json"),
        "split": metadata["split"],
        "samples": samples,
        "input_shape": list(arrays["inputs"].shape),
        "action_shape": list(arrays["action_history"].shape),
        "candidate_coverage": candidate_coverage,
        "scenario_sample_counts": {str(int(key)): int(value) for key, value in zip(scenario_ids, scenario_counts)},
        "candidate_sample_counts": {str(int(key)): int(value) for key, value in zip(candidate_ids, candidate_counts)},
        "labels": labels,
        "all_finite": True,
        "action_history_contract": {
            "normalization": metadata["action_history_normalization"],
            "action_scale": action_scale,
            "frozen_actor_action_scale": frozen_action_scale,
            "maximum_absolute_normalized_action": maximum_normalized_action,
            "within_expected_range": True,
            "frozen_actor_checkpoint_sha256": metadata.get("frozen_actor_checkpoint_sha256"),
        },
        "information_boundary": boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--compare-dataset-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset_dir.resolve())
    if args.compare_dataset_dir is not None:
        other_arrays, other_metadata, _other_path = load(args.compare_dataset_dir.resolve())
        current_arrays, current_metadata, _current_path = load(args.dataset_dir.resolve())
        overlap = np.intersect1d(np.unique(current_arrays["episode_seed"]), np.unique(other_arrays["episode_seed"]))
        report["split_comparison"] = {
            "other_directory": str(args.compare_dataset_dir.resolve()),
            "other_split": other_metadata.get("split"),
            "overlapping_episode_seed_count": int(overlap.size),
            "episode_seeds_disjoint": bool(overlap.size == 0),
        }
        if overlap.size:
            raise ValueError("Counterfactual train/validation episode seeds overlap.")
        if current_metadata.get("split") == other_metadata.get("split"):
            raise ValueError("Comparison datasets must come from different splits.")
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
