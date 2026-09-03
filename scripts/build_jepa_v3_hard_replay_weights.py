"""Create train-only hard-example replay weights for JEPA-v3 multitask training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hard_example_masks(arrays: dict[str, np.ndarray], extent: float, policy: dict[str, Any]) -> dict[str, np.ndarray]:
    """Derive the fixed replay predicates without changing the sample split.

    The same pure predicates are reused by the held-out P4 evaluator.  That
    evaluator never builds weights or samples validation data; it only reports
    the performance of a train-defined definition on the validation split.
    """
    required = {
        "labels_obstacle_clearance",
        "labels_inter_agent_clearance",
        "labels_cbf_correction",
        "labels_collision",
        "labels_boundary",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"Counterfactual dataset is missing replay labels: {sorted(missing)}")
    low_clearance = np.minimum(arrays["labels_obstacle_clearance"], arrays["labels_inter_agent_clearance"]).min(axis=1) * extent
    high_cbf_correction = arrays["labels_cbf_correction"].max(axis=1)
    collision_or_boundary = np.logical_or(arrays["labels_collision"].max(axis=1) > 0.0, arrays["labels_boundary"].max(axis=1) > 0.0)
    hard_mask = np.logical_or(
        low_clearance < float(policy["low_clearance_threshold_m"]),
        high_cbf_correction > float(policy["high_cbf_correction_threshold_mps"]),
    )
    if bool(policy["include_collision_or_boundary_labels"]):
        hard_mask = np.logical_or(hard_mask, collision_or_boundary)
    return {
        "hard": hard_mask,
        "low_clearance": low_clearance < float(policy["low_clearance_threshold_m"]),
        "high_cbf_correction": high_cbf_correction > float(policy["high_cbf_correction_threshold_mps"]),
        "collision_or_boundary": collision_or_boundary,
    }


def build_weights(arrays: dict[str, np.ndarray], extent: float, policy: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    masks = hard_example_masks(arrays, extent, policy)
    hard_mask = masks["hard"]
    weights = np.where(hard_mask, float(policy["hard_sample_weight"]), 1.0).astype(np.float32)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("Replay weights are invalid.")
    return weights, {
        "samples": int(weights.size),
        "hard_samples": int(np.sum(hard_mask)),
        "hard_fraction": float(np.mean(hard_mask)),
        "low_clearance_samples": int(np.sum(masks["low_clearance"])),
        "high_cbf_correction_samples": int(np.sum(masks["high_cbf_correction"])),
        "collision_or_boundary_samples": int(np.sum(masks["collision_or_boundary"])),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite replay weights or their manifest.")
    metadata = json.loads(args.metadata.resolve().read_text(encoding="utf-8"))
    if metadata.get("split") != "train":
        raise ValueError("Hard-example replay weights may only be built from the train split.")
    boundary = metadata.get("information_boundary", {})
    if boundary.get("development_s3_or_locked_data_used_for_training") is not False:
        raise ValueError("Train metadata does not prove S3/locked exclusion.")
    protocol = yaml.safe_load(args.protocol.resolve().read_text(encoding="utf-8"))
    policy = protocol.get("hard_example_replay", {})
    if policy.get("source_split") != "train_only" or policy.get("do_not_change_validation_or_development_data") is not True:
        raise ValueError("Protocol does not authorize train-only hard-example replay.")
    uniform_fraction = float(policy["uniform_fraction"])
    if not 0.50 <= uniform_fraction <= 1.0:
        raise ValueError("Protocol must preserve at least 50% uniform replay draws.")
    collection_config = yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))
    extent = float(collection_config["world"]["half_extent_xy"])
    with np.load(args.dataset.resolve()) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    weights, diagnostics = build_weights(arrays, extent, policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sample_weights=weights)
    manifest = {
        "replay_type": "jepa_v3_train_only_hard_example_weights",
        "not_a_locked_test": True,
        "source_split": "train",
        "source_dataset": str(args.dataset.resolve()),
        "source_dataset_sha256": sha256(args.dataset.resolve()),
        "source_metadata_sha256": sha256(args.metadata.resolve()),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol.resolve()),
        "uniform_fraction": uniform_fraction,
        "policy": policy,
        "diagnostics": diagnostics,
        "weights_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
