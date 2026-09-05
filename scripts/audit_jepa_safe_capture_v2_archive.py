"""Audit P1 safe-capture archives and prove split/group integrity."""

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
    "labels_target_velocity",
    "labels_target_acceleration",
    "labels_obstacle_clearance",
    "labels_inter_agent_clearance",
    "labels_pairwise_ttc",
    "labels_target_visible",
    "labels_observation_age",
    "labels_cbf_correction",
    "labels_cbf_intervention",
    "labels_cbf_qp_feasible",
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
ALLOWED_SPLITS = {"train", "validation", "calibration"}
SUPPORTED_DATASET_VERSIONS = {
    "jepa_safe_capture_v2_p1",
    "jepa_safe_capture_v2_p1_corrected_frame",
    # Stratified development archives keep an explicit collection version so
    # their scenario contract can evolve without reusing a historical hash.
    "jepa_safe_capture_l0_l3_v1",
    "jepa_safe_capture_l0_l3_v2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    dataset = next((directory / name for name in ("counterfactual_safe_capture_v2.npz", "counterfactual_multitask_dataset.npz") if (directory / name).is_file()), None)
    if dataset is None or not (directory / "metadata.json").is_file():
        raise FileNotFoundError(f"Archive {directory} must contain a dataset and metadata.json.")
    with np.load(dataset) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(f"Archive is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return arrays, json.loads((directory / "metadata.json").read_text(encoding="utf-8")), dataset


def candidate_group_audit(arrays: dict[str, np.ndarray], candidate_count: int) -> dict[str, Any]:
    keys = np.stack([arrays["episode_seed"], arrays["time_index"], arrays["agent_id"]], axis=1)
    groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=groups.shape[0])
    complete = True
    invalid_groups = 0
    for index in range(groups.shape[0]):
        candidates = np.sort(arrays["candidate_index"][inverse == index])
        if candidates.shape != (candidate_count,) or not np.array_equal(candidates, np.arange(candidate_count)):
            complete = False
            invalid_groups += 1
    return {
        "state_agent_groups": int(groups.shape[0]),
        "minimum_candidates_per_state_agent": int(np.min(counts)),
        "maximum_candidates_per_state_agent": int(np.max(counts)),
        "all_groups_have_expected_candidate_count": bool(complete),
        "invalid_group_count": int(invalid_groups),
        "nominal_fraction": float(np.mean(arrays["candidate_is_nominal"])),
        "expected_nominal_fraction": 1.0 / candidate_count,
        "nominal_fraction_matches": bool(np.isclose(np.mean(arrays["candidate_is_nominal"]), 1.0 / candidate_count)),
    }


def audit(directory: Path) -> dict[str, Any]:
    arrays, metadata, dataset_path = load(directory)
    samples = int(arrays["inputs"].shape[0])
    if samples <= 0 or arrays["inputs"].shape[1:] != (8, 63) or arrays["action_history"].shape[1:] != (8, 3):
        raise ValueError("Archive does not match the frozen 8x63 and 8x3 input contract.")
    if metadata.get("dataset_version") not in SUPPORTED_DATASET_VERSIONS:
        raise ValueError("Archive metadata is not a supported P1 v2 dataset version.")
    if metadata.get("dataset_version") == "jepa_safe_capture_v2_p1_corrected_frame":
        if metadata.get("target_relative_frame") != "post_action_defender_position" or int(metadata.get("label_frame_correction_version", 0)) < 1:
            raise ValueError("Corrected-frame archive metadata is incomplete.")
    if metadata.get("split") not in ALLOWED_SPLITS:
        raise ValueError("P1 archive split must be train, validation, or calibration.")
    boundary = metadata.get("information_boundary", {})
    if boundary.get("target_truth_used_only_for_offline_labels") is not True or boundary.get("locked_test_opened") is not False:
        raise ValueError("Archive metadata does not prove the information boundary.")
    for name, array in arrays.items():
        if array.shape[0] != samples:
            raise ValueError(f"Array {name} has inconsistent sample count.")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"Array {name} contains non-finite values.")
    action_scale = float(metadata.get("action_scale", 0.0))
    frozen_scale = float(metadata.get("frozen_actor_action_scale", 0.0))
    if metadata.get("action_history_normalization") != "actions_divided_by_frozen_actor_action_scale" or action_scale <= 0 or not np.isclose(action_scale, frozen_scale):
        raise ValueError("Action history normalization does not match the frozen actor contract.")
    if float(np.max(np.abs(arrays["action_history"]))) > 1.05:
        raise ValueError("Normalized action history exceeds the expected range.")
    if metadata.get("episode_seed_count") != len(metadata.get("episode_seeds", [])):
        raise ValueError("Metadata episode_seed_count does not match episode_seeds.")
    if len(set(metadata.get("episode_seeds", []))) != len(metadata.get("episode_seeds", [])):
        raise ValueError("An archive contains duplicate episode seeds.")
    collection_path = Path(metadata.get("collection_config", ""))
    protocol_path = Path(metadata.get("protocol", ""))
    if not collection_path.is_file() or not protocol_path.is_file():
        raise ValueError("Archive provenance does not point to existing collection/protocol files.")
    if metadata.get("collection_config_sha256") != sha256(collection_path) or metadata.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("Archive provenance hash does not match collection/protocol files.")
    for relative, expected in metadata.get("source_hashes", {}).items():
        source = Path(__file__).resolve().parents[1] / relative
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Archive source hash mismatch: {relative}")
    candidate_count = int(metadata["candidate_count"])
    groups = candidate_group_audit(arrays, candidate_count)
    if not groups["all_groups_have_expected_candidate_count"] or not groups["nominal_fraction_matches"]:
        raise ValueError(f"Candidate group contract failed: {groups}")
    binary = {name: bool(np.all(np.isin(arrays[name], [0.0, 1.0]))) for name in ("labels_target_visible", "labels_cbf_intervention", "labels_cbf_qp_feasible", "labels_collision", "labels_boundary")}
    if not all(binary.values()):
        raise ValueError(f"Binary P1 labels are not indicators: {binary}")
    ttc_clip = float(metadata.get("ttc_clip_seconds", 0.0))
    if ttc_clip <= 0.0 or np.min(arrays["labels_pairwise_ttc"]) < 0.0 or np.max(arrays["labels_pairwise_ttc"]) > ttc_clip + 1e-6:
        raise ValueError("Pairwise TTC labels fall outside the declared clip range.")
    if np.min(arrays["labels_observation_age"]) < 0.0 or np.min(arrays["labels_cbf_correction"]) < 0.0:
        raise ValueError("Observation age or CBF correction labels are negative.")
    labels = {
        "target_velocity_max_abs": float(np.max(np.abs(arrays["labels_target_velocity"]))),
        "target_acceleration_max_abs": float(np.max(np.abs(arrays["labels_target_acceleration"]))),
        "obstacle_clearance_min_normalized": float(np.min(arrays["labels_obstacle_clearance"])),
        "inter_agent_clearance_min_normalized": float(np.min(arrays["labels_inter_agent_clearance"])),
        "pairwise_ttc_min_seconds": float(np.min(arrays["labels_pairwise_ttc"])),
        "observation_age_max_steps": float(np.max(arrays["labels_observation_age"])),
        "cbf_intervention_fraction": float(np.mean(arrays["labels_cbf_intervention"])),
        "cbf_qp_infeasible_fraction": float(np.mean(1.0 - arrays["labels_cbf_qp_feasible"])),
        "collision_fraction": float(np.mean(arrays["labels_collision"])),
        "boundary_fraction": float(np.mean(arrays["labels_boundary"])),
    }
    archive_manifest_path = directory / "archive_manifest.json"
    manifest_report: dict[str, Any] = {"present": False}
    if archive_manifest_path.is_file():
        manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != sha256(dataset_path) or manifest.get("metadata_sha256") != sha256(directory / "metadata.json"):
            raise ValueError("Archive manifest hashes do not match generated files.")
        if manifest.get("split") != metadata["split"] or manifest.get("locked_test_opened") is not False:
            raise ValueError("Archive manifest split/locked contract is invalid.")
        manifest_report = {"present": True, "sha256": sha256(archive_manifest_path), "locked_test_opened": False}
    tensorboard_report: dict[str, Any] = {"present": False}
    tensorboard_path = Path(metadata.get("tensorboard_logdir", ""))
    if not tensorboard_path.is_dir():
        # The TensorBoard path is stored in the archive manifest because the
        # generated metadata is intentionally immutable after collection.
        archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8")) if archive_manifest_path.is_file() else {}
        tensorboard_path = Path(archive_manifest.get("tensorboard_logdir", ""))
    if tensorboard_path.is_dir():
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(tensorboard_path), size_guidance={"scalars": 0, "tensors": 0, "histograms": 0})
        accumulator.Reload()
        tags = accumulator.Tags()
        required_scalars = {
            "Data/sample_count",
            "Data/episode_count",
            "Data/scenario_count",
            "Data/candidate_count",
            "Data/nominal_fraction",
            "Data/labels_target_velocity/coverage",
        }
        missing_scalars = sorted(required_scalars.difference(tags.get("scalars", [])))
        if missing_scalars:
            raise ValueError(f"TensorBoard data audit is missing scalar tags: {missing_scalars}")
        # PyTorch SummaryWriter stores add_text payloads as tensor tags with a
        # ``/text_summary`` suffix in TensorBoard's event accumulator.
        required_text = {
            "Config/protocol/text_summary",
            "Config/collection/text_summary",
            "Data/metadata/text_summary",
            "Provenance/source_hashes/text_summary",
        }
        missing_text = sorted(required_text.difference(tags.get("tensors", [])))
        if missing_text:
            raise ValueError(f"TensorBoard data audit is missing text tags: {missing_text}")
        tensorboard_report = {
            "present": True,
            "path": str(tensorboard_path.resolve()),
            "scalar_tag_count": len(tags.get("scalars", [])),
            "histogram_tag_count": len(tags.get("histograms", [])),
            "text_tag_count": len(tags.get("tensors", [])),
            "required_tags_complete": True,
        }
    else:
        raise ValueError("P1 archive is missing its TensorBoard data logdir.")
    return {
        "directory": str(directory.resolve()),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256(dataset_path),
        "metadata_sha256": sha256(directory / "metadata.json"),
        "dataset_version": metadata["dataset_version"],
        "split": metadata["split"],
        "samples": samples,
        "episode_seeds": metadata.get("episode_seeds", []),
        "episode_seed_count": len(metadata.get("episode_seeds", [])),
        "scenario_count": len(metadata.get("scenario_records", [])),
        "candidate_coverage": groups,
        "binary_label_contract": binary,
        "labels": labels,
        "archive_manifest": manifest_report,
        "tensorboard": tensorboard_report,
        "all_finite": True,
        "locked_test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--compare-dataset-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = audit(args.dataset_dir.resolve())
    all_reports = [current]
    for other_directory in args.compare_dataset_dir:
        other = audit(other_directory.resolve())
        overlap = sorted(set(current["episode_seeds"]).intersection(other["episode_seeds"]))
        if overlap:
            raise ValueError(f"P1 archive split overlap detected between {current['split']} and {other['split']}: {overlap[:8]}")
        if current["split"] == other["split"]:
            raise ValueError("Compared P1 archives must have distinct split names.")
        current.setdefault("split_comparisons", []).append({"other_split": other["split"], "overlap_count": 0, "episode_seeds_disjoint": True})
        all_reports.append(other)
    report = {"archive_reports": all_reports, "all_episode_seeds_disjoint": True, "locked_test_opened": False}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
