"""Audit route-identity hard-negative archive labels before JEPA training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DATASET_VERSION = "jepa_safe_capture_route_identity_hard_negative_v2"
EXPECTED_SPLITS = ("train", "validation", "calibration")
RISK_LABELS = (
    "labels_stopping_distance",
    "labels_obstacle_ttc",
    "labels_boundary_ttc",
    "labels_pairwise_ttc",
    "labels_acceleration_slack",
)
TTC_LABELS = (
    "labels_obstacle_ttc",
    "labels_boundary_ttc",
    "labels_pairwise_ttc",
)
REQUIRED_TENSORBOARD_SCALARS = {
    "Archive/sample_count",
    "Archive/route_geometry_valid_fraction",
    "Archive/cbf_first_step_feasible_fraction",
    "Archive/branch_failure_within_horizon_fraction",
    "Archive/negative_boundary_ttc_fraction",
    "Archive/negative_pairwise_ttc_fraction",
    "Archive/negative_acceleration_slack_fraction",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must be an object: {path}")
    return value


def audit_archive(directory: Path, *, expected_dataset_version: str = DATASET_VERSION) -> dict[str, Any]:
    directory = directory.resolve()
    dataset_path = directory / "route_identity_counterfactual.npz"
    metadata_path = directory / "metadata.json"
    provenance_path = directory / "provenance.json"
    if not dataset_path.is_file() or not metadata_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"Archive is missing dataset/metadata/provenance: {directory}")
    metadata = _load_metadata(metadata_path)
    if metadata.get("dataset_version") != expected_dataset_version:
        raise ValueError(f"Unexpected dataset version in {directory}")
    split = str(metadata.get("split", ""))
    if split not in EXPECTED_SPLITS:
        raise ValueError(f"Archive split must be one of {EXPECTED_SPLITS}: {split!r}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Archive safety phase is invalid: {directory}")
    arrays = np.load(dataset_path, allow_pickle=False)
    required = {
        "sample_type",
        "route_geometry_valid",
        "route_side_index",
        "episode_seed",
        "labels_cbf_feasible",
        "labels_boundary_clearance",
        "labels_target_visible",
        *RISK_LABELS,
    }
    missing = sorted(required.difference(arrays.files))
    if missing:
        raise ValueError(f"Archive is missing arrays {missing}: {directory}")
    sample_count = int(arrays["sample_type"].shape[0])
    if sample_count <= 0:
        raise ValueError(f"Archive is empty: {directory}")
    for label in RISK_LABELS:
        values = np.asarray(arrays[label])
        if values.shape != (sample_count, 5):
            raise ValueError(f"{label} must have shape [N,5], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains NaN/Inf")
    for label in TTC_LABELS:
        values = np.asarray(arrays[label])
        if float(values.min()) < 0.0 or float(values.max()) > 10.0:
            raise ValueError(f"{label} is outside [0,10]")
    stopping = np.asarray(arrays["labels_stopping_distance"])
    if float(stopping.min()) < 0.0:
        raise ValueError("stopping distance is negative")
    visibility = np.asarray(arrays["labels_target_visible"])
    if not np.isin(visibility, (0.0, 1.0)).all():
        raise ValueError("target visibility must be binary")
    runtime = np.asarray(arrays["sample_type"]) == 0
    if not runtime.any():
        raise ValueError("archive has no runtime route rows")
    feasibility_min = np.asarray(arrays["labels_cbf_feasible"])[runtime].min(axis=1)
    route_geometry = np.asarray(arrays["route_geometry_valid"])[runtime] > 0.5
    boundary_clearance = np.asarray(arrays["labels_boundary_clearance"]).min(axis=1)
    route_sides = np.asarray(arrays["route_side_index"])[runtime]
    cbf_feasible = np.asarray(arrays["labels_cbf_feasible"])[runtime]
    earliest_failure = np.asarray(arrays["earliest_failure_step"])[runtime]
    runtime_boundary_negative = int(np.sum(boundary_clearance[runtime] < 0.0))
    shadow_boundary_negative = int(np.sum(boundary_clearance[~runtime] < 0.0))
    if runtime_boundary_negative:
        raise ValueError(f"Runtime rows contain negative boundary clearance: {directory}")
    failure_mask = earliest_failure <= 5
    feasibility_negative = cbf_feasible.min(axis=1) <= 0.5
    if int(np.sum(failure_mask != feasibility_negative)):
        raise ValueError(f"Failure and CBF feasibility labels disagree: {directory}")
    acceleration = np.asarray(arrays["labels_acceleration_slack"])[runtime]
    acceleration_sentinel = acceleration == -1.0
    if float(acceleration.min()) < -1.0:
        raise ValueError(f"Acceleration slack is below the -1 sentinel: {directory}")
    event_dir = Path(str(json.loads(provenance_path.read_text(encoding="utf-8"))["tensorboard_logdir"]))
    if not event_dir.is_dir():
        raise FileNotFoundError(f"TensorBoard directory is missing: {event_dir}")
    accumulator = EventAccumulator(str(event_dir))
    accumulator.Reload()
    missing_scalars = sorted(REQUIRED_TENSORBOARD_SCALARS.difference(accumulator.Tags().get("scalars", [])))
    if missing_scalars:
        raise ValueError(f"TensorBoard is missing archive scalars: {missing_scalars}")
    return {
        "directory": str(directory),
        "split": split,
        "dataset_sha256": _sha256(dataset_path),
        "sample_count": sample_count,
        "runtime_sample_count": int(runtime.sum()),
        "episode_seeds": sorted({int(value) for value in np.asarray(arrays["episode_seed"]).tolist()}),
        "route_geometry_valid": int(route_geometry.sum()),
        "route_geometry_invalid": int((~route_geometry).sum()),
        "feasibility_positive_rows": int((feasibility_min > 0.5).sum()),
        "feasibility_negative_rows": int((feasibility_min <= 0.5).sum()),
        "boundary_clearance_negative_rows": int((boundary_clearance < 0.0).sum()),
        "runtime_boundary_clearance_negative_rows": runtime_boundary_negative,
        "offline_shadow_boundary_clearance_negative_rows": shadow_boundary_negative,
        "acceleration_slack_sentinel_values": int(acceleration_sentinel.sum()),
        "acceleration_slack_measured_negative_values": int(
            np.sum((acceleration < 0.0) & ~acceleration_sentinel)
        ),
        "visibility_values": sorted({float(value) for value in visibility.reshape(-1).tolist()}),
        "route_side_count": int(np.unique(route_sides).size),
        "risk_label_ranges": {
            label: {"min": float(np.asarray(arrays[label]).min()), "max": float(np.asarray(arrays[label]).max())}
            for label in RISK_LABELS
        },
        "tensorboard_scalar_count": len(accumulator.Tags().get("scalars", [])),
        "tensorboard_logdir": str(event_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        audit_archive(path, expected_dataset_version=args.dataset_version)
        for path in args.archive
    ]
    by_split = {item["split"]: item for item in reports}
    if set(by_split) != set(EXPECTED_SPLITS):
        raise ValueError(f"Expected exactly train/validation/calibration archives, got {sorted(by_split)}")
    seed_sets = {split: set(item["episode_seeds"]) for split, item in by_split.items()}
    for first in EXPECTED_SPLITS:
        for second in EXPECTED_SPLITS:
            if first < second and seed_sets[first] & seed_sets[second]:
                raise ValueError(f"Episode seed overlap between {first} and {second}")
    validation = by_split["validation"]
    if validation["feasibility_positive_rows"] == 0 or validation["feasibility_negative_rows"] == 0:
        raise ValueError("Validation feasibility labels do not contain both classes")
    if validation["boundary_clearance_negative_rows"] == 0:
        raise ValueError("Validation has no boundary-negative examples")
    if validation["visibility_values"] != [0.0, 1.0]:
        raise ValueError("Validation visibility labels do not contain both classes")
    result = {
        "dataset_version": args.dataset_version,
        "archives": reports,
        "seed_disjoint": True,
        "validation_class_coverage": True,
        "tensorboard_contract": True,
        "locked_test_opened": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
