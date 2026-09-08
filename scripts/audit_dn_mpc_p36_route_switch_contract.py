"""Audit the P36 route-identity archive contract without opening evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REQUIRED_ARRAYS = {
    "inputs",
    "route_action_chunk",
    "route_relative_action_chunk",
    "route_pairwise_relative_action_chunk",
    "executed_route_index",
    "previous_executed_route_index",
    "route_switch_outcome",
    "route_match_residual_mps",
    "labels_target_escape_cost",
    "independent_cbf_trace_present",
    "labels_cbf_feasible",
    "route_geometry_valid",
    "earliest_failure_step",
}
REQUIRED_SCALARS = {
    "Archive/sample_count",
    "Archive/runtime_sample_count",
    "Archive/cbf_first_step_feasible_fraction",
    "RouteIdentity/matched_fraction",
    "RouteIdentity/previous_known_fraction",
    "RouteIdentity/switch_fraction",
    "RouteIdentity/match_residual_p95_mps",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_archive(
    directory: Path,
    expected_split: str | None = None,
    expected_dataset_version: str = "dn_mpc_route_identity_chunk5_p36_route_switch_v1",
    require_pairwise: bool = False,
) -> dict[str, Any]:
    directory = directory.resolve()
    dataset_path = directory / "route_identity_counterfactual.npz"
    metadata_path = directory / "metadata.json"
    provenance_path = directory / "provenance.json"
    if not all(path.is_file() for path in (dataset_path, metadata_path, provenance_path)):
        raise FileNotFoundError(f"missing P36 archive artifact in {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_version") != expected_dataset_version:
        raise ValueError(f"unexpected dataset version: {directory}")
    split = str(metadata.get("split"))
    if expected_split is not None and split != expected_split:
        raise ValueError(f"expected split {expected_split}, got {split}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"P36 archive is not development-only/closed: {directory}")
    safety = metadata.get("safety_contract", {})
    # The collector stores the executable subset in metadata; the full YAML
    # contract remains bound through the archive-config hash in provenance.
    information_boundary = metadata.get("information_boundary", {})
    cbf_contract = metadata.get("cbf_contract", {})
    if safety:
        for key in ("cbf_margin_changed", "stale_ood_nonfinite_gates_changed", "raw_unverified_execution_allowed"):
            if safety.get(key) is not False:
                raise ValueError(f"safety contract violation {key} in {directory}")
        if safety.get("controlled_abort_preserved") is not True:
            raise ValueError(f"controlled abort was not preserved: {directory}")
    if information_boundary.get("raw_unverified_action_executed") is not False:
        raise ValueError(f"raw unverified execution boundary is invalid: {directory}")
    if cbf_contract.get("controlled_abort_preserved") is not True:
        raise ValueError(f"controlled abort was not preserved: {directory}")
    identity = metadata.get("route_execution_identity_contract", {})
    expected_fields = {"previous_executed_route_index", "executed_route_index", "route_switch_outcome", "route_match_residual_mps"}
    declared_fields = set(identity.get("fields", ()))
    declared_fields.update(
        value for value in (
            identity.get("previous_field"),
            identity.get("current_field"),
            identity.get("switch_field"),
            identity.get("match_residual_field"),
        ) if value
    )
    if identity.get("enabled") is not True or declared_fields != expected_fields:
        raise ValueError(f"route identity contract is incomplete: {directory}")
    if float(identity.get("match_tolerance_mps", 0.0)) != 1.0:
        raise ValueError(f"unexpected route match tolerance: {directory}")
    if metadata.get("target_escape_label_contract", {}).get("enabled") is not True:
        raise ValueError(f"target escape label contract is disabled: {directory}")
    if require_pairwise:
        if metadata.get("pairwise_action_conditioned_route_chunk") is not True:
            raise ValueError(f"pairwise route feature is disabled: {directory}")
        interaction = metadata.get("interaction_hard_negatives", {})
        if interaction.get("enabled") is not True or interaction.get("offline_only") is not True:
            raise ValueError(f"pairwise interaction hard-negative contract is invalid: {directory}")

    arrays = np.load(dataset_path, allow_pickle=False)
    missing = sorted(REQUIRED_ARRAYS.difference(arrays.files))
    if missing:
        raise ValueError(f"missing required arrays {missing}: {directory}")
    sample_count = int(arrays["inputs"].shape[0])
    if sample_count <= 0:
        raise ValueError(f"empty archive: {directory}")
    for name in arrays.files:
        values = np.asarray(arrays[name])
        if values.dtype.kind in "fiu" and not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {name}: {directory}")
    if arrays["route_action_chunk"].shape[:2] != (sample_count, 5):
        raise ValueError(f"unexpected route chunk shape: {arrays['route_action_chunk'].shape}")
    if arrays["route_relative_action_chunk"].shape != arrays["route_action_chunk"].shape:
        raise ValueError("route relative action shape does not match route action shape")
    if arrays["route_pairwise_relative_action_chunk"].shape != (sample_count, 5, 9):
        raise ValueError("pairwise route action shape must be [N,5,9]")

    sample_type = np.asarray(arrays["sample_type"], dtype=np.int64)
    runtime = sample_type == 0
    if not runtime.any():
        raise ValueError(f"archive has no runtime rows: {directory}")
    matched = np.asarray(arrays["executed_route_index"])[runtime] >= 0
    previous_known = np.asarray(arrays["previous_executed_route_index"])[runtime] >= 0
    residual = np.asarray(arrays["route_match_residual_mps"])[runtime]
    first_step_feasible = np.asarray(arrays["labels_cbf_feasible"])[runtime, 0] >= 0.5
    unknown = ~matched
    if np.any(unknown):
        # An unknown route is valid only when no verified action was executed.
        # These rows are explicit CBF abstentions, not missing labels.
        selected_feasible = np.asarray(arrays["selected_cbf_feasible"])[runtime]
        if np.any(selected_feasible[unknown] >= 0.5):
            raise ValueError(f"feasible runtime row has unknown route identity: {directory}")
        if np.any(np.asarray(arrays["earliest_failure_step"])[runtime][unknown] <= 0):
            raise ValueError(f"unknown route identity lacks a branch failure marker: {directory}")
    if np.any(residual[matched] > 1.0 + 1e-9):
        raise ValueError(f"runtime route residual exceeds match tolerance: {directory}")
    switch = np.asarray(arrays["route_switch_outcome"])[runtime]
    if not np.isin(switch[previous_known & matched], (0, 1)).all():
        raise ValueError(f"route switch outcome is undefined for known previous routes: {directory}")
    if np.any(switch[unknown] != -1):
        raise ValueError(f"unknown current route must have switch outcome -1: {directory}")
    trace_present = np.asarray(arrays["independent_cbf_trace_present"])[runtime].astype(bool)
    if np.any(~trace_present & ~unknown):
        raise ValueError(f"CBF traces are incomplete for route-identified rows: {directory}")
    if np.any(np.asarray(arrays["labels_target_escape_cost"]) < 0.0):
        raise ValueError(f"target escape labels contain negative values: {directory}")

    event_dir = Path(str(provenance.get("tensorboard_logdir", "")))
    if not event_dir.is_dir():
        raise FileNotFoundError(f"TensorBoard logdir missing: {event_dir}")
    accumulator = EventAccumulator(str(event_dir))
    accumulator.Reload()
    missing_scalars = sorted(REQUIRED_SCALARS.difference(accumulator.Tags().get("scalars", [])))
    if missing_scalars:
        raise ValueError(f"TensorBoard missing scalars {missing_scalars}: {directory}")
    return {
        "directory": str(directory),
        "split": split,
        "dataset_sha256": _sha256(dataset_path),
        "metadata_sha256": _sha256(metadata_path),
        "sample_count": sample_count,
        "runtime_sample_count": int(runtime.sum()),
        "route_match_fraction": float(matched.mean()),
        "unknown_route_fraction": float(unknown.mean()),
        "unknown_rows_all_cbf_infeasible": bool(np.all(~unknown | ~first_step_feasible)),
        "previous_known_fraction": float(previous_known.mean()),
        "switch_fraction_previous_known": float(switch[previous_known].mean()) if previous_known.any() else None,
        "route_match_residual_p95_mps": float(np.percentile(residual[matched], 95)),
        "independent_cbf_trace_fraction": float(trace_present.mean()),
        "geometry_valid_fraction": float(np.asarray(arrays["route_geometry_valid"])[runtime].mean()),
        "branch_failure_count": int(np.sum(np.asarray(arrays["earliest_failure_step"])[runtime] <= 5)),
        "episode_seeds": sorted({int(value) for value in np.asarray(arrays["episode_seed"]).tolist()}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--dataset-version", default="dn_mpc_route_identity_chunk5_p36_route_switch_v1")
    parser.add_argument("--require-pairwise", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "protocol": str(args.dataset_version),
        "development_only": True,
        "locked_test_opened": False,
        "train": audit_archive(args.train, "train", args.dataset_version, args.require_pairwise),
        "calibration": audit_archive(args.calibration, "calibration", args.dataset_version, args.require_pairwise),
    }
    if args.validation is not None:
        report["validation"] = audit_archive(args.validation, "validation", args.dataset_version, args.require_pairwise)
    split_names = tuple(name for name in ("train", "validation", "calibration") if name in report)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap = sorted(set(report[left_name]["episode_seeds"]) & set(report[right_name]["episode_seeds"]))
            if overlap:
                raise ValueError(f"episode seed overlap between {left_name} and {right_name}: {overlap}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
