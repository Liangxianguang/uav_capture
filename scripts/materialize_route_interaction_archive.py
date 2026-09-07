"""Add teammate-relative candidate-action features to a frozen route archive.

The dense route archive already contains one row for every defender, route
candidate and route-state group.  This utility derives the public,
action-conditioned interaction feature without rerunning the environment:
each row receives its candidate chunk minus the mean chunk of the other
defenders in the same route-state group.  The source archive is never
modified, and the derived archive remains development-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


FEATURE_NAME = "route_relative_action_chunk"
GROUP_FIELDS = ("scenario_index", "time_index", "route_candidate_index", "sample_type")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fresh(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _derive_feature(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    required = ("route_action_chunk", *GROUP_FIELDS)
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"Source archive is missing required fields: {missing}")
    route_actions = np.asarray(arrays["route_action_chunk"], dtype=np.float32)
    if route_actions.ndim != 3 or route_actions.shape[-1] != 3:
        raise ValueError(f"route_action_chunk must have shape [N,H,3], got {route_actions.shape}")
    samples = int(route_actions.shape[0])
    if not np.isfinite(route_actions).all():
        raise ValueError("route_action_chunk contains non-finite values")
    keys = [np.asarray(arrays[name]).reshape(-1) for name in GROUP_FIELDS]
    if any(values.shape[0] != samples for values in keys):
        raise ValueError("Route group fields must have the same sample count")
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(*(values.tolist() for values in keys))):
        groups[tuple(int(value) for value in key)].append(index)
    feature = np.zeros_like(route_actions, dtype=np.float32)
    group_sizes: dict[int, int] = defaultdict(int)
    for key, indices in groups.items():
        # Four defenders are part of the route archive contract.  Requiring
        # the complete group prevents silently mixing a teammate from another
        # scenario or time step.
        if len(indices) != 4:
            raise ValueError(f"Route group {key} has {len(indices)} rows; expected four defenders")
        index_array = np.asarray(indices, dtype=np.int64)
        group_actions = route_actions[index_array]
        group_mean = group_actions.mean(axis=0, keepdims=True)
        feature[index_array] = group_actions - (group_mean * 4.0 - group_actions) / 3.0
        group_sizes[len(indices)] += 1
    if not np.isfinite(feature).all():
        raise ValueError("Derived route interaction feature is non-finite")
    return feature, {str(size): count for size, count in sorted(group_sizes.items())}


def materialize(
    source_dataset: Path,
    source_metadata: Path,
    output_dir: Path,
    tensorboard_logdir: Path,
    dataset_version: str,
) -> dict[str, Any]:
    source_dataset = source_dataset.resolve()
    source_metadata = source_metadata.resolve()
    output_dir = _fresh(output_dir)
    tensorboard_logdir = _fresh(tensorboard_logdir)
    metadata = _load_json(source_metadata)
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError("Only closed development archives may be materialized")
    if metadata.get("interaction_action_conditioned_route_chunk") is True:
        raise ValueError("Source archive already contains the interaction-conditioned feature")
    with np.load(source_dataset) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    feature, group_sizes = _derive_feature(arrays)
    arrays[FEATURE_NAME] = feature
    dataset_path = output_dir / "route_identity_counterfactual.npz"
    np.savez_compressed(dataset_path, **arrays)
    output_metadata = dict(metadata)
    output_metadata["dataset_version"] = str(dataset_version)
    output_metadata["interaction_action_conditioned_route_chunk"] = True
    output_metadata["array_shapes"] = {
        name: list(value.shape) for name, value in arrays.items()
    }
    output_metadata["derived_feature_contract"] = {
        "name": FEATURE_NAME,
        "definition": "candidate_action_minus_mean_of_other_three_defenders_in_same_route_state",
        "source_field": "route_action_chunk",
        "group_fields": list(GROUP_FIELDS),
        "defender_count": 4,
        "source_dataset": str(source_dataset),
        "source_dataset_sha256": _sha256(source_dataset),
        "source_metadata": str(source_metadata),
        "source_metadata_sha256": _sha256(source_metadata),
        "group_size_counts": group_sizes,
    }
    output_metadata["source"] = dict(output_metadata.get("source", {}))
    output_metadata["source"]["derived_feature_script"] = "scripts/materialize_route_interaction_archive.py"
    output_metadata["source"]["source_dataset_sha256"] = _sha256(source_dataset)
    (output_dir / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "dataset_sha256": _sha256(dataset_path),
        "metadata_sha256": _sha256(output_dir / "metadata.json"),
        "source_dataset_sha256": _sha256(source_dataset),
        "source_metadata_sha256": _sha256(source_metadata),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorboard_logdir": str(tensorboard_logdir),
        "development_only": True,
        "locked_test_opened": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with SummaryWriter(log_dir=str(tensorboard_logdir), flush_secs=1) as writer:
        sample_type = np.asarray(arrays["sample_type"])
        runtime = sample_type == 0
        writer.add_scalar("Archive/sample_count", float(len(feature)), 0)
        writer.add_scalar("Archive/runtime_sample_count", float(runtime.sum()), 0)
        writer.add_scalar(
            "Archive/route_geometry_valid_fraction",
            float(np.mean(np.asarray(arrays["route_geometry_valid"])[runtime])),
            0,
        )
        writer.add_scalar(
            "Archive/cbf_first_step_feasible_fraction",
            float(np.mean(np.asarray(arrays["labels_cbf_feasible"])[runtime, 0])),
            0,
        )
        writer.add_scalar(
            "Archive/branch_failure_within_horizon_fraction",
            float(np.mean(np.asarray(arrays["earliest_failure_step"])[runtime] <= 5)),
            0,
        )
        writer.add_scalar(
            "Archive/negative_boundary_ttc_fraction",
            float(np.mean(np.asarray(arrays["labels_boundary_ttc"]).min(axis=1) < 10.0)),
            0,
        )
        writer.add_scalar(
            "Archive/negative_pairwise_ttc_fraction",
            float(np.mean(np.asarray(arrays["labels_pairwise_ttc"]).min(axis=1) < 10.0)),
            0,
        )
        writer.add_scalar(
            "Archive/negative_acceleration_slack_fraction",
            float(np.mean(np.asarray(arrays["labels_acceleration_slack"]).min(axis=1) < 0.0)),
            0,
        )
        writer.add_scalar(
            "Archive/boundary_clearance_negative_fraction",
            float(np.mean(np.asarray(arrays["labels_boundary_clearance"]).min(axis=1) < 0.0)),
            0,
        )
        for index, label in enumerate(metadata.get("route_labels", [])):
            writer.add_scalar(
                f"Archive/route_samples/{label}",
                float(metadata.get("route_counts", {}).get(label, 0)),
                index,
            )
        writer.add_scalar("Archive/route_interaction_group_count", float(sum(group_sizes.values())), 0)
        writer.add_scalar("Archive/route_interaction_feature_abs_mean", float(np.abs(feature).mean()), 0)
        writer.add_scalar("Archive/route_interaction_feature_max_abs", float(np.abs(feature).max()), 0)
        writer.add_text("Archive/dataset_version", str(dataset_version), 0)
        writer.add_text("Archive/source_dataset_sha256", _sha256(source_dataset), 0)
        writer.add_text("Archive/output_dataset_sha256", _sha256(dataset_path), 0)
        writer.add_text("Archive/feature_contract", json.dumps(output_metadata["derived_feature_contract"], sort_keys=True), 0)
    return {
        "dataset": str(dataset_path),
        "metadata": str(output_dir / "metadata.json"),
        "sample_count": int(len(feature)),
        "group_sizes": group_sizes,
        "dataset_sha256": _sha256(dataset_path),
        "tensorboard": str(tensorboard_logdir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()
    result = materialize(
        args.source_dataset,
        args.source_metadata,
        args.output_dir,
        args.tensorboard_logdir,
        args.dataset_version,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
