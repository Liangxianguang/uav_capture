"""Build a source-bound calibration bundle from P14 and P15 label archives.

The bundle is calibration-only.  P14 virtual probes provide physical
strict-margin positives without executing unsafe actions; P15 route branches
provide verified CBF/branch outcomes.  Their labels are concatenated without
reinterpreting either source, and source identifiers are stored explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_VERSION = "dn_mpc_pairwise_multisource_calibration_v1"
SOURCE_NAMES = ("p14_virtual_probe", "p15_route_outcomes")
REQUIRED_FIELDS = (
    "inputs",
    "action_history",
    "route_action_chunk",
    "route_relative_action_chunk",
    "route_pairwise_relative_action_chunk",
    "labels_predicted_ttc_hazard",
    "labels_strict_margin_violation",
    "labels_cbf_infeasible",
    "labels_branch_failure",
    "labels_inter_agent_clearance",
    "labels_cbf_feasible",
    "sample_type",
    "earliest_failure_step",
    "branch_terminated",
)
OPTIONAL_FIELDS = ("virtual_probe_mode", "virtual_probe_pair_index")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fresh(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {label}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_source(
    dataset: Path,
    metadata_path: Path,
    expected_source: str,
    *,
    expected_split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset = dataset.resolve()
    metadata_path = metadata_path.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{expected_source} metadata must be an object")
    if metadata.get("split") != expected_split:
        raise ValueError(f"{expected_source} must be a {expected_split} archive")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"{expected_source} must be closed development-only data")
    with np.load(dataset, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    missing = [name for name in REQUIRED_FIELDS if name not in arrays]
    if missing:
        raise ValueError(f"{expected_source} is missing fields: {missing}")
    rows = int(arrays["inputs"].shape[0])
    if rows <= 0:
        raise ValueError(f"{expected_source} is empty")
    for name in REQUIRED_FIELDS:
        if arrays[name].shape[0] != rows:
            raise ValueError(f"{expected_source} field {name} has inconsistent row count")
    if "virtual_probe" in expected_source and not np.all(arrays["sample_type"] == 5):
        raise ValueError("virtual probe source must use sample_type=5")
    return arrays, {
        "name": expected_source,
        "dataset": str(dataset),
        "metadata": str(metadata_path),
        "dataset_sha256": _sha256(dataset),
        "metadata_sha256": _sha256(metadata_path),
        "rows": rows,
        "metadata_value": metadata,
    }


def _fill_optional(name: str, rows: int, source: str, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if name in arrays:
        return np.asarray(arrays[name])
    if name in OPTIONAL_FIELDS:
        return np.full(rows, -1, dtype=np.int64)
    raise ValueError(f"Cannot fill missing required field {name} for {source}")


def build_bundle(
    p14_dataset: Path,
    p14_metadata: Path,
    p15_dataset: Path,
    p15_metadata: Path,
    *,
    expected_split: str = "calibration",
    source_names: tuple[str, str] = SOURCE_NAMES,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sources = [
        _load_source(p14_dataset, p14_metadata, source_names[0], expected_split=expected_split),
        _load_source(p15_dataset, p15_metadata, source_names[1], expected_split=expected_split),
    ]
    all_names = list(REQUIRED_FIELDS) + list(OPTIONAL_FIELDS)
    arrays: dict[str, np.ndarray] = {}
    for name in all_names:
        values = [_fill_optional(name, info["rows"], info["name"], source) for source, info in sources]
        reference_shape = values[0].shape[1:]
        if any(value.shape[1:] != reference_shape for value in values):
            raise ValueError(f"source field {name} has incompatible shapes")
        arrays[name] = np.concatenate(values, axis=0)
    source_ids = np.concatenate(
        [np.full(info["rows"], index, dtype=np.int64) for index, (_source, info) in enumerate(sources)]
    )
    arrays["calibration_source_id"] = source_ids
    arrays["calibration_source_name"] = np.asarray(
        [source_names[index] for index in source_ids], dtype="U32"
    )
    metadata = {
        "dataset_version": CONTRACT_VERSION,
        "task": "dn_mpc_pairwise_multisource_calibration",
        "split": expected_split,
        "development_only": True,
        "locked_test_opened": False,
        "offline_only": True,
        "raw_unverified_action_executed": False,
        "label_semantics_preserved": True,
        "source_names": {str(index): name for index, name in enumerate(source_names)},
        "source_archives": [
            {key: value for key, value in info.items() if key != "metadata_value"}
            for _source, info in sources
        ],
        "source_sample_type_contract": {
            source_names[0]: {"sample_type": 5, "unsafe_actions_executed": False},
            source_names[1]: {"sample_type": "0..4", "unsafe_actions_executed": False},
        },
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "rows": int(arrays["inputs"].shape[0]),
        "source_rows": {info["name"]: info["rows"] for _source, info in sources},
        "expected_split": expected_split,
        "information_boundary": {
            "target_truth_used_only_for_offline_labels": True,
            "calibration_only": expected_split == "calibration",
            "locked_test_opened": False,
            "virtual_probe_actions_executed": False,
            "routes_advanced_only_after_verified_cbf": True,
        },
        "source": {
            "builder": "scripts/build_dn_mpc_pairwise_multisource_calibration.py",
            "builder_git_revision": _git_revision(),
        },
    }
    return arrays, metadata


def _rate(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values)
    row_mask = np.asarray(mask, dtype=bool)
    if row_mask.shape != values.shape:
        row_mask = np.broadcast_to(row_mask, values.shape)
    selected = values[row_mask]
    return float(selected.mean()) if selected.size else 0.0


def _write_tensorboard(
    logdir: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    source_names: tuple[str, str],
    namespace: str,
) -> None:
    source = np.asarray(arrays["calibration_source_id"], dtype=np.int64)
    strict = np.asarray(arrays["labels_strict_margin_violation"], dtype=np.float32)
    branch = np.asarray(arrays["labels_branch_failure"], dtype=np.float32)
    cbf = np.asarray(arrays["labels_cbf_infeasible"], dtype=np.float32)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_scalar(f"{namespace}/rows", float(len(source)), 0)
        for index, name in enumerate(source_names):
            mask = source == index
            writer.add_scalar(f"{namespace}/{name}/rows", float(mask.sum()), 0)
            writer.add_scalar(f"{namespace}/{name}/strict_margin_cell_rate", _rate(strict, mask[:, None]), 0)
            writer.add_scalar(f"{namespace}/{name}/strict_margin_row_rate", float(np.any(strict[mask] > 0.5, axis=1).mean()), 0)
            writer.add_scalar(f"{namespace}/{name}/branch_failure_row_rate", float(np.any(branch[mask] > 0.5, axis=1).mean()), 0)
            writer.add_scalar(f"{namespace}/{name}/cbf_infeasible_cell_rate", _rate(cbf, mask[:, None]), 0)
        if "virtual_probe_mode" in arrays:
            safe_hold = (source == 0) & (np.asarray(arrays["virtual_probe_mode"]) == 1)
            writer.add_scalar(f"{namespace}/{source_names[0]}/safe_hold_rows", float(safe_hold.sum()), 0)
        writer.add_text(f"{namespace}/contract", json.dumps(metadata, sort_keys=True), 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p14-dataset", type=Path, required=True)
    parser.add_argument("--p14-metadata", type=Path, required=True)
    parser.add_argument("--p15-dataset", type=Path, required=True)
    parser.add_argument("--p15-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    parser.add_argument("--split", choices=("validation", "calibration"), default="calibration")
    parser.add_argument("--source-0-name", default="p14_virtual_probe")
    parser.add_argument("--source-1-name", default="p15_route_outcomes")
    parser.add_argument("--tensorboard-namespace", default="P16")
    parser.add_argument("--dataset-version")
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("multisource calibration requires --development-only")
    output_dir = _fresh(args.output_dir, "multisource calibration output")
    tensorboard_dir = _fresh(args.tensorboard_logdir, "multisource calibration TensorBoard logdir")
    arrays, metadata = build_bundle(
        args.p14_dataset,
        args.p14_metadata,
        args.p15_dataset,
        args.p15_metadata,
        expected_split=args.split,
        source_names=(args.source_0_name, args.source_1_name),
    )
    dataset_version = args.dataset_version or f"dn_mpc_pairwise_multisource_{args.split}_v1"
    metadata["dataset_version"] = dataset_version
    metadata["task"] = f"dn_mpc_pairwise_multisource_{args.split}"
    dataset_path = output_dir / f"pairwise_multisource_{args.split}.npz"
    metadata_path = output_dir / "metadata.json"
    np.savez_compressed(dataset_path, **arrays)
    metadata = dict(metadata)
    metadata["dataset_sha256"] = _sha256(dataset_path)
    metadata["metadata_path"] = str(metadata_path)
    metadata["tensorboard_logdir"] = str(tensorboard_dir)
    metadata["tensorboard_namespace"] = str(args.tensorboard_namespace)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "dataset_sha256": metadata["dataset_sha256"],
        "metadata_sha256": _sha256(metadata_path),
        "git_revision": _git_revision(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "development_only": True,
        "locked_test_opened": False,
        "offline_only": True,
        "raw_unverified_action_executed": False,
        "split": args.split,
        "dataset_version": dataset_version,
        "tensorboard_namespace": str(args.tensorboard_namespace),
        "tensorboard_logdir": str(tensorboard_dir),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    namespace = str(args.tensorboard_namespace).strip()
    if not namespace or any(char in namespace for char in "/\\"):
        raise ValueError("tensorboard namespace must be a non-empty path-safe token")
    _write_tensorboard(
        tensorboard_dir,
        arrays,
        metadata,
        source_names=(args.source_0_name, args.source_1_name),
        namespace=namespace,
    )
    summary = {
        "dataset": str(dataset_path),
        "metadata": str(metadata_path),
        "provenance": str(output_dir / "provenance.json"),
        "tensorboard": str(tensorboard_dir),
        "dataset_sha256": metadata["dataset_sha256"],
        "source_rows": metadata["source_rows"],
        "raw_unverified_action_executed": False,
        "locked_test_opened": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
