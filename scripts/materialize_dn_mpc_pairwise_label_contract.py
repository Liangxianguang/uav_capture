"""Materialize the P13 pairwise outcome-label contract.

The source archive is immutable.  This utility appends explicit, non-
interchangeable labels for (1) constant-velocity TTC warning, (2) strict
operational-margin violation, (3) verified CBF infeasibility, and (4) actual
counterfactual branch failure.  It is intentionally offline-only and refuses
to process a locked archive.
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
SAMPLE_TYPES = {
    "runtime": 0,
    "boundary_shadow": 1,
    "near_pass": 2,
    "formation_crossing": 3,
    "split_merge": 4,
}
CONTRACT_VERSION = "dn_mpc_pairwise_outcome_labels_v1"
REQUIRED_FIELDS = (
    "labels_pairwise_ttc",
    "labels_inter_agent_clearance",
    "labels_cbf_feasible",
    "earliest_failure_step",
    "branch_terminated",
    "sample_type",
)


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


def _load_source(dataset: Path, metadata_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("source metadata must be a JSON object")
    if metadata.get("split") != "calibration":
        raise ValueError(f"P13 requires split=calibration, got {metadata.get('split')!r}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError("P13 requires a closed development archive")
    with np.load(dataset, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    missing = [name for name in REQUIRED_FIELDS if name not in arrays]
    if missing:
        raise ValueError(f"source archive is missing fields: {missing}")
    ttc = np.asarray(arrays["labels_pairwise_ttc"])
    if ttc.ndim != 2:
        raise ValueError("labels_pairwise_ttc must have shape [rows, horizon]")
    rows, horizon = ttc.shape
    for name in ("labels_inter_agent_clearance", "labels_cbf_feasible"):
        if np.asarray(arrays[name]).shape != (rows, horizon):
            raise ValueError(f"{name} must have shape {(rows, horizon)}")
    for name in ("earliest_failure_step", "branch_terminated", "sample_type"):
        if np.asarray(arrays[name]).shape != (rows,):
            raise ValueError(f"{name} must have shape {(rows,)}")
    return arrays, metadata


def materialize_labels(
    arrays: Mapping[str, np.ndarray],
    *,
    pairwise_margin_m: float = 0.35,
    ttc_threshold_s: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = {name: np.asarray(value) for name, value in arrays.items()}
    ttc = np.asarray(source["labels_pairwise_ttc"], dtype=np.float64)
    clearance = np.asarray(source["labels_inter_agent_clearance"], dtype=np.float64)
    cbf_feasible = np.asarray(source["labels_cbf_feasible"], dtype=np.float64)
    failure_step = np.asarray(source["earliest_failure_step"], dtype=np.int64)
    sample_type = np.asarray(source["sample_type"], dtype=np.int64)
    horizon = int(ttc.shape[1])
    shadow = sample_type == SAMPLE_TYPES["boundary_shadow"]
    failure_by_horizon = (
        failure_step[:, None] <= (np.arange(horizon, dtype=np.int64)[None, :] + 1)
    ) & ~shadow[:, None]
    source["labels_predicted_ttc_hazard"] = (
        np.isfinite(ttc) & (ttc <= float(ttc_threshold_s))
    ).astype(np.float32)
    source["labels_strict_margin_violation"] = (
        np.isfinite(clearance) & (clearance < float(pairwise_margin_m))
    ).astype(np.float32)
    source["labels_cbf_infeasible"] = (
        ~np.isfinite(cbf_feasible) | (cbf_feasible <= 0.5)
    ).astype(np.float32)
    source["labels_branch_failure"] = failure_by_horizon.astype(np.float32)
    contract = {
        "version": CONTRACT_VERSION,
        "horizon_axis": "labels[..., horizon_step]",
        "labels": {
            "labels_predicted_ttc_hazard": {
                "definition": "pairwise TTC under constant-relative-velocity extrapolation <= threshold",
                "source": "labels_pairwise_ttc",
                "threshold_seconds": float(ttc_threshold_s),
                "is_warning_not_failure": True,
            },
            "labels_strict_margin_violation": {
                "definition": "physical inter-agent clearance below operational CBF margin",
                "source": "labels_inter_agent_clearance",
                "threshold_m": float(pairwise_margin_m),
            },
            "labels_cbf_infeasible": {
                "definition": "primary CBF feasibility label is false or non-finite",
                "source": "labels_cbf_feasible",
            },
            "labels_branch_failure": {
                "definition": "counterfactual branch fails at or before this horizon step",
                "source": "earliest_failure_step",
                "boundary_shadow_excluded": True,
                "boundary_shadow_reason": "offline synthetic negative, never executed",
            },
        },
        "sample_type_mapping": SAMPLE_TYPES,
        "rows": int(ttc.shape[0]),
        "horizon": horizon,
        "boundary_shadow_rows": int(shadow.sum()),
    }
    return source, contract


def _rate(array: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    values = np.asarray(array, dtype=bool)
    if mask is not None:
        row_mask = np.asarray(mask, dtype=bool)
        if row_mask.shape != values.shape:
            try:
                row_mask = np.broadcast_to(row_mask, values.shape)
            except ValueError as error:
                raise ValueError(f"mask shape {row_mask.shape} cannot broadcast to {values.shape}") from error
        values = values[row_mask]
    return float(values.mean()) if values.size else None


def _write_tensorboard(
    logdir: Path,
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    *,
    source_dataset: Path,
    source_metadata: Path,
    source_dataset_sha256: str,
    source_metadata_sha256: str,
    tensorboard_namespace: str,
) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    namespace = str(tensorboard_namespace).strip()
    if not namespace or any(char in namespace for char in "/\\"):
        raise ValueError("tensorboard namespace must be a non-empty path-safe token")
    sample_type = np.asarray(arrays["sample_type"], dtype=np.int64)
    runtime = sample_type == SAMPLE_TYPES["runtime"]
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text(
            f"{namespace}/provenance",
            json.dumps(
                {
                    "source_dataset": str(source_dataset),
                    "source_dataset_sha256": source_dataset_sha256,
                    "source_metadata": str(source_metadata),
                    "source_metadata_sha256": source_metadata_sha256,
                    "contract": contract,
                },
                sort_keys=True,
            ),
            0,
        )
        for name in (
            "labels_predicted_ttc_hazard",
            "labels_strict_margin_violation",
            "labels_cbf_infeasible",
            "labels_branch_failure",
        ):
            writer.add_scalar(f"{namespace}/runtime/{name}_cell_rate", _rate(arrays[name], runtime[:, None]) or 0.0, 0)
            for sample_name, sample_id in SAMPLE_TYPES.items():
                mask = sample_type == sample_id
                if mask.any():
                    writer.add_scalar(
                        f"{namespace}/{sample_name}/{name}_cell_rate",
                        _rate(np.asarray(arrays[name]), mask[:, None]) or 0.0,
                        0,
                    )
        writer.add_scalar(f"{namespace}/runtime/rows", float(runtime.sum()), 0)
        writer.add_scalar(f"{namespace}/boundary_shadow/rows", float((~runtime).sum()), 0)
        writer.add_text(f"{namespace}/contract", json.dumps(contract, sort_keys=True), 0)


def materialize(
    source_dataset: Path,
    source_metadata: Path,
    output_dir: Path,
    tensorboard_logdir: Path,
    *,
    dataset_version: str,
    pairwise_margin_m: float,
    ttc_threshold_s: float,
    tensorboard_namespace: str = "P13",
) -> dict[str, Any]:
    source_dataset = source_dataset.resolve()
    source_metadata = source_metadata.resolve()
    output_dir = output_dir.resolve()
    tensorboard_logdir = tensorboard_logdir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if tensorboard_logdir.exists() and any(tensorboard_logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard logdir: {tensorboard_logdir}")
    arrays, source_metadata_value = _load_source(source_dataset, source_metadata)
    arrays, contract = materialize_labels(
        arrays,
        pairwise_margin_m=pairwise_margin_m,
        ttc_threshold_s=ttc_threshold_s,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "route_identity_pairwise_outcome_labels.npz"
    metadata_path = output_dir / "metadata.json"
    np.savez_compressed(dataset_path, **arrays)
    output_metadata = dict(source_metadata_value)
    output_metadata["dataset_version"] = str(dataset_version)
    output_metadata["pairwise_label_contract"] = contract
    output_metadata["source"] = dict(output_metadata.get("source", {}))
    output_metadata["source"]["label_contract_script"] = "scripts/materialize_dn_mpc_pairwise_label_contract.py"
    output_metadata["source"]["source_dataset_sha256"] = _sha256(source_dataset)
    output_metadata["source"]["source_metadata_sha256"] = _sha256(source_metadata)
    output_metadata["array_shapes"] = {name: list(value.shape) for name, value in arrays.items()}
    metadata_path.write_text(json.dumps(output_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "dataset_sha256": _sha256(dataset_path),
        "metadata_sha256": _sha256(metadata_path),
        "source_dataset_sha256": _sha256(source_dataset),
        "source_metadata_sha256": _sha256(source_metadata),
        "git_revision": _git_revision(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "tensorboard_logdir": str(tensorboard_logdir),
        "tensorboard_namespace": str(tensorboard_namespace),
        "development_only": True,
        "locked_test_opened": False,
        "pairwise_label_contract": contract,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tensorboard(
        tensorboard_logdir,
        arrays,
        contract,
        source_dataset=source_dataset,
        source_metadata=source_metadata,
        source_dataset_sha256=provenance["source_dataset_sha256"],
        source_metadata_sha256=provenance["source_metadata_sha256"],
        tensorboard_namespace=tensorboard_namespace,
    )
    summary = {
        "dataset": str(dataset_path),
        "metadata": str(metadata_path),
        "provenance": str(output_dir / "provenance.json"),
        "tensorboard": str(tensorboard_logdir),
        "dataset_sha256": provenance["dataset_sha256"],
        "metadata_sha256": provenance["metadata_sha256"],
        "source_dataset_sha256": provenance["source_dataset_sha256"],
        "source_metadata_sha256": provenance["source_metadata_sha256"],
        "contract": contract,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--dataset-version", default="dn_mpc_route_identity_chunk5_pairwise_outcome_labels_v1")
    parser.add_argument("--pairwise-margin-m", type=float, default=0.35)
    parser.add_argument("--ttc-threshold-s", type=float, default=1.0)
    parser.add_argument(
        "--tensorboard-namespace",
        default="P13",
        help="Namespace prefix for label metrics, e.g. P15 for a later phase.",
    )
    args = parser.parse_args()
    if args.pairwise_margin_m < 0.0 or args.ttc_threshold_s < 0.0:
        raise ValueError("pairwise margin and TTC threshold must be non-negative")
    result = materialize(
        args.source_dataset,
        args.source_metadata,
        args.output_dir,
        args.tensorboard_logdir,
        dataset_version=args.dataset_version,
        pairwise_margin_m=args.pairwise_margin_m,
        ttc_threshold_s=args.ttc_threshold_s,
        tensorboard_namespace=args.tensorboard_namespace,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
