"""Audit source stability between P16 calibration and P17 validation bundles.

This is a model-independent distributional gate.  It does not claim that a
JEPA prediction is calibrated; it only checks whether each source role has
stable observed outcome rates on disjoint seed blocks before model training.
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
METRICS = (
    "strict_margin_cell_rate",
    "strict_margin_row_rate",
    "branch_failure_row_rate",
    "cbf_infeasible_cell_rate",
)
DEFAULT_THRESHOLDS = {
    "strict_margin_cell_rate": 0.05,
    "strict_margin_row_rate": 0.10,
    "branch_failure_row_rate": 0.05,
    "cbf_infeasible_cell_rate": 0.10,
}


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


def _role(name: str) -> str:
    value = str(name).lower()
    if "virtual_probe" in value:
        return "virtual_probe"
    if "route_outcomes" in value:
        return "route_outcomes"
    raise ValueError(f"cannot infer source role from {name!r}")


def _load_bundle(dataset: Path, metadata_path: Path, expected_split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset = dataset.resolve()
    metadata_path = metadata_path.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("bundle metadata must be an object")
    if metadata.get("split") != expected_split:
        raise ValueError(f"expected {expected_split} bundle, got {metadata.get('split')!r}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError("source-stability audit requires a closed development bundle")
    with np.load(dataset, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "calibration_source_id",
        "calibration_source_name",
        "labels_strict_margin_violation",
        "labels_branch_failure",
        "labels_cbf_infeasible",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"bundle is missing fields: {missing}")
    rows = int(arrays["calibration_source_id"].shape[0])
    if rows <= 0 or arrays["calibration_source_name"].shape != (rows,):
        raise ValueError("bundle source fields have invalid shapes")
    for field in ("labels_strict_margin_violation", "labels_branch_failure", "labels_cbf_infeasible"):
        if arrays[field].shape[0] != rows or arrays[field].ndim != 2:
            raise ValueError(f"bundle field {field} has invalid shape")
    return arrays, {
        "dataset": str(dataset),
        "metadata": str(metadata_path),
        "dataset_sha256": _sha256(dataset),
        "metadata_sha256": _sha256(metadata_path),
        "rows": rows,
        "metadata_value": metadata,
    }


def _source_rates(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    names = np.asarray(arrays["calibration_source_name"]).astype(str)
    result: dict[str, dict[str, float | int]] = {}
    for source_name in np.unique(names):
        role = _role(source_name)
        mask = names == source_name
        strict = np.asarray(arrays["labels_strict_margin_violation"])[mask] > 0.5
        branch = np.asarray(arrays["labels_branch_failure"])[mask] > 0.5
        cbf = np.asarray(arrays["labels_cbf_infeasible"])[mask] > 0.5
        result[role] = {
            "source_name": source_name,
            "rows": int(mask.sum()),
            "strict_margin_cell_rate": float(strict.mean()),
            "strict_margin_row_rate": float(np.any(strict, axis=1).mean()),
            "branch_failure_row_rate": float(np.any(branch, axis=1).mean()),
            "cbf_infeasible_cell_rate": float(cbf.mean()),
        }
    return result


def compare_bundles(
    calibration_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    *,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
    minimum_rows: int = 1000,
) -> dict[str, Any]:
    calibration = _source_rates(calibration_arrays)
    validation = _source_rates(validation_arrays)
    required_roles = ("virtual_probe", "route_outcomes")
    per_source: dict[str, Any] = {}
    checks: list[bool] = []
    for role in required_roles:
        if role not in calibration or role not in validation:
            per_source[role] = {"status": "missing"}
            checks.append(False)
            continue
        cal = calibration[role]
        val = validation[role]
        deltas = {metric: abs(float(val[metric]) - float(cal[metric])) for metric in METRICS}
        metric_checks = {
            metric: {
                "absolute_delta": deltas[metric],
                "threshold": float(thresholds[metric]),
                "passed": deltas[metric] <= float(thresholds[metric]),
            }
            for metric in METRICS
        }
        source_passed = int(cal["rows"]) >= minimum_rows and int(val["rows"]) >= minimum_rows and all(
            item["passed"] for item in metric_checks.values()
        )
        checks.append(source_passed)
        per_source[role] = {
            "calibration": cal,
            "validation": val,
            "deltas": deltas,
            "metric_checks": metric_checks,
            "minimum_rows": minimum_rows,
            "passed": source_passed,
        }
    return {
        "metric_type": "model_independent_source_stability",
        "thresholds": {name: float(value) for name, value in thresholds.items()},
        "minimum_rows": int(minimum_rows),
        "sources": per_source,
        "gate_passed": bool(all(checks) and len(checks) == len(required_roles)),
        "online_training_authorized": False,
    }


def _write_tensorboard(logdir: Path, comparison: Mapping[str, Any]) -> None:
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_scalar("P18/gate_passed", float(bool(comparison["gate_passed"])), 0)
        for role, result in comparison["sources"].items():
            if result.get("status") == "missing":
                continue
            writer.add_scalar(f"P18/{role}/calibration_rows", float(result["calibration"]["rows"]), 0)
            writer.add_scalar(f"P18/{role}/validation_rows", float(result["validation"]["rows"]), 0)
            for metric in METRICS:
                writer.add_scalar(f"P18/{role}/{metric}_calibration", float(result["calibration"][metric]), 0)
                writer.add_scalar(f"P18/{role}/{metric}_validation", float(result["validation"][metric]), 0)
                writer.add_scalar(f"P18/{role}/{metric}_absolute_delta", float(result["deltas"][metric]), 0)
        writer.add_text("P18/comparison", json.dumps(comparison, sort_keys=True), 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dataset", type=Path, required=True)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=1000)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("source stability audit requires --development-only")
    if args.minimum_rows <= 0:
        raise ValueError("minimum rows must be positive")
    output_dir = args.output_dir.resolve()
    tensorboard_dir = args.tensorboard_logdir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {tensorboard_dir}")
    calibration_arrays, calibration_info = _load_bundle(args.calibration_dataset, args.calibration_metadata, "calibration")
    validation_arrays, validation_info = _load_bundle(args.validation_dataset, args.validation_metadata, "validation")
    comparison = compare_bundles(calibration_arrays, validation_arrays, minimum_rows=args.minimum_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "calibration": {key: value for key, value in calibration_info.items() if key != "metadata_value"},
        "validation": {key: value for key, value in validation_info.items() if key != "metadata_value"},
    }
    result = {
        "protocol": "dn_mpc_pairwise_source_stability_v1",
        "metric_type": comparison["metric_type"],
        "comparison": comparison,
        "source": source,
        "git_revision": _git_revision(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "development_only": True,
        "locked_test_opened": False,
        "online_training_authorized": False,
        "tensorboard_logdir": str(tensorboard_dir),
    }
    (output_dir / "source_stability_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(
        json.dumps({
            "report_sha256": _sha256(output_dir / "source_stability_report.json"),
            "git_revision": _git_revision(),
            "calibration_dataset_sha256": calibration_info["dataset_sha256"],
            "validation_dataset_sha256": validation_info["dataset_sha256"],
            "tensorboard_logdir": str(tensorboard_dir),
            "development_only": True,
            "locked_test_opened": False,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_tensorboard(tensorboard_dir, comparison)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
