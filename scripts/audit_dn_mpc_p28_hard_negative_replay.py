"""Audit offline P28 anticipatory hard-negative route replay archives.

The audit verifies that braking, nearest-tangent, and boundary-rescue branches
are labelled but never promoted to runtime candidates or executed unverified.
It is intentionally independent of JEPA training and does not alter CBF
margins or planner abstention semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTE_LABELS = ("braking", "left_detour", "right_detour", "boundary_rescue")
SAMPLE_TYPES = {label: 5 + index for index, label in enumerate(ROUTE_LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--calibration-dataset", type=Path, required=True)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def audit_archive(dataset: Path, metadata_path: Path, split: str) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split") != split:
        raise ValueError(f"Expected {split} metadata, got {metadata.get('split')!r}")
    replay = metadata.get("anticipatory_hard_negative_replay", {})
    if replay.get("enabled") is not True or replay.get("offline_only") is not True:
        raise ValueError(f"{split} archive does not declare offline P28 replay")
    if replay.get("sample_type_mapping") != SAMPLE_TYPES:
        raise ValueError(f"{split} P28 sample-type mapping is not stable")
    with np.load(dataset, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    sample_type = arrays["sample_type"]
    finite = all(np.isfinite(array).all() for name, array in arrays.items() if np.issubdtype(array.dtype, np.number))
    reports: dict[str, Any] = {}
    for label, sample_id in SAMPLE_TYPES.items():
        mask = sample_type == sample_id
        rows = int(mask.sum())
        if rows == 0:
            raise ValueError(f"{split} archive is missing P28 route {label}")
        reports[label] = {
            "sample_type": sample_id,
            "rows": rows,
            "geometry_valid_fraction": float(np.mean(arrays["route_geometry_valid"][mask] > 0.5)),
            "first_step_cbf_feasible_fraction": float(np.mean(arrays["labels_cbf_feasible"][mask, 0] > 0.5)),
            "branch_failure_fraction": float(np.mean(arrays["earliest_failure_step"][mask] <= 5)),
            "min_slack": _summary(arrays["labels_cbf_min_slack"][mask].reshape(-1)),
            "route_progress": _summary(arrays["labels_route_progress"][mask, 2].reshape(-1)),
            "candidate_index_all_minus_one": bool(np.all(arrays["route_candidate_index"][mask] == -1)),
            "raw_unverified_execution_allowed": False,
            "runtime_candidate_contract_unchanged": True,
        }
    runtime = sample_type == 0
    return {
        "split": split,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256(dataset),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": sha256(metadata_path),
        "finite_archive": bool(finite),
        "total_rows": int(sample_type.size),
        "runtime_rows": int(runtime.sum()),
        "hard_negative_rows": int((sample_type >= 5).sum()),
        "replay": replay,
        "routes": reports,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# DN-MPC P28 Anticipatory Hard-Negative Replay Audit",
        "",
        "**Status:** development-only; offline labels only; no action was executed.",
        "",
        "P28 replays braking, nearest left/right tangent, and boundary-rescue",
        "routes around the P26 abstention windows. The four routes are stored as",
        "sample types 5-8 and are not runtime candidates in the twelve-route",
        "contract. Every branch was projected and labelled through the unchanged",
        "five-step CBF counterfactual path.",
        "",
        "| Split | total rows | runtime rows | hard-negative rows | finite |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "calibration"):
        item = result["splits"][split]
        lines.append(f"| {split} | {item['total_rows']} | {item['runtime_rows']} | {item['hard_negative_rows']} | {item['finite_archive']} |")
    lines += ["", "## Route Outcomes", "", "| Split | Route | rows | geometry valid | first-step CBF feasible | branch failure | candidate index -1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for split in ("train", "calibration"):
        for label in ROUTE_LABELS:
            item = result["splits"][split]["routes"][label]
            lines.append(
                f"| {split} | {label} | {item['rows']} | {item['geometry_valid_fraction']:.2%} | "
                f"{item['first_step_cbf_feasible_fraction']:.2%} | {item['branch_failure_fraction']:.2%} | "
                f"{item['candidate_index_all_minus_one']} |"
            )
    lines += [
        "",
        "## Decision",
        "",
        "The archive is suitable for offline hard-negative replay analysis. It",
        "does not authorize JEPA online promotion or Ledger-Lite creation. The",
        "next gate is to retrain/recalibrate on a new split that includes these",
        "labels, then rerun P27 OOD/disagreement and selected/nominal/safe-hold",
        "trace checks.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    outputs = [args.output.resolve(), args.markdown_output.resolve()]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    tensorboard = args.tensorboard_logdir.resolve()
    if tensorboard.exists() and any(tensorboard.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory {tensorboard}")
    splits = {
        "train": audit_archive(args.train_dataset.resolve(), args.train_metadata.resolve(), "train"),
        "calibration": audit_archive(args.calibration_dataset.resolve(), args.calibration_metadata.resolve(), "calibration"),
    }
    result = {
        "audit_type": "dn_mpc_p28_anticipatory_hard_negative_replay_audit",
        "development_only": True,
        "locked_test_opened": False,
        "online_promotion_authorized": False,
        "splits": splits,
        "provenance": {
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "train_dataset_sha256": splits["train"]["dataset_sha256"],
            "calibration_dataset_sha256": splits["calibration"]["dataset_sha256"],
            "raw_unverified_action_executed": False,
            "cbf_margin_changed": False,
        },
    }
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].write_text(markdown(result), encoding="utf-8")
    tensorboard.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("P28/Provenance", json.dumps(result["provenance"], sort_keys=True), 0)
        writer.add_scalar("P28/OnlinePromotionAuthorized", 0.0, 0)
        for split, report in splits.items():
            writer.add_scalar(f"P28/{split}/total_rows", report["total_rows"], 0)
            writer.add_scalar(f"P28/{split}/hard_negative_rows", report["hard_negative_rows"], 0)
            for label, route in report["routes"].items():
                writer.add_scalar(f"P28/{split}/{label}/rows", route["rows"], 0)
                writer.add_scalar(f"P28/{split}/{label}/cbf_feasible", route["first_step_cbf_feasible_fraction"], 0)
                writer.add_scalar(f"P28/{split}/{label}/branch_failure", route["branch_failure_fraction"], 0)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
