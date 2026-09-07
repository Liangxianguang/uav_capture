"""Audit pairwise TTC label semantics against clearance and CBF outcomes.

This is a read-only P12 audit.  It deliberately does not train a model,
change a safety margin, or make any runtime checkpoint eligible.  The archive
stores one row per route/defender and one label per rollout horizon; the
report therefore exposes both cell-level and candidate-level rates so a high
rate cannot be hidden by an ambiguous aggregation.
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
REQUIRED_KEYS = (
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
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _counts(mask: np.ndarray) -> dict[str, int | float | None]:
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.size)
    positive = int(mask.sum())
    return {
        "count": total,
        "positive_count": positive,
        "positive_rate": _ratio(positive, total),
    }


def _validate_archive(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> tuple[int, int]:
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Archive is missing required fields: {missing}")
    ttc = np.asarray(arrays["labels_pairwise_ttc"])
    clearance = np.asarray(arrays["labels_inter_agent_clearance"])
    cbf = np.asarray(arrays["labels_cbf_feasible"])
    if ttc.ndim != 2 or clearance.shape != ttc.shape or cbf.shape != ttc.shape:
        raise ValueError(
            "pairwise TTC, inter-agent clearance, and CBF feasibility must all have shape [rows, horizon]"
        )
    rows, horizon = ttc.shape
    for key in ("sample_type", "earliest_failure_step", "branch_terminated"):
        value = np.asarray(arrays[key])
        if value.shape != (rows,):
            raise ValueError(f"{key} must have shape [{rows}], got {value.shape}")
    if metadata.get("split") != "calibration":
        raise ValueError(f"P12 requires a calibration archive, got split={metadata.get('split')!r}")
    if metadata.get("locked_test_opened") is True:
        raise ValueError("P12 refuses an archive that opened a locked test")
    if not bool(metadata.get("development_only", False)):
        raise ValueError("P12 requires development_only=true")
    return rows, horizon


def _cell_metrics(
    ttc: np.ndarray,
    clearance: np.ndarray,
    cbf_feasible: np.ndarray,
    failure_by_horizon: np.ndarray,
    *,
    pairwise_margin_m: float,
    ttc_threshold_s: float,
) -> dict[str, Any]:
    hazard = np.isfinite(ttc) & (ttc <= float(ttc_threshold_s))
    strict_violation = np.isfinite(clearance) & (clearance < float(pairwise_margin_m))
    cbf_ok = np.isfinite(cbf_feasible) & (cbf_feasible > 0.5)
    cbf_infeasible = ~cbf_ok
    safe_clearance = ~strict_violation

    def event(mask: np.ndarray) -> dict[str, int | float | None]:
        return _counts(mask)

    hazard_count = int(hazard.sum())
    return {
        "cells": int(hazard.size),
        "pairwise_ttc_hazard": event(hazard),
        "strict_pairwise_margin_violation": event(strict_violation),
        "hazard_and_clearance_safe": event(hazard & safe_clearance),
        "hazard_and_cbf_feasible": event(hazard & cbf_ok),
        "hazard_and_cbf_infeasible": event(hazard & cbf_infeasible),
        "hazard_and_branch_failure": event(hazard & failure_by_horizon),
        "conditional_rates": {
            "clearance_safe_given_hazard": _ratio(int((hazard & safe_clearance).sum()), hazard_count),
            "cbf_feasible_given_hazard": _ratio(int((hazard & cbf_ok).sum()), hazard_count),
            "cbf_infeasible_given_hazard": _ratio(int((hazard & cbf_infeasible).sum()), hazard_count),
            "branch_failure_given_hazard": _ratio(int((hazard & failure_by_horizon).sum()), hazard_count),
        },
        "thresholds": {
            "pairwise_ttc_seconds": float(ttc_threshold_s),
            "strict_pairwise_margin_m": float(pairwise_margin_m),
        },
    }


def _candidate_metrics(
    ttc: np.ndarray,
    clearance: np.ndarray,
    cbf_feasible: np.ndarray,
    failure_by_horizon: np.ndarray,
    *,
    pairwise_margin_m: float,
    ttc_threshold_s: float,
) -> dict[str, Any]:
    hazard = np.isfinite(ttc) & (ttc <= float(ttc_threshold_s))
    strict_violation = np.isfinite(clearance) & (clearance < float(pairwise_margin_m))
    cbf_ok = np.isfinite(cbf_feasible) & (cbf_feasible > 0.5)
    safe_clearance = ~strict_violation
    hazard_candidate = np.any(hazard, axis=1)
    return {
        "candidates": int(hazard.shape[0]),
        "pairwise_ttc_hazard": _counts(hazard_candidate),
        "strict_pairwise_margin_violation": _counts(np.any(strict_violation, axis=1)),
        "hazard_and_clearance_safe": _counts(np.any(hazard & safe_clearance, axis=1)),
        "hazard_and_cbf_feasible": _counts(np.any(hazard & cbf_ok, axis=1)),
        "hazard_and_cbf_infeasible": _counts(np.any(hazard & ~cbf_ok, axis=1)),
        "hazard_and_branch_failure": _counts(np.any(hazard & failure_by_horizon, axis=1)),
        "conditional_rates": {
            "clearance_safe_given_hazard": _ratio(
                int(np.any(hazard & safe_clearance, axis=1).sum()), int(hazard_candidate.sum())
            ),
            "cbf_feasible_given_hazard": _ratio(
                int(np.any(hazard & cbf_ok, axis=1).sum()), int(hazard_candidate.sum())
            ),
            "cbf_infeasible_given_hazard": _ratio(
                int(np.any(hazard & ~cbf_ok, axis=1).sum()), int(hazard_candidate.sum())
            ),
            "branch_failure_given_hazard": _ratio(
                int(np.any(hazard & failure_by_horizon, axis=1).sum()), int(hazard_candidate.sum())
            ),
        },
        "thresholds": {
            "pairwise_ttc_seconds": float(ttc_threshold_s),
            "strict_pairwise_margin_m": float(pairwise_margin_m),
        },
    }


def summarize_archive(
    archive_path: Path,
    metadata_path: Path,
    *,
    pairwise_margin_m: float = 0.35,
    ttc_threshold_s: float = 1.0,
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    metadata_path = metadata_path.resolve()
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(value) for key, value in loaded.items()}
    rows, horizon = _validate_archive(arrays, metadata)
    ttc = np.asarray(arrays["labels_pairwise_ttc"], dtype=np.float64)
    clearance = np.asarray(arrays["labels_inter_agent_clearance"], dtype=np.float64)
    cbf_feasible = np.asarray(arrays["labels_cbf_feasible"], dtype=np.float64)
    failure_step = np.asarray(arrays["earliest_failure_step"], dtype=np.int64)
    branch_terminated = np.asarray(arrays["branch_terminated"], dtype=bool)
    # A failed branch is invalid from its first rejected horizon onward.  A
    # branch that only terminates after all labels were emitted is retained as
    # a separate provenance count, not silently treated as an early failure.
    sample_type = np.asarray(arrays["sample_type"], dtype=np.int64)
    boundary_shadow_rows = sample_type == SAMPLE_TYPES["boundary_shadow"]
    failure_by_horizon = (
        failure_step[:, None] <= (np.arange(horizon, dtype=np.int64)[None, :] + 1)
    ) & ~boundary_shadow_rows[:, None]
    reports: dict[str, Any] = {}
    for name, sample_type_id in SAMPLE_TYPES.items():
        mask = sample_type == sample_type_id
        if not mask.any():
            reports[name] = {"rows": 0, "cell_level": None, "candidate_level": None}
            continue
        reports[name] = {
            "rows": int(mask.sum()),
            "branch_terminated_rows": int(branch_terminated[mask].sum()),
            "early_failure_rows": int(np.any(failure_by_horizon[mask], axis=1).sum()),
            "boundary_shadow_rows": int(np.sum(mask & boundary_shadow_rows)),
            "cell_level": _cell_metrics(
                ttc[mask],
                clearance[mask],
                cbf_feasible[mask],
                failure_by_horizon[mask],
                pairwise_margin_m=pairwise_margin_m,
                ttc_threshold_s=ttc_threshold_s,
            ),
            "candidate_level": _candidate_metrics(
                ttc[mask],
                clearance[mask],
                cbf_feasible[mask],
                failure_by_horizon[mask],
                pairwise_margin_m=pairwise_margin_m,
                ttc_threshold_s=ttc_threshold_s,
            ),
        }
    return {
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "dataset_version": metadata.get("dataset_version"),
        "split": metadata.get("split"),
        "rows": rows,
        "horizon": horizon,
        "sample_type_mapping": SAMPLE_TYPES,
        "pairwise_margin_m": float(pairwise_margin_m),
        "ttc_threshold_s": float(ttc_threshold_s),
        "branch_terminated_rows": int(branch_terminated.sum()),
        "early_failure_rows": int(np.any(failure_by_horizon, axis=1).sum()),
        "boundary_shadow_rows_excluded_from_branch_failure": int(boundary_shadow_rows.sum()),
        "reports": reports,
    }


def _add_tensorboard(writer: SummaryWriter, block: str, result: Mapping[str, Any]) -> None:
    writer.add_text(
        f"P12/{block}/provenance",
        json.dumps(
            {
                "archive": result["archive"],
                "archive_sha256": result["archive_sha256"],
                "metadata": result["metadata"],
                "metadata_sha256": result["metadata_sha256"],
                "dataset_version": result["dataset_version"],
            },
            sort_keys=True,
        ),
        0,
    )
    for name, report in result["reports"].items():
        if report["rows"] == 0:
            continue
        for level in ("cell_level", "candidate_level"):
            prefix = f"P12/{block}/{name}/{level}"
            level_report = report[level]
            writer.add_scalar(f"{prefix}/pairwise_ttc_hazard_rate", level_report["pairwise_ttc_hazard"]["positive_rate"] or 0.0, 0)
            writer.add_scalar(f"{prefix}/strict_margin_violation_rate", level_report["strict_pairwise_margin_violation"]["positive_rate"] or 0.0, 0)
            for metric_name, metric_value in level_report["conditional_rates"].items():
                if metric_value is not None:
                    writer.add_scalar(f"{prefix}/{metric_name}", metric_value, 0)
        writer.add_scalar(f"P12/{block}/{name}/rows", float(report["rows"]), 0)
        writer.add_scalar(f"P12/{block}/{name}/branch_terminated_rows", float(report["branch_terminated_rows"]), 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-archive", type=Path, required=True)
    parser.add_argument("--original-metadata", type=Path, required=True)
    parser.add_argument("--fresh-archive", type=Path, required=True)
    parser.add_argument("--fresh-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--pairwise-margin-m", type=float, default=0.35)
    parser.add_argument("--ttc-threshold-s", type=float, default=1.0)
    args = parser.parse_args()
    if args.pairwise_margin_m < 0.0 or args.ttc_threshold_s < 0.0:
        raise ValueError("pairwise margin and TTC threshold must be non-negative")
    output = args.output.resolve()
    tensorboard = args.tensorboard_logdir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if tensorboard.exists() and any(tensorboard.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard logdir: {tensorboard}")
    original = summarize_archive(
        args.original_archive,
        args.original_metadata,
        pairwise_margin_m=args.pairwise_margin_m,
        ttc_threshold_s=args.ttc_threshold_s,
    )
    fresh = summarize_archive(
        args.fresh_archive,
        args.fresh_metadata,
        pairwise_margin_m=args.pairwise_margin_m,
        ttc_threshold_s=args.ttc_threshold_s,
    )
    result = {
        "audit": "dn_mpc_pairwise_ttc_label_semantics_p12",
        "development_only": True,
        "locked_test_opened": False,
        "git_revision": _git_revision(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pairwise_margin_m": float(args.pairwise_margin_m),
        "ttc_threshold_s": float(args.ttc_threshold_s),
        "blocks": {"original": original, "fresh": fresh},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tensorboard.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("P12/audit_contract", json.dumps({"development_only": True, "locked_test_opened": False}, sort_keys=True), 0)
        writer.add_text("P12/thresholds", json.dumps({"pairwise_margin_m": args.pairwise_margin_m, "ttc_threshold_s": args.ttc_threshold_s}, sort_keys=True), 0)
        for block, block_result in result["blocks"].items():
            _add_tensorboard(writer, block, block_result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
