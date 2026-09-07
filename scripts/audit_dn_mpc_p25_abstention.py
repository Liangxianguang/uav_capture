"""Audit explicit planner abstention groups in the P25 route archives.

This is a read-only development audit.  It identifies states where no route
candidate is simultaneously geometrically valid and first-step CBF feasible,
without fabricating a selected route or changing any safety threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from train_route_identity_jepa import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
ROUTE_LABELS = (
    "nominal",
    "left_detour",
    "right_detour",
    "upper_detour",
    "lower_detour",
    "radial_out",
    "formation_split",
    "formation_contract",
    "braking",
    "safe_intercept",
    "visibility_hold",
    "verified_safe_hold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "calibration"):
        parser.add_argument(f"--{split}-dataset", type=Path, required=True)
        parser.add_argument(f"--{split}-metadata", type=Path, required=True)
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


def _group_report(tensors: dict[str, Any]) -> dict[str, Any]:
    sample_type = tensors["sample_type"].numpy()
    runtime = sample_type == 0
    candidate = tensors["route_candidate_index"].numpy()
    scenario = tensors["scenario_index"].numpy()
    time_index = tensors["time_index"].numpy()
    geometry = tensors["route_geometry_valid"].numpy() >= 0.5
    cbf = tensors["labels_cbf_feasible"].numpy()[:, 0] >= 0.5
    trace = tensors.get("independent_cbf_trace_present")
    trace_mask = runtime & (trace.numpy() > 0) if trace is not None else np.zeros_like(runtime, dtype=bool)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero(runtime):
        groups[(int(scenario[index]), int(time_index[index]))].append(int(index))
    zero_groups: list[dict[str, Any]] = []
    eligible_counts: list[int] = []
    trace_missing = 0
    for key, indices in sorted(groups.items()):
        candidate_ids = sorted({int(candidate[index]) for index in indices if 0 <= int(candidate[index]) < 12})
        eligible: list[int] = []
        reasons: Counter[str] = Counter()
        for candidate_id in candidate_ids:
            rows = [index for index in indices if int(candidate[index]) == candidate_id]
            geometry_ok = bool(np.all(geometry[rows]))
            cbf_ok = bool(np.all(cbf[rows]))
            if geometry_ok and cbf_ok:
                eligible.append(candidate_id)
            else:
                if not geometry_ok:
                    reasons[f"{ROUTE_LABELS[candidate_id]}:geometry_invalid"] += 1
                if not cbf_ok:
                    reasons[f"{ROUTE_LABELS[candidate_id]}:first_step_cbf_infeasible"] += 1
        eligible_counts.append(len(eligible))
        group_trace = bool(np.any(trace_mask[indices]))
        if not group_trace:
            trace_missing += 1
        if not eligible:
            zero_groups.append(
                {
                    "scenario_index": key[0],
                    "time_index": key[1],
                    "candidate_count": len(candidate_ids),
                    "eligible_count": 0,
                    "candidate_labels": [ROUTE_LABELS[item] for item in candidate_ids],
                    "rejection_reasons": dict(sorted(reasons.items())),
                    "independent_trace_present": group_trace,
                }
            )
    return {
        "runtime_group_count": len(groups),
        "zero_eligible_group_count": len(zero_groups),
        "min_eligible_candidates": min(eligible_counts, default=0),
        "mean_eligible_candidates": float(np.mean(eligible_counts)) if eligible_counts else None,
        "trace_missing_group_count": trace_missing,
        "zero_eligible_groups": zero_groups,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# DN-MPC P26 Planner-Abstention Audit",
        "",
        "**Status:** development-only; read-only archive diagnosis.",
        "",
        "No route, CBF margin, fallback, or controlled-abort behavior was changed.",
        "The audit treats a group with zero jointly geometry-valid and first-step",
        "CBF-feasible candidates as an explicit planner abstention.",
        "",
        "| Split | runtime groups | zero-eligible groups | min eligible | mean eligible | missing trace groups |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "calibration"):
        item = result["splits"][split]
        mean = "n/a" if item["mean_eligible_candidates"] is None else f"{item['mean_eligible_candidates']:.2f}"
        lines.append(
            f"| {split} | {item['runtime_group_count']} | {item['zero_eligible_group_count']} | {item['min_eligible_candidates']} | {mean} | {item['trace_missing_group_count']} |"
        )
    lines += ["", "## Abstention Groups", ""]
    for split in ("train", "validation", "calibration"):
        lines.append(f"### {split}")
        groups = result["splits"][split]["zero_eligible_groups"]
        if not groups:
            lines.append("- none")
            continue
        for group in groups:
            lines.append(
                f"- scenario `{group['scenario_index']}`, time `{group['time_index']}`: "
                f"{group['rejection_reasons']}"
            )
    lines += ["", "## Decision", "", "Keep abstention explicit. Do not fabricate a selected route or bypass CBF when the eligible set is empty.", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    outputs = (args.output.resolve(), args.markdown_output.resolve(), args.tensorboard_logdir.resolve())
    for path in outputs[:2]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if outputs[2].exists() and any(outputs[2].iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory {outputs[2]}")
    splits: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for split in ("train", "validation", "calibration"):
        dataset = getattr(args, f"{split}_dataset").resolve()
        metadata = getattr(args, f"{split}_metadata").resolve()
        tensors, metadata_payload = load_dataset(dataset, metadata, split)
        splits[split] = _group_report(tensors)
        provenance[split] = {
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "metadata": str(metadata),
            "metadata_sha256": sha256(metadata),
            "dataset_version": metadata_payload.get("dataset_version"),
        }
    result = {
        "audit_type": "dn_mpc_p26_planner_abstention_audit",
        "development_only": True,
        "locked_test_opened": False,
        "promotion_eligible": False,
        "provenance": provenance,
        "splits": splits,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.resolve().write_text(markdown(result), encoding="utf-8")
    args.tensorboard_logdir.resolve().mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=1) as writer:
        writer.add_text("P26/Provenance", json.dumps(provenance, sort_keys=True), 0)
        writer.add_scalar("P26/PromotionEligible", 0.0, 0)
        for split, report in splits.items():
            for name in ("runtime_group_count", "zero_eligible_group_count", "min_eligible_candidates", "trace_missing_group_count"):
                writer.add_scalar(f"P26/{name}/{split}", float(report[name]), 0)
            if report["mean_eligible_candidates"] is not None:
                writer.add_scalar(f"P26/mean_eligible_candidates/{split}", float(report["mean_eligible_candidates"]), 0)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
