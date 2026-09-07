"""Diagnose P20 route-progress labels and top-1 versus pairwise behavior.

This consumes the P19/P20 ranking-detail CSV produced by the offline audit. It
does not execute a policy, alter CBF constraints, or authorize promotion. The
purpose is to distinguish near-tied labels from genuine score-direction
errors before retraining the route-progress head.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--tie-margin", type=float, default=0.005)
    parser.add_argument("--informative-margin", type=float, default=0.005)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} is non-finite")
    return value


def _group_report(rows: list[dict[str, str]], tie_margin: float, informative_margin: float) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("eligible") != "1":
            continue
        groups[(int(row["scenario_index"]), int(row["time_index"]))].append(row)
    total = exact = tie_aware = informative_groups = informative_exact = 0
    top_gaps: list[float] = []
    top_regrets: list[float] = []
    pair_total = pair_correct = 0
    label_values: list[float] = []
    prediction_values: list[float] = []
    for key, candidates in sorted(groups.items()):
        if len(candidates) < 2:
            continue
        truth = np.asarray([_float(row, "truth_progress") for row in candidates], dtype=np.float64)
        prediction = np.asarray([_float(row, "predicted_progress") for row in candidates], dtype=np.float64)
        labels = [int(row["candidate_index"]) for row in candidates]
        truth_order = np.lexsort((np.asarray(labels), -truth))
        prediction_order = np.lexsort((np.asarray(labels), -prediction))
        truth_best_index = int(truth_order[0])
        prediction_best_index = int(prediction_order[0])
        total += 1
        exact += int(labels[truth_best_index] == labels[prediction_best_index])
        top_gap = float(truth[truth_order[0]] - truth[truth_order[1]])
        top_gaps.append(top_gap)
        top_regrets.append(float(truth[truth_best_index] - truth[prediction_best_index]))
        target_tie = truth >= truth[truth_best_index] - tie_margin
        tie_aware += int(bool(target_tie[prediction_best_index]))
        if top_gap > informative_margin:
            informative_groups += 1
            informative_exact += int(labels[truth_best_index] == labels[prediction_best_index])
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                delta = float(truth[left] - truth[right])
                if abs(delta) <= informative_margin:
                    continue
                pair_total += 1
                pair_correct += int(float(prediction[left] - prediction[right]) * delta > 0.0)
        label_values.extend(truth.tolist())
        prediction_values.extend(prediction.tolist())
    truth_array = np.asarray(label_values, dtype=np.float64)
    prediction_array = np.asarray(prediction_values, dtype=np.float64)
    slope = None
    if truth_array.size >= 2 and np.var(truth_array) > 1e-12:
        slope = float(np.cov(truth_array, prediction_array, ddof=0)[0, 1] / np.var(truth_array))
    return {
        "group_count": total,
        "exact_top1": float(exact / total) if total else None,
        "tie_aware_top1": float(tie_aware / total) if total else None,
        "informative_group_count": informative_groups,
        "informative_exact_top1": float(informative_exact / informative_groups) if informative_groups else None,
        "pairwise_agreement": float(pair_correct / pair_total) if pair_total else None,
        "pairwise_correct": pair_correct,
        "pairwise_total": pair_total,
        "top_gap_le_tie_margin_fraction": float(np.mean(np.asarray(top_gaps) <= tie_margin)) if top_gaps else None,
        "top_gap_quantiles": np.quantile(np.asarray(top_gaps), [0.0, 0.25, 0.5, 0.75, 1.0]).tolist() if top_gaps else [],
        "mean_top_regret": float(np.mean(top_regrets)) if top_regrets else None,
        "label_std": float(np.std(truth_array)) if truth_array.size else None,
        "prediction_std": float(np.std(prediction_array)) if prediction_array.size else None,
        "prediction_on_label_slope": slope,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# DN-MPC P21 Route-Progress Label and Score-Direction Audit",
        "",
        "**Status:** development-only; no online action or CBF policy was changed.",
        "",
        f"The audit consumes `{result['details']}` and uses a tie margin of `{result['tie_margin']}`.",
        "",
        "| Split | groups | exact top-1 | tie-aware top-1 | informative top-1 | pairwise agreement | top-gap <= margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "calibration"):
        item = result["splits"].get(split)
        if not item:
            continue
        def pct(name: str) -> str:
            value = item.get(name)
            return "n/a" if value is None else f"{float(value) * 100:.2f}%"
        lines.append(
            f"| {split} | {item['group_count']} | {pct('exact_top1')} | {pct('tie_aware_top1')} | {pct('informative_exact_top1')} | {pct('pairwise_agreement')} | {pct('top_gap_le_tie_margin_fraction')} |"
        )
    validation = result["splits"].get("validation", {})
    lines += [
        "",
        "## Interpretation",
        "",
        f"- `{validation.get('top_gap_le_tie_margin_fraction')}` of validation groups have a top-label gap no larger than the registered tie margin.",
        f"- Tie-aware top-1 is `{validation.get('tie_aware_top1')}`, while exact top-1 is `{validation.get('exact_top1')}`.",
        f"- Among groups whose best label is separated by more than the informative margin, exact top-1 is `{validation.get('informative_exact_top1')}`.",
        f"- Pairwise agreement is `{validation.get('pairwise_agreement')}`; it is the more stable signal in this archive.",
        "",
        "## Decision",
        "",
        "Near ties explain part of the exact top-1 failure, but the informative-group top-1 result remains insufficient for direct route selection. Do not promote P18. Recalibrate or retrain route-progress with explicit tie-aware supervision, and keep DN-MPC/CBF as the executable authority.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    details = args.details.resolve()
    if args.tie_margin < 0.0 or args.informative_margin < 0.0:
        raise ValueError("margins must be non-negative")
    for output in (args.output.resolve(), args.markdown_output.resolve()):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite output: {output}")
    logdir = args.tensorboard_logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard logdir: {logdir}")
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    with details.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "scenario_index", "time_index", "candidate_index", "eligible", "truth_progress", "predicted_progress"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Ranking detail CSV is missing fields: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            rows_by_split[str(row["split"])].append(row)
    reports = {split: _group_report(rows, args.tie_margin, args.informative_margin) for split, rows in rows_by_split.items()}
    result = {
        "audit_type": "dn_mpc_p21_route_progress_label_score_direction_audit",
        "development_only": True,
        "locked_test_opened": False,
        "details": str(details),
        "details_sha256": sha256(details),
        "tie_margin": float(args.tie_margin),
        "informative_margin": float(args.informative_margin),
        "splits": reports,
        "promotion_eligible": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = args.markdown_output.resolve()
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(result), encoding="utf-8")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("P21/Provenance", json.dumps({"details": str(details), "sha256": result["details_sha256"]}, sort_keys=True), 0)
        writer.add_scalar("P21/PromotionEligible", 0.0, 0)
        for split, report in reports.items():
            for name in ("exact_top1", "tie_aware_top1", "informative_exact_top1", "pairwise_agreement", "top_gap_le_tie_margin_fraction", "mean_top_regret", "prediction_on_label_slope"):
                value = report.get(name)
                if value is not None:
                    writer.add_scalar(f"P21/{name}/{split}", float(value), 0)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
