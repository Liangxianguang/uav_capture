"""Offline utility-weight sweep for P18 route-progress plus route length.

Route length is public candidate geometry, not target future truth. This audit
tests whether an explicit nearest-route cost stabilizes P18 ranking. It is not
an online integration and does not execute an action or change CBF policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--archive", type=Path, action="append", nargs=2, metavar=("SPLIT", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--lambdas", default="0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,1,2")
    parser.add_argument("--normalization-m", type=float, default=10.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_lengths(path: Path) -> dict[tuple[int, int, int], float]:
    result: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    with np.load(path) as archive:
        required = ("scenario_index", "time_index", "route_candidate_index", "sample_type", "route_length_m")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"Archive is missing route length fields: {missing}")
        for scenario, time_index, candidate, sample_type, length in zip(
            archive["scenario_index"], archive["time_index"], archive["route_candidate_index"], archive["sample_type"], archive["route_length_m"]
        ):
            if int(sample_type) == 0 and 0 <= int(candidate) < 12:
                result[(int(scenario), int(time_index), int(candidate))].append(float(length))
    return {key: float(np.mean(values)) for key, values in result.items()}


def _report(rows: list[dict[str, str]], lengths: dict[tuple[int, int, int], float], weight: float, normalization_m: float) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("eligible") != "1":
            continue
        key = (int(row["scenario_index"]), int(row["time_index"]))
        candidate = int(row["candidate_index"])
        length = lengths.get((*key, candidate))
        if length is None:
            raise ValueError(f"No route length for {key}, candidate {candidate}")
        groups[key].append({
            "candidate": candidate,
            "truth": float(row["truth_progress"]) - weight * length / normalization_m,
            "prediction": float(row["predicted_progress"]) - weight * length / normalization_m,
        })
    total = exact = tie_aware = pair_total = pair_correct = informative_groups = informative_exact = 0
    top_gaps: list[float] = []
    for candidates in groups.values():
        if len(candidates) < 2:
            continue
        total += 1
        truth_best = max(candidates, key=lambda item: (item["truth"], -item["candidate"]))
        prediction_best = max(candidates, key=lambda item: (item["prediction"], -item["candidate"]))
        exact += int(truth_best["candidate"] == prediction_best["candidate"])
        ordered = sorted(candidates, key=lambda item: (-item["truth"], item["candidate"]))
        gap = float(ordered[0]["truth"] - ordered[1]["truth"])
        top_gaps.append(gap)
        tie_aware += int(truth_best["truth"] - next(item["truth"] for item in candidates if item["candidate"] == prediction_best["candidate"]) <= 0.005)
        if gap > 0.005:
            informative_groups += 1
            informative_exact += int(truth_best["candidate"] == prediction_best["candidate"])
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                delta = left["truth"] - right["truth"]
                if abs(delta) <= 0.005:
                    continue
                pair_total += 1
                pair_correct += int((left["prediction"] - right["prediction"]) * delta > 0.0)
    return {
        "weight": float(weight),
        "group_count": total,
        "exact_top1": float(exact / total) if total else None,
        "tie_aware_top1": float(tie_aware / total) if total else None,
        "informative_group_count": informative_groups,
        "informative_exact_top1": float(informative_exact / informative_groups) if informative_groups else None,
        "pairwise_agreement": float(pair_correct / pair_total) if pair_total else None,
        "pairwise_total": pair_total,
        "top_gap_le_005": float(np.mean(np.asarray(top_gaps) <= 0.005)) if top_gaps else None,
    }


def _select(calibration: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in calibration if item.get("informative_exact_top1") is not None]
    if not valid:
        raise ValueError("Calibration has no informative route groups")
    return max(valid, key=lambda item: (item["informative_exact_top1"], item["pairwise_agreement"] or -1.0, -item["weight"]))


def _markdown(result: dict[str, Any]) -> str:
    selected = result["selected_weight"]
    lines = [
        "# DN-MPC P22 Route-Utility Weight Audit",
        "",
        "**Status:** development-only; no action was executed and no CBF rule was changed.",
        "",
        "The sweep subtracts `lambda * route_length_m / 10` from both the offline route-progress label and P18 prediction. Route length is public candidate geometry and is available before CBF verification.",
        "",
        f"Calibration-selected weight: **`{selected}`** (selected only from calibration).",
        "",
        "| Split | lambda | exact top-1 | tie-aware top-1 | informative top-1 | pairwise agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, reports in result["splits"].items():
        for item in reports:
            if float(item["weight"]) != selected and split == "validation":
                continue
            def pct(name: str) -> str:
                value = item.get(name)
                return "n/a" if value is None else f"{float(value) * 100:.2f}%"
            lines.append(f"| {split} | {item['weight']:g} | {pct('exact_top1')} | {pct('tie_aware_top1')} | {pct('informative_exact_top1')} | {pct('pairwise_agreement')} |" )
    chosen = next(item for item in result["splits"]["validation"] if item["weight"] == selected)
    lines += [
        "",
        f"On validation at the calibration-selected weight, informative top-1 is `{chosen['informative_exact_top1']}` and pairwise agreement is `{chosen['pairwise_agreement']}`.",
        "",
        "## Decision",
        "",
        "The explicit route-length prior improves pairwise utility direction but does not reach the direct-selection promotion gate. Keep route length as a planner cost/regularizer, not as evidence that P18 is ready for online JEPA selection. Retraining remains gated on selected/nominal/safe-hold CBF traces and fresh OOD/disagreement calibration.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.normalization_m <= 0.0:
        raise ValueError("normalization-m must be positive")
    weights = [float(value) for value in args.lambdas.split(",") if value.strip()]
    if not weights or any(value < 0.0 for value in weights):
        raise ValueError("lambdas must contain non-negative values")
    output = args.output.resolve()
    markdown = args.markdown_output.resolve()
    if output.exists() or markdown.exists():
        raise FileExistsError("Refusing to overwrite P22 outputs")
    logdir = args.tensorboard_logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError("Refusing to overwrite P22 TensorBoard logdir")
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.details.resolve().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_split[str(row["split"])].append(row)
    lengths_by_split = {str(split): _load_lengths(Path(path).resolve()) for split, path in args.archive}
    reports = {split: [_report(rows_by_split[split], lengths_by_split[split], weight, args.normalization_m) for weight in weights] for split in rows_by_split}
    selected = _select(reports["calibration"])
    result = {
        "audit_type": "dn_mpc_p22_route_utility_weight_audit",
        "development_only": True,
        "locked_test_opened": False,
        "details": str(args.details.resolve()),
        "details_sha256": sha256(args.details.resolve()),
        "normalization_m": float(args.normalization_m),
        "weights": weights,
        "selected_weight": float(selected["weight"]),
        "splits": reports,
        "promotion_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(result), encoding="utf-8")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("P22/Provenance", json.dumps({"details": str(args.details.resolve()), "sha256": result["details_sha256"]}, sort_keys=True), 0)
        writer.add_scalar("P22/SelectedRouteLengthWeight", selected["weight"], 0)
        writer.add_scalar("P22/PromotionEligible", 0.0, 0)
        for split, values in reports.items():
            for item in values:
                step = int(round(item["weight"] * 1000))
                for name in ("exact_top1", "tie_aware_top1", "informative_exact_top1", "pairwise_agreement", "top_gap_le_005"):
                    if item.get(name) is not None:
                        writer.add_scalar(f"P22/{name}/{split}", float(item[name]), step)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
