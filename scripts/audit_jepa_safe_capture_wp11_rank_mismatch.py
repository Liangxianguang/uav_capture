"""Quantify candidate-ranking mismatch in the frozen WP-7 tie3 M3 block.

This read-only audit joins M3/M0 paired outcomes with canonical WP-8 replay
records. It reports ranking stability, ledger credit, prediction gaps, and
CBF intervention by paired outcome label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from aggregate_jepa_safe_capture_v2_paired import sha256  # noqa: E402

PAIR_LABELS = ("degraded", "improved", "tied")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-index", type=Path, required=True)
    parser.add_argument("--aggregate-summary", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _p95(values: list[float]) -> float | None:
    return float(np.quantile(values, 0.95)) if values else None


def _pair_labels(summary: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    if summary.get("stage") != "full" or summary.get("locked_test_opened") is not False:
        raise ValueError("Aggregate summary is not development-only full output")
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for comparison in summary.get("paired_comparisons", []):
        if not isinstance(comparison, Mapping) or comparison.get("candidate_variant") != "m3":
            continue
        seed = int(comparison["training_seed"])
        for raw in comparison.get("pairs", []):
            key = (seed, int(raw["episode_index"]))
            if key in pairs:
                raise ValueError(f"Duplicate M3 pair: {key}")
            delta = int(raw["delta"])
            pairs[key] = {
                "training_seed": seed,
                "episode_index": key[1],
                "episode_seed": int(raw["episode_seed"]),
                "base_safe_capture": bool(raw["base_safe_capture"]),
                "candidate_safe_capture": bool(raw["candidate_safe_capture"]),
                "delta": delta,
                "pair_label": "improved" if delta > 0 else "degraded" if delta < 0 else "tied",
            }
    if len(pairs) != 120:
        raise ValueError(f"Expected 120 M3 pairs, found {len(pairs)}")
    counts = Counter(item["pair_label"] for item in pairs.values())
    if counts != Counter({"degraded": 30, "improved": 10, "tied": 80}):
        raise ValueError(f"Unexpected pair counts: {dict(counts)}")
    return pairs


def _index_rows(path: Path, pairs: Mapping[tuple[int, int], Mapping[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    report = _read_json(path)
    if report.get("development_only") is not True or report.get("locked_test_opened") is not False:
        raise ValueError("Failure index crossed the locked-test boundary")
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in report.get("rows", []):
        if not isinstance(raw, Mapping) or str(raw.get("variant")) != "m3":
            continue
        key = (int(raw["training_seed"]), int(raw["episode_index"]))
        if key in indexed:
            raise ValueError(f"Duplicate failure-index M3 row: {key}")
        pair = pairs.get(key)
        if pair is None or int(raw["episode_seed"]) != pair["episode_seed"]:
            raise ValueError(f"Failure-index/aggregate mismatch: {key}")
        indexed[key] = dict(raw)
    if set(indexed) != set(pairs):
        raise ValueError("Failure index does not contain exactly the paired M3 block")
    return indexed


def _payload(replay_dir: Path, seed: int, episode_index: int) -> list[dict[str, Any]]:
    path = replay_dir.resolve() / "replays" / f"{seed}_m3_{episode_index:04d}" / "replay_1.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid replay payload: {path}")
    return rows


def _episode_metrics(row: Mapping[str, Any], pair: Mapping[str, Any], payload: list[Mapping[str, Any]]) -> dict[str, Any]:
    margins: list[float] = []
    credits: list[float] = []
    corrections: list[float] = []
    latencies: list[float] = []
    selected: list[int] = []
    selected_not_best = 0
    unverified = 0
    for record in payload:
        ranking = record.get("candidate_ranking")
        cbf = record.get("cbf")
        if isinstance(ranking, Mapping):
            if (margin := _finite(ranking.get("top_two_score_margin"))) is not None:
                margins.append(margin)
            selected_index = ranking.get("selected_index")
            best_index = ranking.get("best_score_index")
            if selected_index is not None:
                selected.append(int(selected_index))
            if selected_index is not None and best_index is not None and int(selected_index) != int(best_index):
                selected_not_best += 1
            if isinstance(ranking.get("ledger_credits"), list):
                credits.extend(number for item in ranking["ledger_credits"] if (number := _finite(item)) is not None)
        if isinstance(cbf, Mapping):
            if (value := _finite(cbf.get("action_correction_norm"))) is not None:
                corrections.append(value)
            if (value := _finite(cbf.get("solve_latency_ms"))) is not None:
                latencies.append(value)
            unverified += int(bool(cbf.get("unverified", not bool(cbf.get("verified_feasible", False)))))
    labels = set()
    try:
        parsed = json.loads(str(row.get("diagnostic_labels_json", "[]")))
        if isinstance(parsed, list):
            labels = {str(value) for value in parsed}
    except json.JSONDecodeError:
        pass
    switches = sum(left != right for left, right in zip(selected, selected[1:]))
    return {
        "training_seed": int(pair["training_seed"]),
        "episode_index": int(pair["episode_index"]),
        "episode_seed": int(pair["episode_seed"]),
        "pair_label": str(pair["pair_label"]),
        "delta": int(pair["delta"]),
        "base_safe_capture": bool(pair["base_safe_capture"]),
        "candidate_safe_capture": bool(pair["candidate_safe_capture"]),
        "scenario": str(row.get("scenario", "")),
        "target_motion_mode": str(row.get("target_motion_mode", "")),
        "observation_condition": str(row.get("observation_condition", "")),
        "trace_steps": len(payload),
        "mean_top_two_margin": _mean(margins),
        "min_top_two_margin": min(margins) if margins else None,
        "selected_not_best_fraction": float(selected_not_best / len(payload)),
        "selected_switch_rate": float(switches / (len(selected) - 1)) if len(selected) > 1 else 0.0,
        "mean_ledger_credit": _mean(credits),
        "min_ledger_credit": min(credits) if credits else None,
        "cbf_correction_p95_mps": _p95(corrections),
        "cbf_latency_p95_ms": _p95(latencies),
        "cbf_unverified_steps": unverified,
        "candidate_capture_regression": "candidate_capture_regression" in labels,
        "high_credit_failure": "high_credit_failure" in labels,
        "candidate_oscillation": "candidate_oscillation" in labels,
        "stale_observation": "stale_observation" in labels,
        "clearance_prediction_gap_m": _finite(row.get("clearance_prediction_gap_mean_m")),
        "visibility_prediction_gap": _finite(row.get("visibility_prediction_gap_mean_m")),
    }


def _group_stats(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "selected_switch_rate", "selected_not_best_fraction", "mean_top_two_margin",
        "min_top_two_margin", "mean_ledger_credit", "min_ledger_credit",
        "cbf_correction_p95_mps", "cbf_latency_p95_ms",
        "clearance_prediction_gap_m", "visibility_prediction_gap",
    )
    result: dict[str, Any] = {"episodes": len(rows)}
    for field in fields:
        values = [number for row in rows if (number := _finite(row.get(field))) is not None]
        result[field] = {"mean": _mean(values), "p50": float(np.quantile(values, 0.5)) if values else None, "p95": _p95(values)}
    for field in ("candidate_capture_regression", "high_credit_failure", "candidate_oscillation", "stale_observation"):
        result[f"{field}_count"] = sum(bool(row.get(field)) for row in rows)
    return result


def _tensorboard(result: Mapping[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/wp11_rank_mismatch", json.dumps({"development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(result["inputs"], indent=2), 0)
        writer.add_text("Provenance/pair_counts", json.dumps(result["pair_counts"], indent=2), 0)
        for label in PAIR_LABELS:
            stats = result["by_pair_label"][label]
            writer.add_scalar(f"Pairs/{label}/switch_rate_mean", float(stats["selected_switch_rate"]["mean"] or 0.0), 0)
            writer.add_scalar(f"Pairs/{label}/not_best_mean", float(stats["selected_not_best_fraction"]["mean"] or 0.0), 0)
            writer.add_scalar(f"Pairs/{label}/high_credit_failure_count", float(stats["high_credit_failure_count"]), 0)
            writer.add_scalar(f"Pairs/{label}/oscillation_count", float(stats["candidate_oscillation_count"]), 0)
        for index, row in enumerate(result["rows"]):
            writer.add_scalar(f"Episodes/{index:03d}/switch_rate", float(row["selected_switch_rate"]), 0)
            writer.add_scalar(f"Episodes/{index:03d}/not_best", float(row["selected_not_best_fraction"]), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {"Config/wp11_rank_mismatch/text_summary", "Provenance/inputs/text_summary", "Provenance/pair_counts/text_summary"}
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    missing = sorted(required.difference(tags.get("tensors", [])))
    if missing or not events:
        raise ValueError(f"WP-11 TensorBoard audit failed: missing_text={missing}, event_files={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", [])), "required_provenance": True}


def audit_rank_mismatch(failure_index: Path, aggregate_summary: Path, replay_dir: Path, output_dir: Path, tensorboard_logdir: Path) -> dict[str, Any]:
    pairs = _pair_labels(_read_json(aggregate_summary))
    rows = _index_rows(failure_index, pairs)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [_episode_metrics(rows[key], pairs[key], _payload(replay_dir, key[0], key[1])) for key in sorted(pairs)]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metrics:
        groups[str(row["pair_label"])].append(row)
    result: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_wp11_rank_mismatch",
        "development_only": True,
        "locked_test_opened": False,
        "episode_count": len(metrics),
        "pair_counts": dict(Counter(row["pair_label"] for row in metrics)),
        "by_pair_label": {label: _group_stats(groups[label]) for label in PAIR_LABELS},
        "rows": metrics,
        "inputs": {
            "failure_index": str(failure_index.resolve()),
            "failure_index_sha256": sha256(failure_index.resolve()),
            "aggregate_summary": str(aggregate_summary.resolve()),
            "aggregate_summary_sha256": sha256(aggregate_summary.resolve()),
            "replay_dir": str(replay_dir.resolve()),
            "replay_summary_sha256": sha256(replay_dir.resolve() / "replay_summary.json"),
        },
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "python": platform.python_version(),
            "source_hashes": {"scripts/audit_jepa_safe_capture_wp11_rank_mismatch.py": sha256(Path(__file__).resolve())},
        },
    }
    (output_dir / "rank_mismatch_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output_dir / "rank_mismatch_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(metrics[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    lines = [
        "# WP-11 Candidate Rank Mismatch Audit", "",
        "Status: development-only; locked_test_opened=false", "",
        "| Pair label | Episodes | Switch rate mean | Selected-not-best mean | Margin mean | High-credit failures | Oscillation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in PAIR_LABELS:
        stats = result["by_pair_label"][label]
        lines.append(f"| {label} | {stats['episodes']} | {stats['selected_switch_rate']['mean']:.4f} | {stats['selected_not_best_fraction']['mean']:.4f} | {stats['mean_top_two_margin']['mean']:.6f} | {stats['high_credit_failure_count']} | {stats['candidate_oscillation_count']} |")
    lines.extend(["", "This diagnostic correlates trace signals with paired outcomes; it does not prove target-drift causality."])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["tensorboard"] = _tensorboard(result, tensorboard_logdir)
    (output_dir / "rank_mismatch_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("WP-11 requires --development-only")
    result = audit_rank_mismatch(args.failure_index, args.aggregate_summary, args.replay_dir, args.output_dir, args.tensorboard_logdir)
    print(json.dumps({"episodes": result["episode_count"], "pair_counts": result["pair_counts"], "tensorboard": result["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
