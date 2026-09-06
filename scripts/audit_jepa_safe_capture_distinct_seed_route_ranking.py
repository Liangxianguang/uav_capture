"""Audit a negative distinct-seed replay without changing online behavior.

This bounded audit consumes existing M0/M3 traces only.  It reports score-order,
route-switching, Ledger fallback, and target-clearance signals; it does not
claim a settled-best route because it never re-simulates counterfactual routes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    x_arr, y_arr = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _episode_audit(run: Path, episode_index: int) -> dict[str, Any]:
    trace_path = run / "step_traces" / f"episode_{episode_index:04d}.jsonl"
    traces = _trace_rows(trace_path)
    labels: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    fallback: Counter[str] = Counter()
    score_argmin_matches = 0
    score_argmin_comparisons = 0
    eligible_steps = 0
    all_ineligible_steps = 0
    route_switches = 0
    previous_label: str | None = None
    predicted_progress: list[float] = []
    actual_progress: list[float] = []
    target_clearance = [_safe_float(item.get("target_clearance_m")) for item in traces]
    for index, trace in enumerate(traces):
        ranking = trace.get("candidate_ranking") or {}
        mode = str(ranking.get("execution_mode", "missing"))
        modes[mode] += 1
        reason = ranking.get("fallback_reason")
        if reason:
            fallback[str(reason)] += 1
        selected_route = trace.get("selected_route") or {}
        label = str(selected_route.get("label", "none"))
        labels[label] += 1
        if previous_label is not None and label != previous_label:
            route_switches += 1
        previous_label = label
        eligible = [i for i, value in enumerate(ranking.get("eligible_mask", [])) if bool(value)]
        selected_index = ranking.get("selected_index")
        scores = ranking.get("scores", [])
        if eligible:
            eligible_steps += 1
            finite = [i for i in eligible if _safe_float(scores[i]) is not None]
            if finite and isinstance(selected_index, int) and selected_index in finite:
                score_argmin = min(finite, key=lambda i: (float(scores[i]), i))
                score_argmin_comparisons += 1
                score_argmin_matches += int(selected_index == score_argmin)
        else:
            all_ineligible_steps += 1
        if index + 1 < len(traces) and isinstance(selected_index, int):
            progress_values = ranking.get("predicted_route_progress_m", [])
            if 0 <= selected_index < len(progress_values):
                predicted = _safe_float(progress_values[selected_index])
                current = target_clearance[index]
                following = target_clearance[index + 1]
                if predicted is not None and current is not None and following is not None:
                    predicted_progress.append(predicted)
                    actual_progress.append(current - following)
    finite_clearance = [value for value in target_clearance if value is not None]
    return {
        "episode_index": episode_index,
        "steps": len(traces),
        "labels": dict(labels),
        "execution_modes": dict(modes),
        "fallback_reasons": dict(fallback),
        "eligible_steps": eligible_steps,
        "all_ineligible_steps": all_ineligible_steps,
        "score_argmin_comparisons": score_argmin_comparisons,
        "score_argmin_match_rate": score_argmin_matches / max(score_argmin_comparisons, 1),
        "route_switches": route_switches,
        "predicted_progress_actual_clearance_delta_correlation": _corr(predicted_progress, actual_progress),
        "target_clearance_first_m": finite_clearance[0] if finite_clearance else None,
        "target_clearance_last_m": finite_clearance[-1] if finite_clearance else None,
        "target_clearance_min_m": min(finite_clearance) if finite_clearance else None,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    m0, m3 = args.m0_run.resolve(), args.m3_run.resolve()
    for run in (m0, m3):
        for name in ("summary.json", "provenance.json", "episodes.csv", "scene_manifest.jsonl"):
            if not (run / name).is_file():
                raise FileNotFoundError(run / name)
    m0_summary, m3_summary = _json(m0 / "summary.json"), _json(m3 / "summary.json")
    m0_prov, m3_prov = _json(m0 / "provenance.json"), _json(m3 / "provenance.json")
    if any(value.get("development_only") is not True or value.get("locked_test_opened") is not False
           for value in (m0_summary.get("metadata", {}), m3_summary.get("metadata", {}), m0_prov, m3_prov)):
        raise ValueError("Audit accepts development-only, unlocked runs only.")
    m0_manifest = _sha256(m0 / "scene_manifest.jsonl")
    m3_manifest = _sha256(m3 / "scene_manifest.jsonl")
    if m0_manifest != m3_manifest:
        raise ValueError("M0/M3 manifests differ.")
    m0_rows, m3_rows = _rows(m0 / "episodes.csv"), _rows(m3 / "episodes.csv")
    if len(m0_rows) != len(m3_rows):
        raise ValueError("M0/M3 episode counts differ.")
    episodes: list[dict[str, Any]] = []
    for index, (row0, row3) in enumerate(zip(m0_rows, m3_rows)):
        if row0.get("episode_seed") != row3.get("episode_seed"):
            raise ValueError(f"Episode seed mismatch at index {index}.")
        item = _episode_audit(m3, index)
        item.update({
            "episode_seed": int(row3["episode_seed"]),
            "m0_safe_capture": row0.get("safe_capture_success", "false").lower() == "true",
            "m3_safe_capture": row3.get("safe_capture_success", "false").lower() == "true",
            "m0_termination": row0.get("termination_reason"),
            "m3_termination": row3.get("termination_reason"),
            "m3_target_boundary_violation": row3.get("target_boundary_violation", "false").lower() == "true",
        })
        episodes.append(item)
    result = {
        "stage": "distinct_model_seed_route_ranking_audit",
        "development_only": True,
        "locked_test_opened": False,
        "m0_run": str(m0),
        "m3_run": str(m3),
        "m0_summary_sha256": _sha256(m0 / "summary.json"),
        "m3_summary_sha256": _sha256(m3 / "summary.json"),
        "scene_manifest_sha256": m0_manifest,
        "episodes": episodes,
        "interpretation": {
            "settled_best_route_not_available": True,
            "online_contract_modified": False,
            "next_gate": "offline route-regret and target-boundary-aware progress audit before recalibration",
        },
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.tensorboard_dir.exists() and any(args.tensorboard_dir.iterdir()):
        raise FileExistsError(args.tensorboard_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ranking_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Distinct-Seed Route Ranking Audit",
        "",
        "This is a read-only development audit. It does not re-simulate candidates, so it does not claim a settled-best route.",
        "",
        "| Episode | M0 | M3 | M3 termination | M3 score-argmin match | M3 route switches | M3 target clearance first -> last |",
        "|---:|:---:|:---:|---|---:|---:|---:|",
    ]
    for item in episodes:
        lines.append(
            f"| {item['episode_index']} ({item['episode_seed']}) | {item['m0_safe_capture']} | {item['m3_safe_capture']} | "
            f"{item['m3_termination']} | {item['score_argmin_match_rate']:.3f} | {item['route_switches']} | "
            f"{item['target_clearance_first_m']:.2f} -> {item['target_clearance_last_m']:.2f} m |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The negative paired gate is a route-ranking/target-motion generalization signal only when it co-occurs with available verified candidates and zero safety violations.",
        "The next permitted change is target-boundary-aware progress/escape supervision or checkpoint-bound recalibration after a settled counterfactual audit; CBF margins and stale/OOD gates remain frozen.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(args.tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/audit", json.dumps(result["interpretation"], indent=2), 0)
        writer.add_text("Provenance/manifest", m0_manifest, 0)
        for item in episodes:
            prefix = f"Ranking/episode_{item['episode_index']}"
            writer.add_scalar(f"{prefix}/score_argmin_match_rate", item["score_argmin_match_rate"], 0)
            writer.add_scalar(f"{prefix}/route_switches", item["route_switches"], 0)
            writer.add_scalar(f"{prefix}/eligible_steps", item["eligible_steps"], 0)
            writer.add_scalar(f"{prefix}/all_ineligible_steps", item["all_ineligible_steps"], 0)
            correlation = item["predicted_progress_actual_clearance_delta_correlation"]
            if correlation is not None:
                writer.add_scalar(f"{prefix}/predicted_progress_actual_clearance_delta_correlation", correlation, 0)
        writer.add_scalar("Gates/online_contract_modified", 0.0, 0)
        writer.add_scalar("Gates/settled_best_claim_available", 0.0, 0)
    result["tensorboard"] = {
        "logdir": str(args.tensorboard_dir),
        "event_files": sorted(path.name for path in args.tensorboard_dir.glob("events.out.tfevents.*")),
    }
    (args.output_dir / "ranking_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-run", type=Path, required=True)
    parser.add_argument("--m3-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("This audit requires --development-only.")
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
