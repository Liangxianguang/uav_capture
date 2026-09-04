"""Audit a development smoke pair produced by the P11 rank guard.

The audit is intentionally outcome-descriptive: it verifies scene pairing,
step-trace invariants, rank-guard observability, CBF safety and TensorBoard
provenance, but it does not claim a task improvement from a smoke block.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _trace_stats(run_dir: Path) -> dict[str, Any]:
    trace_dir = run_dir / "step_traces"
    files = sorted(trace_dir.glob("episode_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No step traces found: {trace_dir}")
    ranking_steps = 0
    selected: list[int] = []
    abstention_reasons: dict[str, int] = {}
    hysteresis_steps = 0
    hold_steps = 0
    margin_values: list[float] = []
    invalid_selection_count = 0
    nonfinite_score_count = 0
    cbf_trace_steps = 0
    unverified_trace_steps = 0
    raw_unverified_trace_steps = 0
    missing_raw_unverified_trace_steps = 0
    for path in files:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            ranking = record.get("candidate_ranking")
            if not isinstance(ranking, dict):
                continue
            ranking_steps += 1
            labels = ranking.get("candidate_labels")
            if not isinstance(labels, list) or len(labels) != 5:
                raise ValueError(f"Ranking trace does not contain five candidates: {path}")
            index = int(ranking.get("selected_index", -1))
            if not 0 <= index < 5:
                invalid_selection_count += 1
            selected.append(index)
            for value in ranking.get("scores", []):
                if not _finite(value):
                    nonfinite_score_count += 1
            reason = ranking.get("rank_abstention_reason")
            if reason:
                abstention_reasons[str(reason)] = abstention_reasons.get(str(reason), 0) + 1
            if _bool(ranking.get("hysteresis_applied", False)):
                hysteresis_steps += 1
            if int(ranking.get("hold_steps_remaining", 0)) > 0:
                hold_steps += 1
            margin = ranking.get("top_two_margin_m")
            if margin is not None and _finite(margin):
                margin_values.append(float(margin))
            cbf = record.get("cbf")
            if isinstance(cbf, dict):
                cbf_trace_steps += 1
                if not _bool(cbf.get("verified_feasible", False)):
                    unverified_trace_steps += 1
            raw_flag = record.get("raw_unverified_executed")
            if raw_flag is None:
                missing_raw_unverified_trace_steps += 1
            elif _bool(raw_flag):
                raw_unverified_trace_steps += 1
    switches = sum(first != second for first, second in zip(selected, selected[1:]))
    return {
        "trace_files": len(files),
        "ranking_steps": ranking_steps,
        "selected_steps": len(selected),
        "candidate_switch_count": switches,
        "candidate_switch_rate": float(switches / max(len(selected) - 1, 1)),
        "abstention_count": int(sum(abstention_reasons.values())),
        "abstention_reasons": dict(sorted(abstention_reasons.items())),
        "hysteresis_steps": hysteresis_steps,
        "hold_steps": hold_steps,
        "mean_top_two_margin_m": float(np.mean(margin_values)) if margin_values else None,
        "minimum_top_two_margin_m": float(np.min(margin_values)) if margin_values else None,
        "invalid_selection_count": invalid_selection_count,
        "nonfinite_score_count": nonfinite_score_count,
        "cbf_trace_steps": cbf_trace_steps,
        "cbf_unverified_trace_steps": unverified_trace_steps,
        "raw_unverified_trace_steps": raw_unverified_trace_steps,
        "missing_raw_unverified_trace_steps": missing_raw_unverified_trace_steps,
    }


def _run_record(run_dir: Path) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json")
    metadata = summary.get("metadata")
    overall = summary.get("overall")
    if not isinstance(metadata, dict) or not isinstance(overall, dict):
        raise ValueError(f"Invalid paired evaluator summary: {run_dir}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run is not development-only: {run_dir}")
    candidate = metadata.get("candidate_contract", {})
    if candidate.get("candidate_count") != 5 or candidate.get("chunk_length_steps") != 3:
        raise ValueError(f"Candidate contract mismatch: {run_dir}")
    return {
        "run_dir": str(run_dir.resolve()),
        "variant": metadata.get("variant", {}).get("variant"),
        "training_seed": int(metadata.get("training_seed", -1)),
        "episodes": int(overall.get("episodes", 0)),
        "safe_capture_count": int(overall.get("safe_capture_count", 0)),
        "safe_capture_rate": float(overall.get("safe_capture_rate", 0.0)),
        "collision_count": int(overall.get("collision_count", 0)),
        "boundary_violation_count": int(overall.get("boundary_violation_count", 0)),
        "pairwise_violation_count": int(overall.get("pairwise_violation_count", 0)),
        "cbf_infeasible_steps": int(overall.get("cbf_infeasible_steps", 0)),
        "cbf_timeout_steps": int(overall.get("cbf_timeout_steps", 0)),
        "cbf_unverified_steps": int(overall.get("cbf_unverified_steps", 0)),
        "cbf_controlled_abort_steps": int(overall.get("cbf_controlled_abort_steps", 0)),
        "raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        "mean_capture_time_seconds": overall.get("mean_capture_time_seconds"),
        "mean_cbf_action_correction_norm": float(overall.get("mean_cbf_action_correction_norm", 0.0)),
        "protocol_sha256": metadata.get("inputs", {}).get("protocol_sha256"),
        "scene_manifest_sha256": metadata.get("inputs", {}).get("scene_manifest_sha256"),
        "actor_checkpoint_sha256": metadata.get("inputs", {}).get("actor_checkpoint_sha256"),
        "jepa_checkpoint_sha256": metadata.get("inputs", {}).get("jepa_checkpoint_sha256"),
        "ledger_sha256": metadata.get("inputs", {}).get("reliability_ledger_sha256"),
        "git_revision": metadata.get("git_revision"),
        "tensorboard_dir": metadata.get("tensorboard_dir"),
        "ranker_config": candidate,
        "trace": _trace_stats(run_dir),
    }


def _episodes(path: Path) -> dict[int, dict[str, Any]]:
    with (path / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["episode_index"])
        result[index] = {
            "safe_capture": _bool(row.get("safe_capture_success")),
            "scene_hash": row.get("scene_hash"),
            "episode_seed": int(row.get("episode_seed", -1)),
        }
    return result


def _paired(baseline: dict[int, dict[str, Any]], candidate: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("Paired runs do not contain the same episode indices.")
    if any(baseline[index]["scene_hash"] != candidate[index]["scene_hash"] for index in baseline):
        raise ValueError("Paired runs do not share scene hashes.")
    improved = sum(not baseline[index]["safe_capture"] and candidate[index]["safe_capture"] for index in baseline)
    degraded = sum(baseline[index]["safe_capture"] and not candidate[index]["safe_capture"] for index in baseline)
    tied = len(baseline) - improved - degraded
    return {
        "episodes": len(baseline),
        "improved": int(improved),
        "degraded": int(degraded),
        "tied": int(tied),
        "delta_percentage_points": float(100.0 * (improved - degraded) / max(len(baseline), 1)),
    }


def _write_tensorboard(logdir: Path, report: dict[str, Any]) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/rank_guard_audit", json.dumps({"development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/runs", json.dumps(report["runs"], indent=2), 0)
        writer.add_text("Provenance/paired", json.dumps(report.get("paired"), indent=2), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2), 0)
        for run in report["runs"]:
            prefix = str(run["variant"]).upper()
            writer.add_scalar(f"Safety/{prefix}/safe_capture_rate", run["safe_capture_rate"], 0)
            writer.add_scalar(f"Safety/{prefix}/collision_count", run["collision_count"], 0)
            writer.add_scalar(f"Safety/{prefix}/boundary_violation_count", run["boundary_violation_count"], 0)
            writer.add_scalar(f"Safety/{prefix}/pairwise_violation_count", run["pairwise_violation_count"], 0)
            writer.add_scalar(f"CBF/{prefix}/unverified_steps", run["cbf_unverified_steps"], 0)
            writer.add_scalar(f"Safety/{prefix}/raw_unverified_executed_steps", run["raw_unverified_executed_steps"], 0)
            writer.add_scalar(f"CBF/{prefix}/timeout_steps", run["cbf_timeout_steps"], 0)
            writer.add_scalar(f"CBF/{prefix}/mean_correction_norm", run["mean_cbf_action_correction_norm"], 0)
            writer.add_scalar(f"Ranking/{prefix}/switch_rate", run["trace"]["candidate_switch_rate"], 0)
            writer.add_scalar(f"Ranking/{prefix}/abstention_count", run["trace"]["abstention_count"], 0)
            writer.add_scalar(f"Ranking/{prefix}/hysteresis_steps", run["trace"]["hysteresis_steps"], 0)
        if report.get("paired"):
            writer.add_scalar("Paired/delta_percentage_points", report["paired"]["delta_percentage_points"], 0)
            writer.add_scalar("Paired/improved", report["paired"]["improved"], 0)
            writer.add_scalar("Paired/degraded", report["paired"]["degraded"], 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/rank_guard_audit/text_summary",
        "Provenance/runs/text_summary",
        "Provenance/paired/text_summary",
        "Gates/status/text_summary",
    }
    missing = sorted(required.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"Rank guard TensorBoard validation failed: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", []))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("Rank guard smoke audit requires --development-only.")
    baseline = _run_record(args.baseline_run.resolve())
    candidate = _run_record(args.candidate_run.resolve())
    if baseline["episodes"] != candidate["episodes"] or baseline["training_seed"] != candidate["training_seed"]:
        raise ValueError("Baseline and candidate runs are not episode/seed matched.")
    if baseline["scene_manifest_sha256"] != candidate["scene_manifest_sha256"]:
        raise ValueError("Baseline and candidate scene manifests differ.")
    paired = _paired(_episodes(args.baseline_run.resolve()), _episodes(args.candidate_run.resolve()))
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "scene_pairing": True,
        "baseline_safety_hard_gate": all(baseline[field] == 0 for field in ("collision_count", "boundary_violation_count", "pairwise_violation_count")),
        "candidate_safety_hard_gate": all(candidate[field] == 0 for field in ("collision_count", "boundary_violation_count", "pairwise_violation_count")),
        "baseline_raw_unverified": baseline["raw_unverified_executed_steps"] == 0,
        "candidate_raw_unverified": candidate["raw_unverified_executed_steps"] == 0,
        "baseline_trace_invariants": baseline["trace"]["invalid_selection_count"] == 0 and baseline["trace"]["nonfinite_score_count"] == 0,
        "candidate_trace_invariants": candidate["trace"]["invalid_selection_count"] == 0 and candidate["trace"]["nonfinite_score_count"] == 0,
        "baseline_raw_trace_observable": baseline["trace"]["missing_raw_unverified_trace_steps"] == 0,
        "candidate_raw_trace_observable": candidate["trace"]["missing_raw_unverified_trace_steps"] == 0,
        "rank_guard_observable": candidate["trace"]["ranking_steps"] > 0,
    }
    report = {
        "audit_type": "jepa_safe_capture_v5_rank_guard_smoke",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "runs": [baseline, candidate],
        "paired": paired,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "source_hashes": {
                "scripts/audit_jepa_safe_capture_v5_rank_guard_smoke.py": sha256(Path(__file__).resolve()),
            },
        },
    }
    report["tensorboard"] = _write_tensorboard(args.tensorboard_logdir, report)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rank_guard_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "rank_guard_audit.md").write_text(
        "# P11 Rank Guard Smoke Audit\n\n"
        "> Development-only paired smoke evidence; not a locked test and not a task-improvement claim.\n\n"
        f"Paired result: improved/degraded/tied = `{paired['improved']}/{paired['degraded']}/{paired['tied']}`; "
        f"delta = `{paired['delta_percentage_points']:.1f} pp`.\n\n"
        f"All gates pass: `{report['all_gates_pass']}`.\n\n"
        "| Run | Safe capture | Collision | Boundary | Pairwise | CBF unverified | Raw unverified | Switch rate | Abstentions | Hysteresis steps |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {run['variant']} | {run['safe_capture_rate']:.3f} | {run['collision_count']} | "
            f"{run['boundary_violation_count']} | {run['pairwise_violation_count']} | "
            f"{run['cbf_unverified_steps']} | {run['raw_unverified_executed_steps']} | "
            f"{run['trace']['candidate_switch_rate']:.4f} | "
            f"{run['trace']['abstention_count']} | {run['trace']['hysteresis_steps']} |"
            for run in report["runs"]
        )
        + "\n\n`locked_test_opened=false`. Capture time and correction norm remain secondary diagnostics.\n",
        encoding="utf-8",
    )
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "paired": paired, "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
