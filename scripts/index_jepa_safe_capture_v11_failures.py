"""Index failures in the v11 corrected-frame paired smoke matrix.

This audit is read-only.  It consumes the twelve development-only M0/M3/A1/A2
runs, classifies complete episode outcomes, and summarizes observed ranking,
ledger, CBF, visibility, and staleness signals.  It never reads online target
ground truth, modifies source runs, changes thresholds, or opens a locked split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from index_jepa_safe_capture_failures import classify_episode, read_trace, summarize_trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m0", "m3", "a1", "a2")
EXPECTED_EPISODES = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer {label}: {value!r}") from error


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _run_path(input_root: Path, seed: int, variant: str) -> Path:
    return input_root / f"jepa_safe_capture_v5_v11_corrected_frame_smoke_{variant}_seed{seed}"


def load_run(input_root: Path, seed: int, variant: str) -> dict[str, Any]:
    path = _run_path(input_root, seed, variant).resolve()
    required = ("summary.json", "provenance.json", "episodes.csv", "scene_manifest.jsonl")
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing v11 smoke artifact: {path / name}")
    summary = read_json(path / "summary.json")
    provenance = read_json(path / "provenance.json")
    metadata = summary.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Missing summary metadata: {path}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run crossed the locked-test boundary: {path}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Provenance crossed the locked-test boundary: {path}")
    declared = metadata.get("variant")
    if not isinstance(declared, Mapping) or declared.get("variant") != variant:
        raise ValueError(f"Variant metadata mismatch: {path}")
    if as_int(metadata.get("training_seed", -1), "training_seed") != seed:
        raise ValueError(f"Training seed metadata mismatch: {path}")
    if as_int(metadata.get("episodes", -1), "episodes") != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} episodes: {path}")
    manifest_hash = str(metadata.get("inputs", {}).get("scene_manifest_sha256", ""))
    if len(manifest_hash) != 64 or manifest_hash != sha256(path / "scene_manifest.jsonl"):
        raise ValueError(f"Scene manifest hash mismatch: {path}")
    rows: dict[int, dict[str, Any]] = {}
    with (path / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = as_int(row.get("episode_index"), "episode_index")
            if index in rows:
                raise ValueError(f"Duplicate episode index {index}: {path}")
            rows[index] = dict(row)
    if sorted(rows) != list(range(EXPECTED_EPISODES)):
        raise ValueError(f"Episode indices are not contiguous: {path}")
    trace_dir = path / "step_traces"
    traces = sorted(trace_dir.glob("episode_*.jsonl")) if trace_dir.is_dir() else []
    if len(traces) != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} traces, found {len(traces)}: {path}")
    return {
        "path": path,
        "seed": seed,
        "variant": variant,
        "summary": summary,
        "provenance": provenance,
        "rows": rows,
        "manifest_sha256": manifest_hash,
        "summary_sha256": sha256(path / "summary.json"),
        "provenance_sha256": sha256(path / "provenance.json"),
    }


def canonical_episode(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map the evaluator CSV naming to the classifier's stable contract."""

    result = dict(row)
    result["safe_capture"] = as_bool(row.get("safe_capture_success"))
    result["collision"] = as_bool(row.get("collision"))
    result["boundary_violation"] = as_bool(row.get("defender_boundary_violation", row.get("boundary_violation")))
    result["target_boundary_violation"] = as_bool(row.get("target_boundary_violation"))
    result["pairwise_violation"] = as_bool(row.get("pairwise_violation"))
    result["cbf_infeasible_steps"] = as_int(row.get("cbf_infeasible_steps", 0), "cbf_infeasible_steps")
    result["cbf_timeout_steps"] = as_int(row.get("cbf_timeout_steps", 0), "cbf_timeout_steps")
    result["cbf_unverified_steps"] = as_int(row.get("cbf_unverified_steps", 0), "cbf_unverified_steps")
    result["cbf_controlled_abort_steps"] = as_int(row.get("cbf_controlled_abort_steps", 0), "cbf_controlled_abort_steps")
    result["termination_reason"] = str(row.get("termination_reason", row.get("task_termination_reason", "")))
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def build_index(input_root: Path) -> dict[str, Any]:
    runs = [load_run(input_root, seed, variant) for seed in SEEDS for variant in VARIANTS]
    manifest_hashes = {run["manifest_sha256"] for run in runs}
    protocol_hashes = {str(run["summary"]["metadata"]["inputs"].get("protocol_sha256", "")) for run in runs}
    if len(manifest_hashes) != 3:
        raise ValueError(f"Expected one paired manifest per seed, got {sorted(manifest_hashes)}")
    if len(protocol_hashes) != 1 or "" in protocol_hashes:
        raise ValueError(f"Protocol hashes are inconsistent: {sorted(protocol_hashes)}")
    baselines = {
        run["seed"]: {index: as_bool(row.get("safe_capture_success")) for index, row in run["rows"].items()}
        for run in runs
        if run["variant"] == "m0"
    }
    rows: list[dict[str, Any]] = []
    for run in runs:
        for index in range(EXPECTED_EPISODES):
            source = canonical_episode(run["rows"][index])
            trace = read_trace(run["path"] / "step_traces" / f"episode_{index:04d}.jsonl", index)
            trace_summary = summarize_trace(trace)
            primary, labels = classify_episode(
                source,
                trace_summary,
                baseline_safe_capture=baselines[run["seed"]].get(index),
                variant=run["variant"],
            )
            row: dict[str, Any] = {
                "training_seed": run["seed"],
                "variant": run["variant"],
                "episode_index": index,
                "episode_seed": as_int(source.get("episode_seed"), "episode_seed"),
                "layout_seed": as_int(source.get("layout_seed", -1), "layout_seed"),
                "layout_signature": str(source.get("layout_signature", "")),
                "scenario": str(source.get("scenario", "")),
                "observation_condition": str(source.get("observation_condition", "")),
                "target_motion_mode": str(source.get("target_motion_mode", "")),
                "obstacle_count": as_int(source.get("obstacle_count", -1), "obstacle_count"),
                "safe_capture": bool(source["safe_capture"]),
                "baseline_safe_capture": baselines[run["seed"]].get(index),
                "termination_reason": source["termination_reason"],
                "primary_cause": primary,
                "diagnostic_labels": labels,
                "source_manifest_sha256": run["manifest_sha256"],
                "source_summary_sha256": run["summary_sha256"],
                "source_provenance_sha256": run["provenance_sha256"],
            }
            row.update(trace_summary)
            rows.append(row)
    failures = [row for row in rows if not row["safe_capture"]]
    safe = [row for row in rows if row["safe_capture"]]
    primary_counts = Counter(str(row["primary_cause"]) for row in failures)
    label_counts = Counter(label for row in failures for label in row["diagnostic_labels"])
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        failed = [row for row in subset if not row["safe_capture"]]
        by_variant[variant] = {
            "episodes": len(subset),
            "safe_capture_count": sum(row["safe_capture"] for row in subset),
            "safe_capture_rate": float(np.mean([row["safe_capture"] for row in subset])),
            "failure_count": len(failed),
            "primary_causes": dict(Counter(row["primary_cause"] for row in failed)),
            "high_credit_failure_count": sum("high_credit_failure" in row["diagnostic_labels"] for row in failed),
            "fallback_episode_count": sum("low_credit_or_nominal_fallback" in row["diagnostic_labels"] for row in failed),
            "cbf_abort_episode_count": sum("cbf_controlled_abort" in row["diagnostic_labels"] for row in failed),
            "mean_candidate_switch_rate": float(np.mean([row["candidate_switch_rate"] for row in subset])),
            "mean_cbf_correction_p95_mps": float(np.mean([row["cbf_correction_p95_mps"] for row in subset if row["cbf_correction_p95_mps"] is not None])) if subset else None,
        }
    by_condition: dict[str, Any] = {}
    for condition in sorted({str(row["observation_condition"]) for row in rows}):
        subset = [row for row in rows if row["observation_condition"] == condition]
        by_condition[condition] = {
            "episodes": len(subset),
            "safe_capture_rate": float(np.mean([row["safe_capture"] for row in subset])),
            "failure_count": sum(not row["safe_capture"] for row in subset),
            "primary_causes": dict(Counter(row["primary_cause"] for row in subset if not row["safe_capture"])),
        }
    return {
        "index_type": "jepa_safe_capture_v11_corrected_frame_failure_index",
        "development_only": True,
        "locked_test_opened": False,
        "input_root": str(input_root.resolve()),
        "run_count": len(runs),
        "episode_count": len(rows),
        "safe_capture_count": len(safe),
        "failure_count": len(failures),
        "safe_capture_rate": float(len(safe) / len(rows)),
        "primary_cause_counts": dict(primary_counts),
        "diagnostic_label_counts": dict(label_counts),
        "by_variant": by_variant,
        "by_observation_condition": by_condition,
        "manifest_sha256_by_seed": {str(run["seed"]): run["manifest_sha256"] for run in runs if run["variant"] == "m0"},
        "protocol_sha256": next(iter(protocol_hashes)),
        "target_drift_observable": False,
        "target_drift_note": "Online traces contain no offline future target labels; target drift is not inferred from proxy fields.",
        "runs": [
            {
                "training_seed": run["seed"],
                "variant": run["variant"],
                "path": str(run["path"]),
                "manifest_sha256": run["manifest_sha256"],
                "summary_sha256": run["summary_sha256"],
                "provenance_sha256": run["provenance_sha256"],
            }
            for run in runs
        ],
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "training_seed", "variant", "episode_index", "episode_seed", "layout_seed", "layout_signature",
        "scenario", "observation_condition", "target_motion_mode", "obstacle_count", "safe_capture",
        "baseline_safe_capture", "termination_reason", "primary_cause", "diagnostic_labels",
        "trace_steps", "candidate_switch_count", "candidate_switch_rate", "non_nominal_selection_rate",
        "ranking_modes", "ledger_state_counts", "ledger_credit_mean", "ledger_credit_min",
        "cbf_fallback_mode_counts", "cbf_unverified_steps_trace", "cbf_infeasible_steps_trace",
        "cbf_timeout_steps_trace", "cbf_correction_mean_mps", "cbf_correction_p95_mps", "cbf_correction_max_mps",
        "cbf_latency_p95_ms", "predicted_clearance_mean_m", "observed_clearance_mean_m",
        "clearance_prediction_gap_mean_m", "clearance_overoptimism_max_m", "predicted_visibility_mean",
        "observed_visibility_mean", "visibility_prediction_gap_mean", "observation_age_max_steps",
        "message_age_max_steps", "prediction_target_drift_observable", "source_manifest_sha256",
        "source_summary_sha256", "source_provenance_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values = dict(row)
            for key in ("diagnostic_labels", "ranking_modes", "ledger_state_counts", "cbf_fallback_mode_counts", "selected_candidate_indices"):
                if key in values:
                    values[key] = json.dumps(values[key], ensure_ascii=True, separators=(",", ":"))
            writer.writerow(values)


def write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/v11_failure_index", json.dumps({"index_type": report["index_type"], "development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/runs", json.dumps(report["runs"], indent=2), 0)
        writer.add_text("Provenance/limits", report["target_drift_note"], 0)
        writer.add_scalar("Episodes/total", report["episode_count"], 0)
        writer.add_scalar("Episodes/safe_capture", report["safe_capture_count"], 0)
        writer.add_scalar("Episodes/failure", report["failure_count"], 0)
        writer.add_scalar("Episodes/safe_capture_rate", report["safe_capture_rate"], 0)
        for cause, count in report["primary_cause_counts"].items():
            writer.add_scalar(f"Failure/primary/{cause}", count, 0)
        for label, count in report["diagnostic_label_counts"].items():
            writer.add_scalar(f"Failure/diagnostic/{label}", count, 0)
        for variant, values in report["by_variant"].items():
            writer.add_scalar(f"Variant/{variant}/safe_capture_rate", values["safe_capture_rate"], 0)
            writer.add_scalar(f"Variant/{variant}/failure_count", values["failure_count"], 0)
            writer.add_scalar(f"Variant/{variant}/cbf_abort_episode_count", values["cbf_abort_episode_count"], 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/v11_failure_index/text_summary",
        "Provenance/runs/text_summary",
        "Provenance/limits/text_summary",
        "Episodes/safe_capture_rate",
    }
    missing = sorted(required.difference(set(tags.get("tensors", [])) | set(tags.get("scalars", []))))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"v11 failure-index TensorBoard validation failed: missing={missing}, events={events}")
    return {
        "logdir": str(logdir),
        "event_files": events,
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_provenance": not missing,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# v11 Corrected-Frame Failure Index",
        "",
        "> Development-only read-only audit; `locked_test_opened=false`. Target drift is not inferred without offline future labels.",
        "",
        f"Runs: `{report['run_count']}`; episodes: `{report['episode_count']}`; safe capture: `{report['safe_capture_count']}/{report['episode_count']}` (`{report['safe_capture_rate']:.1%}`).",
        "",
        "## Primary Causes",
        "",
        "| Primary cause | Episodes |",
        "|---|---:|",
    ]
    for cause, count in sorted(report["primary_cause_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{cause}` | {count} |")
    lines.extend(["", "## Diagnostic Labels", "", "| Label | Episodes |", "|---|---:|"])
    for label, count in sorted(report["diagnostic_label_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(["", "## By Variant", "", "| Variant | Episodes | Safe capture | Failures | High-credit failures | Fallback episodes | CBF-abort episodes |", "|---|---:|---:|---:|---:|---:|---:|"])
    for variant, values in report["by_variant"].items():
        lines.append(f"| `{variant}` | {values['episodes']} | {values['safe_capture_count']}/{values['episodes']} ({values['safe_capture_rate']:.1%}) | {values['failure_count']} | {values['high_credit_failure_count']} | {values['fallback_episode_count']} | {values['cbf_abort_episode_count']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `primary_cause` is a deterministic label from episode summary and trace solver fields; it is not an unmeasured causal claim.",
        "- Ranking mismatch and ledger state are diagnostic signals. They must be checked against settled counterfactual rows before changing a score.",
        "- `target_drift_observable=false`: the online trace has no future target labels, so no target-motion drift claim is made.",
        "- All source runs remain development-only; this report does not authorize a locked test.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("v11 failure index requires --development-only")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    report = build_index(args.input_root.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "failure_index.csv", report["rows"])
    report["provenance"] = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "command": " ".join(__import__("sys").argv),
    }
    report["tensorboard"] = write_tensorboard(args.tensorboard_logdir, report)
    (output_dir / "failure_index.json").write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps(report["provenance"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"episodes": report["episode_count"], "failures": report["failure_count"], "primary_causes": report["primary_cause_counts"], "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
