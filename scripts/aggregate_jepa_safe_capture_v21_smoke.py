"""Aggregate the frozen V21 M0/M3/A1/A2 paired smoke matrix.

This is intentionally separate from the historical v2 P6 aggregator: V21
uses four pre-registered variants and a new run naming contract.  It consumes
only episode-level summaries and paired CSV outcomes; no result is rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import binomtest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from aggregate_jepa_safe_capture_v2_paired import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    _as_bool,
    _as_int,
    _bootstrap_delta,
    canonical_scene_manifest_sha256,
    sha256,
)


SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m0", "m3", "a1", "a2")
RUN_PATTERN = re.compile(r"^jepa_safe_capture_v21_smoke_(?P<variant>m0|m3|a1|a2)_seed(?P<seed>\d+)$")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_run(path: Path) -> dict[str, Any]:
    path = path.resolve()
    match = RUN_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Invalid V21 smoke run name: {path}")
    seed = int(match.group("seed"))
    variant = match.group("variant")
    if seed not in SEEDS:
        raise ValueError(f"Unexpected V21 training seed: {seed}")
    required = ("summary.json", "provenance.json", "episodes.csv", "scene_manifest.jsonl")
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    summary = _json(path / "summary.json")
    provenance = _json(path / "provenance.json")
    metadata = summary.get("metadata")
    overall = summary.get("overall")
    if not isinstance(metadata, Mapping) or not isinstance(overall, Mapping):
        raise ValueError(f"Missing metadata/overall: {path}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run crossed development boundary: {path}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Provenance crossed development boundary: {path}")
    declared = metadata.get("variant", {})
    if declared.get("variant") != variant or int(metadata.get("training_seed", -1)) != seed:
        raise ValueError(f"Run identity mismatch: {path}")
    with (path / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise ValueError(f"V21 smoke requires 20 episodes: {path}")
    episodes: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = _as_int(row.get("episode_index"), "episode_index")
        if index in episodes:
            raise ValueError(f"Duplicate episode index: {path}")
        episodes[index] = {
            "episode_index": index,
            "episode_seed": _as_int(row.get("episode_seed"), "episode_seed"),
            "safe_capture": _as_bool(row.get("safe_capture_success")),
            "collision": _as_bool(row.get("collision")),
            "boundary_violation": _as_bool(row.get("boundary_violation")),
            "pairwise_violation": _as_bool(row.get("pairwise_violation")),
            "transit_success": _as_bool(row.get("transit_success")),
        }
    if sorted(episodes) != list(range(20)):
        raise ValueError(f"Episode indices are not contiguous: {path}")
    inputs = metadata.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError(f"Missing inputs: {path}")
    manifest = path / "scene_manifest.jsonl"
    declared_manifest = str(inputs.get("scene_manifest_sha256", ""))
    if declared_manifest != sha256(manifest):
        raise ValueError(f"Scene manifest hash mismatch: {path}")
    return {
        "path": str(path),
        "seed": seed,
        "variant": variant,
        "summary": summary,
        "metadata": metadata,
        "overall": overall,
        "episodes": episodes,
        "summary_sha256": sha256(path / "summary.json"),
        "provenance_sha256": sha256(path / "provenance.json"),
        "manifest_sha256": declared_manifest,
        "canonical_manifest_sha256": canonical_scene_manifest_sha256(manifest),
    }


def _discover(root: Path) -> list[dict[str, Any]]:
    runs = [_load_run(path) for path in sorted(root.resolve().glob("jepa_safe_capture_v21_smoke_*_seed*")) if path.is_dir() and RUN_PATTERN.match(path.name)]
    expected = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
    actual = {(run["seed"], run["variant"]) for run in runs}
    if actual != expected:
        raise ValueError(f"Incomplete V21 matrix; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return runs


def _check_pairing(runs: list[dict[str, Any]]) -> tuple[dict[str, str], str]:
    canonical_by_seed: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    for run in runs:
        canonical_by_seed[run["seed"]].add(run["canonical_manifest_sha256"])
    invalid = {seed: sorted(values) for seed, values in canonical_by_seed.items() if len(values) != 1}
    if invalid:
        raise ValueError(f"Scene geometry differs within a V21 seed block: {invalid}")
    protocol = {str(run["metadata"]["inputs"].get("protocol_sha256")) for run in runs}
    if len(protocol) != 1:
        raise ValueError(f"Protocol differs across V21 runs: {sorted(protocol)}")
    reference = next(run for run in runs if run["seed"] == SEEDS[0] and run["variant"] == "m0")
    reference_seeds = [reference["episodes"][index]["episode_seed"] for index in range(20)]
    for run in runs:
        seeds = [run["episodes"][index]["episode_seed"] for index in range(20)]
        if seeds != reference_seeds:
            raise ValueError(f"Episode pairing differs: {run['path']}")
    return {str(seed): next(iter(canonical_by_seed[seed])) for seed in SEEDS}, next(iter(protocol))


def _paired(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    if base["seed"] != candidate["seed"]:
        raise ValueError("Paired runs must use the same training seed")
    deltas: list[float] = []
    pairs: list[dict[str, Any]] = []
    for index in range(20):
        left = base["episodes"][index]
        right = candidate["episodes"][index]
        if left["episode_seed"] != right["episode_seed"]:
            raise ValueError("Episode seed mismatch")
        delta = int(right["safe_capture"]) - int(left["safe_capture"])
        deltas.append(float(delta))
        pairs.append({"episode_index": index, "episode_seed": left["episode_seed"], "base_safe_capture": left["safe_capture"], "candidate_safe_capture": right["safe_capture"], "delta": delta})
    improved = sum(not pair["base_safe_capture"] and pair["candidate_safe_capture"] for pair in pairs)
    degraded = sum(pair["base_safe_capture"] and not pair["candidate_safe_capture"] for pair in pairs)
    discordant = improved + degraded
    return {
        "training_seed": base["seed"],
        "base_variant": base["variant"],
        "candidate_variant": candidate["variant"],
        "episodes": 20,
        "base_safe_capture_count": sum(pair["base_safe_capture"] for pair in pairs),
        "candidate_safe_capture_count": sum(pair["candidate_safe_capture"] for pair in pairs),
        "improved": int(improved),
        "degraded": int(degraded),
        "tied": int(20 - improved - degraded),
        "delta_rate": float(np.mean(deltas)),
        "mcnemar_exact_two_sided_p": float(binomtest(min(improved, degraded), n=discordant, p=0.5).pvalue) if discordant else 1.0,
        "bootstrap": _bootstrap_delta(np.asarray(deltas, dtype=np.float64)),
        "pairs": pairs,
    }


def _metric(run: Mapping[str, Any]) -> dict[str, Any]:
    overall = run["overall"]
    return {
        "training_seed": run["seed"],
        "variant": run["variant"],
        "episodes": 20,
        "safe_capture_count": int(overall["safe_capture_count"]),
        "safe_capture_rate": float(overall["safe_capture_rate"]),
        "collision_count": int(overall["collision_count"]),
        "boundary_violation_count": int(overall["boundary_violation_count"]),
        "pairwise_violation_count": int(overall["pairwise_violation_count"]),
        "cbf_infeasible_steps": int(overall.get("cbf_infeasible_steps", 0)),
        "cbf_timeout_steps": int(overall.get("cbf_timeout_steps", 0)),
        "cbf_controlled_abort_steps": int(overall.get("cbf_controlled_abort_steps", 0)),
        "cbf_fallback_steps": int(overall.get("cbf_fallback_steps", 0)),
        "cbf_unverified_steps": int(overall.get("cbf_unverified_steps", 0)),
        "raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        "transit_success_rate": float(overall.get("transit_success_rate", 0.0)),
        "mean_capture_time_seconds": overall.get("mean_capture_time_seconds"),
        "cycle_p95_ms": float(overall.get("latency_breakdown", {}).get("cycle_total", {}).get("max_episode_p95_ms", 0.0)),
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    manifests, protocol = _check_pairing(runs)
    by_key = {(run["seed"], run["variant"]): run for run in runs}
    metrics = [_metric(run) for run in sorted(runs, key=lambda item: (item["variant"], item["seed"]))]
    comparisons = [_paired(by_key[(seed, "m0")], by_key[(seed, variant)]) for seed in SEEDS for variant in ("m3", "a1", "a2")]
    m3 = [item for item in comparisons if item["candidate_variant"] == "m3"]
    all_deltas = np.asarray([pair["delta"] for item in m3 for pair in item["pairs"]], dtype=np.float64)
    safe_metrics = [item for item in metrics]
    safety_gate = all(item["collision_count"] == item["boundary_violation_count"] == item["pairwise_violation_count"] == 0 for item in safe_metrics)
    reliability_gate = all(item["cbf_timeout_steps"] == 0 and item["cbf_fallback_steps"] >= item["cbf_infeasible_steps"] and item["cbf_controlled_abort_steps"] == item["cbf_unverified_steps"] and item["raw_unverified_executed_steps"] == 0 for item in safe_metrics)
    deltas = [item["delta_rate"] for item in m3]
    decision = {
        "safety_hard_gate": safety_gate,
        "reliability_observability_gate": reliability_gate,
        "m3_mean_paired_delta_rate": float(np.mean(deltas)),
        "m3_seed_delta_rates": {str(seed): float(item["delta_rate"]) for seed, item in zip(SEEDS, m3)},
        "m3_seeds_nonnegative": int(sum(value >= 0.0 for value in deltas)),
        "m3_cross_seed_improved": int(sum(item["improved"] for item in m3)),
        "m3_cross_seed_degraded": int(sum(item["degraded"] for item in m3)),
        "m3_cross_seed_tied": int(len(all_deltas) - sum(item["improved"] for item in m3) - sum(item["degraded"] for item in m3)),
        "m3_cross_seed_bootstrap": _bootstrap_delta(all_deltas),
    }
    if not safety_gate or not reliability_gate:
        classification = "rejected_for_safety"
    elif decision["m3_mean_paired_delta_rate"] > 0.0 and decision["m3_seeds_nonnegative"] >= 2:
        classification = "positive_development_evidence"
    elif decision["m3_mean_paired_delta_rate"] >= 0.0 and decision["m3_seeds_nonnegative"] >= 2:
        classification = "safety_preserving_non_inferiority"
    else:
        classification = "useful_safety_fallback_only"
    decision["classification"] = classification
    return {
        "aggregation_type": "jepa_safe_capture_v21_four_variant_paired_smoke",
        "development_only": True,
        "locked_test_opened": False,
        "training_seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "episodes_per_run": 20,
        "run_count": len(runs),
        "scene_manifest_canonical_sha256_by_seed": manifests,
        "protocol_sha256": protocol,
        "run_metrics": metrics,
        "paired_comparisons": comparisons,
        "decision": decision,
        "inputs": [{"path": run["path"], "summary_sha256": run["summary_sha256"], "provenance_sha256": run["provenance_sha256"], "manifest_sha256": run["manifest_sha256"], "training_seed": run["seed"], "variant": run["variant"], "jepa_checkpoint_sha256": run["metadata"]["inputs"].get("jepa_checkpoint_sha256")} for run in sorted(runs, key=lambda item: (item["variant"], item["seed"]))],
        "environment": {"python": platform.python_version(), "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
    }


def _write_tensorboard(report: Mapping[str, Any], logdir: Path) -> dict[str, Any]:
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/aggregate", json.dumps({"aggregation_type": report["aggregation_type"], "variants": report["variants"]}, indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(report["inputs"], indent=2), 0)
        writer.add_text("Provenance/decision", json.dumps(report["decision"], indent=2), 0)
        writer.add_text("Provenance/protocol", str(report["protocol_sha256"]), 0)
        for row in report["run_metrics"]:
            tag = f"{row['variant']}/seed{row['training_seed']}"
            writer.add_scalar(f"SafeCapture/{tag}", row["safe_capture_rate"], 0)
            writer.add_scalar(f"Safety/{tag}/raw_unverified", row["raw_unverified_executed_steps"], 0)
            writer.add_scalar(f"CBF/{tag}/controlled_abort", row["cbf_controlled_abort_steps"], 0)
            writer.add_scalar(f"Latency/{tag}/cycle_p95_ms", row["cycle_p95_ms"], 0)
        writer.add_scalar("Paired/m3/delta_rate_mean", report["decision"]["m3_mean_paired_delta_rate"], 0)
        writer.add_scalar("Paired/m3/improved", report["decision"]["m3_cross_seed_improved"], 0)
        writer.add_scalar("Paired/m3/degraded", report["decision"]["m3_cross_seed_degraded"], 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {"Config/aggregate/text_summary", "Provenance/inputs/text_summary", "Provenance/decision/text_summary", "Provenance/protocol/text_summary"}
    missing = sorted(required.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"V21 aggregate TensorBoard incomplete: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", []))}


def _markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    manifest_text = "; ".join(f"seed {seed}: `{value}`" for seed, value in report["scene_manifest_canonical_sha256_by_seed"].items())
    lines = ["# JEPA Safe-Capture V21 Four-Variant Paired Smoke", "", "Development-only aggregate; `locked_test_opened=false`.", "", f"Canonical scene manifest SHA-256 by seed: {manifest_text}", "", "| Variant | Seed 20260911 | Seed 20260912 | Seed 20260913 | Mean +/- SD |", "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        values = [row["safe_capture_rate"] for row in report["run_metrics"] if row["variant"] == variant]
        lines.append(f"| {variant.upper()} | " + " | ".join(f"{int(value * 20)}/20" for value in values) + f" | {np.mean(values):.3f} +/- {np.std(values, ddof=1):.3f} |" )
    lines += ["", "## M3 paired delta", "", f"Mean per-seed delta: `{decision['m3_mean_paired_delta_rate']:.3f}`; non-negative seeds: `{decision['m3_seeds_nonnegative']}/3`.", f"Improved/degraded/tied: `{decision['m3_cross_seed_improved']}/{decision['m3_cross_seed_degraded']}/{decision['m3_cross_seed_tied']}`.", f"Bootstrap 95% CI: `[{decision['m3_cross_seed_bootstrap']['ci95_low']:.3f}, {decision['m3_cross_seed_bootstrap']['ci95_high']:.3f}]`.", "", "| Seed | M0 | M3 | Delta | Improved | Degraded | McNemar p |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for item in report["paired_comparisons"]:
        if item["candidate_variant"] == "m3":
            lines.append(f"| {item['training_seed']} | {item['base_safe_capture_count']}/20 | {item['candidate_safe_capture_count']}/20 | {item['delta_rate']:.3f} | {item['improved']} | {item['degraded']} | {item['mcnemar_exact_two_sided_p']:.4f} |")
    lines += ["", "## Gates", "", f"Safety hard gate: `{'PASS' if decision['safety_hard_gate'] else 'FAIL'}`.", f"Reliability observability gate: `{'PASS' if decision['reliability_observability_gate'] else 'FAIL'}`.", f"Classification: `{decision['classification']}`.", "", "This smoke is not locked-test evidence and does not establish a formal JEPA safe-capture improvement."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("V21 aggregate requires --development-only")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    report = _aggregate(_discover(args.input_root))
    output.mkdir(parents=True, exist_ok=True)
    report["tensorboard"] = _write_tensorboard(report, output / "tensorboard")
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    (output / "paired_comparisons.json").write_text(json.dumps(report["paired_comparisons"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
