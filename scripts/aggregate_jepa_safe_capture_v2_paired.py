"""Aggregate paired P6 safe-capture development runs.

The aggregator consumes evaluator summaries and episode tables only. It
requires one fixed scene manifest across all runs, preserves the development
only/locked-test boundary, and reports paired binary outcomes before any
across-seed average. A3 is retained as a raw/no-CBF diagnostic and is never
included in the safety-preserving decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


VARIANTS = ("m0", "m1", "m2", "m3", "a1", "a2", "a3")
SEEDS = (20260911, 20260912, 20260913)
SAFE_VARIANTS = tuple(variant for variant in VARIANTS if variant != "a3")
RUN_PATTERN = re.compile(r"^jepa_safe_capture_v2_p6_paired_(?P<stage>smoke|full)_seed(?P<seed>\d+)_(?P<variant>m0|m1|m2|m3|a1|a2|a3)$")
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_SAMPLES = 4000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_scene_manifest_sha256(path: Path) -> str:
    """Hash scene content while excluding per-run training provenance."""

    canonical_records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Scene manifest record is not an object: {path}")
        record = dict(record)
        record.pop("training_seed", None)
        canonical_records.append(record)
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for record in canonical_records
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def finite_stats(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("Cannot summarize an empty or non-finite metric list.")
    return {
        "mean": float(np.mean(data)),
        "sample_std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer {label}: {value!r}") from error


def load_run(path: Path, *, stage: str, expected_episodes: int) -> dict[str, Any]:
    path = path.resolve()
    match = RUN_PATTERN.match(path.name)
    if match is None or match.group("stage") != stage:
        raise ValueError(f"Run directory does not match {stage} contract: {path}")
    seed = int(match.group("seed"))
    variant = match.group("variant")
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError(f"Unexpected run identity: {path.name}")
    summary_path = path / "summary.json"
    provenance_path = path / "provenance.json"
    episodes_path = path / "episodes.csv"
    manifest_path = path / "scene_manifest.jsonl"
    for required in (summary_path, provenance_path, episodes_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing P6 artifact: {required}")
    summary = read_json(summary_path)
    provenance = read_json(provenance_path)
    metadata = summary.get("metadata", {})
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Summary crossed the locked-test boundary: {summary_path}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Invalid provenance boundary: {provenance_path}")
    if int(metadata.get("episodes", -1)) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes in {summary_path}")
    declared = metadata.get("variant", {})
    if declared.get("variant") != variant or int(metadata.get("training_seed", -1)) != seed:
        raise ValueError(f"Run identity metadata mismatch: {summary_path}")
    inputs = metadata.get("inputs", {})
    manifest_hash = str(inputs.get("scene_manifest_sha256", ""))
    if len(manifest_hash) != 64 or manifest_hash != sha256(manifest_path):
        raise ValueError(f"Scene manifest provenance mismatch: {summary_path}")
    rows: list[dict[str, Any]] = []
    with episodes_path.open("r", newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if len(rows) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episode rows in {episodes_path}, got {len(rows)}")
    episode_map: dict[int, dict[str, Any]] = {}
    for row in rows:
        episode_index = _as_int(row.get("episode_index"), "episode_index")
        episode_seed = _as_int(row.get("episode_seed"), "episode_seed")
        if episode_index in episode_map:
            raise ValueError(f"Duplicate episode index in {episodes_path}: {episode_index}")
        episode_map[episode_index] = {
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "safe_capture": _as_bool(row.get("safe_capture_success")),
            "collision": _as_bool(row.get("collision")),
            "boundary_violation": _as_bool(row.get("boundary_violation")),
            "pairwise_violation": _as_bool(row.get("pairwise_violation")),
            "transit_success": _as_bool(row.get("transit_success")),
            "cbf_infeasible_steps": _as_int(row.get("cbf_infeasible_steps", 0), "cbf_infeasible_steps"),
            "cbf_timeout_steps": _as_int(row.get("cbf_timeout_steps", 0), "cbf_timeout_steps"),
            "cbf_unverified_steps": _as_int(row.get("cbf_unverified_steps", 0), "cbf_unverified_steps"),
            "raw_unverified_executed_steps": _as_int(
                row.get("raw_unverified_executed_steps", 0), "raw_unverified_executed_steps"
            ),
            "termination_reason": str(row.get("termination_reason", "")),
        }
    if sorted(episode_map) != list(range(expected_episodes)):
        raise ValueError(f"Episode indices are not contiguous in {episodes_path}")
    return {
        "path": str(path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "provenance_sha256": sha256(provenance_path),
        "manifest_sha256": manifest_hash,
        "canonical_manifest_sha256": canonical_scene_manifest_sha256(manifest_path),
        "seed": seed,
        "variant": variant,
        "stage": stage,
        "episodes": episode_map,
        "overall": summary["overall"],
        "metadata": metadata,
    }


def discover_runs(input_root: Path, *, stage: str, expected_episodes: int) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(input_root.resolve().glob(f"jepa_safe_capture_v2_p6_paired_{stage}_seed*_*")):
        if path.is_dir() and RUN_PATTERN.match(path.name):
            runs.append(load_run(path, stage=stage, expected_episodes=expected_episodes))
    expected = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
    actual = {(run["seed"], run["variant"]) for run in runs}
    missing = sorted(expected.difference(actual))
    if missing or len(runs) != len(actual):
        raise ValueError(f"P6 matrix incomplete: missing={missing}, duplicate={len(runs) != len(actual)}")
    return runs


def _check_pairing(runs: list[dict[str, Any]]) -> str:
    manifests = {run["canonical_manifest_sha256"] for run in runs}
    if len(manifests) != 1:
        raise ValueError(f"Runs are not paired to one scene manifest: {sorted(manifests)}")
    manifest_hash = next(iter(manifests))
    reference = next(run for run in runs if run["seed"] == SEEDS[0] and run["variant"] == "m0")
    reference_seeds = [reference["episodes"][index]["episode_seed"] for index in range(len(reference["episodes"]))]
    for run in runs:
        seeds = [run["episodes"][index]["episode_seed"] for index in range(len(run["episodes"]))]
        if seeds != reference_seeds:
            raise ValueError(f"Episode seed pairing mismatch: {run['path']}")
    return manifest_hash


def _run_metric(run: dict[str, Any]) -> dict[str, Any]:
    overall = run["overall"]
    return {
        "stage": run["stage"],
        "training_seed": run["seed"],
        "variant": run["variant"],
        "episodes": len(run["episodes"]),
        "safe_capture_count": int(overall["safe_capture_count"]),
        "safe_capture_rate": float(overall["safe_capture_rate"]),
        "collision_count": int(overall["collision_count"]),
        "boundary_violation_count": int(overall["boundary_violation_count"]),
        "pairwise_violation_count": int(overall["pairwise_violation_count"]),
        "cbf_infeasible_steps": int(overall.get("cbf_infeasible_steps", 0)),
        "cbf_timeout_steps": int(overall.get("cbf_timeout_steps", 0)),
        "cbf_unverified_steps": int(overall.get("cbf_unverified_steps", 0)),
        "raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        "cbf_fallback_steps": int(overall.get("cbf_fallback_steps", 0)),
        "cbf_controlled_abort_steps": int(overall.get("cbf_controlled_abort_steps", 0)),
        "transit_success_rate": float(overall.get("transit_success_rate", 0.0)),
        "mean_capture_time_seconds": overall.get("mean_capture_time_seconds"),
        "mean_cbf_p95_solve_latency_ms": float(overall.get("mean_cbf_p95_solve_latency_ms", 0.0)),
    }


def _bootstrap_delta(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> dict[str, float | int]:
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty paired delta.")
    rng = np.random.default_rng(seed)
    observed = float(np.mean(values))
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    samples = np.mean(values[indices], axis=1)
    return {
        "observed": observed,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "unit": "episode_pair",
    }


def paired_comparison(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if base["seed"] != candidate["seed"]:
        raise ValueError("Paired comparison requires the same training seed.")
    pairs = []
    for index in range(len(base["episodes"])):
        left = base["episodes"][index]
        right = candidate["episodes"][index]
        if left["episode_seed"] != right["episode_seed"]:
            raise ValueError("Episode seed mismatch in paired comparison.")
        pairs.append({
            "episode_index": index,
            "episode_seed": left["episode_seed"],
            "base_safe_capture": left["safe_capture"],
            "candidate_safe_capture": right["safe_capture"],
            "delta": int(right["safe_capture"]) - int(left["safe_capture"]),
        })
    improved = sum(int(not item["base_safe_capture"] and item["candidate_safe_capture"]) for item in pairs)
    degraded = sum(int(item["base_safe_capture"] and not item["candidate_safe_capture"]) for item in pairs)
    tied = len(pairs) - improved - degraded
    discordant = improved + degraded
    pvalue = float(binomtest(min(improved, degraded), n=discordant, p=0.5).pvalue) if discordant else 1.0
    deltas = np.asarray([item["delta"] for item in pairs], dtype=np.float64)
    return {
        "training_seed": base["seed"],
        "base_variant": base["variant"],
        "candidate_variant": candidate["variant"],
        "episodes": len(pairs),
        "improved": improved,
        "degraded": degraded,
        "tied": tied,
        "candidate_safe_capture_count": int(sum(item["candidate_safe_capture"] for item in pairs)),
        "base_safe_capture_count": int(sum(item["base_safe_capture"] for item in pairs)),
        "delta_count": int(improved - degraded),
        "delta_rate": float(np.mean(deltas)),
        "mcnemar_exact_two_sided_p": pvalue,
        "bootstrap": _bootstrap_delta(deltas),
        "pairs": pairs,
    }


def aggregate(runs: list[dict[str, Any]], *, stage: str, expected_episodes: int) -> dict[str, Any]:
    manifest_hash = _check_pairing(runs)
    ordered = sorted(runs, key=lambda run: (run["variant"], run["seed"]))
    metrics = [_run_metric(run) for run in ordered]
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        rows = [row for row in metrics if row["variant"] == variant]
        by_variant[variant] = {
            "training_seeds": [row["training_seed"] for row in rows],
            "safe_capture_rate": finite_stats(float(row["safe_capture_rate"]) for row in rows),
            "safe_capture_count_total": int(sum(row["safe_capture_count"] for row in rows)),
            "collision_count_total": int(sum(row["collision_count"] for row in rows)),
            "boundary_violation_count_total": int(sum(row["boundary_violation_count"] for row in rows)),
            "pairwise_violation_count_total": int(sum(row["pairwise_violation_count"] for row in rows)),
            "cbf_infeasible_steps_total": int(sum(row["cbf_infeasible_steps"] for row in rows)),
            "cbf_timeout_steps_total": int(sum(row["cbf_timeout_steps"] for row in rows)),
            "cbf_fallback_steps_total": int(sum(row["cbf_fallback_steps"] for row in rows)),
            "cbf_controlled_abort_steps_total": int(sum(row["cbf_controlled_abort_steps"] for row in rows)),
            "raw_unverified_executed_steps_total": int(
                sum(row["raw_unverified_executed_steps"] for row in rows)
            ),
            "transit_success_rate": finite_stats(float(row["transit_success_rate"]) for row in rows),
            "diagnostic_only": variant == "a3",
        }
    comparisons: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_runs = {run["variant"]: run for run in runs if run["seed"] == seed}
        for candidate in ("m1", "m2", "m3", "a1", "a2", "a3"):
            comparisons.append(paired_comparison(seed_runs["m0"], seed_runs[candidate]))
    m3_seed_comparisons = [item for item in comparisons if item["candidate_variant"] == "m3"]
    all_m3_deltas = np.concatenate([
        np.asarray([pair["delta"] for pair in item["pairs"]], dtype=np.float64)
        for item in m3_seed_comparisons
    ])
    m3_improved = sum(item["improved"] for item in m3_seed_comparisons)
    m3_degraded = sum(item["degraded"] for item in m3_seed_comparisons)
    safe_rows = [row for row in metrics if row["variant"] in SAFE_VARIANTS]
    safety_gate = all(
        row["collision_count"] == 0 and row["boundary_violation_count"] == 0 and row["pairwise_violation_count"] == 0
        for row in safe_rows
    )
    # An infeasible request can still resolve through a solver-verified
    # nominal-CBF or hold fallback, so infeasible and unverified counts need
    # not match. Every infeasible request must be visibly routed to fallback;
    # every unverified result must be a controlled abort; no timeout may hide.
    reliability_gate = all(
        row["cbf_timeout_steps"] == 0
        and row["cbf_fallback_steps"] >= row["cbf_infeasible_steps"]
        and row["cbf_controlled_abort_steps"] == row["cbf_unverified_steps"]
        and row["raw_unverified_executed_steps"] == 0
        for row in safe_rows
    )
    m3_seed_deltas = [item["delta_rate"] for item in m3_seed_comparisons]
    decision = {
        "stage": stage,
        "safe_variants_only": list(SAFE_VARIANTS),
        "safety_hard_gate": safety_gate,
        "reliability_observability_gate": reliability_gate,
        "m3_mean_paired_delta_rate": float(np.mean(m3_seed_deltas)),
        "m3_seed_delta_rates": {str(seed): value for seed, value in zip(SEEDS, m3_seed_deltas)},
        "m3_seeds_nonnegative": int(sum(value >= 0.0 for value in m3_seed_deltas)),
        "m3_cross_seed_improved": int(m3_improved),
        "m3_cross_seed_degraded": int(m3_degraded),
        "m3_cross_seed_tied": int(len(all_m3_deltas) - m3_improved - m3_degraded),
        "m3_cross_seed_bootstrap": _bootstrap_delta(all_m3_deltas),
        "a3_excluded_from_safety_decision": True,
    }
    if not safety_gate or not reliability_gate:
        classification = "insufficient_evidence_or_reject"
    elif decision["m3_mean_paired_delta_rate"] > 0.0 and decision["m3_seeds_nonnegative"] >= 2:
        classification = "positive_development_evidence"
    elif decision["m3_mean_paired_delta_rate"] >= 0.0 and decision["m3_seeds_nonnegative"] >= 2:
        classification = "safety_preserving_non_inferiority"
    else:
        classification = "useful_safety_fallback_only"
    decision["classification"] = classification
    return {
        "aggregation_type": "jepa_safe_capture_v2_p6_paired_development",
        "stage": stage,
        "episodes_per_run": expected_episodes,
        "run_count": len(runs),
        "training_seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "scene_manifest_sha256": manifest_hash,
        "scene_manifest_hash_mode": "canonical_without_training_seed",
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "run_metrics": metrics,
        "by_variant": by_variant,
        "paired_comparisons": comparisons,
        "decision": decision,
        "inputs": [
            {
                "path": run["path"],
                "summary_sha256": run["summary_sha256"],
                "provenance_sha256": run["provenance_sha256"],
                "manifest_sha256": run["manifest_sha256"],
                "canonical_manifest_sha256": run["canonical_manifest_sha256"],
                "training_seed": run["seed"],
                "variant": run["variant"],
            }
            for run in ordered
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_tensorboard(report: dict[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/aggregate", json.dumps({"aggregation_type": report["aggregation_type"], "stage": report["stage"]}, indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(report["inputs"], indent=2), 0)
        writer.add_text("Provenance/decision", json.dumps(report["decision"], indent=2), 0)
        writer.add_text("Provenance/scene_manifest", report["scene_manifest_sha256"], 0)
        for variant, item in report["by_variant"].items():
            step = VARIANTS.index(variant)
            writer.add_scalar(f"SafeCapture/{variant}/rate_mean", item["safe_capture_rate"]["mean"], step)
            writer.add_scalar(f"SafeCapture/{variant}/collision_count", item["collision_count_total"], step)
            writer.add_scalar(f"SafeCapture/{variant}/boundary_count", item["boundary_violation_count_total"], step)
            writer.add_scalar(f"SafeCapture/{variant}/pairwise_count", item["pairwise_violation_count_total"], step)
            writer.add_scalar(f"CBF/{variant}/infeasible_steps", item["cbf_infeasible_steps_total"], step)
        writer.add_scalar("Paired/m3/delta_rate_mean", report["decision"]["m3_mean_paired_delta_rate"], 0)
        writer.add_scalar("Paired/m3/improved", report["decision"]["m3_cross_seed_improved"], 0)
        writer.add_scalar("Paired/m3/degraded", report["decision"]["m3_cross_seed_degraded"], 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/aggregate/text_summary",
        "Provenance/inputs/text_summary",
        "Provenance/decision/text_summary",
        "Provenance/scene_manifest/text_summary",
    }
    missing = sorted(required.difference(tags.get("tensors", [])))
    if missing:
        raise ValueError(f"TensorBoard provenance is incomplete: {missing}")
    return {
        "logdir": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "required_provenance": not missing,
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
    }


def render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# JEPA Safe-Capture v2 P6 Paired Development Aggregate",
        "",
        f"Development-only {report['stage']} aggregate. locked_test_opened=false; A3 is raw/no-CBF diagnostic only.",
        "",
        f"Canonical scene manifest SHA-256: {report['scene_manifest_sha256']}",
        f"Episodes per run: {report['episodes_per_run']}",
        "",
        "## Per-Variant Results",
        "",
        "| Variant | Safe capture mean +/- std | Total collision | Boundary | Pairwise | CBF infeasible | Diagnostic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = report["by_variant"][variant]
        rate = item["safe_capture_rate"]
        lines.append(
            f"| {variant.upper()} | {rate['mean']:.3f} +/- {rate['sample_std']:.3f} | "
            f"{item['collision_count_total']} | {item['boundary_violation_count_total']} | "
            f"{item['pairwise_violation_count_total']} | {item['cbf_infeasible_steps_total']} | "
            f"{'yes' if item['diagnostic_only'] else 'no'} |"
        )
    lines += [
        "",
        "## M3 Paired Delta",
        "",
        f"Mean per-seed delta: {decision['m3_mean_paired_delta_rate']:.3f}; non-negative seeds: {decision['m3_seeds_nonnegative']}/3.",
        f"Improved/degraded/tied episodes: {decision['m3_cross_seed_improved']}/{decision['m3_cross_seed_degraded']}/{decision['m3_cross_seed_tied']}.",
        f"Bootstrap 95% CI: [{decision['m3_cross_seed_bootstrap']['ci95_low']:.3f}, {decision['m3_cross_seed_bootstrap']['ci95_high']:.3f}].",
        "",
        "| Seed | M0 safe capture | M3 safe capture | Delta | Improved | Degraded | McNemar exact p |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["paired_comparisons"]:
        if item["candidate_variant"] == "m3":
            lines.append(
                f"| {item['training_seed']} | {item['base_safe_capture_count']}/{item['episodes']} | "
                f"{item['candidate_safe_capture_count']}/{item['episodes']} | {item['delta_rate']:.3f} | "
                f"{item['improved']} | {item['degraded']} | {item['mcnemar_exact_two_sided_p']:.4f} |"
            )
    lines += [
        "",
        "## Gates and Classification",
        "",
        f"Safety hard gate: {'PASS' if decision['safety_hard_gate'] else 'FAIL'}.",
        f"Reliability observability gate: {'PASS' if decision['reliability_observability_gate'] else 'FAIL'}.",
        f"Final development classification: {decision['classification']}.",
        "",
        "A3 collision results are excluded from the safety decision and retained only as evidence that raw actions bypass the CBF boundary.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("P7 aggregate requires --development-only.")
    # The frozen development validation split contains 40 episodes.  The
    # 100-episode locked-test split is intentionally unreachable here.
    expected_episodes = 20 if args.stage == "smoke" else 40
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    runs = discover_runs(args.input_root, stage=args.stage, expected_episodes=expected_episodes)
    report = aggregate(runs, stage=args.stage, expected_episodes=expected_episodes)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "run_metrics.csv", report["run_metrics"])
    write_csv(
        output_dir / "m3_seed_comparisons.csv",
        [item for item in report["paired_comparisons"] if item["candidate_variant"] == "m3"],
    )
    tensorboard = write_tensorboard(report, output_dir / "tensorboard")
    report["tensorboard"] = tensorboard
    (output_dir / "paired_comparison.json").write_text(json.dumps(report["paired_comparisons"], indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "tensorboard": tensorboard}, indent=2))


if __name__ == "__main__":
    main()
