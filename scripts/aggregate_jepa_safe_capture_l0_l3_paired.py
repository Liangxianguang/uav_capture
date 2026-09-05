"""Aggregate paired L0-L3 closed-loop development runs."""

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
from torch.utils.tensorboard import SummaryWriter


VARIANTS = ("m0", "m1", "m2", "m3", "a1", "a2", "a3")
SEEDS = (20260911, 20260912, 20260913)
SAFE_VARIANTS = tuple(value for value in VARIANTS if value != "a3")
RUN_PATTERN = re.compile(
    r"^jepa_safe_capture_l0_l3_paired_(?P<stage>full|smoke)_seed(?P<seed>\d+)_(?P<variant>m0|m1|m2|m3|a1|a2|a3)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error


def canonical_manifest_sha256(path: Path) -> str:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record is not an object: {path}")
        record = dict(record)
        record.pop("training_seed", None)
        records.append(record)
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for record in records
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_run(path: Path, *, stage: str, expected_episodes: int) -> dict[str, Any]:
    path = path.resolve()
    match = RUN_PATTERN.match(path.name)
    if match is None or match.group("stage") != stage:
        raise ValueError(f"Invalid L0-L3 run name: {path}")
    seed = int(match.group("seed"))
    variant = match.group("variant")
    if seed not in SEEDS:
        raise ValueError(f"Unexpected training seed: {seed}")
    summary_path = path / "summary.json"
    provenance_path = path / "provenance.json"
    episodes_path = path / "episodes.csv"
    manifest_path = path / "scene_manifest.jsonl"
    for required in (summary_path, provenance_path, episodes_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    summary = read_json(summary_path)
    provenance = read_json(provenance_path)
    metadata = summary.get("metadata", {})
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run crossed the locked-test boundary: {summary_path}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Invalid provenance boundary: {provenance_path}")
    if as_int(metadata.get("episodes", -1), "episode count") != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes: {summary_path}")
    declared = metadata.get("variant", {})
    if declared.get("variant") != variant or as_int(metadata.get("training_seed", -1), "training seed") != seed:
        raise ValueError(f"Run identity mismatch: {summary_path}")
    inputs = metadata.get("inputs", {})
    manifest_hash = str(inputs.get("scene_manifest_sha256", ""))
    if manifest_hash != sha256(manifest_path):
        raise ValueError(f"Manifest provenance mismatch: {summary_path}")
    rows: list[dict[str, Any]] = []
    with episodes_path.open("r", newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if len(rows) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} rows, got {len(rows)}: {episodes_path}")
    episodes: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = as_int(row.get("episode_index"), "episode_index")
        if index in episodes:
            raise ValueError(f"Duplicate episode index {index}: {episodes_path}")
        episodes[index] = {
            "episode_index": index,
            "episode_seed": as_int(row.get("episode_seed"), "episode_seed"),
            "level": str(row.get("level", "")),
            "scenario_name": str(row.get("scenario_name", "")),
            "obstacle_count": as_int(row.get("obstacle_count", 0), "obstacle_count"),
            "observation_condition": str(row.get("observation_condition", "")),
            "target_motion_mode": str(row.get("target_motion_mode", "")),
            "safe_capture": as_bool(row.get("safe_capture_success")),
            "collision": as_bool(row.get("collision")),
            "boundary": as_bool(row.get("boundary_violation")),
            "pairwise": as_bool(row.get("pairwise_violation")),
            "transit": as_bool(row.get("transit_success")),
            "raw_unverified_steps": as_int(row.get("raw_unverified_executed_steps", 0), "raw unverified steps"),
            "cbf_abort_steps": as_int(row.get("cbf_controlled_abort_steps", 0), "CBF abort steps"),
        }
    if sorted(episodes) != list(range(expected_episodes)):
        raise ValueError(f"Episode indices are not contiguous: {episodes_path}")
    return {
        "path": str(path),
        "summary": summary,
        "metadata": metadata,
        "episodes": episodes,
        "seed": seed,
        "variant": variant,
        "stage": stage,
        "manifest_sha256": manifest_hash,
        "canonical_manifest_sha256": canonical_manifest_sha256(manifest_path),
        "summary_sha256": sha256(summary_path),
        "provenance_sha256": sha256(provenance_path),
    }


def discover_runs(input_root: Path, *, stage: str, expected_episodes: int) -> list[dict[str, Any]]:
    runs = [
        load_run(path, stage=stage, expected_episodes=expected_episodes)
        for path in sorted(input_root.resolve().glob(f"jepa_safe_capture_l0_l3_paired_{stage}_seed*_*"))
        if path.is_dir() and RUN_PATTERN.match(path.name)
    ]
    expected = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
    actual = {(run["seed"], run["variant"]) for run in runs}
    missing = sorted(expected.difference(actual))
    if missing or len(actual) != len(runs):
        raise ValueError(f"L0-L3 matrix incomplete: missing={missing}, duplicate={len(actual) != len(runs)}")
    return runs


def check_pairing(runs: list[dict[str, Any]], expected_episodes: int) -> str:
    manifests = {run["canonical_manifest_sha256"] for run in runs}
    if len(manifests) != 1:
        raise ValueError(f"Runs do not share one scene manifest: {sorted(manifests)}")
    reference = next(run for run in runs if run["seed"] == SEEDS[0] and run["variant"] == "m0")
    seeds = [reference["episodes"][index]["episode_seed"] for index in range(expected_episodes)]
    for run in runs:
        current = [run["episodes"][index]["episode_seed"] for index in range(expected_episodes)]
        if current != seeds:
            raise ValueError(f"Episode seed mismatch: {run['path']}")
    return next(iter(manifests))


def finite_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty/non-finite values.")
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    rows = list(run["episodes"].values())
    overall = run["summary"]["overall"]
    return {
        "stage": run["stage"],
        "training_seed": run["seed"],
        "variant": run["variant"],
        "episodes": len(rows),
        "safe_capture_count": int(overall["safe_capture_count"]),
        "safe_capture_rate": float(overall["safe_capture_rate"]),
        "collision_count": int(overall.get("collision_count", 0)),
        "boundary_count": int(overall.get("boundary_violation_count", 0)),
        "pairwise_count": int(overall.get("pairwise_violation_count", 0)),
        "raw_unverified_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        "cbf_abort_steps": int(overall.get("cbf_controlled_abort_steps", 0)),
        "transit_success_rate": float(overall.get("transit_success_rate", 0.0)),
        "mean_capture_time_seconds": overall.get("mean_capture_time_seconds"),
        "max_cbf_p95_solve_latency_ms": float(overall.get("max_cbf_p95_solve_latency_ms", 0.0)),
    }


def level_metrics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        for row in run["episodes"].values():
            grouped.setdefault((run["variant"], row["level"]), []).append(row)
    result: list[dict[str, Any]] = []
    for (variant, level), rows in sorted(grouped.items()):
        result.append(
            {
                "variant": variant,
                "level": level,
                "episodes": len(rows),
                "safe_capture_count": int(sum(row["safe_capture"] for row in rows)),
                "safe_capture_rate": float(np.mean([row["safe_capture"] for row in rows])),
                "collision_count": int(sum(row["collision"] for row in rows)),
                "boundary_count": int(sum(row["boundary"] for row in rows)),
                "pairwise_count": int(sum(row["pairwise"] for row in rows)),
                "transit_success_rate": float(np.mean([row["transit"] for row in rows])),
            }
        )
    return result


def pair_comparison(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for index in range(len(base["episodes"])):
        left = base["episodes"][index]
        right = candidate["episodes"][index]
        pairs.append(
            {
                "episode_index": index,
                "episode_seed": left["episode_seed"],
                "level": left["level"],
                "base_safe_capture": bool(left["safe_capture"]),
                "candidate_safe_capture": bool(right["safe_capture"]),
                "delta": int(right["safe_capture"]) - int(left["safe_capture"]),
            }
        )
    improved = sum(not item["base_safe_capture"] and item["candidate_safe_capture"] for item in pairs)
    degraded = sum(item["base_safe_capture"] and not item["candidate_safe_capture"] for item in pairs)
    tied = len(pairs) - improved - degraded
    p_value = float(binomtest(min(improved, degraded), improved + degraded, 0.5).pvalue) if improved + degraded else 1.0
    return {
        "training_seed": base["seed"],
        "base_variant": base["variant"],
        "candidate_variant": candidate["variant"],
        "episodes": len(pairs),
        "improved": int(improved),
        "degraded": int(degraded),
        "tied": int(tied),
        "base_safe_capture_count": int(sum(item["base_safe_capture"] for item in pairs)),
        "candidate_safe_capture_count": int(sum(item["candidate_safe_capture"] for item in pairs)),
        "delta_rate": float((improved - degraded) / len(pairs)),
        "mcnemar_exact_two_sided_p": p_value,
        "pairs": pairs,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(runs: list[dict[str, Any]], expected_episodes: int) -> dict[str, Any]:
    manifest_hash = check_pairing(runs, expected_episodes)
    metrics = [run_metrics(run) for run in runs]
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        rows = [row for row in metrics if row["variant"] == variant]
        by_variant[variant] = {
            "safe_capture_rate": finite_stats(row["safe_capture_rate"] for row in rows),
            "safe_capture_count_total": int(sum(row["safe_capture_count"] for row in rows)),
            "collision_count_total": int(sum(row["collision_count"] for row in rows)),
            "boundary_count_total": int(sum(row["boundary_count"] for row in rows)),
            "pairwise_count_total": int(sum(row["pairwise_count"] for row in rows)),
            "raw_unverified_steps_total": int(sum(row["raw_unverified_steps"] for row in rows)),
            "cbf_abort_steps_total": int(sum(row["cbf_abort_steps"] for row in rows)),
            "transit_success_rate": finite_stats(row["transit_success_rate"] for row in rows),
            "max_cbf_p95_latency_ms": float(max(row["max_cbf_p95_solve_latency_ms"] for row in rows)),
        }
    run_lookup = {(run["seed"], run["variant"]): run for run in runs}
    comparisons = [
        pair_comparison(run_lookup[(seed, "m0")], run_lookup[(seed, variant)])
        for seed in SEEDS
        for variant in ("m1", "m2", "m3", "a1", "a2")
    ]
    m3 = [item for item in comparisons if item["candidate_variant"] == "m3"]
    m3_delta = float(np.mean([item["delta_rate"] for item in m3]))
    safety_hard_gate = all(
        by_variant[variant][field] == 0
        for variant in SAFE_VARIANTS
        for field in ("collision_count_total", "boundary_count_total", "pairwise_count_total", "raw_unverified_steps_total")
    )
    reliability_gate = all(bool(run["metadata"].get("locked_test_opened") is False) for run in runs)
    return {
        "evaluation_type": "jepa_safe_capture_l0_l3_paired_development_aggregate",
        "stage": "full",
        "episodes_per_run": expected_episodes,
        "training_seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "canonical_scene_manifest_sha256": manifest_hash,
        "variant_summary": by_variant,
        "level_summary": level_metrics(runs),
        "run_metrics": metrics,
        "paired_comparisons": comparisons,
        "decision": {
            "safety_hard_gate": bool(safety_hard_gate),
            "reliability_observability_gate": bool(reliability_gate),
            "m3_mean_paired_delta_rate": m3_delta,
            "m3_seed_delta_rates": {str(item["training_seed"]): item["delta_rate"] for item in m3},
            "m3_seeds_nonnegative": int(sum(item["delta_rate"] >= 0.0 for item in m3)),
            "classification": "positive_development_evidence" if safety_hard_gate and m3_delta > 0.0 else "inconclusive_development_evidence",
            "a3_excluded_from_safety_decision": True,
        },
        "locked_test_opened": False,
    }


def write_tensorboard(report: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(output), flush_secs=5) as writer:
        writer.add_text("Config/aggregate", json.dumps({"evaluation_type": report["evaluation_type"], "stage": report["stage"]}, indent=2), 0)
        writer.add_text("Provenance/scene_manifest", report["canonical_scene_manifest_sha256"], 0)
        writer.add_text("Provenance/decision", json.dumps(report["decision"], indent=2), 0)
        writer.add_text("Provenance/level_summary", json.dumps(report["level_summary"], indent=2), 0)
        for index, variant in enumerate(VARIANTS):
            item = report["variant_summary"][variant]
            writer.add_scalar(f"SafeCapture/{variant}/rate_mean", item["safe_capture_rate"]["mean"], index)
            writer.add_scalar(f"SafeCapture/{variant}/rate_std", item["safe_capture_rate"]["sample_std"], index)
            writer.add_scalar(f"Safety/{variant}/collision_count", item["collision_count_total"], index)
            writer.add_scalar(f"Safety/{variant}/boundary_count", item["boundary_count_total"], index)
            writer.add_scalar(f"Safety/{variant}/pairwise_count", item["pairwise_count_total"], index)
            writer.add_scalar(f"Safety/{variant}/raw_unverified_steps", item["raw_unverified_steps_total"], index)
        writer.add_scalar("Decision/m3_mean_paired_delta_rate", report["decision"]["m3_mean_paired_delta_rate"], 0)
        writer.add_scalar("Decision/safety_hard_gate", float(report["decision"]["safety_hard_gate"]), 0)
        writer.flush()
    events = sorted(path.name for path in output.glob("events.out.tfevents.*"))
    if not events:
        raise RuntimeError("TensorBoard did not produce an event file.")
    return {"logdir": str(output), "event_files": events, "required_provenance": True}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# JEPA Safe-Capture L0-L3 Full Closed-Loop Aggregate",
        "",
        "Development-only paired result; `locked_test_opened=false`.",
        "",
        f"Canonical scene manifest SHA-256: `{report['canonical_scene_manifest_sha256']}`  ",
        f"Episodes per run: `{report['episodes_per_run']}`",
        "",
        "## Overall Results",
        "",
        "| Variant | Safe capture mean +/- std | Collision | Boundary | Pairwise | Raw-unverified steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = report["variant_summary"][variant]
        rate = item["safe_capture_rate"]
        lines.append(
            f"| {variant.upper()} | {rate['mean']:.3f} +/- {rate['sample_std']:.3f} | "
            f"{item['collision_count_total']} | {item['boundary_count_total']} | "
            f"{item['pairwise_count_total']} | {item['raw_unverified_steps_total']} |"
        )
    lines += ["", "## Level Results", "", "| Variant | Level | Episodes | Safe capture | Collision | Boundary | Pairwise |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in report["level_summary"]:
        lines.append(
            f"| {item['variant'].upper()} | {item['level']} | {item['episodes']} | "
            f"{item['safe_capture_count']}/{item['episodes']} ({item['safe_capture_rate']:.1%}) | "
            f"{item['collision_count']} | {item['boundary_count']} | {item['pairwise_count']} |"
        )
    lines += ["", "## M3 Versus M0", "", "| Seed | M0 | M3 | Delta | Improved | Degraded | Tied | McNemar p |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in report["paired_comparisons"]:
        if item["candidate_variant"] != "m3":
            continue
        lines.append(
            f"| {item['training_seed']} | {item['base_safe_capture_count']}/{item['episodes']} | "
            f"{item['candidate_safe_capture_count']}/{item['episodes']} | {item['delta_rate']:+.3f} | "
            f"{item['improved']} | {item['degraded']} | {item['tied']} | {item['mcnemar_exact_two_sided_p']:.4f} |"
        )
    decision = report["decision"]
    lines += [
        "",
        "## Decision",
        "",
        f"Safety hard gate: **{'PASS' if decision['safety_hard_gate'] else 'FAIL'}**.",
        f"Reliability/provenance gate: **{'PASS' if decision['reliability_observability_gate'] else 'FAIL'}**.",
        f"M3 mean paired delta: `{decision['m3_mean_paired_delta_rate']:+.3f}`; non-negative seeds: `{decision['m3_seeds_nonnegative']}/3`.",
        f"Classification: **{decision['classification']}**.",
        "",
        "A3 is a raw/no-CBF diagnostic and is excluded from the safety decision.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-scenario", type=int, default=8)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("L0-L3 aggregation requires --development-only.")
    if args.episodes_per_scenario <= 0:
        raise ValueError("episodes-per-scenario must be positive.")
    expected_episodes = args.episodes_per_scenario * 8
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    report = aggregate(discover_runs(args.input_root, stage="full", expected_episodes=expected_episodes), expected_episodes)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "run_metrics.csv", report["run_metrics"])
    write_csv(output / "level_metrics.csv", report["level_summary"])
    write_csv(output / "paired_comparison.csv", [{key: value for key, value in item.items() if key != "pairs"} for item in report["paired_comparisons"]])
    report["tensorboard"] = write_tensorboard(report, output / "tensorboard")
    (output / "paired_comparison.json").write_text(json.dumps(report["paired_comparisons"], indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
