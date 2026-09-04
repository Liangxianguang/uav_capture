"""Aggregate the v11 hard-replay four-variant development smoke matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from torch.utils.tensorboard import SummaryWriter

SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m0", "m3", "a1", "a2")
RUN_RE = re.compile(r"^jepa_safe_capture_v11_hard_replay_smoke_(m0|m3|a1|a2)_seed(20260911|20260912|20260913)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    payload = "".join(json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n" for item in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error


def load_run(root: Path, seed: int, variant: str) -> dict[str, Any]:
    path = (root / f"jepa_safe_capture_v11_hard_replay_smoke_{variant}_seed{seed}").resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    required = [path / name for name in ("summary.json", "provenance.json", "episodes.csv", "scene_manifest.jsonl")]
    if any(not item.is_file() for item in required):
        raise FileNotFoundError(f"Incomplete run: {path}")
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    metadata = summary.get("metadata", {})
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Locked-test boundary violation: {path}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Invalid provenance boundary: {path}")
    if as_int(metadata.get("episodes", -1), "episodes") != 20:
        raise ValueError(f"Expected 20 episodes: {path}")
    declared = metadata.get("variant", {})
    if declared.get("variant") != variant or as_int(metadata.get("training_seed", -1), "training_seed") != seed:
        raise ValueError(f"Run identity mismatch: {path}")
    manifest = path / "scene_manifest.jsonl"
    if metadata.get("inputs", {}).get("scene_manifest_sha256") != sha256(manifest):
        raise ValueError(f"Manifest hash mismatch: {path}")
    with (path / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise ValueError(f"Expected 20 episode rows: {path}")
    rows_by_index = {as_int(row.get("episode_index"), "episode_index"): row for row in rows}
    if sorted(rows_by_index) != list(range(20)):
        raise ValueError(f"Episode indices are not contiguous: {path}")
    return {
        "path": str(path),
        "seed": seed,
        "variant": variant,
        "summary": summary,
        "rows": rows_by_index,
        "manifest_sha256": sha256(manifest),
        "canonical_manifest_sha256": canonical_manifest_sha256(manifest),
        "summary_sha256": sha256(path / "summary.json"),
        "provenance_sha256": sha256(path / "provenance.json"),
    }


def episode_safe(row: dict[str, Any]) -> bool:
    return as_bool(row.get("safe_capture_success"))


def paired(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = base["rows"]
    right = candidate["rows"]
    pairs = []
    for index in range(20):
        if left[index].get("episode_seed") != right[index].get("episode_seed"):
            raise ValueError(f"Episode pairing mismatch at {index}: {base['path']} vs {candidate['path']}")
        b = episode_safe(left[index])
        c = episode_safe(right[index])
        pairs.append({"episode_index": index, "base_safe_capture": b, "candidate_safe_capture": c, "delta": int(c) - int(b)})
    improved = sum(not item["base_safe_capture"] and item["candidate_safe_capture"] for item in pairs)
    degraded = sum(item["base_safe_capture"] and not item["candidate_safe_capture"] for item in pairs)
    discordant = improved + degraded
    p_value = float(binomtest(min(improved, degraded), n=discordant, p=0.5).pvalue) if discordant else 1.0
    values = np.asarray([item["delta"] for item in pairs], dtype=np.float64)
    rng = np.random.default_rng(20260905)
    bootstrap = np.mean(values[rng.integers(0, values.size, size=(4000, values.size))], axis=1)
    return {
        "training_seed": base["seed"],
        "base_variant": base["variant"],
        "candidate_variant": candidate["variant"],
        "episodes": 20,
        "base_safe_capture_count": int(sum(item["base_safe_capture"] for item in pairs)),
        "candidate_safe_capture_count": int(sum(item["candidate_safe_capture"] for item in pairs)),
        "improved": int(improved),
        "degraded": int(degraded),
        "tied": int(20 - improved - degraded),
        "delta_rate": float(np.mean(values)),
        "mcnemar_exact_two_sided_p": p_value,
        "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "pairs": pairs,
    }


def metric(run: dict[str, Any]) -> dict[str, Any]:
    overall = run["summary"]["overall"]
    return {
        "training_seed": run["seed"],
        "variant": run["variant"],
        "safe_capture_count": int(overall["safe_capture_count"]),
        "safe_capture_rate": float(overall["safe_capture_rate"]),
        "collision_count": int(overall["collision_count"]),
        "boundary_violation_count": int(overall["boundary_violation_count"]),
        "pairwise_violation_count": int(overall["pairwise_violation_count"]),
        "cbf_infeasible_steps": int(overall.get("cbf_infeasible_steps", 0)),
        "cbf_timeout_steps": int(overall.get("cbf_timeout_steps", 0)),
        "cbf_unverified_steps": int(overall.get("cbf_unverified_steps", 0)),
        "cbf_fallback_steps": int(overall.get("cbf_fallback_steps", 0)),
        "cbf_controlled_abort_steps": int(overall.get("cbf_controlled_abort_steps", 0)),
        "raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        "transit_success_rate": float(overall.get("transit_success_rate", 0.0)),
    }


def aggregate(root: Path) -> dict[str, Any]:
    runs = [load_run(root, seed, variant) for seed in SEEDS for variant in VARIANTS]
    canonical = {run["canonical_manifest_sha256"] for run in runs}
    if len(canonical) != 1:
        raise ValueError(f"Scene manifests are not paired: {sorted(canonical)}")
    protocol_hashes = {run["summary"]["metadata"]["inputs"].get("protocol_sha256") for run in runs}
    if len(protocol_hashes) != 1:
        raise ValueError(f"Protocol hashes differ: {protocol_hashes}")
    by_key = {(run["seed"], run["variant"]): run for run in runs}
    comparisons = [paired(by_key[(seed, "m0")], by_key[(seed, variant)]) for seed in SEEDS for variant in ("m3", "a1", "a2")]
    metrics = [metric(run) for run in sorted(runs, key=lambda item: (item["variant"], item["seed"]))]
    by_variant = {}
    for variant in VARIANTS:
        subset = [item for item in metrics if item["variant"] == variant]
        rates = np.asarray([item["safe_capture_rate"] for item in subset], dtype=np.float64)
        by_variant[variant] = {
            "training_seeds": [item["training_seed"] for item in subset],
            "safe_capture_rate_mean": float(np.mean(rates)),
            "safe_capture_rate_sample_std": float(np.std(rates, ddof=1)),
            "safe_capture_count_total": int(sum(item["safe_capture_count"] for item in subset)),
            "collision_count_total": int(sum(item["collision_count"] for item in subset)),
            "boundary_violation_count_total": int(sum(item["boundary_violation_count"] for item in subset)),
            "pairwise_violation_count_total": int(sum(item["pairwise_violation_count"] for item in subset)),
            "raw_unverified_executed_steps_total": int(sum(item["raw_unverified_executed_steps"] for item in subset)),
            "cbf_controlled_abort_steps_total": int(sum(item["cbf_controlled_abort_steps"] for item in subset)),
            "cbf_infeasible_steps_total": int(sum(item["cbf_infeasible_steps"] for item in subset)),
        }
    m3 = [item for item in comparisons if item["candidate_variant"] == "m3"]
    m3_rates = np.asarray([item["delta_rate"] for item in m3], dtype=np.float64)
    safety_gate = all(item["collision_count"] == 0 and item["boundary_violation_count"] == 0 and item["pairwise_violation_count"] == 0 and item["raw_unverified_executed_steps"] == 0 for item in metrics)
    reliability_gate = all(item["cbf_timeout_steps"] == 0 and item["cbf_fallback_steps"] >= item["cbf_infeasible_steps"] and item["cbf_controlled_abort_steps"] == item["cbf_unverified_steps"] for item in metrics)
    m3_nonnegative = int(np.count_nonzero(m3_rates >= 0.0))
    if not safety_gate or not reliability_gate:
        classification = "insufficient_evidence_or_reject"
    elif float(np.mean(m3_rates)) > 0.0 and m3_nonnegative >= 2:
        classification = "positive_development_evidence"
    elif float(np.mean(m3_rates)) >= 0.0 and m3_nonnegative >= 2:
        classification = "safety_preserving_non_inferiority"
    else:
        classification = "prediction_signal_no_control_gain"
    return {
        "aggregation_type": "jepa_safe_capture_v11_hard_replay_smoke",
        "development_only": True,
        "locked_test_opened": False,
        "episodes_per_run": 20,
        "run_count": len(runs),
        "training_seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "scene_manifest_sha256": next(iter(canonical)),
        "protocol_sha256": next(iter(protocol_hashes)),
        "run_metrics": metrics,
        "by_variant": by_variant,
        "paired_comparisons": comparisons,
        "decision": {
            "safety_hard_gate": safety_gate,
            "reliability_observability_gate": reliability_gate,
            "m3_mean_paired_delta_rate": float(np.mean(m3_rates)),
            "m3_seed_delta_rates": {str(seed): float(item["delta_rate"]) for seed, item in zip(SEEDS, m3)},
            "m3_seeds_nonnegative": m3_nonnegative,
            "m3_cross_seed_improved": int(sum(item["improved"] for item in m3)),
            "m3_cross_seed_degraded": int(sum(item["degraded"] for item in m3)),
            "m3_cross_seed_tied": int(sum(item["tied"] for item in m3)),
            "classification": classification,
        },
        "inputs": [{"path": run["path"], "summary_sha256": run["summary_sha256"], "provenance_sha256": run["provenance_sha256"], "manifest_sha256": run["manifest_sha256"], "training_seed": run["seed"], "variant": run["variant"]} for run in runs],
    }


def write_report(report: dict[str, Any], output: Path, tensorboard: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output / "run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["run_metrics"][0]))
        writer.writeheader()
        writer.writerows(report["run_metrics"])
    comparisons = report["paired_comparisons"]
    with (output / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["training_seed", "candidate_variant", "base_safe_capture_count", "candidate_safe_capture_count", "improved", "degraded", "tied", "delta_rate", "mcnemar_exact_two_sided_p"])
        writer.writeheader()
        for item in comparisons:
            writer.writerow({key: item[key] for key in writer.fieldnames})
    decision = report["decision"]
    lines = ["# V11 Hard-Replay Smoke Aggregate", "", "Development-only; `locked_test_opened=false`.", "", f"Canonical scene manifest SHA-256: `{report['scene_manifest_sha256']}`", "", "| Variant | Safe capture mean +/- seed std | Collision | Boundary | Pairwise | Raw unverified | Controlled abort |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        item = report["by_variant"][variant]
        lines.append(f"| {variant.upper()} | {item['safe_capture_rate_mean']:.3f} +/- {item['safe_capture_rate_sample_std']:.3f} | {item['collision_count_total']} | {item['boundary_violation_count_total']} | {item['pairwise_violation_count_total']} | {item['raw_unverified_executed_steps_total']} | {item['cbf_controlled_abort_steps_total']} |")
    lines += ["", "## M3 paired result", "", f"Per-seed delta: `{decision['m3_seed_delta_rates']}`; improved/degraded/tied = `{decision['m3_cross_seed_improved']}/{decision['m3_cross_seed_degraded']}/{decision['m3_cross_seed_tied']}`.", f"Safety hard gate: `{decision['safety_hard_gate']}`; reliability gate: `{decision['reliability_observability_gate']}`.", f"Classification: `{decision['classification']}`.", "", "No result here authorizes a locked test or treats mean capture time as a safety endpoint."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tensorboard.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("Config/aggregate", json.dumps({"development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/decision", json.dumps(decision, indent=2), 0)
        for index, variant in enumerate(VARIANTS):
            writer.add_scalar(f"SafeCapture/{variant}/rate_mean", report["by_variant"][variant]["safe_capture_rate_mean"], index)
        writer.add_scalar("Paired/m3/delta_rate_mean", decision["m3_mean_paired_delta_rate"], 0)
    report["tensorboard"] = {"logdir": str(tensorboard.resolve()), "event_files": sorted(item.name for item in tensorboard.glob("events.out.tfevents.*"))}
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("This aggregate is development-only.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.tensorboard_dir.exists() and any(args.tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.tensorboard_dir}")
    report = aggregate(args.input_root.resolve())
    report["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    write_report(report, args.output_dir.resolve(), args.tensorboard_dir.resolve())
    print(json.dumps({"classification": report["decision"]["classification"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
