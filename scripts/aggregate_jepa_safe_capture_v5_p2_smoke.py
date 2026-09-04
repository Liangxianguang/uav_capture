"""Aggregate the explicit V5 P2 paired smoke matrix.

This aggregator intentionally does not discover historical V2/V3 directories.
All twelve V5 runs (three seeds x M0/M3/A1/A2) must be supplied explicitly by
their naming convention, and every comparison is made at complete-episode
resolution on the same per-seed scene manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m0", "m3", "a1", "a2")
SAFE_VARIANTS = VARIANTS


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
            raise ValueError(f"Scene manifest record is not an object: {path}")
        record = dict(record)
        record.pop("training_seed", None)
        records.append(record)
    payload = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for item in records
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_run(root: Path, seed: int, variant: str, expected_episodes: int, run_prefix: str) -> dict[str, Any]:
    path = root / f"{run_prefix}_{variant}_seed{seed}"
    required = ("summary.json", "provenance.json", "scene_manifest.jsonl", "episodes.csv")
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    trace_dir = path / "step_traces"
    trace_files = sorted(trace_dir.glob("episode_*.jsonl")) if trace_dir.is_dir() else []
    if len(trace_files) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} step-trace files, found {len(trace_files)}: {path}")
    provenance_path = path / "provenance.json"
    provenance_preview = read_json(provenance_path)
    tensorboard_path = provenance_preview.get("tensorboard_dir")
    if not tensorboard_path:
        tensorboard_path = provenance_preview.get("tensorboard", {}).get("logdir")
    if not tensorboard_path or not list(Path(tensorboard_path).glob("events.out.tfevents.*")):
        raise ValueError(f"TensorBoard event file is missing: {path}")
    summary = read_json(path / "summary.json")
    provenance = provenance_preview
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Closed development boundary failed: {path}")
    declared_variant = provenance.get("variant", {}).get("variant")
    if declared_variant != variant or int(provenance.get("training_seed", -1)) != seed:
        raise ValueError(f"Seed/variant provenance mismatch: {path}")
    if int(provenance.get("episodes", -1)) != expected_episodes:
        raise ValueError(f"Unexpected episode count in provenance: {path}")
    manifest = path / "scene_manifest.jsonl"
    manifest_hash = sha256(manifest)
    declared_manifest_hash = provenance.get("inputs", {}).get("scene_manifest_sha256")
    if declared_manifest_hash != manifest_hash:
        raise ValueError(f"Manifest hash mismatch: {path}")
    inputs = provenance.get("inputs", {})
    ledger_path_value = inputs.get("reliability_ledger")
    ledger_hash_value = inputs.get("reliability_ledger_sha256")
    if variant in {"m3", "a2"}:
        if not ledger_path_value or not ledger_hash_value:
            raise ValueError(f"Ledger provenance is missing for {variant}: {path}")
        ledger_path = Path(str(ledger_path_value))
        if not ledger_path.is_file() or sha256(ledger_path) != ledger_hash_value:
            raise ValueError(f"Ledger hash mismatch: {path}")
        ledger_payload = read_json(ledger_path)
        source = ledger_payload.get("source", {})
        jepa_path = inputs.get("jepa_checkpoint")
        if not jepa_path or source.get("checkpoint_sha256") != sha256(Path(str(jepa_path))):
            raise ValueError(f"Ledger/checkpoint binding mismatch: {path}")
        if source.get("protocol_sha256") != inputs.get("protocol_sha256"):
            raise ValueError(f"Ledger/protocol binding mismatch: {path}")
    rows: list[dict[str, Any]] = []
    with (path / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if len(rows) != expected_episodes:
        raise ValueError(f"Episode table count mismatch: {path}")
    indices = [int(row["episode_index"]) for row in rows]
    if indices != list(range(expected_episodes)):
        raise ValueError(f"Episode indices are not contiguous: {path}")
    rows.sort(key=lambda row: int(row["episode_index"]))
    for row in rows:
        if int(row["training_seed"]) != seed or row["variant"] != variant:
            raise ValueError(f"Episode provenance mismatch: {path}")
    overall = summary.get("overall")
    if not isinstance(overall, dict):
        raise ValueError(f"summary.overall is missing: {path}")
    return {
        "path": str(path.resolve()),
        "seed": seed,
        "variant": variant,
        "summary": summary,
        "provenance": provenance,
        "rows": rows,
        "manifest_sha256": manifest_hash,
        "canonical_manifest_sha256": canonical_manifest_sha256(manifest),
        "summary_sha256": sha256(path / "summary.json"),
        "provenance_sha256": sha256(path / "provenance.json"),
        "step_trace_files": [str(item) for item in trace_files],
        "tensorboard_dir": str(Path(tensorboard_path).resolve()),
    }


def safe_metrics(run: dict[str, Any]) -> dict[str, Any]:
    overall = run["summary"]["overall"]
    fields = {
        "collision_count": "collision_count",
        "boundary_violation_count": "boundary_violation_count",
        "pairwise_violation_count": "pairwise_violation_count",
        "cbf_infeasible_steps": "cbf_infeasible_steps",
        "cbf_timeout_steps": "cbf_timeout_steps",
        "cbf_controlled_abort_steps": "cbf_controlled_abort_steps",
        "cbf_unverified_steps": "cbf_unverified_steps",
        "raw_unverified_executed_steps": "raw_unverified_executed_steps",
        "cbf_fallback_steps": "cbf_fallback_steps",
        "target_boundary_violation_count": "target_boundary_violation_count",
    }
    result = {
        "training_seed": run["seed"],
        "variant": run["variant"],
        "episodes": len(run["rows"]),
        "safe_capture_count": int(overall["safe_capture_count"]),
        "safe_capture_rate": float(overall["safe_capture_rate"]),
    }
    result.update({name: int(overall.get(source, 0)) for name, source in fields.items()})
    result["mean_capture_time_seconds"] = overall.get("mean_capture_time_seconds")
    result["mean_cbf_p95_solve_latency_ms"] = float(overall.get("mean_cbf_p95_solve_latency_ms", 0.0))
    result["p95_cycle_total_ms"] = float(
        overall.get("latency_breakdown", {}).get("cycle_total", {}).get("mean_episode_p95_ms", 0.0)
    )
    return result


def paired(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if [row["episode_seed"] for row in base["rows"]] != [row["episode_seed"] for row in candidate["rows"]]:
        raise ValueError(f"Episode pairing mismatch: {candidate['path']}")
    base_safe = [as_bool(row["cooperative_safe_capture"]) for row in base["rows"]]
    candidate_safe = [as_bool(row["cooperative_safe_capture"]) for row in candidate["rows"]]
    improved = sum((not old) and new for old, new in zip(base_safe, candidate_safe))
    degraded = sum(old and (not new) for old, new in zip(base_safe, candidate_safe))
    tied = len(base_safe) - improved - degraded
    discordant = improved + degraded
    p_value = float(binomtest(improved, discordant, 0.5).pvalue) if discordant else 1.0
    delta = float(np.mean(candidate_safe) - np.mean(base_safe))
    return {
        "training_seed": base["seed"],
        "base_variant": base["variant"],
        "candidate_variant": candidate["variant"],
        "episodes": len(base_safe),
        "base_safe_capture_count": int(sum(base_safe)),
        "candidate_safe_capture_count": int(sum(candidate_safe)),
        "delta_rate": delta,
        "improved": int(improved),
        "degraded": int(degraded),
        "tied": int(tied),
        "mcnemar_exact_two_sided_p": p_value,
    }


def bootstrap(values: np.ndarray, samples: int = 4000, seed: int = 20260904) -> dict[str, Any]:
    if values.size == 0:
        raise ValueError("Cannot bootstrap empty values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    means = values[draws].mean(axis=1)
    return {
        "observed": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "unit": "episode_pair",
        "samples": samples,
        "seed": seed,
    }


def aggregate(runs: list[dict[str, Any]], expected_episodes: int) -> dict[str, Any]:
    if len(runs) != len(SEEDS) * len(VARIANTS):
        raise ValueError("Expected exactly 12 explicit V5 smoke runs")
    keys = {(run["seed"], run["variant"]) for run in runs}
    expected = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
    if keys != expected:
        raise ValueError(f"Run matrix mismatch: missing={sorted(expected - keys)}")
    protocols = {run["provenance"]["inputs"].get("protocol_sha256") for run in runs}
    environments = {run["provenance"]["inputs"].get("environment_config_sha256") for run in runs}
    canonical = {run["canonical_manifest_sha256"] for run in runs}
    if len(protocols) != 1 or None in protocols:
        raise ValueError(f"Protocol hashes are inconsistent: {protocols}")
    if len(environments) != 1 or None in environments:
        raise ValueError(f"Environment hashes are inconsistent: {environments}")
    if len(canonical) != 1:
        raise ValueError(f"Canonical scene manifests are inconsistent: {canonical}")
    metrics = [safe_metrics(run) for run in sorted(runs, key=lambda item: (item["variant"], item["seed"]))]
    safety_gate = all(
        metric["collision_count"] == 0
        and metric["boundary_violation_count"] == 0
        and metric["pairwise_violation_count"] == 0
        and metric["raw_unverified_executed_steps"] == 0
        for metric in metrics
    )
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        values = np.asarray([item["safe_capture_rate"] for item in metrics if item["variant"] == variant], dtype=float)
        by_variant[variant] = {
            "seed_rates": {str(item["training_seed"]): item["safe_capture_rate"] for item in metrics if item["variant"] == variant},
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "total_safe_capture": int(sum(item["safe_capture_count"] for item in metrics if item["variant"] == variant)),
            "total_episodes": int(values.size * expected_episodes),
            "total_collision": int(sum(item["collision_count"] for item in metrics if item["variant"] == variant)),
            "total_boundary": int(sum(item["boundary_violation_count"] for item in metrics if item["variant"] == variant)),
            "total_pairwise": int(sum(item["pairwise_violation_count"] for item in metrics if item["variant"] == variant)),
            "total_cbf_abort": int(sum(item["cbf_controlled_abort_steps"] for item in metrics if item["variant"] == variant)),
            "total_raw_unverified": int(sum(item["raw_unverified_executed_steps"] for item in metrics if item["variant"] == variant)),
        }
    comparisons = []
    for seed in SEEDS:
        base = next(run for run in runs if run["seed"] == seed and run["variant"] == "m0")
        for variant in ("m3", "a1", "a2"):
            comparisons.append(paired(base, next(run for run in runs if run["seed"] == seed and run["variant"] == variant)))
    m3 = [item for item in comparisons if item["candidate_variant"] == "m3"]
    m3_delta = np.asarray([item["delta_rate"] for item in m3], dtype=float)
    all_m3_pairs: list[float] = []
    for seed in SEEDS:
        base_run = next(run for run in runs if run["seed"] == seed and run["variant"] == "m0")
        candidate_run = next(run for run in runs if run["seed"] == seed and run["variant"] == "m3")
        for base_row, candidate_row in zip(base_run["rows"], candidate_run["rows"]):
            all_m3_pairs.append(
                float(as_bool(candidate_row["cooperative_safe_capture"]))
                - float(as_bool(base_row["cooperative_safe_capture"]))
            )
    all_m3_pairs_array = np.asarray(all_m3_pairs, dtype=float)
    reliability_observability = all(
        isinstance(run["provenance"].get("variant"), dict)
        and run["provenance"]["variant"].get("use_ledger") is (run["variant"] in {"m3", "a2"})
        for run in runs
    )
    decision = {
        "safety_hard_gate": bool(safety_gate),
        "reliability_observability_gate": bool(reliability_observability),
        "m3_mean_paired_delta_rate": float(m3_delta.mean()),
        "m3_seed_delta_rates": {str(item["training_seed"]): item["delta_rate"] for item in m3},
        "m3_seeds_nonnegative": int(sum(item["delta_rate"] >= 0.0 for item in m3)),
        "m3_cross_seed_improved": int(sum(item["improved"] for item in m3)),
        "m3_cross_seed_degraded": int(sum(item["degraded"] for item in m3)),
        "m3_cross_seed_tied": int(sum(item["tied"] for item in m3)),
        "m3_bootstrap": bootstrap(m3_delta),
        "m3_episode_pair_mean_delta": float(all_m3_pairs_array.mean()),
        "classification": "safety_preserving_non_inferiority" if safety_gate and reliability_observability and m3_delta.mean() >= 0.0 else "useful_safety_fallback_only",
        "locked_test_opened": False,
    }
    return {
        "aggregation_type": "jepa_safe_capture_v5_p2_explicit_paired_smoke",
        "stage": "smoke",
        "development_only": True,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "episodes_per_run": expected_episodes,
        "training_seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "protocol_sha256": next(iter(protocols)),
        "environment_config_sha256": next(iter(environments)),
        "canonical_scene_manifest_sha256": next(iter(canonical)),
        "run_metrics": metrics,
        "by_variant": by_variant,
        "paired_comparisons": comparisons,
        "decision": decision,
        "safety_gate": safety_gate,
        "inputs": [
            {
                "path": run["path"],
                "seed": run["seed"],
                "variant": run["variant"],
                "summary_sha256": run["summary_sha256"],
                "provenance_sha256": run["provenance_sha256"],
                "manifest_sha256": run["manifest_sha256"],
                "canonical_manifest_sha256": run["canonical_manifest_sha256"],
            }
            for run in sorted(runs, key=lambda item: (item["seed"], item["variant"]))
        ],
    }


def write_tensorboard(report: dict[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/aggregate", json.dumps({"aggregation_type": report["aggregation_type"], "stage": report["stage"]}, indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(report["inputs"], indent=2), 0)
        writer.add_text("Provenance/decision", json.dumps(report["decision"], indent=2), 0)
        writer.add_text("Provenance/scene_manifest", report["canonical_scene_manifest_sha256"], 0)
        for index, variant in enumerate(VARIANTS):
            item = report["by_variant"][variant]
            writer.add_scalar(f"SafeCapture/{variant}/mean_rate", item["mean"], index)
            writer.add_scalar(f"Safety/{variant}/collision", item["total_collision"], index)
            writer.add_scalar(f"Safety/{variant}/boundary", item["total_boundary"], index)
            writer.add_scalar(f"Safety/{variant}/pairwise", item["total_pairwise"], index)
            writer.add_scalar(f"CBF/{variant}/controlled_abort_steps", item["total_cbf_abort"], index)
        writer.add_scalar("Paired/m3/mean_delta", report["decision"]["m3_mean_paired_delta_rate"], 0)
        writer.add_scalar("Paired/m3/improved", report["decision"]["m3_cross_seed_improved"], 0)
        writer.add_scalar("Paired/m3/degraded", report["decision"]["m3_cross_seed_degraded"], 0)
    return {
        "logdir": str(logdir),
        "event_files": sorted(item.name for item in logdir.glob("events.out.tfevents.*")),
        "required_provenance": True,
    }


def render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# V5 P2 Explicit Paired Smoke Aggregate",
        "",
        "> Development-only; locked_test_opened=false. M0/M3/A1/A2 use explicit paired scene manifests.",
        "",
        f"Canonical scene manifest SHA-256: `{report['canonical_scene_manifest_sha256']}`",
        f"Protocol SHA-256: `{report['protocol_sha256']}`",
        "",
        "| Variant | Mean safe-capture | Sample SD | Total safe captures | Collision | Boundary | Pairwise | CBF abort steps | Raw unverified |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = report["by_variant"][variant]
        lines.append(f"| {variant.upper()} | {item['mean']:.3f} | {item['sample_std']:.3f} | {item['total_safe_capture']}/{item['total_episodes']} | {item['total_collision']} | {item['total_boundary']} | {item['total_pairwise']} | {item['total_cbf_abort']} | {item['total_raw_unverified']} |")
    lines += [
        "",
        "## M3 Paired Delta",
        "",
        f"Mean per-seed delta: `{decision['m3_mean_paired_delta_rate']:.3f}`; non-negative seeds: `{decision['m3_seeds_nonnegative']}/3`.",
        f"Improved/degraded/tied episode pairs: `{decision['m3_cross_seed_improved']}/{decision['m3_cross_seed_degraded']}/{decision['m3_cross_seed_tied']}`.",
        f"Bootstrap 95% CI: `[{decision['m3_bootstrap']['ci95_low']:.3f}, {decision['m3_bootstrap']['ci95_high']:.3f}]`.",
        "",
        "| Seed | M0 | M3 | Delta | Improved | Degraded | McNemar p |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["paired_comparisons"]:
        if item["candidate_variant"] == "m3":
            lines.append(f"| {item['training_seed']} | {item['base_safe_capture_count']}/{item['episodes']} | {item['candidate_safe_capture_count']}/{item['episodes']} | {item['delta_rate']:.3f} | {item['improved']} | {item['degraded']} | {item['mcnemar_exact_two_sided_p']:.4f} |")
    lines += [
        "",
        "## Gates",
        "",
        f"- Safety hard gate: `{report['safety_gate']}`.",
        f"- Reliability observability gate: `{decision['reliability_observability_gate']}`.",
        f"- Classification: `{decision['classification']}`.",
        "- This smoke result does not open the locked test and does not use mean capture time as a safety substitute.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--run-prefix", default="jepa_safe_capture_v5_p2_smoke")
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("V5 smoke aggregation requires --development-only")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    runs = [
        load_run(args.input_root.resolve(), seed, variant, args.episodes, args.run_prefix)
        for seed in SEEDS
        for variant in VARIANTS
    ]
    report = aggregate(runs, args.episodes)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["run_metrics"][0]))
        writer.writeheader()
        writer.writerows(report["run_metrics"])
    (output_dir / "paired_comparisons.json").write_text(json.dumps(report["paired_comparisons"], indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    report["tensorboard"] = write_tensorboard(report, args.tensorboard_dir)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
