"""Aggregate the v20 M0/M3 paired development replays.

The v20 replay directories use a deterministic backend contract but do not use
the legacy P6 directory naming. This aggregator validates pairing and
provenance before calculating episode-level safe-capture deltas. It never
treats control cycles or candidates as independent samples.
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


SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m0", "m3")
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_SAMPLES = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def bootstrap(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    samples = np.mean(values[rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))], axis=1)
    return {
        "observed": float(np.mean(values)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "unit": "episode_pair",
    }


def load_run(root: Path, seed: int, variant: str) -> dict[str, Any]:
    path = root / f"results/jepa_safe_capture_v20_cpu_deterministic_replay_{variant}_cuda_seed{seed}"
    if not path.is_dir():
        raise FileNotFoundError(path)
    summary_path = path / "summary.json"
    manifest_path = path / "scene_manifest.jsonl"
    episodes_path = path / "episodes.csv"
    provenance_path = path / "provenance.json"
    for required in (summary_path, manifest_path, episodes_path, provenance_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    summary = read_json(summary_path)
    provenance = read_json(provenance_path)
    metadata = summary.get("metadata", {})
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Development boundary mismatch: {summary_path}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Provenance boundary mismatch: {provenance_path}")
    if int(metadata.get("training_seed", -1)) != seed:
        raise ValueError(f"Training seed mismatch: {summary_path}")
    declared = metadata.get("variant", {})
    if declared.get("variant") != variant:
        raise ValueError(f"Variant mismatch: {summary_path}")
    rows: dict[int, dict[str, Any]] = {}
    with episodes_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["episode_index"])
            if index in rows:
                raise ValueError(f"Duplicate episode index {index}: {episodes_path}")
            rows[index] = {
                "episode_index": index,
                "episode_seed": int(row["episode_seed"]),
                "safe_capture": as_bool(row.get("safe_capture_success")),
                "collision": as_bool(row.get("collision")),
                "boundary_violation": as_bool(row.get("boundary_violation")),
                "pairwise_violation": as_bool(row.get("pairwise_violation")),
                "raw_unverified": int(row.get("raw_unverified_executed_steps", 0)),
                "cbf_infeasible": int(row.get("cbf_infeasible_steps", 0)),
                "cbf_timeout": int(row.get("cbf_timeout_steps", 0)),
                "cbf_unverified": int(row.get("cbf_unverified_steps", 0)),
                "cbf_fallback": int(row.get("cbf_fallback_steps", 0)),
                "cbf_abort": int(row.get("cbf_controlled_abort_steps", 0)),
                "termination_reason": str(row.get("termination_reason", "")),
            }
    expected = list(range(len(rows)))
    if sorted(rows) != expected:
        raise ValueError(f"Episode indices are not contiguous: {episodes_path}")
    inputs = metadata.get("inputs", {})
    return {
        "seed": seed,
        "variant": variant,
        "path": str(path.resolve()),
        "summary_sha256": sha256(summary_path),
        "provenance_sha256": sha256(provenance_path),
        "manifest_sha256": sha256(manifest_path),
        "protocol_sha256": str(inputs.get("protocol_sha256", "")),
        "actor_checkpoint_sha256": str(inputs.get("actor_checkpoint_sha256", "")),
        "jepa_checkpoint_sha256": inputs.get("jepa_checkpoint_sha256"),
        "ledger_sha256": inputs.get("reliability_ledger_sha256"),
        "episodes": rows,
        "overall": summary["overall"],
    }


def paired(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if base["seed"] != candidate["seed"]:
        raise ValueError("Pairing requires equal training seeds")
    pairs: list[dict[str, Any]] = []
    for index in base["episodes"]:
        left = base["episodes"][index]
        right = candidate["episodes"][index]
        if left["episode_seed"] != right["episode_seed"]:
            raise ValueError(f"Episode seed mismatch at {index}, seed {base['seed']}")
        pairs.append({
            "episode_index": index,
            "episode_seed": left["episode_seed"],
            "m0_safe_capture": left["safe_capture"],
            "m3_safe_capture": right["safe_capture"],
            "delta": int(right["safe_capture"]) - int(left["safe_capture"]),
            "m0_termination": left["termination_reason"],
            "m3_termination": right["termination_reason"],
        })
    improved = sum(not item["m0_safe_capture"] and item["m3_safe_capture"] for item in pairs)
    degraded = sum(item["m0_safe_capture"] and not item["m3_safe_capture"] for item in pairs)
    deltas = np.asarray([item["delta"] for item in pairs], dtype=np.float64)
    discordant = improved + degraded
    return {
        "training_seed": base["seed"],
        "episodes": len(pairs),
        "m0_safe_capture_count": sum(item["m0_safe_capture"] for item in pairs),
        "m3_safe_capture_count": sum(item["m3_safe_capture"] for item in pairs),
        "improved": int(improved),
        "degraded": int(degraded),
        "tied": int(len(pairs) - improved - degraded),
        "delta_rate": float(np.mean(deltas)),
        "mcnemar_exact_two_sided_p": float(
            binomtest(min(improved, degraded), n=discordant, p=0.5).pvalue if discordant else 1.0
        ),
        "bootstrap": bootstrap(deltas),
        "pairs": pairs,
    }


def aggregate(project_root: Path, output_dir: Path, tensorboard_dir: Path) -> dict[str, Any]:
    root = project_root.resolve()
    runs = {(seed, variant): load_run(root, seed, variant) for seed in SEEDS for variant in VARIANTS}
    for seed in SEEDS:
        m0, m3 = runs[(seed, "m0")], runs[(seed, "m3")]
        if m0["manifest_sha256"] != m3["manifest_sha256"]:
            raise ValueError(f"Scene manifest mismatch for seed {seed}")
        if m0["protocol_sha256"] != m3["protocol_sha256"]:
            raise ValueError(f"Protocol mismatch for seed {seed}")
        if m0["actor_checkpoint_sha256"] != m3["actor_checkpoint_sha256"]:
            raise ValueError(f"Actor checkpoint mismatch for seed {seed}")
    comparisons = [paired(runs[(seed, "m0")], runs[(seed, "m3")]) for seed in SEEDS]
    pooled = np.asarray([item["delta"] for row in comparisons for item in row["pairs"]], dtype=np.float64)
    run_metrics = []
    for (seed, variant), run in runs.items():
        overall = run["overall"]
        run_metrics.append({
            "training_seed": seed,
            "variant": variant,
            "episodes": len(run["episodes"]),
            "safe_capture_count": int(overall["safe_capture_count"]),
            "safe_capture_rate": float(overall["safe_capture_rate"]),
            "collision_count": int(overall["collision_count"]),
            "boundary_violation_count": int(overall["boundary_violation_count"]),
            "pairwise_violation_count": int(overall["pairwise_violation_count"]),
            "raw_unverified_steps": int(overall.get("raw_unverified_executed_steps", 0)),
            "cbf_infeasible_steps": int(overall.get("cbf_infeasible_steps", 0)),
            "cbf_timeout_steps": int(overall.get("cbf_timeout_steps", 0)),
            "cbf_unverified_steps": int(overall.get("cbf_unverified_steps", 0)),
            "cbf_fallback_steps": int(overall.get("cbf_fallback_steps", 0)),
            "cbf_controlled_abort_steps": int(overall.get("cbf_controlled_abort_steps", 0)),
            "transit_success_rate": float(overall.get("transit_success_rate", 0.0)),
            "mean_capture_time_seconds": overall.get("mean_capture_time_seconds"),
            "mean_cbf_p95_solve_latency_ms": float(overall.get("mean_cbf_p95_solve_latency_ms", 0.0)),
        })
    safety_gate = all(
        row["collision_count"] == 0
        and row["boundary_violation_count"] == 0
        and row["pairwise_violation_count"] == 0
        and row["raw_unverified_steps"] == 0
        for row in run_metrics
    )
    reliability_gate = all(
        row["cbf_timeout_steps"] == 0
        and row["cbf_fallback_steps"] >= row["cbf_infeasible_steps"]
        and row["cbf_controlled_abort_steps"] == row["cbf_unverified_steps"]
        for row in run_metrics
    )
    mean_delta = float(np.mean([row["delta_rate"] for row in comparisons]))
    nonnegative = sum(row["delta_rate"] >= 0.0 for row in comparisons)
    result: dict[str, Any] = {
        "stage": "WP2_v20_m0_m3_paired_development",
        "development_only": True,
        "locked_test_opened": False,
        "protocol_sha256": runs[(SEEDS[0], "m0")]["protocol_sha256"],
        "training_seeds": list(SEEDS),
        "episodes_per_seed": len(comparisons[0]["pairs"]),
        "run_metrics": run_metrics,
        "paired_by_seed": comparisons,
        "pooled": {
            "episodes": int(pooled.size),
            "improved": int(sum(row["improved"] for row in comparisons)),
            "degraded": int(sum(row["degraded"] for row in comparisons)),
            "tied": int(sum(row["tied"] for row in comparisons)),
            "delta_rate": float(np.mean(pooled)),
            "bootstrap": bootstrap(pooled),
        },
        "decision": {
            "safety_hard_gate": safety_gate,
            "reliability_observability_gate": reliability_gate,
            "m3_mean_paired_delta_rate": mean_delta,
            "m3_seed_delta_rates": {str(seed): row["delta_rate"] for seed, row in zip(SEEDS, comparisons)},
            "m3_seeds_nonnegative": int(nonnegative),
            "classification": (
                "positive_development_evidence"
                if safety_gate and reliability_gate and mean_delta > 0 and nonnegative >= 2
                else "safety_preserving_non_inferiority"
                if safety_gate and reliability_gate and mean_delta >= 0 and nonnegative >= 2
                else "useful_safety_fallback_only"
            ),
        },
        "inputs": [
            {key: value for key, value in run.items() if key not in {"episodes", "overall"}}
            for run in runs.values()
        ],
    }
    output = output_dir.resolve()
    tensorboard = tensorboard_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if tensorboard.exists() and any(tensorboard.iterdir()):
        raise FileExistsError(tensorboard)
    output.mkdir(parents=True, exist_ok=True)
    tensorboard.mkdir(parents=True, exist_ok=True)
    (output / "paired_aggregate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows = [item for comparison in comparisons for item in comparison["pairs"]]
    with (output / "paired_episode_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# v20 M0/M3 三 seed paired development aggregate",
        "",
        "`development_only=true`; `locked_test_opened=false`。统计单位为 `(training_seed, episode)`。",
        "",
        "| Seed | M0 safe capture | M3 safe capture | Improved | Degraded | Tied | Delta | McNemar p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['training_seed']} | {row['m0_safe_capture_count']}/{row['episodes']} | "
            f"{row['m3_safe_capture_count']}/{row['episodes']} | {row['improved']} | {row['degraded']} | "
            f"{row['tied']} | {row['delta_rate']:+.3f} | {row['mcnemar_exact_two_sided_p']:.4f} |"
        )
    lines += [
        "",
        f"Pooled delta: `{result['pooled']['delta_rate']:+.4f}`; bootstrap 95% CI "
        f"`[{result['pooled']['bootstrap']['ci95_low']:+.4f}, {result['pooled']['bootstrap']['ci95_high']:+.4f}]`.",
        f"Decision: `{result['decision']['classification']}`; safety gate `{safety_gate}`; reliability gate `{reliability_gate}`.",
        "CBF failures and controlled aborts remain in the episode denominator; mean capture time is diagnostic only.",
        "This is development evidence and does not authorize a locked test.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("Config/aggregate", json.dumps({"stage": result["stage"], "protocol_sha256": result["protocol_sha256"]}, indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(result["inputs"], indent=2), 0)
        writer.add_text("Decision/summary", json.dumps(result["decision"], indent=2), 0)
        for row in comparisons:
            step = SEEDS.index(row["training_seed"])
            writer.add_scalar(f"SafeCapture/seed_{row['training_seed']}/m0_rate", row["m0_safe_capture_count"] / row["episodes"], step)
            writer.add_scalar(f"SafeCapture/seed_{row['training_seed']}/m3_rate", row["m3_safe_capture_count"] / row["episodes"], step)
            writer.add_scalar(f"Paired/seed_{row['training_seed']}/delta_rate", row["delta_rate"], step)
            writer.add_scalar(f"Paired/seed_{row['training_seed']}/improved", row["improved"], step)
            writer.add_scalar(f"Paired/seed_{row['training_seed']}/degraded", row["degraded"], step)
        writer.add_scalar("Paired/pooled_delta_rate", result["pooled"]["delta_rate"], 0)
        writer.add_scalar("Gates/safety_hard_gate", float(safety_gate), 0)
        writer.add_scalar("Gates/reliability_observability_gate", float(reliability_gate), 0)
    result["tensorboard"] = {"logdir": str(tensorboard), "event_files": sorted(path.name for path in tensorboard.glob("events.out.tfevents.*"))}
    (output / "paired_aggregate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.project_root, args.output_dir, args.tensorboard_dir), indent=2))


if __name__ == "__main__":
    main()
