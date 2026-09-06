"""Aggregate one actor-matched M0/M3 paired development block.

The evaluator writes each run into its own directory because generated traces
and TensorBoard event files are intentionally ignored by Git.  This utility
keeps the comparison auditable: it requires identical manifests, protocol,
environment, and actor hashes, then reports episode-level outcomes without
treating control cycles or candidates as independent samples.
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


def _int(row: dict[str, str], key: str) -> int:
    return int(row.get(key, "0") or 0)


def _load_run(path: Path, expected_variant: str) -> dict[str, Any]:
    path = path.resolve()
    required = [path / name for name in ("summary.json", "provenance.json", "episodes.csv", "scene_manifest.jsonl")]
    missing = [item for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")
    summary = read_json(path / "summary.json")
    provenance = read_json(path / "provenance.json")
    metadata = summary.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"summary metadata is missing: {path}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run is outside development boundary: {path}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Provenance is outside development boundary: {path}")
    variant = metadata.get("variant", {})
    if not isinstance(variant, dict) or variant.get("variant") != expected_variant:
        raise ValueError(f"Expected variant {expected_variant!r}: {path}")
    rows: dict[int, dict[str, Any]] = {}
    with (path / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            index = _int(raw, "episode_index")
            if index in rows:
                raise ValueError(f"Duplicate episode index {index}: {path}")
            rows[index] = {
                "episode_index": index,
                "episode_seed": _int(raw, "episode_seed"),
                "safe_capture": as_bool(raw.get("safe_capture_success")),
                "collision": as_bool(raw.get("collision")),
                "defender_boundary_violation": as_bool(raw.get("defender_boundary_violation")),
                "target_boundary_violation": as_bool(raw.get("target_boundary_violation")),
                "pairwise_violation": as_bool(raw.get("pairwise_violation")),
                "raw_unverified_steps": _int(raw, "raw_unverified_executed_steps"),
                "cbf_timeout_steps": _int(raw, "cbf_timeout_steps"),
                "cbf_infeasible_steps": _int(raw, "cbf_infeasible_steps"),
                "cbf_unverified_steps": _int(raw, "cbf_unverified_steps"),
                "cbf_controlled_abort_steps": _int(raw, "cbf_controlled_abort_steps"),
                "cbf_fallback_steps": _int(raw, "cbf_fallback_steps"),
                "route_candidate_generated": _int(raw, "route_candidate_generated"),
                "route_geometry_valid": _int(raw, "route_geometry_valid"),
                "route_geometry_invalid": _int(raw, "route_geometry_invalid"),
                "route_cbf_probe_accepted": _int(raw, "route_cbf_probe_accepted"),
                "route_cbf_probe_rejected": _int(raw, "route_cbf_probe_rejected"),
                "termination_reason": str(raw.get("termination_reason", "")),
                "observation_condition": str(raw.get("observation_condition", "")),
            }
    if sorted(rows) != list(range(len(rows))):
        raise ValueError(f"Episode indices are not contiguous: {path}")
    inputs = metadata.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError(f"Run inputs are missing: {path}")
    return {
        "path": str(path),
        "variant": expected_variant,
        "training_seed": int(metadata.get("training_seed", -1)),
        "episodes": rows,
        "overall": summary.get("overall", {}),
        "metadata": metadata,
        "inputs": inputs,
        "hashes": {
            "summary": sha256(path / "summary.json"),
            "provenance": sha256(path / "provenance.json"),
            "scene_manifest": sha256(path / "scene_manifest.jsonl"),
        },
    }


def _bootstrap(values: np.ndarray, *, seed: int = 20260906, samples: int = 10_000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(samples, values.size))]
    means = np.mean(draws, axis=1)
    return {
        "observed": float(np.mean(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "seed": seed,
        "samples": samples,
        "unit": "episode_pair",
    }


def aggregate(m0_dir: Path, m3_dir: Path, output_dir: Path, tensorboard_dir: Path) -> dict[str, Any]:
    m0 = _load_run(m0_dir, "m0")
    m3 = _load_run(m3_dir, "m3")
    if m0["training_seed"] != m3["training_seed"]:
        raise ValueError("M0 and M3 training seeds differ")
    if m0["hashes"]["scene_manifest"] != m3["hashes"]["scene_manifest"]:
        raise ValueError("M0 and M3 scene manifests differ")
    for key in ("protocol_sha256", "environment_config_sha256", "actor_checkpoint_sha256"):
        if m0["inputs"].get(key) != m3["inputs"].get(key):
            raise ValueError(f"M0/M3 provenance mismatch: {key}")
    if len(m0["episodes"]) != len(m3["episodes"]):
        raise ValueError("M0 and M3 episode counts differ")

    pairs: list[dict[str, Any]] = []
    for index in sorted(m0["episodes"]):
        left = m0["episodes"][index]
        right = m3["episodes"][index]
        if left["episode_seed"] != right["episode_seed"]:
            raise ValueError(f"Episode seed mismatch at index {index}")
        delta = int(right["safe_capture"]) - int(left["safe_capture"])
        pairs.append({
            "episode_index": index,
            "episode_seed": left["episode_seed"],
            "m0_safe_capture": left["safe_capture"],
            "m3_safe_capture": right["safe_capture"],
            "delta": delta,
            "m0_termination": left["termination_reason"],
            "m3_termination": right["termination_reason"],
            "m0_cbf_abort_steps": left["cbf_controlled_abort_steps"],
            "m3_cbf_abort_steps": right["cbf_controlled_abort_steps"],
            "m3_route_geometry_valid": right["route_geometry_valid"],
            "m3_route_geometry_invalid": right["route_geometry_invalid"],
        })
    values = np.asarray([item["delta"] for item in pairs], dtype=np.float64)
    improved = int(sum(not item["m0_safe_capture"] and item["m3_safe_capture"] for item in pairs))
    degraded = int(sum(item["m0_safe_capture"] and not item["m3_safe_capture"] for item in pairs))
    tied = int(len(pairs) - improved - degraded)

    def _overall(run: dict[str, Any]) -> dict[str, Any]:
        overall = run["overall"]
        return {
            "episodes": int(overall.get("episodes", len(run["episodes"]))),
            "safe_capture_count": int(overall.get("safe_capture_count", 0)),
            "safe_capture_rate": float(overall.get("safe_capture_rate", 0.0)),
            "collision_count": int(overall.get("collision_count", 0)),
            "boundary_violation_count": int(overall.get("boundary_violation_count", 0)),
            "target_boundary_violation_count": int(overall.get("target_boundary_violation_count", 0)),
            "pairwise_violation_count": int(overall.get("pairwise_violation_count", 0)),
            "raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", 0)),
            "cbf_timeout_steps": int(overall.get("cbf_timeout_steps", 0)),
            "cbf_controlled_abort_steps": int(overall.get("cbf_controlled_abort_steps", 0)),
            "cbf_unverified_steps": int(overall.get("cbf_unverified_steps", 0)),
            "cbf_fallback_steps": int(overall.get("cbf_fallback_steps", 0)),
            "route_candidate_generated": int(overall.get("route_candidate_generated", 0)),
            "route_geometry_valid": int(overall.get("route_geometry_valid", 0)),
            "route_geometry_invalid": int(overall.get("route_geometry_invalid", 0)),
            "route_cbf_probe_accepted": int(overall.get("route_cbf_probe_accepted", 0)),
            "route_cbf_probe_rejected": int(overall.get("route_cbf_probe_rejected", 0)),
            "transit_success_rate": float(overall.get("transit_success_rate", 0.0)),
            "mean_capture_time_seconds": overall.get("mean_capture_time_seconds"),
            "mean_cbf_p95_solve_latency_ms": float(overall.get("mean_cbf_p95_solve_latency_ms", 0.0)),
        }

    metrics = {"m0": _overall(m0), "m3": _overall(m3)}
    safety_gate = all(
        metrics["m3"][key] == 0
        for key in ("collision_count", "boundary_violation_count", "pairwise_violation_count", "raw_unverified_executed_steps")
    )
    reliability_gate = (
        metrics["m3"]["cbf_timeout_steps"] == 0
        and metrics["m3"]["cbf_controlled_abort_steps"] == metrics["m3"]["cbf_unverified_steps"]
        and metrics["m3"]["cbf_fallback_steps"] >= metrics["m3"]["cbf_controlled_abort_steps"]
    )
    result: dict[str, Any] = {
        "stage": "WP4_actor_matched_m0_m3_paired_development",
        "development_only": True,
        "locked_test_opened": False,
        "training_seed": m0["training_seed"],
        "episodes": len(pairs),
        "inputs": {
            "m0_run": m0["path"],
            "m3_run": m3["path"],
            "protocol_sha256": m0["inputs"].get("protocol_sha256"),
            "environment_config_sha256": m0["inputs"].get("environment_config_sha256"),
            "actor_checkpoint_sha256": m0["inputs"].get("actor_checkpoint_sha256"),
            "jepa_checkpoint_sha256": m3["inputs"].get("jepa_checkpoint_sha256"),
            "reliability_ledger_sha256": m3["inputs"].get("reliability_ledger_sha256"),
            "scene_manifest_sha256": m0["hashes"]["scene_manifest"],
            "m0_summary_sha256": m0["hashes"]["summary"],
            "m3_summary_sha256": m3["hashes"]["summary"],
            "m0_provenance_sha256": m0["hashes"]["provenance"],
            "m3_provenance_sha256": m3["hashes"]["provenance"],
        },
        "metrics": metrics,
        "paired": {
            "improved": improved,
            "degraded": degraded,
            "tied": tied,
            "delta_rate": float(np.mean(values)),
            "mcnemar_exact_two_sided_p": float(binomtest(min(improved, degraded), n=improved + degraded, p=0.5).pvalue if improved + degraded else 1.0),
            "bootstrap": _bootstrap(values),
            "pairs": pairs,
        },
        "decision": {
            "safety_hard_gate": safety_gate,
            "reliability_gate": reliability_gate,
            "m3_delta_rate": float(np.mean(values)),
            "classification": (
                "positive_single_seed_development_evidence"
                if safety_gate and reliability_gate and float(np.mean(values)) > 0.0
                else "safety_preserving_no_control_gain"
                if safety_gate and reliability_gate
                else "execution_contract_failure"
            ),
            "next_step": (
                "run_three_seed_paired_smoke"
                if safety_gate and reliability_gate and float(np.mean(values)) >= 0.0
                else "stop_and_fix_execution_contract"
            ),
        },
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
    with (output / "paired_episode_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    report = [
        "# WP4 Actor-Matched M0/M3 Paired Development",
        "",
        "`development_only=true`; `locked_test_opened=false`. Statistical unit is an episode pair.",
        "",
        "| Variant | Safe capture | Collision | Defender boundary | Target boundary | Pairwise | Raw unverified | CBF abort steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("m0", "m3"):
        item = metrics[variant]
        report.append(
            f"| {variant.upper()} | {item['safe_capture_count']}/{item['episodes']} ({item['safe_capture_rate']:.1%}) | "
            f"{item['collision_count']} | {item['boundary_violation_count']} | {item['target_boundary_violation_count']} | "
            f"{item['pairwise_violation_count']} | {item['raw_unverified_executed_steps']} | {item['cbf_controlled_abort_steps']} |"
        )
    report += [
        "",
        f"Paired M3 versus M0: `{improved} improved / {degraded} degraded / {tied} tied`; delta `{float(np.mean(values)):+.1%}`.",
        f"McNemar exact two-sided p-value: `{result['paired']['mcnemar_exact_two_sided_p']:.4f}`; bootstrap 95% CI: `[{result['paired']['bootstrap']['ci95_low']:+.1%}, {result['paired']['bootstrap']['ci95_high']:+.1%}]`.",
        f"Route candidates generated/geometry-valid/invalid: `{metrics['m3']['route_candidate_generated']}/{metrics['m3']['route_geometry_valid']}/{metrics['m3']['route_geometry_invalid']}`.",
        f"Route CBF probes accepted/rejected: `{metrics['m3']['route_cbf_probe_accepted']}/{metrics['m3']['route_cbf_probe_rejected']}`.",
        f"Safety hard gate: `{safety_gate}`; reliability gate: `{reliability_gate}`.",
        f"Classification: `{result['decision']['classification']}`.",
        "",
        "Target-boundary counts are reported separately from the defender safety gate. Mean capture time is diagnostic only.",
        "This report does not authorize a locked test; the next authorized step is a three-seed paired development smoke.",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(result["inputs"], indent=2),
        "```",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("Config/stage", result["stage"], 0)
        writer.add_text("Provenance/inputs", json.dumps(result["inputs"], indent=2), 0)
        writer.add_text("Decision/summary", json.dumps(result["decision"], indent=2), 0)
        for variant in ("m0", "m3"):
            writer.add_scalar(f"SafeCapture/{variant}", metrics[variant]["safe_capture_rate"], 0)
            writer.add_scalar(f"Safety/{variant}/collision", metrics[variant]["collision_count"], 0)
            writer.add_scalar(f"Safety/{variant}/defender_boundary", metrics[variant]["boundary_violation_count"], 0)
            writer.add_scalar(f"Safety/{variant}/pairwise", metrics[variant]["pairwise_violation_count"], 0)
            writer.add_scalar(f"Safety/{variant}/raw_unverified", metrics[variant]["raw_unverified_executed_steps"], 0)
            writer.add_scalar(f"CBF/{variant}/controlled_abort_steps", metrics[variant]["cbf_controlled_abort_steps"], 0)
        writer.add_scalar("Paired/delta_rate", float(np.mean(values)), 0)
        writer.add_scalar("Paired/improved", improved, 0)
        writer.add_scalar("Paired/degraded", degraded, 0)
        writer.add_scalar("Paired/tied", tied, 0)
        writer.add_scalar("Gates/safety_hard_gate", float(safety_gate), 0)
        writer.add_scalar("Gates/reliability_gate", float(reliability_gate), 0)
    result["tensorboard"] = {"logdir": str(tensorboard), "event_files": sorted(item.name for item in tensorboard.glob("events.out.tfevents.*"))}
    (output / "paired_aggregate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-dir", type=Path, required=True)
    parser.add_argument("--m3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.m0_dir, args.m3_dir, args.output_dir, args.tensorboard_dir)
    print(json.dumps({"decision": result["decision"], "paired": result["paired"], "tensorboard": result["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
