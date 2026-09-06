"""Aggregate the actor-matched M0/M3 development block across three seeds.

The current route checkpoint is fixed, so this report explicitly records
whether the three protocol seeds correspond to distinct manifests and model
checkpoints.  It never upgrades repeated replays into independent training
evidence.
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

from aggregate_actor_matched_paired_development import _load_run


SEEDS = (20260911, 20260912, 20260913)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_manifest(path: Path) -> str:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest record is not an object: {path}")
            value = dict(value)
            value.pop("training_seed", None)
            records.append(value)
    payload = "".join(json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n" for item in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summary(run: dict[str, Any]) -> dict[str, Any]:
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
    }


def _paired(m0: dict[str, Any], m3: dict[str, Any]) -> dict[str, Any]:
    if m0["training_seed"] != m3["training_seed"]:
        raise ValueError("M0/M3 training seeds differ")
    pairs: list[dict[str, Any]] = []
    for index in sorted(m0["episodes"]):
        left = m0["episodes"][index]
        right = m3["episodes"].get(index)
        if right is None or left["episode_seed"] != right["episode_seed"]:
            raise ValueError(f"Episode pairing mismatch at index {index}")
        pairs.append({
            "episode_index": index,
            "episode_seed": left["episode_seed"],
            "m0_safe_capture": left["safe_capture"],
            "m3_safe_capture": right["safe_capture"],
            "delta": int(right["safe_capture"]) - int(left["safe_capture"]),
            "m0_termination": left["termination_reason"],
            "m3_termination": right["termination_reason"],
        })
    improved = int(sum(not item["m0_safe_capture"] and item["m3_safe_capture"] for item in pairs))
    degraded = int(sum(item["m0_safe_capture"] and not item["m3_safe_capture"] for item in pairs))
    return {
        "training_seed": m0["training_seed"],
        "episodes": len(pairs),
        "m0_safe_capture_count": int(sum(item["m0_safe_capture"] for item in pairs)),
        "m3_safe_capture_count": int(sum(item["m3_safe_capture"] for item in pairs)),
        "improved": improved,
        "degraded": degraded,
        "tied": int(len(pairs) - improved - degraded),
        "delta_rate": float(np.mean([item["delta"] for item in pairs])),
        "mcnemar_exact_two_sided_p": float(binomtest(min(improved, degraded), n=improved + degraded, p=0.5).pvalue if improved + degraded else 1.0),
        "pairs": pairs,
    }


def aggregate(project_root: Path, output_dir: Path, tensorboard_dir: Path) -> dict[str, Any]:
    root = project_root.resolve()
    runs: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in SEEDS:
        for variant in ("m0", "m3"):
            runs[(seed, variant)] = _load_run(root / f"results/wp4_actor_matched_paired_{variant}_seed{seed}", variant)
    comparisons = []
    for seed in SEEDS:
        m0, m3 = runs[(seed, "m0")], runs[(seed, "m3")]
        if m0["hashes"]["scene_manifest"] != m3["hashes"]["scene_manifest"]:
            raise ValueError(f"M0/M3 manifest mismatch for seed {seed}")
        for key in ("protocol_sha256", "environment_config_sha256", "actor_checkpoint_sha256"):
            if m0["inputs"].get(key) != m3["inputs"].get(key):
                raise ValueError(f"M0/M3 provenance mismatch for seed {seed}: {key}")
        comparisons.append(_paired(m0, m3))

    manifest_canonical = {
        str(seed): _canonical_manifest(Path(runs[(seed, "m0")]["path"]) / "scene_manifest.jsonl")
        for seed in SEEDS
    }
    checkpoint_hashes = {
        "actor": sorted({str(runs[(seed, "m3")]["inputs"].get("actor_checkpoint_sha256")) for seed in SEEDS}),
        "jepa": sorted({str(runs[(seed, "m3")]["inputs"].get("jepa_checkpoint_sha256")) for seed in SEEDS}),
        "ledger": sorted({str(runs[(seed, "m3")]["inputs"].get("reliability_ledger_sha256")) for seed in SEEDS}),
    }
    pooled_pairs = [item for row in comparisons for item in row["pairs"]]
    pooled_deltas = np.asarray([item["delta"] for item in pooled_pairs], dtype=np.float64)
    metrics = {
        str(seed): {variant: _summary(runs[(seed, variant)]) for variant in ("m0", "m3")}
        for seed in SEEDS
    }
    safety_gate = all(
        metrics[str(seed)]["m3"][key] == 0
        for seed in SEEDS
        for key in ("collision_count", "boundary_violation_count", "pairwise_violation_count", "raw_unverified_executed_steps")
    )
    reliability_gate = all(
        metrics[str(seed)]["m3"]["cbf_timeout_steps"] == 0
        and metrics[str(seed)]["m3"]["cbf_controlled_abort_steps"] == metrics[str(seed)]["m3"]["cbf_unverified_steps"]
        and metrics[str(seed)]["m3"]["cbf_fallback_steps"] >= metrics[str(seed)]["m3"]["cbf_controlled_abort_steps"]
        for seed in SEEDS
    )
    mean_delta = float(np.mean([row["delta_rate"] for row in comparisons]))
    independent_model_seed_evidence = len(checkpoint_hashes["jepa"]) == len(SEEDS)
    independent_scene_seed_evidence = len(set(manifest_canonical.values())) == len(SEEDS)
    result: dict[str, Any] = {
        "stage": "WP4_actor_matched_three_seed_m0_m3_paired_development",
        "development_only": True,
        "locked_test_opened": False,
        "protocol_seeds": list(SEEDS),
        "episodes_per_seed": 20,
        "metrics_by_seed": metrics,
        "paired_by_seed": comparisons,
        "pooled": {
            "episodes": int(pooled_deltas.size),
            "improved": int(sum(row["improved"] for row in comparisons)),
            "degraded": int(sum(row["degraded"] for row in comparisons)),
            "tied": int(sum(row["tied"] for row in comparisons)),
            "delta_rate": float(np.mean(pooled_deltas)),
        },
        "provenance": {
            "canonical_manifest_sha256_by_seed": manifest_canonical,
            "unique_canonical_scene_manifests": int(len(set(manifest_canonical.values()))),
            "checkpoint_sha256_by_kind": checkpoint_hashes,
            "unique_jepa_checkpoints": int(len(checkpoint_hashes["jepa"])),
            "unique_ledgers": int(len(checkpoint_hashes["ledger"])),
            "same_scene_replayed_across_protocol_seeds": len(set(manifest_canonical.values())) == 1,
        },
        "decision": {
            "safety_hard_gate": safety_gate,
            "reliability_gate": reliability_gate,
            "mean_paired_delta_rate": mean_delta,
            "nonnegative_protocol_seeds": int(sum(row["delta_rate"] >= 0.0 for row in comparisons)),
            "independent_model_seed_evidence": independent_model_seed_evidence,
            "independent_scene_seed_evidence": independent_scene_seed_evidence,
            "classification": (
                "positive_repeated_protocol_replay_signal_not_independent_seed_evidence"
                if safety_gate and reliability_gate and mean_delta > 0.0 and not independent_model_seed_evidence
                else "positive_three_seed_development_evidence"
                if safety_gate and reliability_gate and mean_delta > 0.0 and independent_model_seed_evidence and independent_scene_seed_evidence
                else "safety_preserving_no_control_gain"
                if safety_gate and reliability_gate
                else "execution_contract_failure"
            ),
            "next_step": (
                "train_or_evaluate_distinct_model_seeds_before_formal_claim"
                if safety_gate and reliability_gate and not independent_model_seed_evidence
                else "run_larger_development_block"
                if safety_gate and reliability_gate and mean_delta >= 0.0
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
    (output / "three_seed_aggregate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    flat_pairs = []
    for comparison in comparisons:
        flat_pairs.extend({"training_seed": comparison["training_seed"], **pair} for pair in comparison["pairs"])
    with (output / "paired_episode_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_pairs[0]))
        writer.writeheader()
        writer.writerows(flat_pairs)
    lines = [
        "# WP4 Actor-Matched Three-Seed Paired Development",
        "",
        "`development_only=true`; `locked_test_opened=false`. Statistical unit is an episode pair.",
        "",
        "| Seed | M0 safe capture | M3 safe capture | Improved | Degraded | Tied | Delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(f"| {row['training_seed']} | {row['m0_safe_capture_count']}/20 | {row['m3_safe_capture_count']}/20 | {row['improved']} | {row['degraded']} | {row['tied']} | {row['delta_rate']:+.1%} |")
    lines += [
        "",
        f"Pooled M3 versus M0: `{result['pooled']['improved']} improved / {result['pooled']['degraded']} degraded / {result['pooled']['tied']} tied`; delta `{result['pooled']['delta_rate']:+.1%}`.",
        f"Safety hard gate: `{safety_gate}`; reliability gate: `{reliability_gate}`.",
        f"Canonical scene manifests: `{result['provenance']['unique_canonical_scene_manifests']}` unique; same scene replayed across protocol seeds: `{result['provenance']['same_scene_replayed_across_protocol_seeds']}`.",
        f"Unique JEPA checkpoints: `{result['provenance']['unique_jepa_checkpoints']}`; independent model-seed evidence: `{independent_model_seed_evidence}`.",
        f"Classification: `{result['decision']['classification']}`.",
        f"Next step: `{result['decision']['next_step']}`.",
        "",
        "Target-boundary counts remain separate from the defender safety hard gate. Mean capture time is diagnostic only.",
        "This development aggregate does not authorize a locked test.",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(result["provenance"], indent=2),
        "```",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("Config/stage", result["stage"], 0)
        writer.add_text("Provenance/summary", json.dumps(result["provenance"], indent=2), 0)
        writer.add_text("Decision/summary", json.dumps(result["decision"], indent=2), 0)
        for row in comparisons:
            seed = row["training_seed"]
            writer.add_scalar(f"SafeCapture/seed_{seed}/m0", row["m0_safe_capture_count"] / row["episodes"], 0)
            writer.add_scalar(f"SafeCapture/seed_{seed}/m3", row["m3_safe_capture_count"] / row["episodes"], 0)
            writer.add_scalar(f"Paired/seed_{seed}/delta", row["delta_rate"], 0)
            writer.add_scalar(f"Paired/seed_{seed}/improved", row["improved"], 0)
            writer.add_scalar(f"Paired/seed_{seed}/degraded", row["degraded"], 0)
        writer.add_scalar("Paired/pooled_delta", result["pooled"]["delta_rate"], 0)
        writer.add_scalar("Provenance/unique_scene_manifests", result["provenance"]["unique_canonical_scene_manifests"], 0)
        writer.add_scalar("Provenance/unique_jepa_checkpoints", result["provenance"]["unique_jepa_checkpoints"], 0)
        writer.add_scalar("Gates/safety_hard_gate", float(safety_gate), 0)
        writer.add_scalar("Gates/reliability_gate", float(reliability_gate), 0)
    result["tensorboard"] = {"logdir": str(tensorboard), "event_files": sorted(item.name for item in tensorboard.glob("events.out.tfevents.*"))}
    (output / "three_seed_aggregate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.project_root, args.output_dir, args.tensorboard_dir)
    print(json.dumps({"decision": result["decision"], "pooled": result["pooled"], "provenance": result["provenance"], "tensorboard": result["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
