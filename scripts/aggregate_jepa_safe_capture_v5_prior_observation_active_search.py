"""Aggregate the V5 prior-observation active-search paired development block.

This block contains three protocol-seed replays of the same validation
manifest.  The aggregate therefore reports a repeated-replay signal and
explicitly refuses to call it independent model-seed evidence.  It also keeps
the M0 success that regressed under M3 visible, because a safety-preserving
timeout is still a capability regression.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


SEEDS = (20260911, 20260912, 20260913)
PREFIX = "jepa_safe_capture_v5_prior_observation_active_search"


def _exact_two_sided_binom_pvalue(successes: int, trials: int) -> float:
    """Two-sided exact binomial p-value for p=0.5 without SciPy."""
    if trials == 0:
        return 1.0
    probability = 2.0 ** (-trials)
    observed = math.comb(trials, successes) * probability
    return min(1.0, sum(
        math.comb(trials, k) * probability
        for k in range(trials + 1)
        if math.comb(trials, k) * probability <= observed + 1e-15
    ))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _as_int(value: Any) -> int:
    return int(float(value or 0))


def _canonical_manifest_sha256(path: Path) -> str:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Manifest record is not an object: {path}")
            item = dict(item)
            item.pop("training_seed", None)
            records.append(item)
    payload = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for item in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_run(root: Path, variant: str, seed: int) -> dict[str, Any]:
    path = (root / "results" / f"{PREFIX}_{variant}_seed{seed}").resolve()
    required = ("summary.json", "provenance.json", "episodes.csv", "scene_manifest.jsonl")
    missing = [path / name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts for {variant} seed {seed}: {missing}")
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
    declared_variant = metadata.get("variant", {})
    if not isinstance(declared_variant, Mapping) or declared_variant.get("variant") != variant:
        raise ValueError(f"Variant mismatch in {path}")
    if _as_int(metadata.get("training_seed")) != seed:
        raise ValueError(f"Training seed mismatch in {path}")
    episodes: dict[int, dict[str, Any]] = {}
    with (path / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            index = _as_int(raw.get("episode_index"))
            if index in episodes:
                raise ValueError(f"Duplicate episode index {index}: {path}")
            episodes[index] = {
                "episode_index": index,
                "episode_seed": _as_int(raw.get("episode_seed")),
                "safe_capture": _as_bool(raw.get("safe_capture_success")),
                "termination_reason": str(raw.get("termination_reason", "")),
                "collision": _as_bool(raw.get("collision")),
                "boundary_violation": _as_bool(raw.get("boundary_violation")),
                "pairwise_violation": _as_bool(raw.get("pairwise_violation")),
                "raw_unverified_steps": _as_int(raw.get("raw_unverified_executed_steps")),
                "cbf_controlled_abort_steps": _as_int(raw.get("cbf_controlled_abort_steps")),
                "cbf_timeout_steps": _as_int(raw.get("cbf_timeout_steps")),
                "safe_hold_steps": _as_int(raw.get("safe_hold_steps")),
                "reacquisition_steps": _as_int(raw.get("cautious_reacquisition_attempt_steps")),
                "route_geometry_valid": _as_int(raw.get("route_geometry_valid")),
                "route_cbf_probe_accepted": _as_int(raw.get("route_cbf_probe_accepted")),
            }
    if sorted(episodes) != list(range(len(episodes))):
        raise ValueError(f"Episode indices are not contiguous: {path}")
    inputs = metadata.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError(f"Missing inputs: {path}")
    return {
        "path": str(path),
        "variant": variant,
        "seed": seed,
        "summary": summary,
        "metadata": metadata,
        "provenance": provenance,
        "inputs": dict(inputs),
        "overall": dict(overall),
        "episodes": episodes,
        "hashes": {
            "summary": _sha256(path / "summary.json"),
            "provenance": _sha256(path / "provenance.json"),
            "manifest": _sha256(path / "scene_manifest.jsonl"),
            "canonical_manifest": _canonical_manifest_sha256(path / "scene_manifest.jsonl"),
        },
    }


def _metric(run: Mapping[str, Any]) -> dict[str, Any]:
    overall = run["overall"]
    episodes = run["episodes"]
    termination_counts: dict[str, int] = {}
    for episode in episodes.values():
        reason = episode["termination_reason"]
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    return {
        "variant": run["variant"],
        "training_seed": run["seed"],
        "episodes": int(overall.get("episodes", len(episodes))),
        "safe_capture_count": int(overall.get("safe_capture_count", 0)),
        "safe_capture_rate": float(overall.get("safe_capture_rate", 0.0)),
        "collision_count": int(overall.get("collision_count", 0)),
        "boundary_violation_count": int(overall.get("boundary_violation_count", 0)),
        "pairwise_violation_count": int(overall.get("pairwise_violation_count", 0)),
        "raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        "cbf_controlled_abort_steps": int(overall.get("cbf_controlled_abort_steps", 0)),
        "cbf_timeout_steps": int(overall.get("cbf_timeout_steps", 0)),
        "cbf_fallback_steps": int(overall.get("cbf_fallback_steps", 0)),
        "route_candidate_generated": int(overall.get("route_candidate_generated", 0)),
        "route_geometry_valid": int(overall.get("route_geometry_valid", 0)),
        "route_geometry_invalid": int(overall.get("route_geometry_invalid", 0)),
        "route_cbf_probe_accepted": int(overall.get("route_cbf_probe_accepted", 0)),
        "safe_hold_steps": int(sum(item["safe_hold_steps"] for item in episodes.values())),
        "reacquisition_attempt_steps": int(sum(item["reacquisition_steps"] for item in episodes.values())),
        "termination_counts": termination_counts,
        "max_cycle_p95_ms": float(overall.get("latency_breakdown", {}).get("cycle_total", {}).get("max_episode_p95_ms", 0.0)),
    }


def _paired(m0: Mapping[str, Any], m3: Mapping[str, Any]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for index in sorted(m0["episodes"]):
        left = m0["episodes"][index]
        right = m3["episodes"].get(index)
        if right is None or left["episode_seed"] != right["episode_seed"]:
            raise ValueError(f"M0/M3 episode mismatch at index {index}")
        pairs.append({
            "episode_index": index,
            "episode_seed": left["episode_seed"],
            "m0_safe_capture": left["safe_capture"],
            "m3_safe_capture": right["safe_capture"],
            "delta": int(right["safe_capture"]) - int(left["safe_capture"]),
            "m0_termination": left["termination_reason"],
            "m3_termination": right["termination_reason"],
            "m3_safe_hold_steps": right["safe_hold_steps"],
            "m3_reacquisition_steps": right["reacquisition_steps"],
        })
    improved = sum(not item["m0_safe_capture"] and item["m3_safe_capture"] for item in pairs)
    degraded = sum(item["m0_safe_capture"] and not item["m3_safe_capture"] for item in pairs)
    discordant = improved + degraded
    return {
        "training_seed": m0["seed"],
        "episodes": len(pairs),
        "m0_safe_capture_count": sum(item["m0_safe_capture"] for item in pairs),
        "m3_safe_capture_count": sum(item["m3_safe_capture"] for item in pairs),
        "improved": int(improved),
        "degraded": int(degraded),
        "tied": int(len(pairs) - improved - degraded),
        "delta_rate": float(np.mean([item["delta"] for item in pairs])),
        "mcnemar_exact_two_sided_p": _exact_two_sided_binom_pvalue(min(improved, degraded), discordant),
        "pairs": pairs,
    }


def _aggregate(root: Path) -> dict[str, Any]:
    runs = {(seed, variant): _load_run(root, variant, seed) for seed in SEEDS for variant in ("m0", "m3")}
    for seed in SEEDS:
        m0, m3 = runs[(seed, "m0")], runs[(seed, "m3")]
        for key in ("protocol_sha256", "environment_config_sha256", "actor_checkpoint_sha256", "scene_manifest_sha256"):
            if m0["inputs"].get(key) != m3["inputs"].get(key):
                raise ValueError(f"M0/M3 provenance mismatch for seed {seed}: {key}")
    paired = [_paired(runs[(seed, "m0")], runs[(seed, "m3")]) for seed in SEEDS]
    metrics = [_metric(runs[(seed, variant)]) for variant in ("m0", "m3") for seed in SEEDS]
    m3_metrics = [item for item in metrics if item["variant"] == "m3"]
    safety_gate = all(
        item[key] == 0
        for item in m3_metrics
        for key in ("collision_count", "boundary_violation_count", "pairwise_violation_count", "raw_unverified_executed_steps")
    )
    pooled_improved = sum(item["improved"] for item in paired)
    pooled_degraded = sum(item["degraded"] for item in paired)
    pooled_tied = sum(item["tied"] for item in paired)
    pooled_delta = float(np.mean([item["delta_rate"] for item in paired]))
    canonical_manifests = {str(seed): runs[(seed, "m3")]["hashes"]["canonical_manifest"] for seed in SEEDS}
    jepa_hashes = sorted({str(runs[(seed, "m3")]["inputs"].get("jepa_checkpoint_sha256", "")) for seed in SEEDS})
    ledger_hashes = sorted({str(runs[(seed, "m3")]["inputs"].get("reliability_ledger_sha256", "")) for seed in SEEDS})
    same_episode_outcomes = len({tuple((item["m3_safe_capture"], item["m3_termination"]) for item in row["pairs"]) for row in paired}) == 1
    all_reacquisition_zero = all(item["reacquisition_attempt_steps"] == 0 for item in m3_metrics)
    repeated_replay_only = len(jepa_hashes) == 1 or len(set(canonical_manifests.values())) == 1
    decision = {
        "safety_hard_gate": safety_gate,
        "paired_mean_delta_rate": pooled_delta,
        "nonnegative_seed_count": sum(item["delta_rate"] >= 0.0 for item in paired),
        "improved": pooled_improved,
        "degraded": pooled_degraded,
        "tied": pooled_tied,
        "same_canonical_manifest": len(set(canonical_manifests.values())) == 1,
        "unique_jepa_checkpoints": len(jepa_hashes),
        "unique_ledgers": len(ledger_hashes),
        "independent_model_seed_evidence": len(jepa_hashes) == len(SEEDS),
        "same_episode_outcomes_across_replays": same_episode_outcomes,
        "active_search_triggered": not all_reacquisition_zero,
        "classification": (
            "positive_repeated_replay_signal_with_m3_degradation_not_independent_evidence"
            if safety_gate and pooled_delta > 0.0 and pooled_degraded > 0 and repeated_replay_only
            else "positive_development_signal_not_independent_evidence"
            if safety_gate and pooled_delta > 0.0 and repeated_replay_only
            else "safety_preserving_no_control_gain"
            if safety_gate
            else "execution_contract_failure"
        ),
        "continue_expansion": False,
        "stop_reason": (
            "same_manifest_and_checkpoint_replay; one M0 success regresses to M3 timeout; active-search never executed"
            if safety_gate and pooled_degraded > 0 and all_reacquisition_zero
            else "no stable positive paired delta under safety gate"
            if safety_gate and pooled_delta <= 0.0
            else "safety gate failed"
        ),
    }
    return {
        "stage": "v5_prior_observation_active_search_three_seed_paired_development",
        "development_only": True,
        "locked_test_opened": False,
        "protocol_seeds": list(SEEDS),
        "episodes_per_run": int(metrics[0]["episodes"]),
        "metrics": metrics,
        "paired_by_seed": paired,
        "pooled": {"episodes": sum(item["episodes"] for item in paired), "improved": pooled_improved, "degraded": pooled_degraded, "tied": pooled_tied, "delta_rate": pooled_delta},
        "provenance": {"canonical_manifest_sha256_by_seed": canonical_manifests, "jepa_checkpoint_sha256": jepa_hashes, "ledger_sha256": ledger_hashes, "same_episode_outcomes_across_replays": same_episode_outcomes},
        "decision": decision,
        "inputs": [{"path": runs[(seed, variant)]["path"], "summary_sha256": runs[(seed, variant)]["hashes"]["summary"], "provenance_sha256": runs[(seed, variant)]["hashes"]["provenance"], "variant": variant, "training_seed": seed} for seed in SEEDS for variant in ("m0", "m3")],
        "environment": {"python": platform.python_version(), "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
    }


def _write_tensorboard(report: Mapping[str, Any], logdir: Path) -> dict[str, Any]:
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/stage", report["stage"], 0)
        writer.add_text("Provenance/summary", json.dumps(report["provenance"], indent=2), 0)
        writer.add_text("Decision/summary", json.dumps(report["decision"], indent=2), 0)
        for item in report["metrics"]:
            tag = f"{item['variant']}/seed{item['training_seed']}"
            writer.add_scalar(f"SafeCapture/{tag}", item["safe_capture_rate"], 0)
            writer.add_scalar(f"Safety/{tag}/collision", item["collision_count"], 0)
            writer.add_scalar(f"Safety/{tag}/boundary", item["boundary_violation_count"], 0)
            writer.add_scalar(f"Safety/{tag}/pairwise", item["pairwise_violation_count"], 0)
            writer.add_scalar(f"Safety/{tag}/raw_unverified", item["raw_unverified_executed_steps"], 0)
            writer.add_scalar(f"Fallback/{tag}/safe_hold_steps", item["safe_hold_steps"], 0)
            writer.add_scalar(f"Search/{tag}/reacquisition_attempt_steps", item["reacquisition_attempt_steps"], 0)
            writer.add_scalar(f"Failures/{tag}/timeout", item["termination_counts"].get("timeout", 0), 0)
        for item in report["paired_by_seed"]:
            writer.add_scalar(f"Paired/seed{item['training_seed']}/delta_rate", item["delta_rate"], 0)
            writer.add_scalar(f"Paired/seed{item['training_seed']}/improved", item["improved"], 0)
            writer.add_scalar(f"Paired/seed{item['training_seed']}/degraded", item["degraded"], 0)
        writer.add_scalar("Paired/pooled_delta_rate", report["pooled"]["delta_rate"], 0)
        writer.add_scalar("Gates/safety_hard_gate", float(report["decision"]["safety_hard_gate"]), 0)
        writer.add_scalar("Gates/continue_expansion", float(report["decision"]["continue_expansion"]), 0)
        writer.add_scalar("Search/active_search_triggered", float(report["decision"]["active_search_triggered"]), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    required = {"Config/stage/text_summary", "Provenance/summary/text_summary", "Decision/summary/text_summary"}
    missing = sorted(required.difference(tags.get("tensors", [])))
    if missing or not events:
        raise ValueError(f"Incomplete TensorBoard aggregate: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", []))}


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    decision = report["decision"]
    lines = [
        "# V5 Prior-Observation Active-Search Three-Seed Stop Report",
        "",
        "`development_only=true`; `locked_test_opened=false`. This is a paired replay block, not a locked benchmark.",
        "",
        "| Seed | M0 | M3 | Improved | Degraded | Tied | Delta | M3 termination |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["paired_by_seed"]:
        m3 = next(item for item in report["metrics"] if item["variant"] == "m3" and item["training_seed"] == row["training_seed"])
        lines.append(f"| {row['training_seed']} | {row['m0_safe_capture_count']}/{row['episodes']} | {row['m3_safe_capture_count']}/{row['episodes']} | {row['improved']} | {row['degraded']} | {row['tied']} | {row['delta_rate']:+.1%} | {m3['termination_counts']} |")
    lines += [
        "",
        f"Pooled paired outcome: `{report['pooled']['improved']} improved / {report['pooled']['degraded']} degraded / {report['pooled']['tied']} tied`; delta `{report['pooled']['delta_rate']:+.1%}`.",
        f"Safety hard gate: `{'PASS' if decision['safety_hard_gate'] else 'FAIL'}`; collision, boundary, pairwise and raw-unverified counts are all zero for M3.",
        f"Active-search trigger: `{'YES' if decision['active_search_triggered'] else 'NO'}`; all cautious reacquisition attempt steps are zero.",
        f"Canonical manifest count: `{len(set(report['provenance']['canonical_manifest_sha256_by_seed'].values()))}`; unique JEPA checkpoints: `{len(report['provenance']['jepa_checkpoint_sha256'])}`.",
        f"Classification: `{decision['classification']}`.",
        f"Expansion gate: `{'OPEN' if decision['continue_expansion'] else 'STOP'}`. Reason: {decision['stop_reason']}.",
        "",
        "Interpretation: M3 reaches 2/3 on every replay, but the same fixed episode that M0 captures becomes an M3 timeout. This is a deterministic route/Ledger behavior signal, not evidence that the JEPA model seed generalizes. Do not expand to L1-L3 or retrain before diagnosing this degradation and the previously recorded settled route-ranking mismatch.",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(report["provenance"], indent=2, ensure_ascii=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("This aggregate is development-only")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    report = _aggregate(args.project_root.resolve())
    output.mkdir(parents=True, exist_ok=True)
    report["tensorboard"] = _write_tensorboard(report, args.tensorboard_dir.resolve())
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output / "paired_comparisons.json").write_text(json.dumps(report["paired_by_seed"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output / "paired_episode_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [dict(seed=item["training_seed"], **pair) for item in report["paired_by_seed"] for pair in item["pairs"]]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(report, output / "report.md")
    print(json.dumps({"decision": report["decision"], "pooled": report["pooled"], "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
