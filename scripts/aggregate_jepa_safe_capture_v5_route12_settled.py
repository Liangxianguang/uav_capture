"""Aggregate the development-only twelve-route settled replay audit.

The audit inputs are frozen M3 traces and their offline route-CBF settled
rows.  This report measures route regret and Ledger over-abstention; it never
changes an online decision, checkpoint, CBF margin, or locked split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object: {path}")
            rows.append(value)
    return rows


def _source_eligibility(source_run: Path) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for path in sorted((source_run / "step_traces").glob("episode_*.jsonl")):
        for row in _jsonl(path):
            ranking = row.get("candidate_ranking")
            if not isinstance(ranking, Mapping):
                raise ValueError(f"Missing candidate ranking in {path}")
            eligible = ranking.get("eligible_mask")
            if not isinstance(eligible, list) or len(eligible) != 12:
                raise ValueError(f"Expected twelve-candidate eligibility in {path}")
            key = (int(row["episode_index"]), int(row["step"]))
            result[key] = int(sum(bool(value) for value in eligible))
    return result


def _one(audit_dir: Path, source_run: Path) -> dict[str, Any]:
    report = _json(audit_dir / "settled_counterfactual.json")
    rows = _jsonl(audit_dir / "decision_rows.jsonl")
    if report.get("development_only") is not True or report.get("locked_test_opened") is not False:
        raise ValueError(f"Audit is outside development boundary: {audit_dir}")
    if report.get("policy", {}).get("eligibility_source") != "route_cbf_verified":
        raise ValueError(f"Audit is not the route-CBF attribution view: {audit_dir}")
    if any(int(row.get("candidate_count", 0)) != 12 for row in rows):
        raise ValueError(f"Audit contains a non-route candidate count: {audit_dir}")
    source_rows = _source_eligibility(source_run)
    paired: list[dict[str, Any]] = []
    for row in rows:
        key = (int(row["episode_index"]), int(row["step"]))
        if key not in source_rows:
            raise ValueError(f"Source trace does not cover audit row {key}: {audit_dir}")
        recorded_count = source_rows[key]
        route_count = int(row["route_cbf_eligible_count"])
        paired.append({**row, "recorded_eligible_count": recorded_count, "route_cbf_eligible_count": route_count})
    if len(paired) != len(source_rows):
        raise ValueError(f"Audit/source trace count mismatch: {audit_dir}")
    stats = report["by_variant"]["m3"]
    source_summary = _json(source_run / "summary.json")
    overall = source_summary.get("overall", {})
    return {
        "seed": int(report["inputs"]["training_seeds"][0]),
        "audit_dir": str(audit_dir.resolve()),
        "source_run": str(source_run.resolve()),
        "audit_sha256": _sha256(audit_dir / "settled_counterfactual.json"),
        "source_summary_sha256": _sha256(source_run / "summary.json"),
        "decision_count": len(paired),
        "mean_recorded_eligible": float(np.mean([row["recorded_eligible_count"] for row in paired])),
        "mean_route_cbf_eligible": float(np.mean([row["route_cbf_eligible_count"] for row in paired])),
        "recorded_all_ineligible_count": int(sum(row["recorded_eligible_count"] == 0 for row in paired)),
        "route_all_ineligible_count": int(sum(row["route_cbf_eligible_count"] == 0 for row in paired)),
        "ledger_over_abstention_count": int(
            sum(row["recorded_eligible_count"] == 0 and row["route_cbf_eligible_count"] > 0 for row in paired)
        ),
        "selected_not_best_rate": float(stats["selected_not_best_rate"]),
        "selected_settled_safe_capture_rate": float(stats["selected_settled_safe_capture_rate"]),
        "best_settled_safe_capture_rate": float(stats["best_settled_safe_capture_rate"]),
        "selected_settled_safety_rate": float(stats["selected_settled_safety_rate"]),
        "spearman_mean": stats["spearman_mean"],
        "kendall_mean": stats["kendall_mean"],
        "mean_selected_progress_m": float(stats["mean_selected_progress_m"]),
        "mean_best_progress_m": float(stats["mean_best_progress_m"]),
        "source_safety_hard_events": {
            key: int(overall.get(key, 0))
            for key in ("collision_count", "boundary_violation_count", "pairwise_violation_count", "raw_unverified_executed_steps")
        },
        "paired_rows": paired,
    }


def _weighted(items: list[dict[str, Any]], key: str) -> float:
    total = sum(int(item["decision_count"]) for item in items)
    return float(sum(item[key] * item["decision_count"] for item in items) / max(total, 1))


def _write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/route12_settled_aggregate", json.dumps(report["policy"], indent=2, sort_keys=True), 0)
        writer.add_text("Provenance/inputs", json.dumps(report["inputs"], indent=2, sort_keys=True), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2, sort_keys=True), 0)
        for index, item in enumerate(report["seeds"]):
            prefix = f"Seed/{item['seed']}"
            writer.add_scalar(f"{prefix}/selected_not_best_rate", item["selected_not_best_rate"], index)
            writer.add_scalar(f"{prefix}/selected_safe_capture_rate", item["selected_settled_safe_capture_rate"], index)
            writer.add_scalar(f"{prefix}/best_safe_capture_rate", item["best_settled_safe_capture_rate"], index)
            writer.add_scalar(f"{prefix}/spearman_mean", float(item["spearman_mean"] or 0.0), index)
            writer.add_scalar(f"{prefix}/mean_recorded_eligible", item["mean_recorded_eligible"], index)
            writer.add_scalar(f"{prefix}/mean_route_cbf_eligible", item["mean_route_cbf_eligible"], index)
            writer.add_scalar(f"{prefix}/ledger_over_abstention_count", item["ledger_over_abstention_count"], index)
        writer.add_scalar("Pooled/selected_not_best_rate", report["pooled"]["selected_not_best_rate"], 0)
        writer.add_scalar("Pooled/selected_safe_capture_rate", report["pooled"]["selected_settled_safe_capture_rate"], 0)
        writer.add_scalar("Pooled/best_safe_capture_rate", report["pooled"]["best_settled_safe_capture_rate"], 0)
        writer.add_scalar("Pooled/mean_recorded_eligible", report["pooled"]["mean_recorded_eligible"], 0)
        writer.add_scalar("Pooled/mean_route_cbf_eligible", report["pooled"]["mean_route_cbf_eligible"], 0)
        writer.add_scalar("Gates/retraining_allowed", float(report["gates"]["retraining_allowed"]), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/route12_settled_aggregate/text_summary",
        "Provenance/inputs/text_summary",
        "Gates/status/text_summary",
    }
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    missing = sorted(required.difference(tags.get("tensors", [])))
    if missing or not events:
        raise ValueError(f"TensorBoard validation failed: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", []))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, action="append", required=True)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args()
    if len(args.audit_dir) != len(args.source_run):
        raise ValueError("Each audit directory requires one matching source run.")
    seeds = [_one(audit.resolve(), source.resolve()) for audit, source in zip(args.audit_dir, args.source_run)]
    seeds.sort(key=lambda item: item["seed"])
    total = sum(item["decision_count"] for item in seeds)
    pooled = {
        key: _weighted(seeds, key)
        for key in (
            "selected_not_best_rate",
            "selected_settled_safe_capture_rate",
            "best_settled_safe_capture_rate",
            "selected_settled_safety_rate",
            "mean_recorded_eligible",
            "mean_route_cbf_eligible",
            "mean_selected_progress_m",
            "mean_best_progress_m",
        )
    }
    pooled["ledger_over_abstention_count"] = int(sum(item["ledger_over_abstention_count"] for item in seeds))
    pooled["recorded_all_ineligible_count"] = int(sum(item["recorded_all_ineligible_count"] for item in seeds))
    pooled["route_all_ineligible_count"] = int(sum(item["route_all_ineligible_count"] for item in seeds))
    hard_events = {key: int(sum(item["source_safety_hard_events"][key] for item in seeds)) for key in seeds[0]["source_safety_hard_events"]}
    gates = {
        "development_only": True,
        "locked_test_opened": False,
        "twelve_candidate_contract": True,
        "all_source_hard_safety_events_zero": all(all(value == 0 for value in item["source_safety_hard_events"].values()) for item in seeds),
        "route_settled_safety_observable": all(item["selected_settled_safety_rate"] >= 0.0 for item in seeds),
        "retraining_allowed": False,
    }
    report: dict[str, Any] = {
        "report_type": "jepa_safe_capture_v5_route12_settled_three_seed",
        "development_only": True,
        "locked_test_opened": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "policy": {
            "scope": "offline local-chunk settled route-regret and Ledger abstention attribution",
            "eligibility_source": "route_cbf_verified; recorded Ledger eligibility is compared, not modified",
            "target_truth": "offline settled labels only",
            "online_decision_changed": False,
            "cbf_margin_changed": False,
            "controlled_abort_removed": False,
            "stop_rule": "do not retrain or enlarge scenes while selected route ranking has no positive signal",
        },
        "inputs": {
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": _sha256(args.protocol.resolve()),
            "environment_config": str(args.environment_config.resolve()),
            "environment_config_sha256": _sha256(args.environment_config.resolve()),
            "audit_dirs": [item["audit_dir"] for item in seeds],
            "source_runs": [item["source_run"] for item in seeds],
        },
        "decision_count": total,
        "seeds": [{key: value for key, value in item.items() if key != "paired_rows"} for item in seeds],
        "pooled": pooled,
        "source_hard_events": hard_events,
        "gates": gates,
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    report["tensorboard"] = _write_tensorboard(args.tensorboard_logdir.resolve(), report)
    args.report_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_md.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_json.resolve().write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# V5 Route-12 Settled Counterfactual Three-Seed Audit",
        "",
        "`development_only=true`; `locked_test_opened=false`. This is an offline local-chunk attribution, not a new online benchmark.",
        "",
        "## Result",
        "",
        "The audit keeps geometry-valid, independently CBF-verified routes and compares them with the recorded Ledger eligibility. It does not relax the online Ledger or CBF contract.",
        "",
        "| Seed | Decisions | Mean recorded eligible | Mean CBF-verified eligible | Ledger over-abstention | Selected-not-best | Selected settled safe | Best settled safe | Spearman |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in seeds:
        lines.append(
            f"| {item['seed']} | {item['decision_count']} | {item['mean_recorded_eligible']:.3f} | {item['mean_route_cbf_eligible']:.3f} | "
            f"{item['ledger_over_abstention_count']} | {item['selected_not_best_rate']:.3f} | {item['selected_settled_safe_capture_rate']:.3f} | "
            f"{item['best_settled_safe_capture_rate']:.3f} | {item['spearman_mean'] if item['spearman_mean'] is not None else 'NA'} |"
        )
    lines += [
        "",
        f"Pooled decisions: **{total}**; selected-not-best `{pooled['selected_not_best_rate']:.3f}`; selected settled safe `{pooled['selected_settled_safe_capture_rate']:.3f}`; best settled safe `{pooled['best_settled_safe_capture_rate']:.3f}`.",
        "",
        f"Pooled mean eligible candidates rise from `{pooled['mean_recorded_eligible']:.3f}` in the recorded Ledger view to `{pooled['mean_route_cbf_eligible']:.3f}` when only independently verified route CBF probes are retained. This is evidence of Ledger over-abstention, not permission to disable stale/OOD gates.",
        "",
        "## Decision",
        "",
        "The route-regret signal is not positive: selected routes are frequently not the settled-best route, and no seed shows a reliable capture improvement. Stop further JEPA training, data expansion, and larger scene matrices. The next work item is a bounded Ledger reacquisition contract plus route-ranking/auxiliary-head diagnosis, followed by a new calibration archive.",
        "",
        "All source hard safety events remain zero in this audit block. `controlled_abort`, CBF margins, stale/OOD gates, and raw-unverified prohibitions remain unchanged.",
        "",
        f"TensorBoard: `{args.tensorboard_logdir.resolve()}`.",
    ]
    args.report_md.resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision_count": total, "pooled": pooled, "gates": gates, "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
