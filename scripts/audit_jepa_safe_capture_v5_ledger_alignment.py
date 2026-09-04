"""Audit reliability-ledger routing against settled counterfactual evidence.

This is a read-only T3 audit. It consumes T2 decision rows, checks that the
checkpoint-bound ledger is immutable, exercises deterministic OOD/stale/
non-finite/risk fallback cases, and compares settled local failures by credit
bucket. Missing low-credit coverage is reported as insufficient evidence; it
is never silently treated as evidence that the ledger works.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from encirclement3d.reliability import SafeCaptureReliabilityLedger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _read_rows(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit = _json(run_dir / "settled_counterfactual.json")
    if audit.get("development_only") is not True or audit.get("locked_test_opened") is not False:
        raise ValueError(f"T2 audit crossed locked-test boundary: {run_dir}")
    rows: list[dict[str, Any]] = []
    for line in (run_dir / "decision_rows.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid T2 decision row: {run_dir}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No T2 decision rows: {run_dir}")
    return audit, rows


def _credit_stats(rows: Sequence[Mapping[str, Any]], minimum_credit: float) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    states: Counter[str] = Counter()
    for row in rows:
        selected = int(row.get("selected_index", 0))
        credits = row.get("ledger_credits")
        ledger_states = row.get("ledger_states")
        credit = _finite(credits[selected]) if isinstance(credits, list) and selected < len(credits) else None
        state = str(ledger_states[selected]) if isinstance(ledger_states, list) and selected < len(ledger_states) else "missing"
        bucket = "high" if credit is not None and credit >= minimum_credit else "low_or_missing"
        enriched = dict(row)
        enriched["selected_ledger_credit"] = credit
        enriched["selected_ledger_state"] = state
        enriched["credit_bucket"] = bucket
        buckets[bucket].append(enriched)
        states[state] += 1
    result: dict[str, Any] = {"state_counts": dict(sorted(states.items())), "buckets": {}}
    for name, values in sorted(buckets.items()):
        safe_capture_failures = [not bool(row.get("selected_settled_safe_capture")) for row in values]
        safety_failures = [not bool(row.get("selected_settled_safety_ok")) for row in values]
        result["buckets"][name] = {
            "decisions": len(values),
            "episodes": len({(row.get("training_seed"), row.get("variant"), row.get("episode_index")) for row in values}),
            "safe_capture_failure_count": int(sum(safe_capture_failures)),
            "safe_capture_failure_rate": float(np.mean(safe_capture_failures)) if safe_capture_failures else None,
            "failure_count": int(sum(safety_failures)),
            "failure_rate": float(np.mean(safety_failures)) if safety_failures else None,
            "safe_capture_rate": float(np.mean([bool(row.get("selected_settled_safe_capture")) for row in values])) if values else None,
            "selected_safety_failure_rate": float(np.mean(safety_failures)) if safety_failures else None,
            "mean_credit": _mean([credit for row in values if (credit := _finite(row.get("selected_ledger_credit"))) is not None]),
            "pair_labels": dict(Counter(str(row.get("pair_label")) for row in values)),
        }
    high = result["buckets"].get("high")
    low = result["buckets"].get("low_or_missing")
    result["coverage"] = {
        "high_decisions": int(high["decisions"]) if high else 0,
        "low_decisions": int(low["decisions"]) if low else 0,
        "both_buckets_present": bool(high and low),
    }
    if high and low:
        result["high_credit_failure_not_above_low_credit"] = bool(high["failure_rate"] <= low["failure_rate"] + 1e-12)
    else:
        result["high_credit_failure_not_above_low_credit"] = None
    return result


def _fault_audit(ledger_path: Path) -> dict[str, Any]:
    before = sha256(ledger_path)
    ledger = SafeCaptureReliabilityLedger(_json(ledger_path))
    base: dict[str, Any] = {
        "visibility_condition": 1.0,
        "observation_age_steps": 0.0,
        "obstacle_count": 3,
        "layout_signature": "scenario_0",
        "target_motion_mode": "flee_persistence",
        "minimum_clearance_m": 1.0,
        "pairwise_ttc_s": 2.0,
        "uncertainty": 0.05,
        "cbf_risk": 0.10,
        "candidate_separation_m": 0.30,
    }
    cases = {
        "ood": ({**base, "ood": True}, "ood"),
        "stale": ({**base, "observation_age_steps": ledger.maximum_observation_age_steps + 1.0}, "stale_observation"),
        "non_finite": ({**base, "uncertainty": float("nan")}, "non_finite_context"),
        "uncertainty_high": ({**base, "uncertainty": ledger.safe_hold_uncertainty_threshold + 0.01}, "uncertainty_high"),
        "joint_ttc_cbf_risk": ({**base, "pairwise_ttc_s": ledger.safe_hold_ttc_seconds - 0.01, "cbf_risk": 0.80}, "joint_ttc_cbf_risk"),
        "unknown_horizon": (base, "missing_bucket"),
    }
    results: dict[str, Any] = {}
    for name, (context, expected) in cases.items():
        horizon = 999 if name == "unknown_horizon" else 2
        decision = ledger.decision(horizon, context)
        results[name] = {
            "state": decision.state,
            "fallback_reason": decision.fallback_reason,
            "passed": bool(decision.state == "safe_hold" and decision.fallback_reason == expected),
        }
    after = sha256(ledger_path)
    return {
        "ledger_sha256_before": before,
        "ledger_sha256_after": after,
        "immutable_file_hash": before == after,
        "cases": results,
        "all_required_fallbacks_pass": bool(all(item["passed"] for item in results.values())),
        "minimum_credit": ledger.minimum_credit,
        "minimum_sample_count": ledger.minimum_sample_count,
    }


def _write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/ledger_alignment", json.dumps(report["policy"], indent=2, sort_keys=True), 0)
        writer.add_text("Provenance/inputs", json.dumps(report["inputs"], indent=2, sort_keys=True), 0)
        writer.add_text("Reliability/fault_audit", json.dumps(report["fault_audit"], indent=2, sort_keys=True), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2, sort_keys=True), 0)
        for variant, stats in report["by_variant"].items():
            coverage = stats["coverage"]
            writer.add_scalar(f"Reliability/{variant}/high_decisions", coverage["high_decisions"], 0)
            writer.add_scalar(f"Reliability/{variant}/low_decisions", coverage["low_decisions"], 0)
            for bucket, values in stats["buckets"].items():
                writer.add_scalar(f"Reliability/{variant}/{bucket}_safety_failure_rate", float(values["failure_rate"] or 0.0), 0)
                writer.add_scalar(f"Reliability/{variant}/{bucket}_safe_capture_rate", float(values["safe_capture_rate"] or 0.0), 0)
        for name, result in report["fault_audit"]["cases"].items():
            writer.add_scalar(f"Fault/{name}/passed", float(bool(result["passed"])), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/ledger_alignment/text_summary",
        "Provenance/inputs/text_summary",
        "Reliability/fault_audit/text_summary",
        "Gates/status/text_summary",
    }
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    missing = sorted(required.difference(tags.get("tensors", [])))
    if missing or not events:
        raise ValueError(f"Ledger alignment TensorBoard validation failed: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", []))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settled-run", type=Path, action="append", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    parser.add_argument("--min-bucket-decisions", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("Ledger alignment audit requires --development-only.")
    if args.min_bucket_decisions <= 0:
        raise ValueError("min-bucket-decisions must be positive.")
    protocol = yaml.safe_load(args.protocol.resolve().read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("Ledger alignment audit requires a closed development protocol.")
    if not args.ledger.resolve().is_file():
        raise FileNotFoundError(args.ledger)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    ledger_payload = _json(args.ledger)
    ledger = SafeCaptureReliabilityLedger(ledger_payload)
    run_audits: list[dict[str, Any]] = []
    by_variant_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest_hashes: set[str] = set()
    protocol_hashes: set[str] = set()
    for run_dir in args.settled_run:
        audit, rows = _read_rows(run_dir.resolve())
        variant_values = {str(row.get("variant")) for row in rows}
        if len(variant_values) != 1:
            raise ValueError(f"Settled run contains multiple variants: {run_dir}")
        variant = next(iter(variant_values))
        manifest_hashes.add(str(audit.get("inputs", {}).get("scene_manifest_sha256")))
        protocol_hashes.add(str(audit.get("inputs", {}).get("protocol_sha256")))
        ledger_enabled = bool(audit.get("runs") and audit["runs"][0].get("ledger_sha256"))
        run_audits.append({"run_dir": str(run_dir.resolve()), "variant": variant, "ledger_enabled": ledger_enabled, "decision_count": len(rows), "all_gates_pass": bool(audit.get("all_gates_pass"))})
        if ledger_enabled:
            by_variant_rows[variant].extend(rows)
    fault_audit = _fault_audit(args.ledger.resolve())
    by_variant = {variant: _credit_stats(rows, ledger.minimum_credit) for variant, rows in sorted(by_variant_rows.items())}
    coverage_sufficient = all(
        stats["coverage"]["high_decisions"] >= args.min_bucket_decisions
        and stats["coverage"]["low_decisions"] >= args.min_bucket_decisions
        for stats in by_variant.values()
        if stats["coverage"]["high_decisions"] > 0 or stats["coverage"]["low_decisions"] > 0
    )
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "scene_manifest_shared": len(manifest_hashes) == 1 and None not in manifest_hashes,
        "protocol_hash_consistent": len(protocol_hashes) == 1 and None not in protocol_hashes,
        "source_t2_gates_pass": all(run["all_gates_pass"] for run in run_audits),
        "ledger_file_immutable": fault_audit["immutable_file_hash"],
        "fault_fallbacks_pass": fault_audit["all_required_fallbacks_pass"],
        "credit_bucket_coverage_sufficient": bool(coverage_sufficient),
        "high_credit_failure_gate_evaluable": all(
            stats.get("high_credit_failure_not_above_low_credit") is not None for stats in by_variant.values()
        ),
    }
    report: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_v5_ledger_alignment",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "development_only": True,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "policy": {
            "minimum_credit": ledger.minimum_credit,
            "minimum_sample_count": ledger.minimum_sample_count,
            "min_bucket_decisions": args.min_bucket_decisions,
            "failure_label": "not selected_settled_safety_ok",
            "safe_capture_is_separate_diagnostic": True,
            "low_credit_coverage_required": True,
            "online_ledger_update": False,
        },
        "inputs": {
            "settled_runs": [str(path.resolve()) for path in args.settled_run],
            "ledger": str(args.ledger.resolve()),
            "ledger_sha256": sha256(args.ledger.resolve()),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": sha256(args.protocol.resolve()),
            "scene_manifest_sha256": next(iter(manifest_hashes)) if len(manifest_hashes) == 1 else None,
        },
        "runs": run_audits,
        "by_variant": by_variant,
        "fault_audit": fault_audit,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "source_hashes": {"scripts/audit_jepa_safe_capture_v5_ledger_alignment.py": sha256(Path(__file__).resolve())},
        },
    }
    report["tensorboard"] = _write_tensorboard(args.tensorboard_logdir, report)
    (args.output_dir.resolve() / "ledger_alignment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# T3 Reliability Ledger Alignment Audit",
        "",
        "> Development-only audit; low-credit absence is reported as insufficient evidence, not success.",
        "",
        f"All gates pass: `{report['all_gates_pass']}`.",
        "",
        "| Variant | Decisions | High-credit decisions | Low/missing decisions | High-credit failure | Low/missing failure | Evaluable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, stats in sorted(by_variant.items()):
        high = stats["buckets"].get("high", {})
        low = stats["buckets"].get("low_or_missing", {})
        lines.append(
            f"| {variant} | {sum(item['decisions'] for item in stats['buckets'].values())} | "
            f"{high.get('decisions', 0)} | {low.get('decisions', 0)} | "
            f"{high.get('failure_rate') if high.get('failure_rate') is not None else float('nan'):.3f} | "
            f"{low.get('failure_rate') if low.get('failure_rate') is not None else float('nan'):.3f} | "
            f"{stats.get('high_credit_failure_not_above_low_credit') is not None} |"
        )
    lines.extend(
        [
            "",
            "Fault cases: " + ", ".join(f"{name}={value['state']}/{value['fallback_reason']}" for name, value in fault_audit["cases"].items()),
            "",
            "`locked_test_opened=false`; T3 must obtain independent low-credit/fallback coverage before claiming a reliability improvement.",
        ]
    )
    (args.output_dir.resolve() / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "gates": gates, "by_variant": by_variant, "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
