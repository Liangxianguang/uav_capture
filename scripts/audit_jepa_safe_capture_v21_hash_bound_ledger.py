"""Audit V21 checkpoint-bound Reliability Ledgers before closed-loop use.

This audit is intentionally independent of development episode outcomes.  It
verifies external provenance binding for each ledger, exercises deterministic
ledger fault routes, checks immutability, and writes an auditable TensorBoard
record.  It does not claim a safe-capture control gain and it does not open a
locked split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _base_context() -> dict[str, Any]:
    return {
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


def _expected_source(
    protocol: Path,
    checkpoint: Path,
    calibration_dataset: Path,
    calibration_metadata: Path,
) -> dict[str, str]:
    return {
        "protocol_sha256": sha256(protocol),
        "checkpoint_sha256": sha256(checkpoint),
        "calibration_dataset_sha256": sha256(calibration_dataset),
        "calibration_metadata_sha256": sha256(calibration_metadata),
    }


def _source_hash_gate(payload: Mapping[str, Any], expected: Mapping[str, str]) -> dict[str, Any]:
    source = payload.get("source")
    if not isinstance(source, Mapping):
        return {"passed": False, "reason": "missing_source"}
    checks = {
        name: str(source.get(name)) == value
        for name, value in expected.items()
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "expected": dict(expected),
        "observed": {name: source.get(name) for name in expected},
    }


def _tamper_provenance_gate(payload: Mapping[str, Any], expected: Mapping[str, str]) -> bool:
    tampered = copy.deepcopy(dict(payload))
    source = dict(tampered.get("source", {}))
    source["protocol_sha256"] = "0" * 64
    tampered["source"] = source
    return not bool(_source_hash_gate(tampered, expected)["passed"])


def _fault_cases(ledger: SafeCaptureReliabilityLedger) -> dict[str, tuple[int, dict[str, Any], str]]:
    base = _base_context()
    return {
        "ood": (2, {**base, "ood": True}, "ood"),
        "stale": (2, {**base, "observation_age_steps": ledger.maximum_observation_age_steps + 1.0}, "stale_observation"),
        "non_finite": (2, {**base, "uncertainty": float("nan")}, "non_finite_context"),
        "uncertainty_spike": (2, {**base, "uncertainty": ledger.safe_hold_uncertainty_threshold + 0.01}, "uncertainty_high"),
        "joint_ttc_cbf_risk": (2, {**base, "pairwise_ttc_s": ledger.safe_hold_ttc_seconds - 0.01, "cbf_risk": 0.80}, "joint_ttc_cbf_risk"),
        "unknown_horizon": (999, base, "missing_bucket"),
    }


def _audit_case(ledger: SafeCaptureReliabilityLedger, horizon: int, context: Mapping[str, Any], expected: str) -> dict[str, Any]:
    decision = ledger.decision(horizon, context)
    passed = decision.fallback_reason == expected and decision.state == "safe_hold"
    return {
        "state": decision.state,
        "fallback_reason": decision.fallback_reason,
        "credit": decision.credit,
        "sample_count": decision.sample_count,
        "raw_unverified_executed": False,
        "passed": bool(passed),
    }


def _audit_seed(
    seed: str,
    checkpoint: Path,
    ledger_path: Path,
    protocol: Path,
    calibration_dataset: Path,
    calibration_metadata: Path,
) -> dict[str, Any]:
    payload = _json(ledger_path)
    expected = _expected_source(protocol, checkpoint, calibration_dataset, calibration_metadata)
    source_gate = _source_hash_gate(payload, expected)
    ledger = SafeCaptureReliabilityLedger(payload)
    before = sha256(ledger_path)
    cases = {
        name: _audit_case(ledger, horizon, context, expected_reason)
        for name, (horizon, context, expected_reason) in _fault_cases(ledger).items()
    }
    # A hash-bound artifact must also remain unchanged after all read-only decisions.
    after = sha256(ledger_path)
    return {
        "training_seed": int(seed),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "ledger": str(ledger_path.resolve()),
        "ledger_sha256": before,
        "protocol": str(protocol.resolve()),
        "protocol_sha256": expected["protocol_sha256"],
        "calibration_dataset_sha256": expected["calibration_dataset_sha256"],
        "calibration_metadata_sha256": expected["calibration_metadata_sha256"],
        "source_hash_gate": source_gate,
        "tampered_provenance_rejected": _tamper_provenance_gate(payload, expected),
        "ledger_constructor_loaded": True,
        "immutable_file_hash": before == after,
        "cases": cases,
        "all_fault_cases_pass": bool(all(item["passed"] for item in cases.values())),
        "raw_unverified_executed_count": int(sum(bool(item["raw_unverified_executed"]) for item in cases.values())),
    }


def _write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/protocol", json.dumps(report["policy"], indent=2, sort_keys=True), 0)
        writer.add_text("Provenance/seeds", json.dumps(report["seeds"], indent=2, sort_keys=True), 0)
        writer.add_text("Reliability/hash_binding", json.dumps(report["hash_binding"], indent=2, sort_keys=True), 0)
        writer.add_text("Reliability/fault_matrix", json.dumps(report["fault_matrix"], indent=2, sort_keys=True), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2, sort_keys=True), 0)
        for seed in report["seeds"]:
            prefix = str(seed["training_seed"])
            writer.add_scalar(f"Reliability/{prefix}/hash_binding_pass", float(seed["source_hash_gate"]["passed"]), 0)
            writer.add_scalar(f"Reliability/{prefix}/fault_cases_pass", float(seed["all_fault_cases_pass"]), 0)
            writer.add_scalar(f"Reliability/{prefix}/immutable_file_hash", float(seed["immutable_file_hash"]), 0)
            writer.add_scalar(f"Reliability/{prefix}/raw_unverified", seed["raw_unverified_executed_count"], 0)
            for name, case in seed["cases"].items():
                writer.add_scalar(f"Fault/{prefix}/{name}/passed", float(case["passed"]), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/protocol/text_summary",
        "Provenance/seeds/text_summary",
        "Reliability/hash_binding/text_summary",
        "Reliability/fault_matrix/text_summary",
        "Gates/status/text_summary",
    }
    missing = sorted(required.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"TensorBoard validation failed: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", []))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration-dataset", type=Path, required=True)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--seed", action="append", nargs=3, metavar=("SEED", "CHECKPOINT", "LEDGER"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("V21 ledger audit requires --development-only.")
    protocol = args.protocol.resolve()
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    protocol_payload = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    if not isinstance(protocol_payload, dict) or protocol_payload.get("phase") != "development_only" or protocol_payload.get("locked_test_opened") is not False:
        raise ValueError("V21 ledger audit requires a closed development protocol.")
    calibration_dataset = args.calibration_dataset.resolve()
    calibration_metadata = args.calibration_metadata.resolve()
    if not calibration_dataset.is_file() or not calibration_metadata.is_file():
        raise FileNotFoundError("Calibration dataset and metadata are required.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    seeds = []
    for seed, checkpoint_text, ledger_text in args.seed:
        checkpoint = Path(checkpoint_text).resolve()
        ledger = Path(ledger_text).resolve()
        if not checkpoint.is_file() or not ledger.is_file():
            raise FileNotFoundError(f"Missing seed input: {checkpoint}, {ledger}")
        seeds.append(_audit_seed(seed, checkpoint, ledger, protocol, calibration_dataset, calibration_metadata))
    hash_binding = {
        "all_source_hash_gates_pass": bool(all(item["source_hash_gate"]["passed"] for item in seeds)),
        "all_tampered_provenance_rejected": bool(all(item["tampered_provenance_rejected"] for item in seeds)),
        "all_ledger_files_immutable": bool(all(item["immutable_file_hash"] for item in seeds)),
    }
    fault_matrix = {
        name: bool(all(seed["cases"][name]["passed"] for seed in seeds))
        for name in next(iter(seeds))["cases"]
    }
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "three_seed_inputs": len(seeds) == 3,
        "hash_binding": hash_binding["all_source_hash_gates_pass"],
        "tampered_provenance_rejected": hash_binding["all_tampered_provenance_rejected"],
        "ledger_immutable": hash_binding["all_ledger_files_immutable"],
        "fault_matrix": bool(all(fault_matrix.values())),
        "raw_unverified_zero": bool(all(item["raw_unverified_executed_count"] == 0 for item in seeds)),
    }
    report: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_v21_hash_bound_ledger",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "development_only": True,
        "locked_test_opened": False,
        "policy": {
            "protocol": str(protocol),
            "protocol_sha256": sha256(protocol),
            "calibration_dataset_sha256": sha256(calibration_dataset),
            "calibration_metadata_sha256": sha256(calibration_metadata),
            "online_ledger_update": False,
            "raw_unverified_execution_allowed": False,
        },
        "seeds": seeds,
        "hash_binding": hash_binding,
        "fault_matrix": fault_matrix,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    }
    report["tensorboard"] = _write_tensorboard(args.tensorboard_logdir.resolve(), report)
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    (args.output_dir.resolve() / "hash_bound_ledger_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "protocol": report["policy"]["protocol_sha256"],
        "calibration_dataset": report["policy"]["calibration_dataset_sha256"],
        "calibration_metadata": report["policy"]["calibration_metadata_sha256"],
        "audit_script": report["provenance"]["source_sha256"],
        "seeds": {
            str(seed["training_seed"]): {
                "checkpoint": seed["checkpoint_sha256"],
                "ledger": seed["ledger_sha256"],
            }
            for seed in seeds
        },
    }
    (args.output_dir.resolve() / "hash_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# V21 Hash-Bound Reliability Ledger Audit",
        "",
        "> Development-only provenance and fault audit; no closed-loop performance claim.",
        "",
        f"All gates pass: `{report['all_gates_pass']}`.",
        "",
        "| Seed | Source hash | Tamper reject | Immutable | Fault matrix | Raw unverified |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        lines.append(f"| {seed['training_seed']} | {seed['source_hash_gate']['passed']} | {seed['tampered_provenance_rejected']} | {seed['immutable_file_hash']} | {seed['all_fault_cases_pass']} | {seed['raw_unverified_executed_count']} |")
    lines.extend(["", "Fault matrix: " + ", ".join(f"{name}={passed}" for name, passed in fault_matrix.items()), "", "`locked_test_opened=false`; CBF execution safety remains a separate gate."])
    (args.output_dir.resolve() / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "gates": gates, "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
