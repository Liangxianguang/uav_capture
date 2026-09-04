"""Deterministic reliability-ledger and Joint CBF fault-injection audit.

The matrix exercises explicit failure paths on the development environment.
It is not a task success-rate evaluation and never opens a locked-test split.
Every unverified CBF result is checked for finite fallback output and for zero
execution of the original raw request.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, CylinderObstacle
from encirclement3d.reliability import SafeCaptureReliabilityLedger, make_safe_capture_global_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_TYPE = "jepa_safe_capture_v3_wp5_reliability_cbf_fault_injection"


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Environment config must be a mapping: {path}")
    return value


def _env(config_path: Path) -> CaptureRadiusPursuit3DEnv:
    env = CaptureRadiusPursuit3DEnv(copy.deepcopy(_config(config_path)), obstacle_count=0, target_speed_scale=0.45)
    env.reset(20260911)
    env.defender_positions = np.array(
        [[5.0, 0.0, 4.0], [0.0, 5.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]],
        dtype=np.float64,
    )
    env.defender_velocities = np.zeros((4, 3), dtype=np.float64)
    return env


def _ledger() -> SafeCaptureReliabilityLedger:
    return SafeCaptureReliabilityLedger(
        {
            "ledger_type": SafeCaptureReliabilityLedger.LEDGER_TYPE_V3,
            "ledger_version": 3,
            "not_a_locked_test": True,
            "immutable_after_calibration": True,
            "source": {"checkpoint_sha256": "a" * 64, "calibration_dataset_sha256": "b" * 64},
            "entries": {make_safe_capture_global_key(2): {"credit": 0.90, "sample_count": 1000}},
            "decision_policy": {
                "states": ["trusted", "fallback_nominal", "safe_hold"],
                "minimum_sample_count": 128,
                "minimum_credit": 0.65,
                "maximum_observation_age_steps": 45.0,
                "safe_hold_uncertainty_threshold": 0.40,
                "safe_hold_ttc_seconds": 0.30,
            },
        }
    )


def _ledger_context() -> dict[str, Any]:
    return {
        "visibility_condition": 1.0,
        "observation_age_steps": 0.0,
        "obstacle_count": 0,
        "layout_signature": "fault_injection",
        "target_motion_mode": "flee_persistence",
        "minimum_clearance_m": 2.0,
        "pairwise_ttc_s": 10.0,
        "uncertainty": 0.05,
        "cbf_risk": 0.10,
        "candidate_separation_m": 0.3,
    }


def _ledger_cases() -> list[dict[str, Any]]:
    cases = []
    for name, updates, expected in (
        ("trusted_baseline", {}, "trusted"),
        ("ood", {"ood": True}, "safe_hold"),
        ("stale_observation", {"observation_age_steps": 46.0}, "safe_hold"),
        ("high_uncertainty", {"uncertainty": 0.5}, "safe_hold"),
        ("nonfinite_context", {"uncertainty": float("nan")}, "safe_hold"),
    ):
        context = _ledger_context()
        context.update(updates)
        decision = _ledger().decision(2, context)
        cases.append(
            {
                "name": name,
                "state": decision.state,
                "fallback_reason": decision.fallback_reason,
                "credit": float(decision.credit),
                "sample_count": int(decision.sample_count),
                "expected_state": expected,
                "passed": decision.state == expected,
            }
        )
    return cases


def _cbf_case(name: str, config_path: Path) -> dict[str, Any]:
    env = _env(config_path)
    if name == "state_violation":
        env.obstacles = [CylinderObstacle(np.array([5.0, 0.0]), 1.0, 5.0)]
        env.defender_positions[0] = np.array([5.1, 0.0, 4.0])
    elif name == "motion_infeasible":
        env.defender_velocities[:] = 20.0
    observation = env.observe()
    requested = np.ones((4, 3), dtype=np.float64)
    if name == "nonfinite_request":
        requested[0, 0] = np.nan
    max_latency = 1e-12 if name == "solver_timeout" else 100.0
    started = time.perf_counter()
    action, diagnostics = JointCBFQPSafetyFilter(env, max_latency_ms=max_latency).filter(
        requested, observation, nominal_actions=np.full((4, 3), 0.1, dtype=np.float64),
        execution_mode="safe_hold" if name == "safe_hold" else "normal",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    same_as_raw = bool(np.isfinite(requested).all() and np.allclose(action, requested, rtol=0.0, atol=1e-10))
    failed = not diagnostics.verified_feasible
    passed = bool(np.isfinite(action).all()) and (not failed or not same_as_raw)
    return {
        "name": name,
        "solver_status": diagnostics.solver_status,
        "solver_success": bool(diagnostics.solver_success),
        "infeasible": bool(diagnostics.infeasible),
        "timed_out": bool(diagnostics.timed_out),
        "used_fallback": bool(diagnostics.used_fallback),
        "fallback_mode": diagnostics.fallback_mode,
        "verified_feasible": bool(diagnostics.verified_feasible),
        "requested_action_finite": bool(diagnostics.requested_action_finite),
        "raw_unverified_executed": bool(failed and same_as_raw),
        "action_finite": bool(np.isfinite(action).all()),
        "action_correction_norm": float(diagnostics.action_correction_norm) if math.isfinite(float(diagnostics.action_correction_norm)) else None,
        "minimum_constraint_value": float(diagnostics.minimum_constraint_value) if math.isfinite(float(diagnostics.minimum_constraint_value)) else None,
        "solve_latency_ms": float(diagnostics.solve_latency_ms),
        "end_to_end_latency_ms": float(elapsed_ms),
        "active_constraints": list(diagnostics.active_constraints),
        "passed": passed,
    }


def run_fault_injection(config_path: Path) -> dict[str, Any]:
    cbf_cases = [_cbf_case(name, config_path) for name in ("nominal_feasible", "safe_hold", "nonfinite_request", "solver_timeout", "state_violation", "motion_infeasible")]
    ledger_cases = _ledger_cases()
    all_cbf = all(item["passed"] for item in cbf_cases)
    all_ledger = all(item["passed"] for item in ledger_cases)
    failed_cbf = [item for item in cbf_cases if not item["verified_feasible"]]
    return {
        "audit_type": AUDIT_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "cbf_cases": cbf_cases,
        "ledger_cases": ledger_cases,
        "gates": {
            "all_cbf_cases_pass": all_cbf,
            "all_ledger_cases_pass": all_ledger,
            "all_actions_finite": all(item["action_finite"] for item in cbf_cases),
            "raw_unverified_executed_count_zero": sum(bool(item["raw_unverified_executed"]) for item in cbf_cases) == 0,
            "failed_cbf_cases_have_explicit_fallback": all(bool(item["used_fallback"]) and item["fallback_mode"] for item in failed_cbf),
            "cbf_end_to_end_p95_under_100ms": float(np.quantile([item["end_to_end_latency_ms"] for item in cbf_cases], 0.95)) <= 100.0,
        },
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def _write(report: dict[str, Any], output_dir: Path, tensorboard_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {tensorboard_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fault_injection.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = [
        "# WP-D/F Reliability and CBF Fault Injection",
        "",
        "**Status:** development-only; `locked_test_opened=false`",
        "",
        "| CBF scenario | status | fallback | verified | raw/unverified executed | finite action | latency ms | pass |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["cbf_cases"]:
        lines.append(f"| `{item['name']}` | `{item['solver_status']}` | `{item['fallback_mode']}` | {str(item['verified_feasible']).lower()} | {str(item['raw_unverified_executed']).lower()} | {str(item['action_finite']).lower()} | {item['end_to_end_latency_ms']:.3f} | {str(item['passed']).lower()} |")
    lines.extend(["", "| Ledger scenario | state | reason | expected | pass |", "|---|---|---|---|---:|"])
    for item in report["ledger_cases"]:
        lines.append(f"| `{item['name']}` | `{item['state']}` | `{item['fallback_reason']}` | `{item['expected_state']}` | {str(item['passed']).lower()} |")
    lines.extend(["", "## Gates", ""])
    for name, value in report["gates"].items():
        lines.append(f"- {name}: `{str(value).lower()}`")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/fault_injection", json.dumps({"audit_type": AUDIT_TYPE, "development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/report", json.dumps(report["provenance"], indent=2), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2), 0)
        for index, item in enumerate(report["cbf_cases"]):
            writer.add_scalar("CBF/end_to_end_latency_ms", item["end_to_end_latency_ms"], index)
            writer.add_scalar("CBF/verified_feasible", float(item["verified_feasible"]), index)
            writer.add_scalar("CBF/raw_unverified_executed", float(item["raw_unverified_executed"]), index)
            writer.add_scalar("CBF/action_finite", float(item["action_finite"]), index)
        for index, item in enumerate(report["ledger_cases"]):
            writer.add_scalar("Ledger/state_expected", float(item["passed"]), index)
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {"Config/fault_injection/text_summary", "Provenance/report/text_summary", "Gates/status/text_summary"}
    missing = sorted(required.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"Fault-injection TensorBoard validation failed: missing={missing}, events={events}")
    report["tensorboard"] = {"logdir": str(tensorboard_dir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", [])), "required_provenance": True}
    (output_dir / "fault_injection.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    manifest = {str(path.relative_to(output_dir)).replace("\\", "/"): sha256(path) for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "hash_manifest.json"}
    (output_dir / "hash_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-config", type=Path, default=PROJECT_ROOT / "configs/capture_radius_pursuit_central_v4_flee.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("Fault injection requires --development-only")
    config = args.environment_config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    report = run_fault_injection(config)
    _write(report, args.output_dir, args.tensorboard_logdir)
    print(json.dumps({"gates": report["gates"], "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
