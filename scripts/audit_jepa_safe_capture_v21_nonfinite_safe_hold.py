"""Audit non-finite JEPA predictions through the production safe-hold path.

This development-only audit injects non-finite values into clearance,
uncertainty, and auxiliary heads, then verifies that the ranker returns an
explicit ``safe_hold`` decision and that the selected request is sent through
the same Joint CBF-QP used by normal execution.  It never opens a locked-test
split and never executes a raw/unverified request.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.jepa_safe_capture_candidates import (  # noqa: E402
    CANDIDATE_LABELS,
    SafeCaptureCandidateBatch,
)
from encirclement3d.jepa_safe_capture_ranker import SafeCaptureJEPARanker, SafeCaptureRankerConfig  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402


AUDIT_TYPE = "jepa_safe_capture_v21_nonfinite_prediction_safe_hold"
FAULTS = (
    "nan_clearance",
    "inf_uncertainty",
    "nan_auxiliary",
    "raised_nonfinite",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable({name: getattr(value, name) for name in value.__dataclass_fields__})
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class _FaultHistory:
    """Small deterministic history adapter that injects one prediction fault."""

    defender_count = 4
    predictor = SimpleNamespace(action_dim=3, input_dim=63, horizon_count=4)

    def __init__(self, fault: str) -> None:
        self.fault = fault

    def predict_candidates_multitask(self, actions: np.ndarray, *, horizon_index: int):
        if horizon_index != 2:
            raise AssertionError(f"unexpected horizon: {horizon_index}")
        count = int(actions.shape[0])
        means = np.zeros((count, self.defender_count, 3), dtype=np.float32)
        means[..., 0] = 0.10
        std = np.full((count, self.defender_count, 3), 0.01, dtype=np.float32)
        auxiliary = {
            "obstacle_clearance_lower_quantile": np.ones((count, self.defender_count), dtype=np.float32),
            "inter_agent_clearance_lower_quantile": np.ones((count, self.defender_count), dtype=np.float32),
            "pairwise_ttc": np.full((count, self.defender_count), 10.0, dtype=np.float32),
            "target_visibility_logit": np.full((count, self.defender_count), 10.0, dtype=np.float32),
            "cbf_intervention_logit": np.full((count, self.defender_count), -10.0, dtype=np.float32),
            "cbf_correction": np.zeros((count, self.defender_count), dtype=np.float32),
            "cbf_qp_feasibility_logit": np.full((count, self.defender_count), 10.0, dtype=np.float32),
            "action_consistency": np.asarray(actions, dtype=np.float32).copy(),
        }
        if self.fault == "nan_clearance":
            auxiliary["obstacle_clearance_lower_quantile"][0, 0] = np.nan
        elif self.fault == "inf_uncertainty":
            std[0, 0, 0] = np.inf
        elif self.fault == "nan_auxiliary":
            auxiliary["target_visibility_logit"][0, 0] = np.nan
        elif self.fault == "raised_nonfinite":
            raise RuntimeError("Safe-capture v2 candidate prediction emitted non-finite values.")
        else:  # pragma: no cover - fixed audit matrix guard
            raise AssertionError(f"Unknown fault: {self.fault}")
        return means, std, auxiliary


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Environment config must be a mapping: {path}")
    return config


def _make_batch(defender_count: int) -> SafeCaptureCandidateBatch:
    nominal = np.zeros((defender_count, 3), dtype=np.float64)
    chunks = np.repeat(nominal[None, None, :, :], 5, axis=0)
    chunks = np.repeat(chunks, 3, axis=1)
    chunks[1, :, :, 0] = 0.10
    return SafeCaptureCandidateBatch(
        chunks=chunks,
        labels=CANDIDATE_LABELS,
        valid_mask=np.ones(5, dtype=bool),
        rejection_reasons=tuple(() for _ in range(5)),
    )


def _raw_unverified(diagnostics: Any) -> bool:
    if bool(getattr(diagnostics, "verified_feasible", False)):
        return False
    return str(getattr(diagnostics, "fallback_mode", "")) not in {
        "safe_hold",
        "nominal_cbf",
        "controlled_abort",
    }


def _run_case(config: Mapping[str, Any], fault: str) -> dict[str, Any]:
    env = CaptureRadiusPursuit3DEnv(
        copy.deepcopy(dict(config)),
        obstacle_count=0,
        target_speed_scale=0.45,
    )
    observation = env.reset(seed=20260911)
    defender_count = int(env.n_defenders)
    nominal = np.zeros((defender_count, 3), dtype=np.float64)
    batch = _make_batch(defender_count)
    ranker = SafeCaptureJEPARanker(
        _FaultHistory(fault),
        config=SafeCaptureRankerConfig(horizon_index=2, horizon_seconds=0.30, position_extent_m=10.0),
    )
    ranking = ranker.rank(observation, batch)
    safety_filter = JointCBFQPSafetyFilter(env)
    action, diagnostics = safety_filter.filter(
        ranking.selected_action,
        observation,
        nominal_actions=nominal,
        execution_mode=ranking.execution_mode,
    )
    serialized_trace = ranking.trace.as_dict()
    serialized_trace_json = json.dumps(_jsonable(serialized_trace), allow_nan=False)
    fallback_mode = str(diagnostics.fallback_mode)
    allowed_fallbacks = {"safe_hold", "nominal_cbf", "controlled_abort"}
    passed = all(
        (
            ranking.execution_mode == "safe_hold",
            ranking.fallback_reason == "non_finite_prediction",
            not any(ranking.trace.eligible_mask),
            np.isfinite(action).all(),
            not _raw_unverified(diagnostics),
            fallback_mode in allowed_fallbacks,
            serialized_trace_json,
        )
    )
    return {
        "fault": fault,
        "ranking": {
            "execution_mode": ranking.execution_mode,
            "fallback_reason": ranking.fallback_reason,
            "selected_index": int(ranking.selected_index),
            "prediction_fault_fields": list(ranking.trace.prediction_fault_fields),
            "eligible_mask": list(ranking.trace.eligible_mask),
            "trace": _jsonable(serialized_trace),
        },
        "cbf": {
            "execution_mode_requested": ranking.execution_mode,
            "fallback_mode": fallback_mode,
            "verified_feasible": bool(diagnostics.verified_feasible),
            "solver_status": str(diagnostics.solver_status),
            "infeasible": bool(diagnostics.infeasible),
            "timed_out": bool(diagnostics.timed_out),
            "used_fallback": bool(diagnostics.used_fallback),
            "action_finite": bool(np.isfinite(action).all()),
            "raw_unverified_executed": bool(_raw_unverified(diagnostics)),
            "minimum_constraint_value": _jsonable(float(diagnostics.minimum_constraint_value)),
            "action_correction_norm": _jsonable(float(diagnostics.action_correction_norm)),
            "solve_latency_ms": _jsonable(float(diagnostics.solve_latency_ms)),
            "active_constraints": list(diagnostics.active_constraints),
        },
        "passed": bool(passed),
    }


def run_audit(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    cases = [_run_case(config, fault) for fault in FAULTS]
    return {
        "audit_type": AUDIT_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "online_target_truth": False,
        "environment_config": str(config_path.resolve()),
        "environment_config_sha256": sha256(config_path),
        "cases": cases,
        "gates": {
            "all_cases_pass": all(bool(case["passed"]) for case in cases),
            "all_route_to_safe_hold": all(case["ranking"]["execution_mode"] == "safe_hold" for case in cases),
            "all_prediction_fault_reasons_explicit": all(
                case["ranking"]["fallback_reason"] == "non_finite_prediction" for case in cases
            ),
            "all_actions_finite": all(bool(case["cbf"]["action_finite"]) for case in cases),
            "raw_unverified_executed_count_zero": sum(
                bool(case["cbf"]["raw_unverified_executed"]) for case in cases
            )
            == 0,
            "all_cbf_fallbacks_explicit": all(case["cbf"]["fallback_mode"] in {"safe_hold", "nominal_cbf", "controlled_abort"} for case in cases),
        },
        "provenance": {
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def write_outputs(report: dict[str, Any], output_dir: Path, tensorboard_dir: Path) -> None:
    output_dir = output_dir.resolve()
    tensorboard_dir = tensorboard_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {tensorboard_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "audit_type": AUDIT_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "online_target_truth": False,
        "command": " ".join(sys.argv),
        "provenance": report["provenance"],
        "environment_config": report["environment_config"],
        "environment_config_sha256": report["environment_config_sha256"],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "fault_injection.json").write_text(
        json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# v21 non-finite JEPA safe-hold fault audit",
        "",
        "`development_only=true`; `locked_test_opened=false`; `online_target_truth=false`.",
        "",
        "| Fault | Fields | Rank mode | Reason | CBF fallback | Verified | Raw/unverified | Pass |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| `{case['fault']}` | `{','.join(case['ranking']['prediction_fault_fields'])}` | "
            f"`{case['ranking']['execution_mode']}` | `{case['ranking']['fallback_reason']}` | "
            f"`{case['cbf']['fallback_mode']}` | {str(case['cbf']['verified_feasible']).lower()} | "
            f"{str(case['cbf']['raw_unverified_executed']).lower()} | {str(case['passed']).lower()} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, value in report["gates"].items():
        lines.append(f"- {name}: `{str(value).lower()}`")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/fault_audit", json.dumps(metadata, indent=2), 0)
        writer.add_text("Provenance/report", json.dumps(report["provenance"], indent=2), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2), 0)
        for index, case in enumerate(report["cases"]):
            writer.add_scalar("Ranker/safe_hold", float(case["ranking"]["execution_mode"] == "safe_hold"), index)
            writer.add_scalar("Ranker/raw_unverified", float(case["cbf"]["raw_unverified_executed"]), index)
            writer.add_scalar("CBF/action_finite", float(case["cbf"]["action_finite"]), index)
            writer.add_scalar("CBF/verified_feasible", float(case["cbf"]["verified_feasible"]), index)
            writer.add_scalar("CBF/solve_latency_ms", float(case["cbf"]["solve_latency_ms"] or 0.0), index)
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/fault_audit/text_summary",
        "Provenance/report/text_summary",
        "Gates/status/text_summary",
        "Ranker/safe_hold",
        "Ranker/raw_unverified",
    }
    missing = sorted(required.difference(set(tags.get("tensors", [])) | set(tags.get("scalars", []))))
    event_files = sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents.*"))
    if missing or not event_files:
        raise ValueError(f"TensorBoard validation failed: missing={missing}, events={event_files}")
    report["tensorboard"] = {
        "logdir": str(tensorboard_dir),
        "event_files": event_files,
        "required_tags": sorted(required),
    }
    (output_dir / "fault_injection.json").write_text(
        json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest = {
        str(path.relative_to(output_dir)).replace("\\", "/"): sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "hash_manifest.json"
    }
    (output_dir / "hash_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=PROJECT_ROOT / "configs/capture_radius_pursuit_central_v4_flee.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("This audit requires --development-only")
    config_path = args.environment_config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    report = run_audit(config_path)
    write_outputs(report, args.output_dir, args.tensorboard_logdir)
    print(json.dumps(_jsonable({"gates": report["gates"], "tensorboard": report["tensorboard"]}), indent=2))


if __name__ == "__main__":
    main()
