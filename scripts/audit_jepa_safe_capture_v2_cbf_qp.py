"""Audit the joint CBF-QP safety boundary on deterministic environment cases.

This is a P5 development audit. It does not train a policy, use target ground
truth, or open a locked test. Every requested action is checked through the
same joint filter and every failed solve must produce an explicit fallback.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, CylinderObstacle  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def _fresh(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {label}: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_env(config: dict[str, Any], seed: int, obstacles: list[CylinderObstacle] | None = None) -> CaptureRadiusPursuit3DEnv:
    env = CaptureRadiusPursuit3DEnv(copy.deepcopy(config), obstacle_count=0, target_speed_scale=0.45)
    env.reset(seed)
    if obstacles is not None:
        env.obstacles = obstacles
    return env


def observation(
    env: CaptureRadiusPursuit3DEnv,
    positions: np.ndarray,
    velocities: np.ndarray | None = None,
) -> dict[str, Any]:
    env.defender_positions = np.asarray(positions, dtype=np.float64).copy()
    env.defender_velocities = (
        np.zeros_like(env.defender_positions)
        if velocities is None
        else np.asarray(velocities, dtype=np.float64).copy()
    )
    return env.observe()


def standard_positions() -> np.ndarray:
    return np.array(
        [[5.0, 0.0, 4.0], [0.0, 5.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]],
        dtype=np.float64,
    )


def run_case(
    name: str,
    env: CaptureRadiusPursuit3DEnv,
    obs: dict[str, Any],
    desired: np.ndarray,
    *,
    filter_: JointCBFQPSafetyFilter | None = None,
    nominal: np.ndarray | None = None,
) -> dict[str, Any]:
    filter_ = filter_ or JointCBFQPSafetyFilter(env)
    started = time.perf_counter()
    action, diagnostics = filter_.filter(desired, obs, nominal_actions=nominal)
    wall_latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "name": name,
        "action": action.tolist(),
        "requested_action": np.asarray(desired).tolist(),
        "solver_status": diagnostics.solver_status,
        "solver_success": diagnostics.solver_success,
        "verified_feasible": diagnostics.verified_feasible,
        "infeasible": diagnostics.infeasible,
        "timed_out": diagnostics.timed_out,
        "used_fallback": diagnostics.used_fallback,
        "fallback_mode": diagnostics.fallback_mode,
        "requested_action_finite": diagnostics.requested_action_finite,
        "action_correction_norm": diagnostics.action_correction_norm,
        "minimum_constraint_value": diagnostics.minimum_constraint_value,
        "minimum_state_clearance": diagnostics.minimum_state_clearance,
        "state_safety_violation": diagnostics.state_safety_violation,
        "solve_latency_ms": diagnostics.solve_latency_ms,
        "wall_latency_ms": wall_latency_ms,
        "active_constraints": list(diagnostics.active_constraints),
        "constraint_slacks": diagnostics.constraint_slacks,
        "task_constraint_slacks": diagnostics.task_constraint_slacks,
        "output_finite": bool(np.isfinite(action).all()),
        "raw_request_executed": bool(np.allclose(action, desired)) if np.isfinite(desired).all() else False,
    }


def build_cases(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    standard = standard_positions()
    cases: list[dict[str, Any]] = []

    env = make_env(config, seed)
    cases.append(
        {
            "name": "normal_joint",
            "env": env,
            "obs": observation(env, standard),
            "desired": np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]]),
        }
    )

    env = make_env(config, seed + 1, [CylinderObstacle(np.array([0.0, 0.0]), 1.0, 5.0)])
    cases.append(
        {
            "name": "obstacle_cylinder",
            "env": env,
            "obs": observation(env, np.array([[1.65, 0.0, 2.0], [5.0, 0.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]])),
            "desired": np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
    )

    env = make_env(
        config,
        seed + 2,
        [CylinderObstacle(np.array([0.0, 0.0]), 1.0, 5.0, "box", np.array([1.0, 2.0]))],
    )
    cases.append(
        {
            "name": "obstacle_box",
            "env": env,
            "obs": observation(env, np.array([[0.0, 2.8, 2.0], [5.0, 0.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]])),
            "desired": np.array([[0.0, -5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
    )

    env = make_env(config, seed + 3)
    cases.append(
        {
            "name": "pairwise_joint",
            "env": env,
            "obs": observation(env, np.array([[0.5, 0.0, 4.0], [-0.5, 0.0, 4.0], [5.0, 0.0, 4.0], [0.0, -5.0, 4.0]])),
            "desired": np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
    )

    env = make_env(config, seed + 4)
    cases.append(
        {
            "name": "boundary_lower",
            "env": env,
            "obs": observation(env, np.array([[-9.0, 0.0, 4.0], [4.0, 4.0, 4.0], [5.0, 0.0, 4.0], [0.0, -5.0, 4.0]])),
            "desired": np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
    )

    env = make_env(config, seed + 5)
    cases.append(
        {
            "name": "altitude_lower",
            "env": env,
            "obs": observation(env, np.array([[4.0, 4.0, 0.7], [4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [-4.0, -4.0, 4.0]])),
            "desired": np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
    )

    env = make_env(config, seed + 6)
    cases.append(
        {
            "name": "nonfinite_request",
            "env": env,
            "obs": observation(env, standard),
            "desired": np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
    )

    env = make_env(config, seed + 7)
    cases.append(
        {
            "name": "motion_infeasible",
            "env": env,
            "obs": observation(env, standard, np.full((4, 3), 20.0)),
            "desired": np.full((4, 3), -20.0),
        }
    )

    env = make_env(config, seed + 8)
    cases.append(
        {
            "name": "solver_timeout",
            "env": env,
            "obs": observation(env, standard),
            "desired": np.full((4, 3), 1.0),
            "filter": JointCBFQPSafetyFilter(env, max_latency_ms=1e-12),
        }
    )

    return cases


def write_tensorboard(report: dict[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = _fresh(logdir, "TensorBoard logdir")
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/audit", json.dumps({key: report[key] for key in ("audit_type", "locked_test_opened", "case_count", "repeats")}, indent=2), 0)
        writer.add_text("Provenance/sources", json.dumps(report["provenance"], indent=2), 0)
        writer.add_text("Provenance/environment", json.dumps(report["environment"], indent=2), 0)
        for step, row in enumerate(report["cases"]):
            writer.add_scalar("CBF/verified_feasible", float(row["verified_feasible"]), step)
            writer.add_scalar("CBF/infeasible", float(row["infeasible"]), step)
            writer.add_scalar("CBF/used_fallback", float(row["used_fallback"]), step)
            writer.add_scalar("CBF/timed_out", float(row["timed_out"]), step)
            writer.add_scalar("CBF/minimum_constraint_value", float(row["minimum_constraint_value"]), step)
            writer.add_scalar("CBF/minimum_state_clearance", float(row["minimum_state_clearance"]), step)
            writer.add_scalar("CBF/active_constraint_count", len(row["active_constraints"]), step)
            writer.add_scalar("Latency/solve_ms", float(row["solve_latency_ms"]), step)
            writer.add_scalar("Latency/wall_ms", float(row["wall_latency_ms"]), step)
        writer.add_scalar("Aggregate/all_outputs_finite", float(report["aggregate"]["all_outputs_finite"]), 0)
        writer.add_scalar("Aggregate/no_raw_request_on_failure", float(report["aggregate"]["no_raw_request_on_failure"]), 0)
        writer.add_scalar("Aggregate/zero_perturbation_exact", float(report["aggregate"]["zero_perturbation_exact"]), 0)
        writer.add_scalar("Aggregate/repeated_deterministic", float(report["aggregate"]["repeated_deterministic"]), 0)
        writer.add_scalar("Latency/p95_solve_ms", float(report["aggregate"]["p95_solve_latency_ms"]), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/audit/text_summary",
        "Provenance/sources/text_summary",
        "Provenance/environment/text_summary",
    }
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    if missing:
        raise ValueError(f"P5 TensorBoard provenance is incomplete: {missing}")
    return {
        "logdir": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_text_complete": not missing,
    }


def main() -> None:
    args = parse_args()
    if int(args.repeats) <= 0:
        raise ValueError("repeats must be positive")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("CBF-QP config must be a mapping")
    output = _fresh(args.output_dir, "audit output")
    cases = build_cases(config, int(args.seed))
    rows = [
        run_case(case["name"], case["env"], case["obs"], case["desired"], filter_=case.get("filter"))
        for case in cases
    ]

    repeat_env = make_env(config, int(args.seed) + 100)
    repeat_obs = observation(repeat_env, standard_positions())
    repeat_desired = np.full((4, 3), 1.25)
    repeat_filter = JointCBFQPSafetyFilter(repeat_env)
    repeat_rows = [run_case(f"repeat_{index}", repeat_env, repeat_obs, repeat_desired, filter_=repeat_filter) for index in range(int(args.repeats))]
    repeated_deterministic = all(
        np.allclose(repeat_rows[0]["action"], row["action"], atol=1e-8)
        and repeat_rows[0]["constraint_slacks"] == row["constraint_slacks"]
        for row in repeat_rows[1:]
    )

    zero_env = make_env(config, int(args.seed) + 101)
    zero_obs = observation(zero_env, standard_positions())
    zero_desired = np.full((4, 3), 0.35)
    zero_filter = JointCBFQPSafetyFilter(zero_env)
    zero_nominal_action, _zero_nominal_diag = zero_filter.filter(zero_desired, zero_obs)
    zero_candidate_action, _zero_candidate_diag = zero_filter.filter(zero_desired, zero_obs, nominal_actions=zero_desired)
    zero_perturbation_exact = bool(np.array_equal(zero_nominal_action, zero_candidate_action))

    failed_rows = [row for row in rows if row["infeasible"] or row["timed_out"] or not row["verified_feasible"]]
    aggregate = {
        "all_outputs_finite": all(bool(row["output_finite"]) for row in rows),
        "no_raw_request_on_failure": all(not bool(row["raw_request_executed"]) for row in failed_rows),
        "zero_perturbation_exact": zero_perturbation_exact,
        "repeated_deterministic": repeated_deterministic,
        "p95_solve_latency_ms": float(np.percentile([row["solve_latency_ms"] for row in rows + repeat_rows], 95)),
        "case_fallback_count": int(sum(bool(row["used_fallback"]) for row in rows)),
        "case_infeasible_count": int(sum(bool(row["infeasible"]) for row in rows)),
        "case_timeout_count": int(sum(bool(row["timed_out"]) for row in rows)),
        "state_violation_count": int(sum(bool(row["state_safety_violation"]) for row in rows)),
    }
    provenance = {
        "git_revision": git_revision(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "source_hashes": {
            "src/encirclement3d/cbf_qp.py": sha256(PROJECT_ROOT / "src" / "encirclement3d" / "cbf_qp.py"),
            "scripts/audit_jepa_safe_capture_v2_cbf_qp.py": sha256(Path(__file__).resolve()),
        },
        "command": " ".join(sys.argv),
    }
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "tensorboard": version("tensorboard"),
        "numpy": np.__version__,
    }
    report: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_v2_p5_joint_cbf_qp",
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "seed": int(args.seed),
        "repeats": int(args.repeats),
        "case_count": len(rows),
        "cases": rows,
        "aggregate": aggregate,
        "provenance": provenance,
        "environment": environment,
    }
    report["tensorboard"] = write_tensorboard(report, args.tensorboard_logdir)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "repeat_rows.json").write_text(json.dumps(repeat_rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
