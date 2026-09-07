"""Run a simulator-free, public-belief multi-cycle DN-MPC replay.

This is a development gate for route identity, phase progression, terminal
progress, stopping-distance accounting, and hysteresis.  It advances only a
kinematic public belief and the selected first action; it never reads a
simulator future state, executes CBF, or claims safe capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from encirclement3d.dn_mpc import DNMPCConfig, DistributedMinimaxMPC
from encirclement3d.obstacle_route_candidates import ObstacleRouteConfig, make_obstacle_route_candidates
from scripts.smoke_dn_mpc_route_selection import _scenarios


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _observation(positions: np.ndarray, target: np.ndarray, target_velocity: np.ndarray, obstacles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "defender_positions": positions.copy(),
        "defender_velocities": np.zeros_like(positions),
        "target_belief_positions": np.repeat(target[None, :], positions.shape[0], axis=0),
        "target_belief_velocities": np.repeat(target_velocity[None, :], positions.shape[0], axis=0),
        "target_observation_received": np.ones(positions.shape[0], dtype=bool),
        "obstacles": obstacles,
        "world_lower": np.array([-10.0, -10.0, 0.5], dtype=np.float64),
        "world_upper": np.array([10.0, 10.0, 10.0], dtype=np.float64),
    }


def _nominal_action(positions: np.ndarray, target: np.ndarray, speed_mps: float) -> np.ndarray:
    direction = target[None, :] - positions
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / np.maximum(norms, 1e-9) * float(speed_mps)


def run(config_path: Path, output_dir: Path, tensorboard_dir: Path, seed: int, cycles: int = 18) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("development_only") is not True or config.get("locked_test_opened") is not False:
        raise ValueError("DN-MPC replay must remain development-only with locked_test_opened=false.")
    if cycles < 2:
        raise ValueError("cycles must be at least two for a multi-cycle replay.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard dir: {tensorboard_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    planner_config = DNMPCConfig(**dict(config.get("planner", {})))
    route_config = ObstacleRouteConfig(**dict(config.get("route", {})))
    rows: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))

    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Contract/config", config_path.read_text(encoding="utf-8"), 0)
        writer.add_text(
            "Contract/status",
            "development_only=true; locked_test_opened=false; simulator_action_executed=false; cbf_executed=false",
            0,
        )
        global_step = 0
        for scenario_index, (name, obstacles, target_start) in enumerate(_scenarios()):
            planner = DistributedMinimaxMPC(planner_config)
            positions = np.array(
                [[-6.0, -1.0, 4.0], [-6.0, 1.0, 4.0], [-6.0, 0.8, 5.0], [-6.0, -0.8, 5.0]],
                dtype=np.float64,
            )
            target = target_start.astype(np.float64).copy()
            target_velocity = np.array([0.0, 0.15 if scenario_index % 2 else -0.12, 0.0], dtype=np.float64)
            previous_action = np.zeros_like(positions)
            selected_ids: list[str | None] = []
            switches = 0
            valid_cycles = 0
            for cycle in range(cycles):
                observation = _observation(positions, target, target_velocity, obstacles)
                nominal = _nominal_action(positions, target, route_config.nominal_speed_mps)
                batch = make_obstacle_route_candidates(
                    nominal,
                    observation,
                    config=route_config,
                    previous_action=previous_action,
                )
                decision = planner.plan(batch, observation, previous_action=previous_action)
                selected = None if decision.selected_index is None else batch.candidates[decision.selected_index]
                first_action = previous_action if selected is None else selected.action_chunk[0]
                if selected is not None and selected.valid:
                    valid_cycles += 1
                switches += int(decision.switched)
                selected_ids.append(decision.selected_route_id)
                row = {
                    "scenario": name,
                    "scenario_index": scenario_index,
                    "cycle": cycle,
                    "global_step": global_step,
                    "selected_route_id": decision.selected_route_id,
                    "selected_label": None if selected is None else selected.label,
                    "selected_side": decision.selected_side,
                    "route_phase": decision.route_phase,
                    "active_obstacle_id": decision.active_obstacle_id,
                    "reason": decision.reason,
                    "switched": decision.switched,
                    "route_age_steps": decision.route_age_steps,
                    "selected_score": None if decision.selected_index is None else decision.scores[decision.selected_index],
                    "selected_terminal_progress_cost": None if decision.selected_index is None else decision.terminal_progress_costs[decision.selected_index],
                    "selected_stopping_distance_cost": None if decision.selected_index is None else decision.stopping_distance_costs[decision.selected_index],
                    "valid_candidate_count": int(np.sum(batch.valid_mask)),
                    "candidate_count": len(batch.candidates),
                    "defender_positions": positions.copy(),
                    "target_belief_position": target.copy(),
                    "target_belief_velocity": target_velocity.copy(),
                }
                rows.append(row)
                prefix = f"Replay/{name}"
                writer.add_scalar(prefix + "/selected_score", float(row["selected_score"] if row["selected_score"] is not None else 0.0), global_step)
                writer.add_scalar(prefix + "/route_switch", float(decision.switched), global_step)
                writer.add_scalar(prefix + "/route_age_steps", float(decision.route_age_steps), global_step)
                writer.add_scalar(prefix + "/valid_candidates", float(np.sum(batch.valid_mask)), global_step)
                writer.add_scalar(prefix + "/terminal_progress_cost", float(row["selected_terminal_progress_cost"] or 0.0), global_step)
                writer.add_scalar(prefix + "/stopping_distance_cost", float(row["selected_stopping_distance_cost"] or 0.0), global_step)
                writer.add_text(prefix + "/route_phase", decision.route_phase, global_step)
                positions = positions + first_action * float(route_config.dt_seconds)
                target = target + target_velocity * float(route_config.dt_seconds)
                # Keep the public belief bounded and deterministic; this is
                # not a simulator target state or a hidden future rollout.
                target_velocity = target_velocity + rng.normal(0.0, 0.005, size=3)
                previous_action = first_action.copy()
                global_step += 1
            switches_ratio = float(switches / max(cycles - 1, 1))
            summary = {
                "scenario": name,
                "cycles": cycles,
                "valid_cycle_rate": float(valid_cycles / cycles),
                "route_switch_count": switches,
                "route_switch_rate": switches_ratio,
                "selected_route_ids": selected_ids,
                "final_route_phase": rows[-1]["route_phase"],
            }
            scenario_summaries.append(summary)
            writer.add_scalar("Summary/route_switch_rate", switches_ratio, scenario_index)
            writer.add_scalar("Summary/valid_cycle_rate", summary["valid_cycle_rate"], scenario_index)

    summary_payload = {
        "schema_version": "dn_mpc_multicycle_replay_v1",
        "development_only": True,
        "locked_test_opened": False,
        "simulator_action_executed": False,
        "cbf_executed": False,
        "seed": int(seed),
        "cycles_per_scenario": int(cycles),
        "scenario_count": len(scenario_summaries),
        "all_cycles_selected_valid_route": all(item["valid_cycle_rate"] == 1.0 for item in scenario_summaries),
        "max_route_switch_rate": max(item["route_switch_rate"] for item in scenario_summaries),
        "scenarios": scenario_summaries,
    }
    provenance = {
        "schema_version": "dn_mpc_multicycle_replay_provenance_v1",
        "development_only": True,
        "locked_test_opened": False,
        "simulator_action_executed": False,
        "cbf_executed": False,
        "seed": int(seed),
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "planner_source_sha256": _sha256(PROJECT_ROOT / "src" / "encirclement3d" / "dn_mpc.py"),
        "route_source_sha256": _sha256(PROJECT_ROOT / "src" / "encirclement3d" / "obstacle_route_candidates.py"),
        "tensorboard_dir": str(tensorboard_dir.resolve()),
        "contract": "public_belief_to_multi_cycle_route_score; no simulator future state; no CBF or action execution",
    }
    _write_json(output_dir / "summary.json", summary_payload)
    _write_json(output_dir / "provenance.json", provenance)
    with (output_dir / "cycle_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    return {"summary": summary_payload, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dn_mpc_jepa_safe_capture_development.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--cycles", type=int, default=18)
    args = parser.parse_args()
    report = run(args.config.resolve(), args.output_dir.resolve(), args.tensorboard_dir.resolve(), args.seed, args.cycles)
    print(json.dumps({"schema_version": report["summary"]["schema_version"], "scenario_count": report["summary"]["scenario_count"], "max_route_switch_rate": report["summary"]["max_route_switch_rate"], "tensorboard_dir": report["provenance"]["tensorboard_dir"]}, indent=2))


if __name__ == "__main__":
    main()
