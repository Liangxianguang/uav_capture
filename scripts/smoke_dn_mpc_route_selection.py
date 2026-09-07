"""Run a deterministic, simulator-free DN-MPC route-selection smoke.

The smoke consumes the public ``obstacle_route_v1`` candidate contract and
never executes an action.  It exists to validate the planner objective,
route-hold state, provenance and TensorBoard observability before any CBF or
JEPA integration is attempted.
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

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from encirclement3d.dn_mpc import DNMPCConfig, DistributedMinimaxMPC
from encirclement3d.obstacle_route_candidates import (
    ObstacleRouteConfig,
    make_obstacle_route_candidates,
)


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


def _cylinder(center: tuple[float, float], radius: float = 1.0) -> dict[str, Any]:
    return {
        "center_xy": np.asarray(center, dtype=np.float64),
        "radius": radius,
        "height": 5.0,
        "shape": "cylinder",
        "half_extents_xy": None,
    }


def _box(center: tuple[float, float], half: tuple[float, float]) -> dict[str, Any]:
    return {
        "center_xy": np.asarray(center, dtype=np.float64),
        "radius": max(half),
        "height": 5.0,
        "shape": "box",
        "half_extents_xy": np.asarray(half, dtype=np.float64),
    }


def _wall(center: tuple[float, float], half: tuple[float, float]) -> dict[str, Any]:
    return {
        "center_xy": np.asarray(center, dtype=np.float64),
        "radius": min(half),
        "height": 5.0,
        "shape": "wall",
        "half_extents_xy": np.asarray(half, dtype=np.float64),
    }


def _scenarios() -> tuple[tuple[str, list[dict[str, Any]], np.ndarray], ...]:
    target = np.array([6.0, 0.0, 4.0], dtype=np.float64)
    return (
        ("open", [], target.copy()),
        ("single_cylinder", [_cylinder((0.0, 0.0))], target.copy()),
        ("single_wall", [_wall((0.0, 0.0), (0.35, 2.0))], target.copy()),
        (
            "mixed",
            [_cylinder((0.0, 0.0)), _box((2.0, 3.0), (0.8, 1.1)), _wall((-1.0, -3.0), (0.35, 1.5))],
            target.copy(),
        ),
    )


def _observation(obstacles: list[dict[str, Any]], target: np.ndarray) -> dict[str, Any]:
    positions = np.array(
        [[-6.0, -1.0, 4.0], [-6.0, 1.0, 4.0], [-6.0, 0.8, 5.0], [-6.0, -0.8, 5.0]],
        dtype=np.float64,
    )
    return {
        "defender_positions": positions,
        "defender_velocities": np.zeros_like(positions),
        "target_belief_positions": np.repeat(target[None, :], positions.shape[0], axis=0),
        "target_belief_velocities": np.zeros_like(positions),
        "target_observation_received": np.ones(positions.shape[0], dtype=bool),
        "obstacles": obstacles,
        "world_lower": np.array([-10.0, -10.0, 0.5], dtype=np.float64),
        "world_upper": np.array([10.0, 10.0, 10.0], dtype=np.float64),
    }


def _nominal_action(observation: dict[str, Any], speed_mps: float) -> np.ndarray:
    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    target = np.asarray(observation["target_belief_positions"], dtype=np.float64).mean(axis=0)
    direction = target[None, :] - positions
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / np.maximum(norms, 1e-9) * float(speed_mps)


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


def _nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def run(config_path: Path, output_dir: Path, tensorboard_dir: Path, seed: int) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("development_only") is not True or config.get("locked_test_opened") is not False:
        raise ValueError("DN-MPC smoke must remain development-only with locked_test_opened=false.")
    if _nonempty(output_dir) or _nonempty(tensorboard_dir):
        raise FileExistsError("Refusing to overwrite a non-empty smoke output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    planner_values = dict(config.get("planner", {}))
    route_values = dict(config.get("route", {}))
    planner_config = DNMPCConfig(**planner_values)
    route_config = ObstacleRouteConfig(**route_values)
    planner = DistributedMinimaxMPC(planner_config)
    decisions: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Contract/config", config_path.read_text(encoding="utf-8"), 0)
        writer.add_text("Contract/status", "development_only=true; locked_test_opened=false; simulator_action_executed=false", 0)
        for scenario_index, (name, obstacles, target) in enumerate(_scenarios()):
            planner.reset()
            observation = _observation(obstacles, target)
            nominal = _nominal_action(observation, route_config.nominal_speed_mps)
            route_batch = make_obstacle_route_candidates(
                nominal,
                observation,
                config=route_config,
                previous_action=np.zeros_like(nominal),
            )
            decision = planner.plan(route_batch, observation, previous_action=np.zeros_like(nominal))
            decision_dict = decision.as_dict()
            decision_dict.update(
                {
                    "scenario": name,
                    "scenario_index": scenario_index,
                    "candidate_count": len(route_batch.candidates),
                    "valid_candidate_count": int(np.sum(route_batch.valid_mask)),
                    "route_contract": route_batch.route_contract,
                }
            )
            decisions.append(decision_dict)
            for candidate_index, candidate in enumerate(route_batch.candidates):
                row = {
                    "scenario": name,
                    "scenario_index": scenario_index,
                    "candidate_index": candidate_index,
                    "route_id": candidate.route_id,
                    "label": candidate.label,
                    "side": candidate.side,
                    "valid": candidate.valid,
                    "projection_residual": float(np.linalg.norm(candidate.action_chunk - candidate.raw_action_chunk)),
                    "score": decision.scores[candidate_index],
                    "worst_case_escape_cost": decision.worst_case_escape_costs[candidate_index],
                    "capture_cost": decision.capture_costs[candidate_index],
                    "formation_cost": decision.formation_costs[candidate_index],
                    "smoothness_cost": decision.smoothness_costs[candidate_index],
                    "selected": decision.selected_index == candidate_index,
                }
                candidate_rows.append(row)
                prefix = f"Planning/{name}/candidate_{candidate_index}"
                for tag, value in (
                    ("score", row["score"]),
                    ("worst_case_escape", row["worst_case_escape_cost"]),
                    ("capture_cost", row["capture_cost"]),
                    ("formation_cost", row["formation_cost"]),
                    ("smoothness_cost", row["smoothness_cost"]),
                    ("valid", float(row["valid"])),
                    ("selected", float(row["selected"])),
                ):
                    numeric_value = float(value)
                    if math.isfinite(numeric_value):
                        writer.add_scalar(prefix + "/" + tag, numeric_value, scenario_index)
            writer.add_scalar(f"Planning/{name}/selected_index", float(decision.selected_index or -1), 0)
            writer.add_scalar(f"Planning/{name}/route_switch", float(decision.switched), 0)
            writer.add_scalar(f"Planning/{name}/route_age", float(decision.route_age_steps), 0)
            writer.add_scalar(f"Planning/{name}/valid_candidates", float(np.sum(route_batch.valid_mask)), 0)

    summary = {
        "schema_version": "dn_mpc_route_smoke_v1",
        "development_only": True,
        "locked_test_opened": False,
        "simulator_action_executed": False,
        "seed": int(seed),
        "scenario_count": len(decisions),
        "selected_route_ids": [row["selected_route_id"] for row in decisions],
        "all_scenarios_selected_valid_route": all(row["selected_index"] is not None for row in decisions),
        "valid_candidate_counts": [row["valid_candidate_count"] for row in decisions],
        "decisions": decisions,
        "candidate_rows": candidate_rows,
    }
    provenance = {
        "schema_version": "dn_mpc_route_smoke_provenance_v1",
        "development_only": True,
        "locked_test_opened": False,
        "simulator_action_executed": False,
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
        "contract": "public_belief_to_route_score_only; no action execution; CBF not run",
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "provenance.json", provenance)
    with (output_dir / "candidate_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    return {"summary": summary, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dn_mpc_jepa_safe_capture_development.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    report = run(args.config.resolve(), args.output_dir.resolve(), args.tensorboard_dir.resolve(), args.seed)
    print(json.dumps({"schema_version": report["summary"]["schema_version"], "selected_route_ids": report["summary"]["selected_route_ids"], "tensorboard_dir": report["provenance"]["tensorboard_dir"]}, indent=2))


if __name__ == "__main__":
    main()
