"""Replay one frozen L0-open episode and expose the CBF feasibility margin.

This is a development diagnostic only.  It uses the same actor, scene
manifest, kinematic environment, and Joint CBF filter as the paired evaluator,
but stops after the first failed filter call and prints state/constraint data.
"""

from __future__ import annotations

import copy
import json
import sys
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.pursuit_controllers import DynamicEncirclementController  # noqa: E402
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    _opposite_side_positions,
    prepare_showcase_episode,
    scenario_from_metadata,
)
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from evaluate_jepa_safe_capture_v2_paired import _actor_action  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--distance", type=float)
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--barrier-mode", choices=("strict_buffer", "physical_feasibility"), default="strict_buffer")
    parser.add_argument("--actor-checkpoint", type=Path, default=ROOT / "models/v5_development_exact_reactive_seed661606.pt")
    parser.add_argument("--reset-interval", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "results/l0_l3_r2_full_runs/jepa_safe_capture_l0_l3_paired_full_seed20260911_m0/scene_manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    record = records[int(args.episode)]
    spec = record["spec"]
    scenario = scenario_from_metadata(record["scenario"])
    if args.distance is not None:
        defenders, target, escape = _opposite_side_positions(float(args.distance), str(spec["defender_side"]))
        scenario = replace(scenario, defender_positions=defenders, target_position=target, target_escape_direction=escape)
    config_path = ROOT / "configs/capture_radius_pursuit_central_v4_flee.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config.setdefault("task", {}).setdefault("pursuit", {}).update(spec["pursuit_overrides"])
    config["task"]["pursuit"]["target_motion_mode"] = str(spec["target_motion_mode"])
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(spec["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(
        env,
        scenario,
        seed=int(spec["episode_seed"]),
        record_history=True,
        validate_scenario=False,
    )
    device = select_device("cuda")
    policy, action_scale, _metadata = load_policy(
        args.actor_checkpoint.resolve(),
        env,
        observation,
        device,
    )
    safety = JointCBFQPSafetyFilter(
        env,
        anticipatory_horizon_steps=int(args.horizon),
        barrier_mode=str(args.barrier_mode),
    )
    expert = DynamicEncirclementController(env) if args.expert else None
    hidden = policy.initial_actor_hidden(env.n_defenders, device=device) if hasattr(policy, "initial_actor_hidden") else None
    previous_action = np.asarray(env.defender_velocities, dtype=np.float64).copy()
    local = policy_observations(env, observation)

    print(json.dumps({"cbf_contract": safety.contract, "scenario": record["scenario"]}, indent=2))
    while env.step_count < env.max_steps:
        # The paired evaluator uses the checkpoint's declared reset interval
        # (one step for this actor), so reproduce that contract exactly.
        if env.step_count > 0 and not args.expert and args.reset_interval > 0 and env.step_count % args.reset_interval == 0:
            hidden = policy.initial_actor_hidden(env.n_defenders, device=device) if hasattr(policy, "initial_actor_hidden") else None
        if expert is not None:
            desired = expert.act(observation)
        else:
            desired, hidden = _actor_action(policy, local, device, action_scale, hidden)
        requested = np.asarray(desired, dtype=np.float64)
        reachable = env._move_toward_velocity(
            previous_action,
            env._clip_rows(requested, float(env.agents["defender_max_speed"])),
            max_delta=float(env.agents["defender_max_acceleration"]) * float(env.dt),
        )
        before_positions = env.defender_positions.copy()
        before_target = env.target_position.copy()
        before_velocities = env.defender_velocities.copy()
        records = safety._build_barriers(observation)
        primary = safety._solve(
            safety._reachable_reference(before_velocities, requested),
            before_velocities,
            records,
        )
        hold = safety._solve(
            safety._reachable_reference(before_velocities, before_velocities),
            before_velocities,
            records,
        )
        action, diagnostics = safety.filter(requested, observation, nominal_actions=reachable)
        row = {
            "step": int(env.step_count + 1),
            "before_defenders": before_positions.tolist(),
            "before_velocities": before_velocities.tolist(),
            "before_target": before_target.tolist(),
            "target_distance": float(np.min(np.linalg.norm(before_positions - before_target[None, :], axis=1))),
            "requested": requested.tolist(),
            "reachable": reachable.tolist(),
            "executed": np.asarray(action).tolist(),
            "fallback": diagnostics.fallback_mode,
            "solver_status": diagnostics.solver_status,
            "solver_message": diagnostics.solver_message,
            "verified": bool(diagnostics.verified_feasible),
            "minimum_constraint": float(diagnostics.minimum_constraint_value),
            "active_constraints": list(diagnostics.active_constraints),
            "slacks": diagnostics.constraint_slacks,
            "primary_status": primary.status,
            "primary_success": bool(primary.success),
            "primary_verified": bool(primary.verified_feasible),
            "primary_minimum_constraint": float(primary.minimum_constraint_value),
            "primary_active_constraints": list(primary.active_constraints),
            "hold_status": hold.status,
            "hold_verified": bool(hold.verified_feasible),
            "hold_minimum_constraint": float(hold.minimum_constraint_value),
        }
        observation, _reward, terminated, truncated, info = env.step(action, record_history=True)
        row.update(
            {
                "after_defenders": env.defender_positions.tolist(),
                "after_target": env.target_position.tolist(),
                "after_target_distance": float(np.min(np.linalg.norm(env.defender_positions - env.target_position[None, :], axis=1))),
                "boundary_clearance": float(np.min(np.concatenate([env.defender_positions - env.lower, env.upper - env.defender_positions]))),
                "pairwise_clearance": float(min(np.linalg.norm(env.defender_positions[i] - env.defender_positions[j]) - 2.0 * env.agents["drone_radius"] for i in range(env.n_defenders) for j in range(i + 1, env.n_defenders))),
                "termination": info.get("termination_reason"),
            }
        )
        if not args.summary_only:
            print(json.dumps(row, allow_nan=False))
        if not diagnostics.verified_feasible or terminated or truncated:
            if args.summary_only:
                print(json.dumps({
                    "episode": int(args.episode),
                    "steps": int(row["step"]),
                    "termination": row["termination"],
                    "target_distance": float(row["after_target_distance"]),
                    "fallback": row["fallback"],
                    "verified": bool(row["verified"]),
                }, allow_nan=False))
            break
        local = policy_observations(env, observation)
        previous_action = np.asarray(action, dtype=np.float64).copy()


if __name__ == "__main__":
    main()
