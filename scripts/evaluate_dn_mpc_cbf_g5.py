"""Evaluate the development-only DN-MPC + strict Joint CBF controller on G5.

This evaluator intentionally excludes JEPA and the reliability ledger.  It
uses only public belief/obstacle geometry, probes every reachable route with
the primary Joint CBF solve, lets DN-MPC select among eligible routes, then
passes only the first selected step through the normal execution filter.
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
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.dn_mpc import DNMPCConfig, DistributedMinimaxMPC  # noqa: E402
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.obstacle_route_candidates import (  # noqa: E402
    ObstacleRouteConfig,
    make_obstacle_route_candidates,
)
from encirclement3d.obstacle_route_runtime import (  # noqa: E402
    probe_independent_cbf_counterfactuals,
    probe_route_batch_with_cbf,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    prepare_showcase_episode,
    scenario_from_metadata,
    target_min_clearance,
)
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from evaluate_jepa_safe_capture_v2_paired import _load_scene_manifest  # noqa: E402
from evaluate_random_central_mixed_obstacles import config_for_spec, load_protocol  # noqa: E402

from evaluate_jepa_safe_capture_v2_paired import _jsonable, _latency_stats, _safety_observables  # noqa: E402,E501
from evaluate_jepa_safe_capture_v2_paired import _raw_unverified_executed  # noqa: E402

DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_route_v1_protocol.yaml"
DEFAULT_ENVIRONMENT = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"
DEFAULT_ACTOR = PROJECT_ROOT / "models" / "v5_development_exact_reactive_seed661606.pt"
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "jepa_route_recovery_dev_g1b_seed20260911" / "scene_manifest.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _fresh(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {label}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--scene-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cbf-horizon", type=int, default=3)
    parser.add_argument("--route-probe-horizon", type=int, default=3)
    parser.add_argument("--route-corridor-samples", type=int, default=65)
    parser.add_argument("--minimum-hold-steps", type=int, default=2)
    parser.add_argument("--switch-improvement-m", type=float, default=0.25)
    parser.add_argument("--tangent-route-hold-steps", type=int, default=5)
    parser.add_argument("--boundary-rescue-enabled", action="store_true")
    parser.add_argument("--boundary-rescue-trigger-m", type=float, default=3.0)
    parser.add_argument("--boundary-rescue-offset-m", type=float, default=2.0)
    parser.add_argument("--development-only", action="store_true")
    return parser.parse_args()


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "torch": version("torch"),
        "pyyaml": version("PyYAML"),
        "tensorboard": version("tensorboard"),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        values["cuda_device_name"] = torch.cuda.get_device_name(0)
        values["cuda_version"] = torch.version.cuda
    return values


def _buffer_observables(env: CaptureRadiusPursuit3DEnv, safety_filter: JointCBFQPSafetyFilter) -> dict[str, float]:
    positions = np.asarray(env.defender_positions, dtype=np.float64)
    radius = float(env.agents["drone_radius"])
    obstacle = [float(env._obstacle_clearance(p, item) - radius - safety_filter.obstacle_margin_m) for p in positions for item in env.obstacles]
    pairwise = [float(np.linalg.norm(positions[i] - positions[j]) - 2.0 * radius - safety_filter.inter_agent_margin_m) for i in range(env.n_defenders) for j in range(i + 1, env.n_defenders)]
    boundary = [float(v - radius - safety_filter.boundary_margin_m) for v in np.concatenate([positions - env.lower[None, :], env.upper[None, :] - positions]).reshape(-1)]
    return {
        "minimum_obstacle_buffer_clearance_m": min(obstacle) if obstacle else float("inf"),
        "minimum_pairwise_buffer_clearance_m": min(pairwise) if pairwise else float("inf"),
        "minimum_boundary_buffer_clearance_m": min(boundary) if boundary else float("inf"),
    }


def _planner_config(args: argparse.Namespace, dt_seconds: float) -> DNMPCConfig:
    return DNMPCConfig(
        horizon_steps=5,
        dt_seconds=float(dt_seconds),
        minimum_hold_steps=int(args.minimum_hold_steps),
        switch_improvement_m=float(args.switch_improvement_m),
        tangent_route_hold_steps=int(args.tangent_route_hold_steps),
        boundary_rescue_trigger_m=float(args.boundary_rescue_trigger_m),
    )


def _run_episode(*, item: dict[str, Any], config: dict[str, Any], actor: Any, action_scale: float, device: torch.device, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    spec = dict(item["spec"])
    scenario = scenario_from_metadata(dict(item["scenario"]))
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=len(scenario.obstacles), target_speed_scale=float(spec["target_speed_scale"]))
    observation = prepare_showcase_episode(env, scenario, seed=int(spec["episode_seed"]), record_history=True, validate_scenario=False)
    safety_filter = JointCBFQPSafetyFilter(env, anticipatory_horizon_steps=int(args.cbf_horizon), barrier_mode="strict_buffer")
    planner = DistributedMinimaxMPC(_planner_config(args, float(env.dt)))
    previous_action = np.asarray(env.defender_velocities, dtype=np.float64).copy()
    hidden = actor.initial_actor_hidden(env.n_defenders, device=device) if hasattr(actor, "initial_actor_hidden") else None
    route_switches = 0
    last_route: str | None = None
    route_counts: dict[str, int] = {}
    counters = {"cbf_checks": 0, "cbf_accepted": 0, "cbf_rejected": 0, "cbf_timeouts": 0, "controlled_abort": 0, "fallback": 0, "unverified": 0, "raw_unverified": 0, "independent_checks": 0, "independent_accepted": 0, "independent_rejected": 0}
    latency: dict[str, list[float]] = {name: [] for name in ("actor", "route_generation", "route_probe", "planner", "cbf_filter", "cycle")}
    min_obstacle = min_pairwise = min_boundary = float("inf")
    min_buffers = {"minimum_obstacle_buffer_clearance_m": float("inf"), "minimum_pairwise_buffer_clearance_m": float("inf"), "minimum_boundary_buffer_clearance_m": float("inf")}
    traces: list[dict[str, Any]] = []
    local_observation = policy_observations(env, observation)
    while True:
        cycle_start = time.perf_counter_ns()
        # G5's recurrent contract resets the actor state every control step.
        if hasattr(actor, "initial_actor_hidden"):
            hidden = actor.initial_actor_hidden(env.n_defenders, device=device)
        actor_start = time.perf_counter_ns()
        local = torch.as_tensor(local_observation, device=device)
        with torch.no_grad():
            distribution, hidden = actor.distribution_step(local, hidden) if hidden is not None else (actor.distribution(local), None)
            desired = torch.tanh(distribution.mean).cpu().numpy() * float(action_scale)
        desired = np.asarray(desired, dtype=np.float64)
        latency["actor"].append((time.perf_counter_ns() - actor_start) / 1e6)
        reachable = env._move_toward_velocity(previous_action, env._clip_rows(desired, float(env.agents["defender_max_speed"])), max_delta=float(env.agents["defender_max_acceleration"]) * float(env.dt))
        route_start = time.perf_counter_ns()
        route_batch = make_obstacle_route_candidates(
            reachable,
            observation,
            config=ObstacleRouteConfig(
                chunk_length_steps=5,
                dt_seconds=float(env.dt),
                max_speed_mps=float(env.agents["defender_max_speed"]),
                max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
                max_action_change_mps=float(env.agents["defender_max_acceleration"]) * float(env.dt),
                vehicle_radius_m=float(env.agents["drone_radius"]),
                obstacle_margin_m=float(safety_filter.obstacle_margin_m),
                route_buffer_m=0.75,
                nominal_speed_mps=min(2.0, float(env.agents["defender_max_speed"])),
                project_to_reachable_dynamics=True,
                world_lower=tuple(float(value) for value in env.lower),
                world_upper=tuple(float(value) for value in env.upper),
                corridor_samples=int(args.route_corridor_samples),
                boundary_rescue_enabled=bool(args.boundary_rescue_enabled),
                boundary_rescue_trigger_m=float(args.boundary_rescue_trigger_m),
                boundary_rescue_offset_m=float(args.boundary_rescue_offset_m),
            ),
            previous_action=previous_action,
        )
        latency["route_generation"].append((time.perf_counter_ns() - route_start) / 1e6)
        probe_start = time.perf_counter_ns()
        runtime = probe_route_batch_with_cbf(route_batch, safety_filter, observation, horizon_steps=int(args.route_probe_horizon))
        latency["route_probe"].append((time.perf_counter_ns() - probe_start) / 1e6)
        probes = runtime.cbf_counterfactuals
        eligible = np.asarray(runtime.candidate_batch.valid_mask, dtype=bool)
        counters["cbf_checks"] += sum(probe is not None for probe in probes)
        counters["cbf_accepted"] += sum(probe is not None and probe.accepted for probe in probes)
        counters["cbf_rejected"] += sum(probe is not None and not probe.accepted for probe in probes)
        counters["cbf_timeouts"] += sum(probe is not None and probe.timed_out for probe in probes)
        planner_start = time.perf_counter_ns()
        decision = planner.plan(route_batch, observation, previous_action=previous_action, eligible_mask=eligible)
        latency["planner"].append((time.perf_counter_ns() - planner_start) / 1e6)
        if decision.selected_index is None:
            requested = reachable.copy()
            selected = None
        else:
            selected = route_batch.candidates[int(decision.selected_index)]
            requested = np.asarray(selected.action_chunk[0], dtype=np.float64)
            route_counts[selected.label] = route_counts.get(selected.label, 0) + 1
            if last_route is not None and last_route != selected.route_id:
                route_switches += 1
            last_route = selected.route_id
        safe_hold = np.asarray(observation["defender_velocities"], dtype=np.float64)
        independent = probe_independent_cbf_counterfactuals(selected_action=requested, nominal_action=reachable, safe_hold_action=safe_hold, observation=observation, safety_filter=safety_filter, selected_route_id=None if selected is None else selected.route_id)
        counters["independent_checks"] += len(independent)
        counters["independent_accepted"] += sum(item.accepted for item in independent)
        counters["independent_rejected"] += sum(not item.accepted for item in independent)
        cbf_start = time.perf_counter_ns()
        action, diagnostics = safety_filter.filter(requested, observation, nominal_actions=reachable)
        latency["cbf_filter"].append((time.perf_counter_ns() - cbf_start) / 1e6)
        if diagnostics.used_fallback:
            counters["fallback"] += 1
        if not diagnostics.verified_feasible:
            counters["unverified"] += 1
        if diagnostics.fallback_mode == "controlled_abort":
            counters["controlled_abort"] += 1
        if _raw_unverified_executed(safety_filter_enabled=True, diagnostics=diagnostics):
            counters["raw_unverified"] += 1
        action = np.asarray(action, dtype=np.float64)
        step_start = time.perf_counter_ns()
        observation, _reward, terminated, truncated, info = env.step(action, record_history=True)
        latency["cycle"].append((time.perf_counter_ns() - cycle_start) / 1e6)
        safety = _safety_observables(env)
        buffers = _buffer_observables(env, safety_filter)
        min_obstacle = min(min_obstacle, safety["minimum_obstacle_clearance_m"])
        min_pairwise = min(min_pairwise, safety["minimum_pairwise_clearance_m"])
        min_boundary = min(min_boundary, safety["minimum_boundary_clearance_m"])
        for key in min_buffers:
            min_buffers[key] = min(min_buffers[key], buffers[key])
        traces.append({
            "step": int(env.step_count),
            "public_observation": {key: observation.get(key) for key in ("defender_positions", "defender_velocities", "target_belief_positions", "target_belief_velocities", "target_visible", "target_observation_received", "target_observation_age_steps", "message_age_steps", "obstacles", "world_lower", "world_upper")},
            "desired_action": desired,
            "reachable_nominal_action": reachable,
            "requested_action": requested,
            "executed_action": action,
            "selected_route": None if selected is None else selected.as_dict(),
            "decision": decision.as_dict(),
            "route_cbf_counterfactuals": [None if probe is None else probe.as_dict() for probe in probes],
            "independent_cbf_counterfactuals": [probe.as_dict() for probe in independent],
            "cbf": diagnostics,
            "safety_observables": safety,
            "operational_buffer_observables": buffers,
            "latency_ms": {name: latency[name][-1] for name in latency},
        })
        if not diagnostics.verified_feasible or terminated or truncated:
            break
        local_observation = policy_observations(env, observation)
        previous_action = action.copy()
    if counters["controlled_abort"] > 0:
        info = dict(info)
        info["termination_reason"] = "cbf_controlled_abort"
    # Match the established G5 contract: target boundary crossings remain an
    # explicit diagnostic, while defender boundary/pairwise/obstacle safety
    # and verified execution determine safe_capture.
    safe_capture = bool(info.get("safe_capture_success", False)) and not bool(env.defender_boundary_violation) and min_pairwise >= -1e-9 and min_obstacle >= -1e-9 and counters["raw_unverified"] == 0 and counters["controlled_abort"] == 0
    info = dict(info)
    info["safe_capture_success"] = bool(safe_capture)
    row = {
        "episode_index": int(item["episode_index"]),
        "seed": int(spec["episode_seed"]),
        "scenario": scenario.name,
        "safe_capture_success": bool(safe_capture),
        "capture_event": bool(info.get("capture_event", False)),
        "capture_time_seconds": info.get("capture_time_seconds"),
        "collision": bool(info.get("collision", False) or min_obstacle < -1e-9),
        "boundary_violation": bool(env.defender_boundary_violation),
        "target_boundary_violation": bool(env.target_boundary_violation),
        "pairwise_violation": bool(min_pairwise < -1e-9),
        "steps": int(env.step_count),
        "termination_reason": str(info.get("termination_reason", "running")),
        "min_clearance_m": float(env.min_clearance),
        "minimum_obstacle_clearance_m": min_obstacle,
        "minimum_pairwise_clearance_m": min_pairwise,
        "minimum_boundary_clearance_m": min_boundary,
        **min_buffers,
        "route_switch_steps": int(route_switches),
        "route_counts": route_counts,
        "route_counters": counters,
        "latency": {name: _latency_stats(values) for name, values in latency.items()},
        "traces": traces,
    }
    _write_json(output_dir / "step_traces" / f"episode_{int(item['episode_index']):03d}.json", row)
    return row


def main() -> None:
    args = _parse_args()
    if not args.development_only:
        raise SystemExit("S1 is development-only; pass --development-only explicitly.")
    if args.episodes != 4:
        raise SystemExit("The G5 S1 evaluator requires exactly the frozen 4-episode manifest.")
    if args.cbf_horizon <= 0 or args.route_probe_horizon <= 0:
        raise SystemExit("CBF and route probe horizons must be positive.")
    if args.minimum_hold_steps < 0 or args.tangent_route_hold_steps < 0:
        raise SystemExit("Route hold steps must be non-negative.")
    if not np.isfinite(args.switch_improvement_m) or args.switch_improvement_m < 0.0:
        raise SystemExit("Switch improvement must be finite and non-negative.")
    if not np.isfinite(args.boundary_rescue_trigger_m) or args.boundary_rescue_trigger_m <= 0.0:
        raise SystemExit("Boundary rescue trigger must be finite and positive.")
    if not np.isfinite(args.boundary_rescue_offset_m) or args.boundary_rescue_offset_m <= 0.0:
        raise SystemExit("Boundary rescue offset must be finite and positive.")
    output_dir = _fresh(args.output_dir, "output directory")
    (output_dir / "step_traces").mkdir(parents=True, exist_ok=True)
    tensorboard_dir = _fresh(args.tensorboard_dir, "TensorBoard directory")
    device = select_device(args.device)
    protocol = load_protocol(args.protocol)
    manifest = _load_scene_manifest(args.scene_manifest, protocol, args.environment_config, split="validation", episodes=args.episodes)
    probe_spec = dict(manifest[0]["spec"])
    base_config = config_for_spec("f2", probe_spec, args.environment_config)
    probe_env = CaptureRadiusPursuit3DEnv(base_config, obstacle_count=len(manifest[0]["scenario"]["obstacles"]), target_speed_scale=float(probe_spec["target_speed_scale"]))
    actor, action_scale, checkpoint = load_policy(args.actor_checkpoint, probe_env, probe_env.reset(seed=int(probe_spec["episode_seed"]), record_history=True), device)
    rows: list[dict[str, Any]] = []
    for item in manifest:
        spec = dict(item["spec"])
        config = config_for_spec("f2", spec, args.environment_config)
        rows.append(_run_episode(item=item, config=config, actor=actor, action_scale=action_scale, device=device, args=args, output_dir=output_dir))
    overall = {
        "episodes": len(rows),
        "safe_capture_count": int(sum(row["safe_capture_success"] for row in rows)),
        "safe_capture_rate": float(np.mean([row["safe_capture_success"] for row in rows])),
        "collision_count": int(sum(row["collision"] for row in rows)),
        "boundary_violation_count": int(sum(row["boundary_violation"] for row in rows)),
        "target_boundary_violation_count": int(sum(row["target_boundary_violation"] for row in rows)),
        "pairwise_violation_count": int(sum(row["pairwise_violation"] for row in rows)),
        "raw_unverified_executed_steps": int(sum(row["route_counters"]["raw_unverified"] for row in rows)),
        "controlled_abort_steps": int(sum(row["route_counters"]["controlled_abort"] for row in rows)),
        "cbf_fallback_steps": int(sum(row["route_counters"]["fallback"] for row in rows)),
        "route_switch_steps": int(sum(row["route_switch_steps"] for row in rows)),
        "mean_capture_time_seconds": float(np.mean([row["capture_time_seconds"] for row in rows if row["capture_time_seconds"] is not None])) if any(row["capture_time_seconds"] is not None for row in rows) else None,
        "worst_min_clearance_m": float(min(row["min_clearance_m"] for row in rows)),
        "route_probe_checks": int(sum(row["route_counters"]["cbf_checks"] for row in rows)),
        "route_probe_accepted": int(sum(row["route_counters"]["cbf_accepted"] for row in rows)),
        "route_probe_rejected": int(sum(row["route_counters"]["cbf_rejected"] for row in rows)),
    }
    metadata = {
        "evaluation_type": "dn_mpc_cbf_g5_development",
        "development_only": True,
        "locked_test_opened": False,
        "episodes": args.episodes,
        "git_revision": _git_revision(),
        "inputs": {
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "actor_checkpoint_sha256": _sha256(args.actor_checkpoint),
            "environment_config": str(args.environment_config.resolve()),
            "environment_config_sha256": _sha256(args.environment_config),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": _sha256(args.protocol),
            "scene_manifest": str(args.scene_manifest.resolve()),
            "scene_manifest_sha256": _sha256(args.scene_manifest),
        },
        "contract": {
            "planner": asdict(_planner_config(args, 0.1)),
            "cbf": JointCBFQPSafetyFilter(probe_env, anticipatory_horizon_steps=args.cbf_horizon, barrier_mode="strict_buffer").contract,
            "route_probe_horizon": int(args.route_probe_horizon),
            "route_chunk_length_steps": 5,
            "boundary_rescue_enabled": bool(args.boundary_rescue_enabled),
            "boundary_rescue_offset_m": float(args.boundary_rescue_offset_m),
            "execute_first_step_then_replan": True,
            "jepa_enabled": False,
            "ledger_enabled": False,
        },
        "environment": _environment_metadata(device),
        "tensorboard_dir": str(tensorboard_dir),
    }
    _write_json(output_dir / "summary.json", {"overall": overall, "metadata": metadata, "episodes": [{key: value for key, value in row.items() if key != "traces"} for row in rows]})
    _write_json(output_dir / "provenance.json", metadata)
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=2) as writer:
        writer.add_text("Provenance/metadata", json.dumps(_jsonable(metadata), indent=2), 0)
        for key, value in overall.items():
            if isinstance(value, (int, float)) and value is not None:
                writer.add_scalar(f"Aggregate/{key}", float(value), 0)
        for row in rows:
            step = int(row["episode_index"])
            writer.add_scalar("Episode/safe_capture", float(row["safe_capture_success"]), step)
            writer.add_scalar("Episode/route_switch_steps", float(row["route_switch_steps"]), step)
            writer.add_scalar("Episode/controlled_abort_steps", float(row["route_counters"]["controlled_abort"]), step)
            writer.add_scalar("Episode/worst_min_clearance_m", float(row["min_clearance_m"]), step)
        writer.flush()
    print(json.dumps(_jsonable({"overall": overall, "metadata": metadata}), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
