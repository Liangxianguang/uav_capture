"""Paired development replay for DN-MPC, route-JEPA, and strict CBF.

The evaluator keeps the new direction separate from historical V4/V5 scripts.
Both variants use the same frozen G5 manifest, actor, projected route batch,
candidate-level CBF probes, and final Joint CBF filter.  The JEPA variant only
reranks a deterministic DN-MPC shortlist; it never emits or executes actions.
"""

from __future__ import annotations

import argparse
import csv
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
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.dn_mpc import DNMPCConfig, DistributedMinimaxMPC  # noqa: E402
from encirclement3d.jepa_safe_capture_candidates import (  # noqa: E402
    SafeCaptureCandidateBatch,
    SafeCaptureCandidateHistory,
)
from encirclement3d.jepa_safe_capture_ranker import (  # noqa: E402
    SafeCaptureJEPARanker,
    SafeCaptureRankerConfig,
)
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.obstacle_route_candidates import (  # noqa: E402
    ObstacleRouteConfig,
    make_obstacle_route_candidates,
)
from encirclement3d.obstacle_route_runtime import (  # noqa: E402
    probe_independent_cbf_counterfactuals,
    probe_route_batch_with_cbf,
)
from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    build_action_conditioned_predictor,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import prepare_showcase_episode, scenario_from_metadata  # noqa: E402
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from evaluate_jepa_safe_capture_v2_paired import (  # noqa: E402
    _jsonable,
    _latency_stats,
    _load_scene_manifest,
    _raw_unverified_executed,
)
from evaluate_random_central_mixed_obstacles import config_for_spec, load_protocol  # noqa: E402

DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_route_v1_protocol.yaml"
DEFAULT_ENVIRONMENT = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"
DEFAULT_ACTOR = PROJECT_ROOT / "models" / "v5_development_exact_reactive_seed661606.pt"
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "jepa_route_recovery_dev_g1b_seed20260911" / "scene_manifest.jsonl"
DEFAULT_JEPA = PROJECT_ROOT / "results" / "jepa_route_identity_pairwise_pooling_actor_train20_seed20260908" / "checkpoint.pt"


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


def _fresh(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {label}: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--jepa-checkpoint", type=Path, default=DEFAULT_JEPA)
    parser.add_argument("--scene-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--shortlist-size", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cbf-horizon", type=int, default=3)
    parser.add_argument("--route-probe-horizon", type=int, default=3)
    parser.add_argument("--route-corridor-samples", type=int, default=65)
    parser.add_argument("--minimum-hold-steps", type=int, default=3)
    parser.add_argument("--switch-improvement-m", type=float, default=0.35)
    parser.add_argument("--tangent-route-hold-steps", type=int, default=6)
    parser.add_argument("--development-only", action="store_true")
    return parser.parse_args()


def _load_jepa(path: Path, device: torch.device) -> InteractionAwareActionConditionedSafeCaptureJEPAPredictor:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    model_type = str(checkpoint.get("model_type", ""))
    expected = {
        "interaction_aware_action_conditioned_jepa_route_identity_v1",
        "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2",
    }
    if model_type not in expected:
        raise ValueError(f"Expected a route-aware JEPA checkpoint, got {model_type!r}.")
    model_config = checkpoint.get("model")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("JEPA checkpoint must contain model and model_state_dict.")
    model = build_action_conditioned_predictor(model_type, model_config)
    if not isinstance(model, InteractionAwareActionConditionedSafeCaptureJEPAPredictor):
        raise TypeError("Route checkpoint is not compatible with the safe-capture ranker.")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "torch": version("torch"),
        "tensorboard": version("tensorboard"),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        values["cuda_device_name"] = torch.cuda.get_device_name(0)
        values["cuda_version"] = torch.version.cuda
    return values


def _planner_config(args: argparse.Namespace, dt: float) -> DNMPCConfig:
    return DNMPCConfig(
        horizon_steps=5,
        dt_seconds=float(dt),
        minimum_hold_steps=int(args.minimum_hold_steps),
        switch_improvement_m=float(args.switch_improvement_m),
        tangent_route_hold_steps=int(args.tangent_route_hold_steps),
    )


def _route_config(env: CaptureRadiusPursuit3DEnv, safety_filter: JointCBFQPSafetyFilter, args: argparse.Namespace) -> ObstacleRouteConfig:
    return ObstacleRouteConfig(
        chunk_length_steps=3,
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
    )


def _shortlist_batch(
    runtime: Any,
    planner_decision: Any,
    size: int,
    *,
    previous_route_index: int | None = None,
) -> tuple[SafeCaptureCandidateBatch, list[int]]:
    candidates = runtime.route_batch.candidates
    eligible = np.asarray(runtime.candidate_batch.valid_mask, dtype=bool)
    scores = np.asarray(planner_decision.scores, dtype=np.float64)
    finite = np.flatnonzero(eligible & np.isfinite(scores))
    ordered = sorted((int(index) for index in finite), key=lambda index: (float(scores[index]), index))
    # Keep the immutable nominal anchor at index zero for the ranker.  It is
    # retained even when the primary CBF prefilter marks it invalid.  The
    # previous global route is inserted before filling the remaining slots so
    # a changing analytic shortlist cannot erase JEPA hysteresis state.
    selected: list[int] = [0]
    for index in ([previous_route_index] if previous_route_index is not None else []) + ordered:
        if index is None or int(index) in selected:
            continue
        selected.append(int(index))
        if len(selected) >= max(1, int(size)):
            break
    chunks = np.stack([runtime.route_batch.chunks[index] for index in selected], axis=0)
    valid = np.asarray([eligible[index] for index in selected], dtype=bool)
    labels = tuple(runtime.route_batch.labels[index] for index in selected)
    reasons = tuple(runtime.candidate_batch.rejection_reasons[index] for index in selected)
    return SafeCaptureCandidateBatch(chunks=chunks, labels=labels, valid_mask=valid, rejection_reasons=reasons), selected


def _buffer_metrics(env: CaptureRadiusPursuit3DEnv, safety_filter: JointCBFQPSafetyFilter) -> dict[str, float]:
    positions = np.asarray(env.defender_positions, dtype=np.float64)
    radius = float(env.agents["drone_radius"])
    obstacle = [float(env._obstacle_clearance(p, item) - radius - safety_filter.obstacle_margin_m) for p in positions for item in env.obstacles]
    pairwise = [float(np.linalg.norm(positions[i] - positions[j]) - 2 * radius - safety_filter.inter_agent_margin_m) for i in range(env.n_defenders) for j in range(i + 1, env.n_defenders)]
    boundary = [float(value - radius - safety_filter.boundary_margin_m) for value in np.concatenate([positions - env.lower[None, :], env.upper[None, :] - positions]).reshape(-1)]
    return {
        "minimum_obstacle_buffer_clearance_m": min(obstacle) if obstacle else float("inf"),
        "minimum_pairwise_buffer_clearance_m": min(pairwise) if pairwise else float("inf"),
        "minimum_boundary_buffer_clearance_m": min(boundary) if boundary else float("inf"),
    }


def _run_episode(*, item: dict[str, Any], variant: str, config: dict[str, Any], actor: Any, action_scale: float, jepa: Any, device: torch.device, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    spec = dict(item["spec"])
    scenario = scenario_from_metadata(dict(item["scenario"]))
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=len(scenario.obstacles), target_speed_scale=float(spec["target_speed_scale"]))
    observation = prepare_showcase_episode(env, scenario, seed=int(spec["episode_seed"]), record_history=True, validate_scenario=False)
    local_observation = policy_observations(env, observation)
    safety_filter = JointCBFQPSafetyFilter(env, anticipatory_horizon_steps=int(args.cbf_horizon), barrier_mode="strict_buffer")
    planner = DistributedMinimaxMPC(_planner_config(args, float(env.dt)))
    ranker = None
    history = None
    if variant == "jepa":
        history = SafeCaptureCandidateHistory(jepa, defender_count=env.n_defenders, device=device, history_length=8, action_scale=float(action_scale))
        history.reset(local_observation)
        ranker = SafeCaptureJEPARanker(history, config=SafeCaptureRankerConfig(
            fixed_point_score_comparison=True,
            minimum_hold_steps=int(args.minimum_hold_steps),
            route_switch_penalty_m=0.15,
            target_escape_alignment_weight=0.10,
        ))
    previous_action = np.asarray(env.defender_velocities, dtype=np.float64).copy()
    # Keep the previous route in the global route-batch index space.  The JEPA
    # shortlist is re-ordered each cycle, so a local shortlist index cannot be
    # carried across replans without silently disabling hysteresis.
    previous_selected_route_index: int | None = None
    hold_steps_remaining = 0
    last_route_id: str | None = None
    route_switches = 0
    route_counts: dict[str, int] = {}
    counters = {"cbf_checks": 0, "cbf_accepted": 0, "cbf_rejected": 0, "cbf_timeouts": 0, "controlled_abort": 0, "fallback": 0, "unverified": 0, "raw_unverified": 0, "independent_checks": 0, "independent_accepted": 0, "independent_rejected": 0}
    latencies: dict[str, list[float]] = {name: [] for name in ("actor", "route_generation", "route_probe", "planner", "jepa", "cbf_filter", "cycle")}
    traces: list[dict[str, Any]] = []
    min_obstacle = min_pairwise = min_boundary = float("inf")
    min_buffers = {"minimum_obstacle_buffer_clearance_m": float("inf"), "minimum_pairwise_buffer_clearance_m": float("inf"), "minimum_boundary_buffer_clearance_m": float("inf")}
    info: dict[str, Any] = {}
    while True:
        cycle_start = time.perf_counter_ns()
        hidden = actor.initial_actor_hidden(env.n_defenders, device=device) if hasattr(actor, "initial_actor_hidden") else None
        actor_start = time.perf_counter_ns()
        with torch.no_grad():
            distribution, _hidden = actor.distribution_step(torch.as_tensor(local_observation, device=device), hidden) if hidden is not None else (actor.distribution(torch.as_tensor(local_observation, device=device)), None)
            desired = torch.tanh(distribution.mean).cpu().numpy() * float(action_scale)
        latencies["actor"].append((time.perf_counter_ns() - actor_start) / 1e6)
        desired = np.asarray(desired, dtype=np.float64)
        reachable = env._move_toward_velocity(previous_action, env._clip_rows(desired, float(env.agents["defender_max_speed"])), max_delta=float(env.agents["defender_max_acceleration"]) * float(env.dt))
        route_start = time.perf_counter_ns()
        route_batch = make_obstacle_route_candidates(reachable, observation, config=_route_config(env, safety_filter, args), previous_action=previous_action)
        latencies["route_generation"].append((time.perf_counter_ns() - route_start) / 1e6)
        probe_start = time.perf_counter_ns()
        runtime = probe_route_batch_with_cbf(route_batch, safety_filter, observation, horizon_steps=int(args.route_probe_horizon))
        latencies["route_probe"].append((time.perf_counter_ns() - probe_start) / 1e6)
        probes = runtime.cbf_counterfactuals
        counters["cbf_checks"] += sum(probe is not None for probe in probes)
        counters["cbf_accepted"] += sum(probe is not None and probe.accepted for probe in probes)
        counters["cbf_rejected"] += sum(probe is not None and not probe.accepted for probe in probes)
        counters["cbf_timeouts"] += sum(probe is not None and probe.timed_out for probe in probes)
        planner_start = time.perf_counter_ns()
        decision = planner.plan(route_batch, observation, previous_action=previous_action, eligible_mask=runtime.candidate_batch.valid_mask)
        latencies["planner"].append((time.perf_counter_ns() - planner_start) / 1e6)
        selected_index = decision.selected_index
        rank_trace = None
        rank_mode = "analytic"
        shortlist_indices: list[int] = []
        if variant == "jepa" and ranker is not None:
            shortlist, shortlist_indices = _shortlist_batch(
                runtime,
                decision,
                args.shortlist_size,
                previous_route_index=previous_selected_route_index,
            )
            jepa_start = time.perf_counter_ns()
            previous_local_index = (
                shortlist_indices.index(previous_selected_route_index)
                if previous_selected_route_index in shortlist_indices
                else None
            )
            rank = ranker.rank(
                observation,
                shortlist,
                previous_action=previous_action,
                previous_selected_index=previous_local_index,
                hold_steps_remaining=hold_steps_remaining,
            )
            latencies["jepa"].append((time.perf_counter_ns() - jepa_start) / 1e6)
            local_index = int(rank.selected_index)
            selected_index = int(shortlist_indices[local_index])
            rank_trace = rank.trace.as_dict()
            rank_mode = rank.execution_mode
            if rank.execution_mode == "trusted":
                hold_steps_remaining = int(rank.trace.hold_steps_remaining)
                previous_selected_route_index = selected_index
            else:
                hold_steps_remaining = 0
                previous_selected_route_index = shortlist_indices[0]
        else:
            latencies["jepa"].append(0.0)
        if selected_index is None:
            requested = reachable.copy()
            selected_route = None
        else:
            selected_route = route_batch.candidates[int(selected_index)]
            requested = np.asarray(selected_route.action_chunk[0], dtype=np.float64)
            route_counts[selected_route.label] = route_counts.get(selected_route.label, 0) + 1
            if last_route_id is not None and last_route_id != selected_route.route_id:
                route_switches += 1
            last_route_id = selected_route.route_id
        safe_hold = np.asarray(observation["defender_velocities"], dtype=np.float64)
        independent = probe_independent_cbf_counterfactuals(selected_action=requested, nominal_action=reachable, safe_hold_action=safe_hold, observation=observation, safety_filter=safety_filter, selected_route_id=None if selected_route is None else selected_route.route_id)
        counters["independent_checks"] += len(independent)
        counters["independent_accepted"] += sum(probe.accepted for probe in independent)
        counters["independent_rejected"] += sum(not probe.accepted for probe in independent)
        cbf_start = time.perf_counter_ns()
        action, diagnostics = safety_filter.filter(requested, observation, nominal_actions=reachable)
        latencies["cbf_filter"].append((time.perf_counter_ns() - cbf_start) / 1e6)
        counters["fallback"] += int(diagnostics.used_fallback)
        counters["unverified"] += int(not diagnostics.verified_feasible)
        counters["controlled_abort"] += int(diagnostics.fallback_mode == "controlled_abort")
        counters["raw_unverified"] += int(_raw_unverified_executed(safety_filter_enabled=True, diagnostics=diagnostics))
        action = np.asarray(action, dtype=np.float64)
        observation, _reward, terminated, truncated, info = env.step(action, record_history=True)
        safety_obstacle = min((float(env._obstacle_clearance(p, item)) - float(env.agents["drone_radius"]) for p in env.defender_positions for item in env.obstacles), default=float("inf"))
        pairwise = min((float(np.linalg.norm(env.defender_positions[i] - env.defender_positions[j])) - 2 * float(env.agents["drone_radius"]) for i in range(env.n_defenders) for j in range(i + 1, env.n_defenders)), default=float("inf"))
        boundary = float(np.min(np.concatenate([env.defender_positions - env.lower[None, :], env.upper[None, :] - env.defender_positions])))
        min_obstacle = min(min_obstacle, safety_obstacle)
        min_pairwise = min(min_pairwise, pairwise)
        min_boundary = min(min_boundary, boundary)
        buffers = _buffer_metrics(env, safety_filter)
        for key in min_buffers:
            min_buffers[key] = min(min_buffers[key], buffers[key])
        latencies["cycle"].append((time.perf_counter_ns() - cycle_start) / 1e6)
        traces.append({
            "step": int(env.step_count),
            "variant": variant,
            "desired_action": desired,
            "reachable_nominal_action": reachable,
            "requested_action": requested,
            "executed_action": action,
            "selected_route": None if selected_route is None else selected_route.as_dict(),
            "shortlist_indices": shortlist_indices,
            "decision": decision.as_dict(),
            "rank_mode": rank_mode,
            "rank_trace": rank_trace,
            "route_cbf_counterfactuals": [None if probe is None else probe.as_dict() for probe in probes],
            "independent_cbf_counterfactuals": [probe.as_dict() for probe in independent],
            "cbf": diagnostics,
            "minimum_obstacle_clearance_m": safety_obstacle,
            "minimum_pairwise_clearance_m": pairwise,
            "minimum_boundary_clearance_m": boundary,
        })
        if history is not None and not diagnostics.verified_feasible:
            break
        if diagnostics.fallback_mode == "controlled_abort" or terminated or truncated:
            break
        local_observation = policy_observations(env, observation)
        if history is not None:
            history.observe_after_action(local_observation, action)
        previous_action = action.copy()
    safe_capture = bool(info.get("safe_capture_success", False)) and not bool(env.defender_boundary_violation) and min_pairwise >= -1e-9 and min_obstacle >= -1e-9 and counters["raw_unverified"] == 0 and counters["controlled_abort"] == 0
    row = {
        "variant": variant,
        "episode_index": int(item["episode_index"]),
        "seed": int(spec["episode_seed"]),
        "scenario": scenario.name,
        "safe_capture_success": bool(safe_capture),
        "capture_event": bool(info.get("capture_event", False)),
        "capture_time_seconds": info.get("capture_time_seconds"),
        "collision": bool(info.get("collision", False) or min_obstacle < -1e-9),
        "boundary_violation": bool(env.defender_boundary_violation),
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
        "latency": {name: _latency_stats(values) for name, values in latencies.items()},
    }
    trace_dir = output_dir / variant / "step_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    _write_json(trace_dir / f"episode_{int(item['episode_index']):03d}.json", {**row, "traces": traces})
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty paired replay.")
    return {
        "episodes": len(rows),
        "safe_capture_count": int(sum(bool(row["safe_capture_success"]) for row in rows)),
        "safe_capture_rate": float(np.mean([bool(row["safe_capture_success"]) for row in rows])),
        "collision_count": int(sum(bool(row["collision"]) for row in rows)),
        "boundary_violation_count": int(sum(bool(row["boundary_violation"]) for row in rows)),
        "pairwise_violation_count": int(sum(bool(row["pairwise_violation"]) for row in rows)),
        "controlled_abort_steps": int(sum(row["route_counters"]["controlled_abort"] for row in rows)),
        "raw_unverified_executed_steps": int(sum(row["route_counters"]["raw_unverified"] for row in rows)),
        "route_switch_steps": int(sum(row["route_switch_steps"] for row in rows)),
        "mean_capture_time_seconds": float(np.mean([row["capture_time_seconds"] for row in rows if row["capture_time_seconds"] is not None])) if any(row["capture_time_seconds"] is not None for row in rows) else None,
        "worst_min_clearance_m": float(min(row["min_clearance_m"] for row in rows)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["variant", "episode_index", "seed", "scenario", "safe_capture_success", "collision", "boundary_violation", "pairwise_violation", "steps", "termination_reason", "capture_time_seconds", "route_switch_steps"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    args = _parse_args()
    if not args.development_only:
        raise SystemExit("This evaluator is development-only; pass --development-only explicitly.")
    if args.episodes != 4 or args.shortlist_size <= 0:
        raise SystemExit("The frozen G5 paired replay requires exactly 4 episodes and a positive shortlist size.")
    if args.cbf_horizon <= 0 or args.route_probe_horizon <= 0:
        raise SystemExit("CBF and route probe horizons must be positive.")
    output_dir = _fresh(args.output_dir, "output directory")
    tensorboard_dir = _fresh(args.tensorboard_dir, "TensorBoard directory")
    device = select_device(args.device)
    protocol = load_protocol(args.protocol)
    manifest = _load_scene_manifest(args.scene_manifest, protocol, args.environment_config, split="validation", episodes=args.episodes)
    probe_spec = dict(manifest[0]["spec"])
    probe_config = config_for_spec("f2", probe_spec, args.environment_config)
    probe_env = CaptureRadiusPursuit3DEnv(probe_config, obstacle_count=len(manifest[0]["scenario"]["obstacles"]), target_speed_scale=float(probe_spec["target_speed_scale"]))
    actor, action_scale, actor_checkpoint = load_policy(args.actor_checkpoint, probe_env, probe_env.reset(seed=int(probe_spec["episode_seed"]), record_history=True), device)
    jepa = _load_jepa(args.jepa_checkpoint, device)
    rows: list[dict[str, Any]] = []
    for variant in ("dn_mpc_cbf", "dn_mpc_jepa_cbf"):
        for item in manifest:
            config = config_for_spec("f2", dict(item["spec"]), args.environment_config)
            rows.append(_run_episode(item=item, variant="baseline" if variant == "dn_mpc_cbf" else "jepa", config=config, actor=actor, action_scale=action_scale, jepa=jepa, device=device, args=args, output_dir=output_dir))
    summaries = {variant: _summary([row for row in rows if row["variant"] == variant]) for variant in ("baseline", "jepa")}
    metadata = {
        "evaluation_type": "dn_mpc_jepa_cbf_g5_paired_development",
        "development_only": True,
        "locked_test_opened": False,
        "episodes": args.episodes,
        "git_revision": _git_revision(),
        "inputs": {
            "protocol": str(args.protocol.resolve()), "protocol_sha256": _sha256(args.protocol),
            "environment_config": str(args.environment_config.resolve()), "environment_config_sha256": _sha256(args.environment_config),
            "actor_checkpoint": str(args.actor_checkpoint.resolve()), "actor_checkpoint_sha256": _sha256(args.actor_checkpoint),
            "jepa_checkpoint": str(args.jepa_checkpoint.resolve()), "jepa_checkpoint_sha256": _sha256(args.jepa_checkpoint),
            "scene_manifest": str(args.scene_manifest.resolve()), "scene_manifest_sha256": _sha256(args.scene_manifest),
        },
        "contract": {
            "route_chunk_length_steps": 3, "route_probe_horizon": args.route_probe_horizon,
            "cbf_horizon": args.cbf_horizon, "shortlist_size": args.shortlist_size,
            "execute_first_step_then_replan": True, "jepa_only_ranks_dn_mpc_shortlist": True,
            "project_to_reachable_dynamics": True, "raw_unverified_execution_allowed": False,
            "cbf_margin_changed": False, "controlled_abort_preserved": True,
        },
        "environment": _environment_metadata(device),
        "tensorboard_dir": str(tensorboard_dir),
    }
    _write_csv(output_dir / "episodes.csv", rows)
    _write_json(output_dir / "summary.json", {"summaries": summaries, "metadata": metadata, "episodes": [{key: value for key, value in row.items() if key != "traces"} for row in rows]})
    _write_json(output_dir / "provenance.json", metadata)
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=2) as writer:
        writer.add_text("Provenance/metadata", json.dumps(_jsonable(metadata), indent=2), 0)
        for variant, summary in summaries.items():
            prefix = "Baseline" if variant == "baseline" else "JEPA"
            writer.add_scalar(f"{prefix}/safe_capture_rate", summary["safe_capture_rate"], 0)
            writer.add_scalar(f"{prefix}/collision_count", summary["collision_count"], 0)
            writer.add_scalar(f"{prefix}/boundary_violation_count", summary["boundary_violation_count"], 0)
            writer.add_scalar(f"{prefix}/pairwise_violation_count", summary["pairwise_violation_count"], 0)
            writer.add_scalar(f"{prefix}/controlled_abort_steps", summary["controlled_abort_steps"], 0)
            writer.add_scalar(f"{prefix}/route_switch_steps", summary["route_switch_steps"], 0)
        for index, row in enumerate(rows):
            prefix = "Baseline" if row["variant"] == "baseline" else "JEPA"
            writer.add_scalar(f"{prefix}/episode_safe_capture", float(row["safe_capture_success"]), index)
            writer.add_scalar(f"{prefix}/episode_route_switch_steps", float(row["route_switch_steps"]), index)
            writer.add_scalar(f"{prefix}/episode_min_clearance_m", float(row["min_clearance_m"]), index)
        writer.flush()
    print(json.dumps(_jsonable({"summaries": summaries, "metadata": metadata}), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
