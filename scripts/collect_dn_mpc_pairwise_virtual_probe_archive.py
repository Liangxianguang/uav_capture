"""Collect offline-only pairwise virtual probes for a closed development split.

The probe actions are never sent to ``env.step``.  Positions are advanced by
the kinematic equation on an isolated mathematical state copy, while the
current strict Joint CBF verifier is queried read-only for each virtual step.
The resulting archive is never a runtime controller or a safety certificate.
Train/validation/development collection remains closed to locked test data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import collect_jepa_route_identity_archive as archive  # noqa: E402
from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.obstacle_route_candidates import make_obstacle_route_candidates  # noqa: E402
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_controllers import DynamicEncirclementController  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    prepare_showcase_episode,
    random_central_mixed_obstacle_scenario,
    scenario_metadata,
)


SAMPLE_TYPE = 5
SAMPLE_TYPE_NAME = "pairwise_virtual_probe"
PAIRWISE_PAIRS = tuple(itertools.combinations(range(4), 2))
HORIZON = 5
CONTRACT_VERSION = "dn_mpc_pairwise_virtual_probe_archive_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fresh(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {label}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _probe_chunk(
    positions: np.ndarray,
    *,
    pair: tuple[int, int] | None,
    max_speed: float,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Build a deliberately convergent or hold action chunk without execution."""

    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (4, 3):
        raise ValueError(f"positions must have shape (4,3), got {positions.shape}")
    if horizon <= 0 or max_speed <= 0.0:
        raise ValueError("horizon and max_speed must be positive")
    actions = np.zeros((horizon, 4, 3), dtype=np.float32)
    if pair is None:
        return actions
    first, second = pair
    delta = positions[second] - positions[first]
    norm = float(np.linalg.norm(delta))
    direction = delta / norm if norm > 1e-9 and np.isfinite(norm) else np.array([1.0, 0.0, 0.0])
    actions[:, first] = (direction * float(max_speed)).astype(np.float32)
    actions[:, second] = (-direction * float(max_speed)).astype(np.float32)
    return actions


def _clearance_for_positions(env: CaptureRadiusPursuit3DEnv, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    radius = float(env.agents["drone_radius"])
    obstacle = np.full(env.n_defenders, 50.0, dtype=np.float64)
    pairwise = np.full(env.n_defenders, 50.0, dtype=np.float64)
    boundary = np.full(env.n_defenders, 50.0, dtype=np.float64)
    for index, position in enumerate(positions):
        if env.obstacles:
            obstacle[index] = min(float(env._obstacle_clearance(position, item) - radius) for item in env.obstacles)
        teammate = [
            float(np.linalg.norm(position - other) - 2.0 * radius)
            for other_index, other in enumerate(positions)
            if other_index != index
        ]
        if teammate:
            pairwise[index] = min(teammate)
        boundary[index] = min(
            float(value)
            for value in np.concatenate([position - env.lower - radius, env.upper - position - radius])
        )
    return obstacle.astype(np.float32), pairwise.astype(np.float32), boundary.astype(np.float32)


def _virtual_probe_rollout(
    env: CaptureRadiusPursuit3DEnv,
    observation: Mapping[str, Any],
    action_chunk: np.ndarray,
    safety_filter: JointCBFQPSafetyFilter,
    *,
    extent: float,
    pairwise_margin_m: float,
) -> dict[str, np.ndarray | int | bool]:
    """Roll a probe on arrays only; no simulator step and no CBF execution."""

    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    if action_chunk.shape != (HORIZON, env.n_defenders, 3):
        raise ValueError(f"action_chunk must have shape ({HORIZON},{env.n_defenders},3)")
    positions = np.asarray(env.defender_positions, dtype=np.float64).copy()
    target_position = np.asarray(env.target_position, dtype=np.float64).copy()
    initial_distances = np.linalg.norm(target_position[None, :] - positions, axis=1)
    probe_env = copy.deepcopy(env)
    max_speed = float(env.agents["defender_max_speed"])
    dt = float(env.dt)
    max_acceleration = float(env.agents["defender_max_acceleration"])
    labels: dict[str, list[np.ndarray]] = {name: [] for name in (
        "relative", "obstacle_clearance", "inter_agent_clearance", "boundary_clearance",
        "stopping_distance", "obstacle_ttc", "boundary_ttc", "pairwise_ttc",
        "acceleration_slack", "target_visible", "cbf_correction", "cbf_intervention",
        "cbf_feasible", "cbf_min_slack", "route_progress", "rollout_valid",
    )}
    target_visible = np.asarray(observation.get("target_visible", np.ones(env.n_defenders)), dtype=np.float32)
    if target_visible.shape != (env.n_defenders,):
        target_visible = np.ones(env.n_defenders, dtype=np.float32)
    for step in range(HORIZON):
        requested = env._clip_rows(action_chunk[step], max_speed)
        virtual_observation = dict(observation)
        virtual_observation["defender_positions"] = positions.copy()
        virtual_observation["defender_velocities"] = requested.copy()
        diagnostics = safety_filter.verify_requested_action(requested, virtual_observation)
        obstacle, pairwise, boundary = _clearance_for_positions(env, positions)
        probe_env.defender_positions = positions.copy()
        probe_env.defender_velocities = requested.copy()
        risk = archive._risk_labels(
            probe_env,
            requested,
            obstacle_margin_m=float(safety_filter.obstacle_margin_m),
            boundary_margin_m=float(safety_filter.boundary_margin_m),
            inter_agent_margin_m=float(pairwise_margin_m),
        )
        target_distances = np.linalg.norm(target_position[None, :] - positions, axis=1)
        feasible = bool(
            diagnostics.verified_feasible
            and not diagnostics.infeasible
            and not diagnostics.timed_out
            and diagnostics.fallback_mode == "none"
        )
        correction = float(diagnostics.action_correction_norm)
        labels["relative"].append(((target_position[None, :] - positions) / extent).astype(np.float32))
        labels["obstacle_clearance"].append(obstacle)
        labels["inter_agent_clearance"].append(pairwise)
        labels["boundary_clearance"].append(boundary)
        labels["stopping_distance"].append((np.linalg.norm(requested, axis=1) ** 2 / max(2.0 * max_acceleration, 1e-12)).astype(np.float32))
        labels["obstacle_ttc"].append(risk["obstacle_ttc"])
        labels["boundary_ttc"].append(risk["boundary_ttc"])
        labels["pairwise_ttc"].append(risk["pairwise_ttc"])
        labels["acceleration_slack"].append(archive._acceleration_slack_labels(diagnostics, env.n_defenders))
        labels["target_visible"].append(target_visible.copy())
        labels["cbf_correction"].append(np.full(env.n_defenders, correction, dtype=np.float32))
        labels["cbf_intervention"].append(np.full(env.n_defenders, float(correction > 1e-6), dtype=np.float32))
        labels["cbf_feasible"].append(np.full(env.n_defenders, float(feasible), dtype=np.float32))
        labels["cbf_min_slack"].append(np.full(env.n_defenders, float(diagnostics.minimum_constraint_value), dtype=np.float32))
        labels["route_progress"].append(((initial_distances - target_distances) / extent).astype(np.float32))
        labels["rollout_valid"].append(np.ones(env.n_defenders, dtype=np.float32))
        # This is a virtual state transition only.  No CBF-filtered action is
        # applied and no branch is declared to have executed or failed.
        positions = positions + requested * dt
    return {
        key: np.stack(value, axis=0).astype(np.float32)
        for key, value in labels.items()
    } | {
        "earliest_failure_step": HORIZON + 1,
        "branch_terminated": False,
        "cbf_failed": bool(np.any(np.asarray(labels["cbf_feasible"]) < 0.5)),
    }


def _append_virtual_probe(
    samples: dict[str, list[Any]],
    *,
    observation_history: list[np.ndarray],
    executed_action_history: list[np.ndarray],
    route: Any,
    labels: Mapping[str, Any],
    episode_seed: int,
    scenario_index: int,
    time_index: int,
    action_scale: float,
    mode: int,
    pair_index: int,
) -> None:
    archive._append_samples(
        samples,
        observation_history=observation_history,
        executed_action_history=executed_action_history,
        route=route,
        route_index=-1,
        labels=labels,
        episode_seed=episode_seed,
        scenario_index=scenario_index,
        time_index=time_index,
        action_scale=action_scale,
        sample_type=SAMPLE_TYPE,
    )
    samples["virtual_probe_mode"].extend([int(mode)] * 4)
    samples["virtual_probe_pair_index"].extend([pair_index] * 4)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def collect(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    from evaluate_random_central_mixed_obstacles import config_for_spec, episode_spec, load_protocol
    from evaluate_capture_radius_mappo import load_policy, select_device

    protocol = load_protocol(args.protocol.resolve())
    env_config_path = args.environment_config.resolve()
    base_config = yaml.safe_load(env_config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError("environment config must be a mapping")
    if args.episodes <= 0 or args.sample_stride <= 0 or args.chunk_length_steps != HORIZON:
        raise ValueError("P14 requires positive episodes/stride and chunk_length_steps=5")
    samples = archive._empty_samples()
    samples["virtual_probe_mode"] = []
    samples["virtual_probe_pair_index"] = []
    scenes: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    probe_count = 0
    actor_policy = None
    actor_device = select_device(args.actor_device)
    actor_action_scale = 5.0
    actor_metadata: dict[str, Any] = {}
    for scenario_index in range(args.episodes):
        spec = episode_spec(protocol, args.split, scenario_index)
        config = config_for_spec("f2", spec, env_config_path)
        config["task"]["pursuit"]["terminate_on_capture"] = False
        config["task"]["pursuit"]["obstacle_profile"] = "mixed"
        probe_env = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=int(spec["obstacle_count"]),
            target_speed_scale=float(spec["target_speed_scale"]),
        )
        scenario = random_central_mixed_obstacle_scenario(
            probe_env,
            layout_seed=int(spec["layout_seed"]),
            initial_side_distance=float(spec["initial_side_distance"]),
            defender_side=str(spec["defender_side"]),
            target_crossing_required=bool(spec["target_crossing_required"]),
            obstacle_count_range=(int(spec["obstacle_count"]), int(spec["obstacle_count"])),
            max_attempts=int(protocol["s3"].get("max_sampling_attempts", 500)),
            required_defender_zone_entries=int(protocol["s3"].get("required_defender_zone_entries", 1)),
        )
        env = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=len(scenario.obstacles),
            target_speed_scale=float(spec["target_speed_scale"]),
        )
        observation = prepare_showcase_episode(env, scenario, seed=int(spec["episode_seed"]), record_history=False)
        controller = DynamicEncirclementController(env)
        if actor_policy is None:
            actor_policy, actor_action_scale, actor_metadata = load_policy(
                args.actor_checkpoint.resolve(), env, observation, actor_device
            )
        safety_filter = JointCBFQPSafetyFilter(
            env,
            anticipatory_horizon_steps=5,
            barrier_mode="strict_buffer",
        )
        route_config = archive._route_config(env, safety_filter, HORIZON)
        extent = float(config["world"]["half_extent_xy"])
        observation_history = [policy_observations(env, observation).copy()]
        executed_actions: list[np.ndarray] = []
        previous_action = np.asarray(env.defender_velocities, dtype=np.float64).copy()
        hidden = actor_policy.initial_actor_hidden(env.n_defenders, device=actor_device) if hasattr(actor_policy, "initial_actor_hidden") else None
        reset_interval = actor_metadata.get("recurrent_reset_interval_steps")
        sampled_states = 0
        for time_index in range(int(env.max_steps)):
            if hidden is not None and reset_interval is not None and time_index > 0 and time_index % int(reset_interval) == 0:
                hidden = actor_policy.initial_actor_hidden(env.n_defenders, device=actor_device)
            desired, hidden = archive._actor_action(
                actor_policy,
                policy_observations(env, observation),
                actor_device,
                actor_action_scale,
                hidden,
            )
            reachable_nominal = env._move_toward_velocity(
                previous_action,
                env._clip_rows(desired, float(env.agents["defender_max_speed"])),
                max_delta=float(env.agents["defender_max_acceleration"]) * float(env.dt),
            )
            if time_index >= args.history_length - 1 and (time_index - (args.history_length - 1)) % args.sample_stride == 0:
                sampled_states += 1
                route_batch = make_obstacle_route_candidates(
                    reachable_nominal,
                    observation,
                    config=route_config,
                    previous_action=previous_action,
                )
                if not route_batch.candidates:
                    raise RuntimeError("P14 route generator returned no template candidate")
                template = route_batch.candidates[0]
                for pair_index, pair in enumerate(PAIRWISE_PAIRS):
                    chunk = _probe_chunk(
                        env.defender_positions,
                        pair=pair,
                        max_speed=float(env.agents["defender_max_speed"]),
                    )
                    route = archive._offline_interaction_route(template, f"virtual_pair:{pair_index}", chunk)
                    labels = _virtual_probe_rollout(
                        env,
                        observation,
                        chunk,
                        safety_filter,
                        extent=extent,
                        pairwise_margin_m=float(safety_filter.inter_agent_margin_m),
                    )
                    _append_virtual_probe(
                        samples,
                        observation_history=observation_history,
                        executed_action_history=executed_actions,
                        route=route,
                        labels=labels,
                        episode_seed=int(spec["episode_seed"]),
                        scenario_index=scenario_index,
                        time_index=time_index,
                        action_scale=5.0,
                        mode=0,
                        pair_index=pair_index,
                    )
                    mode_counts["converge_pair"] += env.n_defenders
                    probe_count += env.n_defenders
                hold_chunk = _probe_chunk(
                    env.defender_positions,
                    pair=None,
                    max_speed=float(env.agents["defender_max_speed"]),
                )
                hold_chunk[:] = previous_action[None, :, :]
                hold_route = archive._offline_interaction_route(template, "virtual_hold", hold_chunk)
                hold_labels = _virtual_probe_rollout(
                    env,
                    observation,
                    hold_chunk,
                    safety_filter,
                    extent=extent,
                    pairwise_margin_m=float(safety_filter.inter_agent_margin_m),
                )
                _append_virtual_probe(
                    samples,
                    observation_history=observation_history,
                    executed_action_history=executed_actions,
                    route=hold_route,
                    labels=hold_labels,
                    episode_seed=int(spec["episode_seed"]),
                    scenario_index=scenario_index,
                    time_index=time_index,
                    action_scale=5.0,
                    mode=1,
                    pair_index=-1,
                )
                mode_counts["safe_hold"] += env.n_defenders
                probe_count += env.n_defenders
            action, diagnostics = safety_filter.filter(desired, observation, nominal_actions=reachable_nominal)
            if not diagnostics.verified_feasible or diagnostics.fallback_mode == "controlled_abort":
                break
            observation, _reward, terminated, truncated, _info = env.step(action, record_history=False)
            executed_actions.append(np.asarray(action, dtype=np.float32).copy())
            observation_history.append(policy_observations(env, observation).copy())
            previous_action = np.asarray(action, dtype=np.float64).copy()
            if terminated or truncated:
                break
        scene = scenario_metadata(scenario)
        scenes.append({
            "scenario_index": scenario_index,
            "episode_seed": int(spec["episode_seed"]),
            "layout_seed": int(spec["layout_seed"]),
            "scene_hash": hashlib.sha256(json.dumps(_jsonable(scene), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "spec": spec,
            "scenario": scene,
            "sampled_states": sampled_states,
        })
    arrays = archive._arrayize(samples)
    if not arrays["inputs"].size:
        raise RuntimeError("P14 produced no virtual probes")
    metadata = {
        "dataset_version": CONTRACT_VERSION,
        "task": "dn_mpc_pairwise_offline_virtual_probe_archive",
        "split": args.split,
        "development_only": True,
        "locked_test_opened": False,
        "offline_only": True,
        "raw_unverified_action_executed": False,
        "sample_type_mapping": {SAMPLE_TYPE_NAME: SAMPLE_TYPE},
        "virtual_probe_mode_mapping": {"0": "converge_pair", "1": "safe_hold"},
        "virtual_probe_pair_vocab": [list(pair) for pair in PAIRWISE_PAIRS],
        "chunk_length_steps": HORIZON,
        "history_length": args.history_length,
        "sample_count": int(arrays["inputs"].shape[0]),
        "sample_type_counts": {SAMPLE_TYPE_NAME: int(np.sum(arrays["sample_type"] == SAMPLE_TYPE))},
        "mode_counts": dict(mode_counts),
        "episodes": args.episodes,
        "sample_stride": args.sample_stride,
        "pairwise_margin_m": float(args.pairwise_margin_m),
        "cbf_contract": {"anticipatory_horizon_steps": 5, "barrier_mode": "strict_buffer", "controlled_abort_preserved": True},
        "state_distribution_source": {
            "mode": "frozen_runtime_actor",
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "actor_checkpoint_sha256": _sha256(args.actor_checkpoint.resolve()),
            "actor_recurrent_reset_interval_steps": actor_metadata.get("recurrent_reset_interval_steps"),
        },
        "information_boundary": {
            "target_truth_used_only_for_offline_labels": True,
            "virtual_probe_actions_not_executed": True,
            "runtime_route_generation_uses_public_obstacles_and_target_beliefs": True,
        },
        "source": {
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": _sha256(args.protocol.resolve()),
            "environment_config": str(env_config_path),
            "environment_config_sha256": _sha256(env_config_path),
            "collector_git_revision": _git_revision(),
        },
        "tensorboard_namespace": str(args.tensorboard_namespace),
    }
    return arrays, metadata, scenes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--split", choices=("train", "validation", "calibration", "development"), default="calibration")
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--chunk-length-steps", type=int, default=5)
    parser.add_argument("--pairwise-margin-m", type=float, default=0.35)
    parser.add_argument("--actor-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--tensorboard-namespace", default="P14")
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("P14 virtual probes require --development-only")
    if args.pairwise_margin_m < 0.0:
        raise ValueError("pairwise margin must be non-negative")
    output_dir = _fresh(args.output_dir, "P14 output")
    tensorboard_dir = _fresh(args.tensorboard_logdir, "P14 TensorBoard logdir")
    arrays, metadata, scenes = collect(args)
    dataset_path = output_dir / "pairwise_virtual_probe_counterfactual.npz"
    metadata_path = output_dir / "metadata.json"
    np.savez_compressed(dataset_path, **arrays)
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "scenes.jsonl").write_text("".join(json.dumps(_jsonable(scene), sort_keys=True) + "\n" for scene in scenes), encoding="utf-8")
    provenance = {
        "dataset_sha256": _sha256(dataset_path),
        "metadata_sha256": _sha256(metadata_path),
        "scenes_sha256": _sha256(output_dir / "scenes.jsonl"),
        "git_revision": _git_revision(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "tensorboard_logdir": str(tensorboard_dir),
        "development_only": True,
        "locked_test_opened": False,
        "offline_only": True,
        "raw_unverified_action_executed": False,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sample_type = np.asarray(arrays["sample_type"]) == SAMPLE_TYPE
    strict_margin = np.asarray(arrays["labels_inter_agent_clearance"]) < float(args.pairwise_margin_m)
    cbf_infeasible = np.asarray(arrays["labels_cbf_feasible"]) < 0.5
    namespace = str(args.tensorboard_namespace).strip()
    if not namespace or any(char in namespace for char in "/\\"):
        raise ValueError("tensorboard namespace must be a non-empty path-safe token")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_scalar(f"{namespace}/probe_rows", float(sample_type.sum()), 0)
        writer.add_scalar(f"{namespace}/strict_margin_positive_cell_rate", float(strict_margin.mean()), 0)
        writer.add_scalar(f"{namespace}/strict_margin_positive_row_rate", float(np.any(strict_margin, axis=1).mean()), 0)
        writer.add_scalar(f"{namespace}/cbf_infeasible_cell_rate", float(cbf_infeasible.mean()), 0)
        writer.add_text(f"{namespace}/metadata", json.dumps(metadata, sort_keys=True), 0)
        writer.add_text(f"{namespace}/provenance", json.dumps(provenance, sort_keys=True), 0)
    summary = {
        "dataset": str(dataset_path),
        "metadata": str(metadata_path),
        "provenance": str(output_dir / "provenance.json"),
        "tensorboard": str(tensorboard_dir),
        "dataset_sha256": provenance["dataset_sha256"],
        "strict_margin_positive_cell_rate": float(strict_margin.mean()),
        "strict_margin_positive_row_rate": float(np.any(strict_margin, axis=1).mean()),
        "cbf_infeasible_cell_rate": float(cbf_infeasible.mean()),
        "mode_counts": metadata["mode_counts"],
        "offline_only": True,
        "raw_unverified_action_executed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
