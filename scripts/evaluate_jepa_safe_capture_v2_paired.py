"""Run paired development rollouts for JEPA safe-capture v2.

This evaluator is deliberately separate from the prediction-only evaluator and
from the historical locked-test scripts.  It creates one deterministic S3
scene manifest and runs one execution variant on that manifest.  Every
candidate action is either filtered by the joint multi-agent CBF-QP or, for
the explicit A3 diagnostic, marked as raw/no-CBF.  The script never accepts
the locked_test split and requires an explicit development-only flag.
"""

from __future__ import annotations

import argparse
import copy
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
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.jepa_safe_capture_candidates import (  # noqa: E402
    SafeCaptureCandidateConfig,
    SafeCaptureCandidateHistory,
    make_safe_capture_candidate_chunks,
)
from encirclement3d.jepa_safe_capture_ranker import (  # noqa: E402
    SafeCaptureJEPARanker,
    SafeCaptureRankerConfig,
)
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    build_action_conditioned_predictor,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.reliability import SafeCaptureReliabilityLedger  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    capture_contract_metrics,
    crossing_metrics,
    prepare_showcase_episode,
    random_central_mixed_obstacle_scenario,
    scenario_from_metadata,
    scenario_metadata,
    target_min_clearance,
    transit_execution_metrics,
    transit_route_metrics,
)
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from evaluate_random_central_mixed_obstacles import (  # noqa: E402
    config_for_spec,
    episode_spec,
    load_protocol,
)


VARIANTS = ("m0", "m1", "m2", "m3", "a1", "a2", "a3")
TRAINING_SEEDS = (20260911, 20260912, 20260913)
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_protocol.yaml"
DEFAULT_ENVIRONMENT_CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"
DEFAULT_ACTOR_CHECKPOINT = PROJECT_ROOT / "models" / "v5_development_exact_reactive_seed661606.pt"
DEFAULT_JEPA_ROOT = PROJECT_ROOT / "results"
DEFAULT_LEDGER_ROOT = PROJECT_ROOT / "results"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(value: Any) -> Any:
    """Convert numpy/dataclass values and non-finite floats to JSON values."""

    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fresh(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {label}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _scene_hash(metadata: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "torch": version("torch"),
        "tensorboard": version("tensorboard"),
        "pyyaml": version("PyYAML"),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        values["cuda_device_name"] = torch.cuda.get_device_name(0)
        values["cuda_version"] = torch.version.cuda
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR_CHECKPOINT)
    parser.add_argument("--jepa-checkpoint", type=Path)
    parser.add_argument("--reliability-ledger", type=Path)
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--jepa-history-length", type=int, default=8)
    parser.add_argument("--recurrent-reset-interval", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Required guard; this evaluator cannot open a locked test.",
    )
    return parser.parse_args()


def _load_jepa(
    checkpoint_path: Path,
    device: torch.device,
) -> InteractionAwareActionConditionedSafeCaptureJEPAPredictor:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=True)
    model_type = checkpoint.get("model_type")
    expected = "interaction_aware_action_conditioned_jepa_safe_capture_v2"
    if model_type != expected:
        raise ValueError(f"Expected {expected}, got {model_type!r}.")
    model_config = checkpoint.get("model")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("JEPA checkpoint must contain model and model_state_dict.")
    model = build_action_conditioned_predictor(str(model_type), model_config)
    if not isinstance(model, InteractionAwareActionConditionedSafeCaptureJEPAPredictor):
        raise TypeError("P6 requires the safe-capture v2 multitask JEPA predictor.")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _load_ledger(path: Path, checkpoint_path: Path) -> SafeCaptureReliabilityLedger:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reliability ledger must be a JSON object.")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Reliability ledger source metadata is missing.")
    checkpoint_hash = _sha256(checkpoint_path.resolve())
    if source.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Reliability ledger checkpoint hash does not match the JEPA checkpoint.")
    ledger = SafeCaptureReliabilityLedger(payload)
    if payload.get("locked_test_opened") is not False:
        raise ValueError("P6 requires locked_test_opened=false in the ledger.")
    return ledger


def _ranker_config(variant: str) -> SafeCaptureRankerConfig:
    # M1/M2 retain target and uncertainty terms but remove auxiliary safety
    # terms from the score. M3/A1 use the complete frozen score. A2 removes
    # only clearance and visibility terms while retaining TTC and CBF-risk.
    if variant in {"m1", "m2"}:
        return SafeCaptureRankerConfig(
            clearance_weight=0.0,
            ttc_weight=0.0,
            visibility_weight=0.0,
            cbf_risk_weight=0.0,
        )
    if variant == "a2":
        return SafeCaptureRankerConfig(clearance_weight=0.0, visibility_weight=0.0)
    return SafeCaptureRankerConfig()


def _variant_contract(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown P6 variant: {variant}")
    return {
        "variant": variant,
        "label": variant.upper(),
        "use_jepa": variant not in {"m0", "a3"},
        "use_ledger": variant in {"m2", "m3", "a2"},
        "use_cbf": variant != "a3",
        "use_auxiliary_score": variant in {"m3", "a1"},
        "diagnostic_only": variant == "a3",
    }


def _build_scene_manifest(
    protocol: dict[str, Any],
    environment_config: Path,
    *,
    split: str,
    episodes: int,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        spec = episode_spec(protocol, split, episode_index)
        config = config_for_spec("f2", spec, environment_config)
        sampler_env = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=0,
            target_speed_scale=float(spec["target_speed_scale"]),
        )
        scenario = random_central_mixed_obstacle_scenario(
            sampler_env,
            layout_seed=int(spec["layout_seed"]),
            initial_side_distance=float(spec["initial_side_distance"]),
            defender_side=str(spec["defender_side"]),
            target_crossing_required=bool(spec["target_crossing_required"]),
            obstacle_count_range=(int(spec["obstacle_count"]), int(spec["obstacle_count"])),
            max_attempts=int(protocol["s3"].get("max_sampling_attempts", 500)),
            required_defender_zone_entries=int(protocol["s3"].get("required_defender_zone_entries", 1)),
        )
        metadata = scenario_metadata(scenario)
        manifest.append(
            {
                "episode_index": int(episode_index),
                "episode_seed": int(spec["episode_seed"]),
                "layout_seed": int(spec["layout_seed"]),
                "spec": spec,
                "scenario": metadata,
                "scene_hash": _scene_hash(metadata),
            }
        )
    return manifest


def _load_scene_manifest(
    path: Path,
    protocol: dict[str, Any],
    environment_config: Path,
    *,
    split: str,
    episodes: int,
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != episodes:
        raise ValueError(f"Scene manifest has {len(records)} records; expected {episodes}.")
    expected_specs = [episode_spec(protocol, split, index) for index in range(episodes)]
    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if int(record.get("episode_index", -1)) != index:
            raise ValueError("Scene manifest episode indices are not contiguous.")
        if record.get("spec") != expected_specs[index]:
            raise ValueError("Scene manifest specification does not match the frozen protocol.")
        scenario_metadata_value = record.get("scenario")
        if not isinstance(scenario_metadata_value, Mapping):
            raise ValueError("Scene manifest scenario metadata is missing.")
        scene_hash = _scene_hash(scenario_metadata_value)
        if record.get("scene_hash") != scene_hash:
            raise ValueError("Scene manifest scene_hash is invalid.")
        # Constructing the scenario and validating once prevents malformed
        # external manifests from changing the runtime contract.
        scenario = scenario_from_metadata(dict(scenario_metadata_value))
        config = config_for_spec("f2", expected_specs[index], environment_config)
        validation_env = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=0,
            target_speed_scale=float(expected_specs[index]["target_speed_scale"]),
        )
        from encirclement3d.showcase import validate_showcase_scenario

        validate_showcase_scenario(validation_env, scenario)
        result.append(dict(record))
    return result


def _safety_observables(env: CaptureRadiusPursuit3DEnv) -> dict[str, float]:
    positions = np.asarray(env.defender_positions, dtype=np.float64)
    radius = float(env.agents["drone_radius"])
    obstacle_values = [
        float(env._obstacle_clearance(position, obstacle) - radius)
        for position in positions
        for obstacle in env.obstacles
    ]
    pairwise_values = [
        float(np.linalg.norm(positions[first] - positions[second]) - 2.0 * radius)
        for first in range(env.n_defenders)
        for second in range(first + 1, env.n_defenders)
    ]
    boundary_values = [
        float(value)
        for value in np.concatenate([positions - env.lower[None, :], env.upper[None, :] - positions]).reshape(-1)
    ]
    return {
        "minimum_obstacle_clearance_m": min(obstacle_values) if obstacle_values else float("inf"),
        "minimum_pairwise_clearance_m": min(pairwise_values) if pairwise_values else float("inf"),
        "minimum_boundary_clearance_m": min(boundary_values) if boundary_values else float("inf"),
    }


def _actor_action(
    policy: Any,
    local_observation: np.ndarray,
    device: torch.device,
    action_scale: float,
    hidden: torch.Tensor | None,
) -> tuple[np.ndarray, torch.Tensor | None]:
    local = torch.as_tensor(local_observation, device=device)
    with torch.no_grad():
        if hidden is not None:
            distribution, hidden = policy.distribution_step(local, hidden)
        else:
            distribution = policy.distribution(local)
        action = torch.tanh(distribution.mean).cpu().numpy() * float(action_scale)
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (local_observation.shape[0], 3) or not np.isfinite(action).all():
        raise RuntimeError("Frozen actor emitted a non-finite or malformed action.")
    return action, hidden


def _run_episode(
    *,
    manifest_item: dict[str, Any],
    config: dict[str, Any],
    policy: Any,
    action_scale: float,
    device: torch.device,
    contract: dict[str, Any],
    jepa: InteractionAwareActionConditionedSafeCaptureJEPAPredictor | None,
    ledger: SafeCaptureReliabilityLedger | None,
    history_length: int,
    recurrent_reset_interval: int | None,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_index = int(manifest_item["episode_index"])
    spec = dict(manifest_item["spec"])
    scenario = scenario_from_metadata(dict(manifest_item["scenario"]))
    seed = int(spec["episode_seed"])
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(spec["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(env, scenario, seed=seed, record_history=True, validate_scenario=False)
    local_observation = policy_observations(env, observation)
    candidate_history: SafeCaptureCandidateHistory | None = None
    ranker: SafeCaptureJEPARanker | None = None
    if contract["use_jepa"]:
        if jepa is None:
            raise RuntimeError("This variant requires a JEPA checkpoint.")
        candidate_history = SafeCaptureCandidateHistory(
            jepa,
            defender_count=env.n_defenders,
            device=device,
            history_length=history_length,
            action_scale=float(action_scale),
        )
        candidate_history.reset(local_observation)
        context_defaults = {
            "layout_signature": "+".join(
                f"{shape}{sum(item['shape'] == shape for item in manifest_item['scenario']['obstacles'])}"
                for shape in ("cylinder", "box", "wall")
                if any(item["shape"] == shape for item in manifest_item["scenario"]["obstacles"])
            ),
            "target_motion_mode": str(spec["target_motion_mode"]),
        }
        ranker = SafeCaptureJEPARanker(
            candidate_history,
            config=_ranker_config(str(contract["variant"])),
            reliability_ledger=ledger if contract["use_ledger"] else None,
            context_defaults=context_defaults,
        )
    safety_filter = JointCBFQPSafetyFilter(env) if contract["use_cbf"] else None
    hidden = policy.initial_actor_hidden(env.n_defenders, device=device) if hasattr(policy, "initial_actor_hidden") else None
    # Candidate validity is defined on the first-step command that is
    # reachable from the last executed velocity.  The frozen actor emits a
    # desired velocity and may legitimately request a large initial change;
    # rejecting that request before CBF would trap every JEPA variant in
    # permanent safe-hold from step one.
    previous_action = np.asarray(env.defender_velocities, dtype=np.float64).copy()
    candidate_config = SafeCaptureCandidateConfig(
        candidate_count=5,
        chunk_length_steps=3,
        perturbation_mps=0.10,
        max_speed_mps=float(env.agents["defender_max_speed"]),
        max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
        dt_seconds=float(env.dt),
        max_action_change_mps=float(env.agents["defender_max_acceleration"]) * float(env.dt),
        project_to_reachable_dynamics=True,
    )
    visible_fractions: list[float] = []
    message_ages: list[float] = []
    observation_ages: list[float] = []
    path_lengths = np.zeros(env.n_defenders, dtype=np.float64)
    previous_positions = env.defender_positions.copy()
    step_records: list[dict[str, Any]] = []
    cbf_latencies: list[float] = []
    cbf_corrections: list[float] = []
    selected_indices: list[int] = []
    ledger_states: list[str] = []
    fallback_modes: list[str] = []
    minimum_obstacle = float("inf")
    minimum_pairwise = float("inf")
    minimum_boundary = float("inf")
    cbf_infeasible_steps = 0
    cbf_timeout_steps = 0
    cbf_abort_steps = 0
    cbf_unverified_steps = 0
    cbf_fallback_steps = 0
    cbf_intervention_steps = 0
    rank_fallback_steps = 0
    safe_hold_steps = 0
    target_collision = False
    forced_termination_reason: str | None = None
    final_info: dict[str, Any] = {}
    while True:
        if (
            hidden is not None
            and recurrent_reset_interval is not None
            and env.step_count > 0
            and env.step_count % recurrent_reset_interval == 0
        ):
            hidden = policy.initial_actor_hidden(env.n_defenders, device=device)
        desired_action, hidden = _actor_action(policy, local_observation, device, action_scale, hidden)
        reachable_nominal_action = env._move_toward_velocity(
            previous_action,
            env._clip_rows(desired_action, float(env.agents["defender_max_speed"])),
            max_delta=float(env.agents["defender_max_acceleration"]) * float(env.dt),
        )
        requested_action = desired_action.copy()
        rank_result = None
        if ranker is not None and candidate_history is not None:
            batch = make_safe_capture_candidate_chunks(
                reachable_nominal_action,
                observation,
                config=candidate_config,
                previous_action=previous_action,
            )
            rank_result = ranker.rank(observation, batch, previous_action=previous_action)
            requested_action = np.asarray(rank_result.selected_action, dtype=np.float64)
            selected_indices.append(int(rank_result.selected_index))
            ledger_states.extend(list(rank_result.trace.ledger_states))
            if rank_result.execution_mode != "trusted":
                rank_fallback_steps += 1
            if rank_result.execution_mode == "safe_hold":
                safe_hold_steps += 1
        diagnostics = None
        if safety_filter is not None:
            execution_mode = "safe_hold" if rank_result is not None and rank_result.execution_mode == "safe_hold" else "normal"
            action, diagnostics = safety_filter.filter(
                requested_action,
                observation,
                nominal_actions=reachable_nominal_action,
                execution_mode=execution_mode,
            )
            action = np.asarray(action, dtype=np.float64)
            cbf_latencies.append(float(diagnostics.solve_latency_ms))
            cbf_corrections.append(float(diagnostics.action_correction_norm))
            fallback_modes.append(str(diagnostics.fallback_mode))
            if diagnostics.infeasible:
                cbf_infeasible_steps += 1
            if diagnostics.timed_out:
                cbf_timeout_steps += 1
            if diagnostics.used_fallback:
                cbf_fallback_steps += 1
            if diagnostics.action_correction_norm > 1e-8:
                cbf_intervention_steps += 1
            if not diagnostics.verified_feasible:
                cbf_unverified_steps += 1
            if diagnostics.fallback_mode == "controlled_abort":
                cbf_abort_steps += 1
        else:
            action = requested_action
            fallback_modes.append("none")
        if not np.isfinite(action).all():
            raise RuntimeError("Execution action became non-finite.")
        observation, _reward, terminated, truncated, final_info = env.step(action, record_history=True)
        path_lengths += np.linalg.norm(env.defender_positions - previous_positions, axis=1)
        previous_positions = env.defender_positions.copy()
        visible_fractions.append(float(final_info["target_visible_fraction"]))
        message_ages.append(float(final_info["mean_message_age_steps"]))
        observation_ages.append(float(final_info["mean_observation_age_steps"]))
        safety_values = _safety_observables(env)
        minimum_obstacle = min(minimum_obstacle, safety_values["minimum_obstacle_clearance_m"])
        minimum_pairwise = min(minimum_pairwise, safety_values["minimum_pairwise_clearance_m"])
        minimum_boundary = min(minimum_boundary, safety_values["minimum_boundary_clearance_m"])
        target_clearance = min(
            float(env._obstacle_clearance(env.target_position, obstacle)) for obstacle in env.obstacles
        ) if env.obstacles else float("inf")
        if target_clearance < 0.0:
            target_collision = True
        step_records.append(
            {
                "episode_index": episode_index,
                "step": int(env.step_count),
                "desired_action": desired_action,
                "reachable_nominal_action": reachable_nominal_action,
                "requested_action": requested_action,
                "executed_action": action,
                "observation": {
                    "target_visible": observation.get("target_visible"),
                    "target_observation_age_steps": observation.get("target_observation_age_steps"),
                    "message_age_steps": observation.get("message_age_steps"),
                },
                "safety_observables": safety_values,
                "target_clearance_m": target_clearance,
                "candidate_ranking": rank_result.trace if rank_result is not None else None,
                "cbf": diagnostics,
            }
        )
        if diagnostics is not None and not diagnostics.verified_feasible:
            forced_termination_reason = "cbf_controlled_abort"
            break
        if target_collision:
            forced_termination_reason = "target_safety_failure"
            break
        if terminated or truncated:
            break
        local_observation = policy_observations(env, observation)
        if candidate_history is not None:
            candidate_history.observe_after_action(local_observation, action)
        previous_action = action.copy()
    if not final_info:
        final_info = {
            "safe_capture_success": False,
            "capture_event": False,
            "capture_time_seconds": None,
            "capturing_defender_id": None,
            "collision": True,
            "world_violation_steps": int(env.world_violation_steps),
            "target_world_violation_steps": int(env.target_world_violation_steps),
            "defender_world_violation_steps": int(env.defender_world_violation_steps),
            "target_boundary_violation": bool(env.target_boundary_violation),
            "defender_boundary_violation": bool(env.defender_boundary_violation),
            "min_clearance_so_far": float(env.min_clearance),
            "termination_reason": forced_termination_reason or "empty_rollout",
            "target_visible_fraction": 0.0,
            "mean_message_age_steps": 0.0,
            "mean_observation_age_steps": 0.0,
        }
    if forced_termination_reason is not None:
        final_info = dict(final_info)
        final_info["termination_reason"] = forced_termination_reason
    target_clearance_over_run = target_min_clearance(env)
    target_collision = bool(target_collision or target_clearance_over_run < 0.0)
    collision = bool(final_info.get("collision", False))
    # ``world_violation_steps`` also includes target boundary crossings for
    # historical compatibility.  The safe-capture contract constrains UAVs,
    # so only defender boundary crossings are a safety failure; target
    # crossings remain an explicit diagnostic field.
    boundary_violation = bool(env.defender_boundary_violation)
    target_boundary_violation = bool(env.target_boundary_violation)
    pairwise_violation = bool(minimum_pairwise < -1e-9)
    if pairwise_violation or minimum_obstacle < -1e-9:
        collision = True
    safe_capture = bool(final_info.get("safe_capture_success", False)) and not (
        collision
        or boundary_violation
        or pairwise_violation
        or target_collision
        or cbf_unverified_steps > 0
    )
    final_info["safe_capture_success"] = safe_capture
    row: dict[str, Any] = {
        "seed": seed,
        "scenario": scenario.name,
        "variant": contract["variant"],
        "training_seed": int(manifest_item.get("training_seed", -1)),
        "safe_capture_success": safe_capture,
        "capture_event": bool(final_info.get("capture_event", False)),
        "capture_time_seconds": final_info.get("capture_time_seconds"),
        "capturing_defender_id": final_info.get("capturing_defender_id"),
        "collision": collision,
        "target_obstacle_collision": target_collision,
        "boundary_violation": boundary_violation,
        "world_violation_steps": int(env.world_violation_steps),
        "target_world_violation_steps": int(env.target_world_violation_steps),
        "defender_world_violation_steps": int(env.defender_world_violation_steps),
        "target_boundary_violation": target_boundary_violation,
        "defender_boundary_violation": boundary_violation,
        "pairwise_violation": pairwise_violation,
        "steps": int(env.step_count),
        "termination_reason": str(final_info.get("termination_reason", "running")),
        "min_clearance_m": float(env.min_clearance),
        "minimum_obstacle_clearance_m": minimum_obstacle,
        "minimum_pairwise_clearance_m": minimum_pairwise,
        "minimum_boundary_clearance_m": minimum_boundary,
        "target_min_obstacle_clearance_m": target_clearance_over_run,
        "mean_visible_fraction": float(np.mean(visible_fractions)) if visible_fractions else 0.0,
        "mean_message_age_steps": float(np.mean(message_ages)) if message_ages else 0.0,
        "mean_observation_age_steps": float(np.mean(observation_ages)) if observation_ages else 0.0,
        "defender_path_length_m": path_lengths.tolist(),
        "mean_defender_path_length_m": float(np.mean(path_lengths)),
        "total_defender_path_length_m": float(np.sum(path_lengths)),
        "cbf_infeasible_steps": cbf_infeasible_steps,
        "cbf_timeout_steps": cbf_timeout_steps,
        "cbf_controlled_abort_steps": cbf_abort_steps,
        "cbf_unverified_steps": cbf_unverified_steps,
        "cbf_fallback_steps": cbf_fallback_steps,
        "cbf_intervention_steps": cbf_intervention_steps,
        "cbf_mean_solve_latency_ms": float(np.mean(cbf_latencies)) if cbf_latencies else 0.0,
        "cbf_p95_solve_latency_ms": float(np.percentile(cbf_latencies, 95)) if cbf_latencies else 0.0,
        "mean_cbf_action_correction_norm": float(np.mean(cbf_corrections)) if cbf_corrections else 0.0,
        "max_cbf_action_correction_norm": float(max(cbf_corrections)) if cbf_corrections else 0.0,
        "jepa_enabled": bool(contract["use_jepa"]),
        "ledger_enabled": bool(contract["use_ledger"]),
        "cbf_enabled": bool(contract["use_cbf"]),
        "rank_fallback_steps": rank_fallback_steps,
        "safe_hold_steps": safe_hold_steps,
        "selected_candidate_indices": selected_indices,
        "selected_candidate_mean_index": float(np.mean(selected_indices)) if selected_indices else None,
        "ledger_state_counts": {
            state: int(ledger_states.count(state))
            for state in ("trusted", "fallback_nominal", "safe_hold")
        },
        "fallback_mode_counts": {
            mode: int(fallback_modes.count(mode))
            for mode in sorted(set(fallback_modes))
        },
        "use_cbf": bool(contract["use_cbf"]),
    }
    crossing = crossing_metrics(env, scenario.obstacle_zone_x)
    contract_metrics = capture_contract_metrics(
        final_info,
        crossing,
        target_collision=target_collision,
        target_crossing_required=bool(scenario.target_crossing_required),
        required_defender_zone_entries=int(scenario.required_defender_zone_entries),
        require_target_zone_entry=scenario.require_target_zone_entry,
    )
    row.update(crossing)
    row.update(contract_metrics)
    row.update(transit_route_metrics(env, scenario))
    row.update(transit_execution_metrics(env, scenario))
    row["showcase_success"] = bool(contract_metrics["cooperative_safe_capture"])
    row["safe_capture_success"] = bool(row["safe_capture_success"] and row["showcase_success"])
    trace_dir = output_dir / "step_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"episode_{episode_index:04d}.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), allow_nan=False) + "\n" for item in step_records),
        encoding="utf-8",
    )
    scene_record = {
        "episode_index": episode_index,
        "episode_seed": seed,
        "layout_seed": int(spec["layout_seed"]),
        "scene_hash": manifest_item["scene_hash"],
        "spec": spec,
        "scenario": scenario_metadata(scenario),
        "outcome": row,
    }
    return row, scene_record


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty P6 evaluation.")

    def rate(field: str) -> float:
        return float(np.mean([bool(row.get(field, False)) for row in rows]))

    def count(field: str) -> int:
        return int(sum(int(row.get(field, 0)) for row in rows))

    capture_times = [
        float(row["capture_time_seconds"])
        for row in rows
        if row.get("capture_time_seconds") is not None
    ]
    return {
        "episodes": len(rows),
        "safe_capture_count": int(sum(bool(row.get("safe_capture_success", False)) for row in rows)),
        "safe_capture_rate": rate("safe_capture_success"),
        "capture_event_rate": rate("capture_event"),
        "showcase_success_rate": rate("showcase_success"),
        "collision_count": int(sum(bool(row.get("collision", False)) for row in rows)),
        "collision_rate": rate("collision"),
        "boundary_violation_count": int(sum(bool(row.get("boundary_violation", False)) for row in rows)),
        "boundary_violation_rate": rate("boundary_violation"),
        "target_boundary_violation_count": int(
            sum(bool(row.get("target_boundary_violation", False)) for row in rows)
        ),
        "target_boundary_violation_rate": rate("target_boundary_violation"),
        "pairwise_violation_count": int(sum(bool(row.get("pairwise_violation", False)) for row in rows)),
        "pairwise_violation_rate": rate("pairwise_violation"),
        "cbf_infeasible_steps": count("cbf_infeasible_steps"),
        "cbf_timeout_steps": count("cbf_timeout_steps"),
        "cbf_controlled_abort_steps": count("cbf_controlled_abort_steps"),
        "cbf_unverified_steps": count("cbf_unverified_steps"),
        "cbf_fallback_steps": count("cbf_fallback_steps"),
        "transit_success_rate": rate("transit_success"),
        "mean_capture_time_seconds": float(np.mean(capture_times)) if capture_times else None,
        "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in rows])),
        "worst_min_clearance_m": float(min(float(row["min_clearance_m"]) for row in rows)),
        "mean_obstacle_clearance_m": float(
            np.mean([float(row["minimum_obstacle_clearance_m"]) for row in rows])
        ),
        "worst_obstacle_clearance_m": float(
            min(float(row["minimum_obstacle_clearance_m"]) for row in rows)
        ),
        "mean_pairwise_clearance_m": float(
            np.mean([float(row["minimum_pairwise_clearance_m"]) for row in rows])
        ),
        "worst_pairwise_clearance_m": float(
            min(float(row["minimum_pairwise_clearance_m"]) for row in rows)
        ),
        "mean_cbf_p95_solve_latency_ms": float(
            np.mean([float(row.get("cbf_p95_solve_latency_ms", 0.0)) for row in rows])
        ),
        "max_cbf_p95_solve_latency_ms": float(
            max(float(row.get("cbf_p95_solve_latency_ms", 0.0)) for row in rows)
        ),
        "mean_cbf_action_correction_norm": float(
            np.mean([float(row.get("mean_cbf_action_correction_norm", 0.0)) for row in rows])
        ),
        "termination_reasons": {
            reason: int(sum(str(row.get("termination_reason")) == reason for row in rows))
            for reason in sorted({str(row.get("termination_reason")) for row in rows})
        },
    }


def _write_tensorboard(
    *,
    logdir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    logdir = _fresh(logdir, "TensorBoard directory")
    with SummaryWriter(log_dir=str(logdir), flush_secs=2) as writer:
        writer.add_text("Config/evaluation", json.dumps(_jsonable(metadata), indent=2), 0)
        writer.add_text("Provenance/environment", json.dumps(_jsonable(metadata["environment"]), indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(_jsonable(metadata["inputs"]), indent=0), 0)
        writer.add_text("Provenance/summary", json.dumps(_jsonable(summary), indent=2), 0)
        for index, row in enumerate(rows):
            writer.add_scalar("Episode/safe_capture", float(bool(row["safe_capture_success"])), index)
            writer.add_scalar("Episode/capture_event", float(bool(row["capture_event"])), index)
            writer.add_scalar("Safety/collision", float(bool(row["collision"])), index)
            writer.add_scalar("Safety/boundary_violation", float(bool(row["boundary_violation"])), index)
            writer.add_scalar(
                "Diagnostic/target_boundary_violation",
                float(bool(row.get("target_boundary_violation", False))),
                index,
            )
            writer.add_scalar("Safety/pairwise_violation", float(bool(row["pairwise_violation"])), index)
            writer.add_scalar("Safety/min_obstacle_clearance_m", float(row["minimum_obstacle_clearance_m"]), index)
            writer.add_scalar("Safety/min_pairwise_clearance_m", float(row["minimum_pairwise_clearance_m"]), index)
            writer.add_scalar("CBF/infeasible_steps", float(row["cbf_infeasible_steps"]), index)
            writer.add_scalar("CBF/timeout_steps", float(row["cbf_timeout_steps"]), index)
            writer.add_scalar("CBF/fallback_steps", float(row["cbf_fallback_steps"]), index)
            writer.add_scalar("CBF/unverified_steps", float(row["cbf_unverified_steps"]), index)
            writer.add_scalar("CBF/intervention_steps", float(row["cbf_intervention_steps"]), index)
            writer.add_scalar("CBF/p95_solve_latency_ms", float(row["cbf_p95_solve_latency_ms"]), index)
            writer.add_scalar("Fallback/rank_steps", float(row["rank_fallback_steps"]), index)
            writer.add_scalar("Fallback/safe_hold_steps", float(row["safe_hold_steps"]), index)
            writer.add_scalar("Ranking/selected_candidate_mean_index", float(row["selected_candidate_mean_index"] or 0.0), index)
            writer.add_scalar("Latency/mean_cbf_correction", float(row["mean_cbf_action_correction_norm"]), index)
        writer.add_scalar("Aggregate/safe_capture_rate", float(summary["safe_capture_rate"]), 0)
        writer.add_scalar("Aggregate/collision_rate", float(summary["collision_rate"]), 0)
        writer.add_scalar("Aggregate/boundary_violation_rate", float(summary["boundary_violation_rate"]), 0)
        writer.add_scalar(
            "Aggregate/target_boundary_violation_rate",
            float(summary.get("target_boundary_violation_rate", 0.0)),
            0,
        )
        writer.add_scalar("Aggregate/pairwise_violation_rate", float(summary["pairwise_violation_rate"]), 0)
        writer.add_scalar("Aggregate/p95_cbf_latency_ms", float(summary["max_cbf_p95_solve_latency_ms"]), 0)
        writer.flush()
    event_files = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("TensorBoard writer did not create an event file.")
    return {
        "logdir": str(logdir),
        "event_files": event_files,
        "required_provenance": True,
    }


def _write_episodes_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_jsonable(row.get(key)), separators=(",", ":"))
                    if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                    else row.get(key)
                    for key in keys
                }
            )


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("P6 evaluator requires --development-only.")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.jepa_history_length <= 0:
        raise ValueError("--jepa-history-length must be positive.")
    if args.recurrent_reset_interval is not None and args.recurrent_reset_interval <= 0:
        raise ValueError("--recurrent-reset-interval must be positive.")
    contract = _variant_contract(args.variant)
    protocol_path = args.protocol.resolve()
    environment_config = args.environment_config.resolve()
    actor_checkpoint = args.actor_checkpoint.resolve()
    for path, label in (
        (protocol_path, "protocol"),
        (environment_config, "environment config"),
        (actor_checkpoint, "actor checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    protocol = load_protocol(protocol_path)
    configured_episodes = int(protocol["episodes_per_split"][args.split])
    if args.episodes > configured_episodes:
        raise ValueError(f"Requested {args.episodes} episodes but validation has {configured_episodes}.")
    output_dir = _fresh(args.output_dir, "P6 output directory")
    tensorboard_dir = args.tensorboard_dir.resolve()
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {tensorboard_dir}")
    tensorboard_dir.parent.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    jepa_checkpoint = args.jepa_checkpoint.resolve() if args.jepa_checkpoint else None
    ledger_path = args.reliability_ledger.resolve() if args.reliability_ledger else None
    if contract["use_jepa"] and jepa_checkpoint is None:
        jepa_checkpoint = (
            DEFAULT_JEPA_ROOT / f"jepa_safe_capture_v2_p2_seed{args.training_seed}" / "checkpoint.pt"
        ).resolve()
    if contract["use_ledger"] and ledger_path is None:
        ledger_path = (
            DEFAULT_LEDGER_ROOT
            / f"jepa_safe_capture_v2_p3_rerun_ledger_seed{args.training_seed}"
            / "reliability_ledger.json"
        ).resolve()
    if jepa_checkpoint is not None and not jepa_checkpoint.is_file():
        raise FileNotFoundError(f"JEPA checkpoint does not exist: {jepa_checkpoint}")
    if ledger_path is not None and not ledger_path.is_file():
        raise FileNotFoundError(f"Reliability ledger does not exist: {ledger_path}")
    if contract["use_ledger"] and not contract["use_jepa"]:
        raise ValueError("A ledger-enabled variant must also enable JEPA.")
    jepa = _load_jepa(jepa_checkpoint, device) if contract["use_jepa"] and jepa_checkpoint else None
    ledger = _load_ledger(ledger_path, jepa_checkpoint) if contract["use_ledger"] and ledger_path and jepa_checkpoint else None
    first_spec = episode_spec(protocol, args.split, 0)
    prototype_config = config_for_spec("f2", first_spec, environment_config)
    prototype = CaptureRadiusPursuit3DEnv(
        prototype_config,
        obstacle_count=0,
        target_speed_scale=float(first_spec["target_speed_scale"]),
    )
    cbf_contract = JointCBFQPSafetyFilter(prototype).contract if contract["use_cbf"] else None
    prototype_observation = prototype.reset(seed=int(first_spec["episode_seed"]))
    policy, action_scale, actor_metadata = load_policy(
        actor_checkpoint,
        prototype,
        prototype_observation,
        device,
    )
    metadata_reset_interval = actor_metadata.get("recurrent_reset_interval_steps")
    recurrent_reset_interval = (
        int(args.recurrent_reset_interval)
        if args.recurrent_reset_interval is not None
        else int(metadata_reset_interval)
        if metadata_reset_interval is not None
        else None
    )
    if args.scene_manifest is None:
        manifest = _build_scene_manifest(
            protocol,
            environment_config,
            split=args.split,
            episodes=args.episodes,
        )
    else:
        manifest = _load_scene_manifest(
            args.scene_manifest.resolve(),
            protocol,
            environment_config,
            split=args.split,
            episodes=args.episodes,
        )
    for item in manifest:
        item["training_seed"] = int(args.training_seed)
    (output_dir / "scene_manifest.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), allow_nan=False) + "\n" for item in manifest),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    started = time.perf_counter()
    for item in manifest:
        spec = item["spec"]
        config = config_for_spec("f2", spec, environment_config)
        row, scene = _run_episode(
            manifest_item=item,
            config=config,
            policy=policy,
            action_scale=action_scale,
            device=device,
            contract=contract,
            jepa=jepa,
            ledger=ledger,
            history_length=args.jepa_history_length,
            recurrent_reset_interval=recurrent_reset_interval,
            output_dir=output_dir,
        )
        row["training_seed"] = int(args.training_seed)
        row["scene_hash"] = item["scene_hash"]
        row["layout_seed"] = int(spec["layout_seed"])
        row["episode_index"] = int(item["episode_index"])
        row["episode_seed"] = int(spec["episode_seed"])
        row["defender_side"] = str(spec["defender_side"])
        row["obstacle_count"] = int(spec["obstacle_count"])
        row["target_speed_scale"] = float(spec["target_speed_scale"])
        row["target_motion_mode"] = str(spec["target_motion_mode"])
        row["observation_condition"] = str(spec["observation_condition"])
        row["layout_signature"] = "+".join(
            f"{shape}{sum(obstacle['shape'] == shape for obstacle in item['scenario']['obstacles'])}"
            for shape in ("cylinder", "box", "wall")
            if any(obstacle["shape"] == shape for obstacle in item["scenario"]["obstacles"])
        )
        scene["outcome"] = row
        rows.append(row)
        scenes.append(scene)
    elapsed_seconds = time.perf_counter() - started
    summary = _metric_summary(rows)
    inputs = {
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "environment_config": str(environment_config),
        "environment_config_sha256": _sha256(environment_config),
        "actor_checkpoint": str(actor_checkpoint),
        "actor_checkpoint_sha256": _sha256(actor_checkpoint),
        "jepa_checkpoint": str(jepa_checkpoint) if jepa_checkpoint else None,
        "jepa_checkpoint_sha256": _sha256(jepa_checkpoint) if jepa_checkpoint else None,
        "reliability_ledger": str(ledger_path) if ledger_path else None,
        "reliability_ledger_sha256": _sha256(ledger_path) if ledger_path else None,
        "scene_manifest_sha256": _sha256(output_dir / "scene_manifest.jsonl"),
    }
    metadata = {
        "evaluation_type": "jepa_safe_capture_v2_p6_paired_development",
        "development_only": True,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "variant": contract,
        "training_seed": int(args.training_seed),
        "split": args.split,
        "episodes": int(args.episodes),
        "candidate_contract": {
            "candidate_count": 5,
            "chunk_length_steps": 3,
            "perturbation_mps": 0.10,
            "execute_first_step_then_replan": True,
            "project_to_reachable_dynamics": True,
            "score_tie_tolerance_m": 5e-4,
        },
        "cbf_contract": cbf_contract,
        "recurrent_reset_interval_steps": recurrent_reset_interval,
        "action_scale": float(action_scale),
        "git_revision": _git_revision(),
        "inputs": inputs,
        "environment": _environment_metadata(device),
        "elapsed_seconds": elapsed_seconds,
        "tensorboard_dir": str(tensorboard_dir),
    }
    _write_episodes_csv(output_dir / "episodes.csv", rows)
    (output_dir / "scenes.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), allow_nan=False) + "\n" for item in scenes),
        encoding="utf-8",
    )
    _write_json(output_dir / "summary.json", {"overall": summary, "metadata": metadata})
    _write_json(output_dir / "provenance.json", metadata)
    tb_info = _write_tensorboard(logdir=tensorboard_dir, metadata=metadata, rows=rows, summary=summary)
    metadata["tensorboard"] = tb_info
    _write_json(output_dir / "provenance.json", metadata)
    _write_json(output_dir / "summary.json", {"overall": summary, "metadata": metadata})
    print(json.dumps(_jsonable({"overall": summary, "metadata": metadata}), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
