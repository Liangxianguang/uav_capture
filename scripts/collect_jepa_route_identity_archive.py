"""Collect a train-only route-identity counterfactual archive.

The archive is built from the frozen dynamic expert on the train split.  Each
sample contains one public observation history and one geometry-conditioned
route candidate.  Simulator truth is used only for offline labels.  A
counterfactual branch is advanced only when its requested action passes the
same Joint CBF verifier used by the runtime; a rejected branch is retained as
an explicit negative feasibility example and is never executed unverified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import yaml

try:  # Keep pure archive-contract helpers importable for lightweight tests.
    import torch
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - exercised by minimal CI images
    torch = None  # type: ignore[assignment]
    SummaryWriter = None  # type: ignore[assignment,misc]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.obstacle_route_candidates import (  # noqa: E402
    ROUTE_LABELS,
    ObstacleRouteCandidate,
    ObstacleRouteConfig,
    make_obstacle_route_candidates,
)
from encirclement3d.pursuit_controllers import DynamicEncirclementController  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    prepare_showcase_episode,
    random_central_mixed_obstacle_scenario,
    scenario_metadata,
)


DATASET_VERSION = "jepa_safe_capture_route_identity_hard_negative_v2"
HORIZON_STEPS = (1, 2, 3, 5)
TTC_CLIP_SECONDS = 10.0
ROUTE_SIDES = (
    "nominal",
    "left",
    "right",
    "upper",
    "lower",
    "radial_out",
    "split",
    "contract",
    "hold",
    "intercept",
    "visibility_hold",
    "boundary_shadow",
)
INTERACTION_HARD_NEGATIVE_MODES = ("near_pass", "formation_crossing", "split_merge")
INTERACTION_SAMPLE_TYPES = {
    mode: 2 + index for index, mode in enumerate(INTERACTION_HARD_NEGATIVE_MODES)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/central_random_mixed_obstacle_s3_route_v1_protocol.yaml",
    )
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=PROJECT_ROOT / "configs/capture_radius_pursuit_central_v4_flee.yaml",
    )
    parser.add_argument(
        "--archive-config",
        type=Path,
        default=PROJECT_ROOT / "configs/jepa_safe_capture_route_identity_archive_v1.yaml",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--split", choices=("train", "validation", "calibration", "development"), default="train")
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--chunk-length-steps", type=int, default=3)
    parser.add_argument(
        "--interaction-hard-negatives",
        action="store_true",
        help="Add offline-only pairwise interaction transition branches to each sampled state.",
    )
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument(
        "--actor-checkpoint",
        type=Path,
        help=(
            "Optional frozen actor checkpoint. When supplied, archive states "
            "follow the same actor used by runtime evaluation instead of the "
            "rule controller."
        ),
    )
    parser.add_argument("--actor-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


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


def _actor_action(
    policy: Any,
    local_observation: np.ndarray,
    device: Any,
    action_scale: float,
    hidden: Any,
) -> tuple[np.ndarray, Any]:
    """Evaluate the frozen runtime actor without sampling its distribution."""

    if torch is None:
        raise RuntimeError("--actor-checkpoint requires the PyTorch environment.")
    local = torch.as_tensor(local_observation, device=device)
    with torch.no_grad():
        if hidden is not None:
            distribution, hidden = policy.distribution_step(local, hidden)
        else:
            distribution = policy.distribution(local)
        action = torch.tanh(distribution.mean).cpu().numpy() * float(action_scale)
    value = np.asarray(action, dtype=np.float64)
    expected = (local_observation.shape[0], 3)
    if value.shape != expected or not np.isfinite(value).all():
        raise RuntimeError(f"Frozen actor emitted an invalid action: {value.shape}")
    return value, hidden


def _jsonable(value: Any) -> Any:
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
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _fresh(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {label}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _route_config(
    env: CaptureRadiusPursuit3DEnv,
    safety_filter: JointCBFQPSafetyFilter,
    chunk_length_steps: int,
) -> ObstacleRouteConfig:
    return ObstacleRouteConfig(
        chunk_length_steps=int(chunk_length_steps),
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
    )


def _reachable_interaction_chunk(
    env: CaptureRadiusPursuit3DEnv,
    previous_action: np.ndarray,
    desired_chunk: np.ndarray,
) -> np.ndarray:
    """Project a synthetic interaction maneuver through reachable dynamics."""

    previous = np.asarray(previous_action, dtype=np.float64).copy()
    desired_chunk = np.asarray(desired_chunk, dtype=np.float64)
    if desired_chunk.ndim != 3 or desired_chunk.shape[1:] != previous.shape:
        raise ValueError("Synthetic interaction chunk has an invalid action shape.")
    projected: list[np.ndarray] = []
    max_speed = float(env.agents["defender_max_speed"])
    max_delta = float(env.agents["defender_max_acceleration"]) * float(env.dt)
    for desired in desired_chunk:
        previous = env._move_toward_velocity(
            previous,
            env._clip_rows(desired, max_speed),
            max_delta=max_delta,
        )
        projected.append(np.asarray(previous, dtype=np.float64).copy())
    return np.stack(projected, axis=0)


def _unit_direction(start: np.ndarray, end: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-9 or not np.isfinite(norm):
        return np.asarray(fallback, dtype=np.float64).copy()
    return delta / norm


def _interaction_hard_negative_chunks(
    env: CaptureRadiusPursuit3DEnv,
    nominal: np.ndarray,
    previous_action: np.ndarray,
    chunk_length_steps: int,
) -> dict[str, np.ndarray]:
    """Create public-state pairwise transition probes for offline labels.

    These probes are deliberately not runtime candidates. They are passed
    through the same reachable projection and CBF counterfactual as normal
    routes so the risk heads see the interaction transitions that nominal
    replay rarely visits.
    """

    positions = np.asarray(env.defender_positions, dtype=np.float64)
    defender_count = int(positions.shape[0])
    if defender_count < 2:
        return {}
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    speed = min(float(env.agents["defender_max_speed"]), 4.5)
    centroid = positions.mean(axis=0)
    pairings = ((0, 1), (2, 3)) if defender_count >= 4 else ((0, 1),)
    result: dict[str, np.ndarray] = {}
    for mode in INTERACTION_HARD_NEGATIVE_MODES:
        desired = np.repeat(np.asarray(nominal, dtype=np.float64)[None, :, :], chunk_length_steps, axis=0)
        if mode == "near_pass":
            for first, second in pairings:
                line = _unit_direction(positions[first], positions[second], fallback)
                tangent = np.array([-line[1], line[0], 0.0], dtype=np.float64)
                tangent = _unit_direction(np.zeros(3, dtype=np.float64), tangent, np.array([0.0, 1.0, 0.0]))
                # Both vehicles pass along a common tangent while retaining a
                # small closing component, creating a close-but-reachable
                # interaction tail rather than a duplicate swap maneuver.
                desired[:, first] = speed * (0.85 * tangent + 0.15 * line)
                desired[:, second] = speed * (0.85 * tangent - 0.15 * line)
        elif mode == "formation_crossing":
            for first, second in pairings:
                first_to_second = _unit_direction(positions[first], positions[second], fallback)
                second_to_first = -first_to_second
                desired[:, first] = speed * first_to_second
                desired[:, second] = speed * second_to_first
        else:  # split_merge: one pair converges while the other opens outward.
            first, second = pairings[0]
            first_to_second = _unit_direction(positions[first], positions[second], fallback)
            desired[:, first] = speed * first_to_second
            desired[:, second] = -speed * first_to_second
            for agent in range(2, defender_count):
                outward = _unit_direction(centroid, positions[agent], fallback)
                desired[:, agent] = speed * outward
        result[mode] = _reachable_interaction_chunk(env, previous_action, desired)
    return result


def _offline_interaction_route(
    template: ObstacleRouteCandidate,
    mode: str,
    action_chunk: np.ndarray,
) -> ObstacleRouteCandidate:
    """Wrap a synthetic action block as an explicitly offline route branch."""

    return replace(
        template,
        route_id=f"offline_interaction:{mode}",
        action_chunk=np.asarray(action_chunk, dtype=np.float64).copy(),
        raw_action_chunk=np.asarray(action_chunk, dtype=np.float64).copy(),
        projected=True,
        reachable=True,
        geometric_feasible=False,
        minimum_geometric_clearance_m=0.0,
        route_length_m=float(np.linalg.norm(action_chunk, axis=-1).sum()),
        rejection_reasons=("offline_interaction_transition",),
        fallback_only=False,
    )


def _clearance_labels(env: CaptureRadiusPursuit3DEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(env.defender_positions, dtype=np.float64)
    radius = float(env.agents["drone_radius"])
    obstacle = np.full(env.n_defenders, 50.0, dtype=np.float64)
    pairwise = np.full(env.n_defenders, 50.0, dtype=np.float64)
    boundary = np.full(env.n_defenders, 50.0, dtype=np.float64)
    for index, position in enumerate(positions):
        if env.obstacles:
            obstacle[index] = min(
                float(env._obstacle_clearance(position, item) - radius) for item in env.obstacles
            )
        teammate = [
            float(np.linalg.norm(position - other) - 2.0 * radius)
            for other_index, other in enumerate(positions)
            if other_index != index
        ]
        if teammate:
            pairwise[index] = min(teammate)
        boundary[index] = min(
            float(value)
            for value in np.concatenate(
                [position - env.lower - radius, env.upper - position - radius]
            )
        )
    return obstacle, pairwise, boundary


def _target_relative(env: CaptureRadiusPursuit3DEnv, extent: float) -> np.ndarray:
    return ((env.target_position[None, :] - env.defender_positions) / extent).astype(np.float32)


def _target_progress(env: CaptureRadiusPursuit3DEnv, initial_distances: np.ndarray, extent: float) -> np.ndarray:
    distances = np.linalg.norm(env.target_position[None, :] - env.defender_positions, axis=1)
    return ((initial_distances - distances) / extent).astype(np.float32)


def _pairwise_ttc_labels(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    radius: float,
    margin: float,
    clip_seconds: float = TTC_CLIP_SECONDS,
) -> np.ndarray:
    """Return conservative per-defender TTC to the operational pairwise set."""

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions and velocities must have shape [defenders, 3]")
    result = np.full(positions.shape[0], float(clip_seconds), dtype=np.float64)
    safe_distance = 2.0 * float(radius) + float(margin)
    for first in range(positions.shape[0]):
        for second in range(first + 1, positions.shape[0]):
            relative_position = positions[first] - positions[second]
            relative_velocity = velocities[first] - velocities[second]
            speed_squared = float(np.dot(relative_velocity, relative_velocity))
            if speed_squared <= 1e-12:
                continue
            closing = float(np.dot(relative_position, relative_velocity))
            c = float(np.dot(relative_position, relative_position) - safe_distance**2)
            if c <= 0.0:
                time = 0.0
            elif closing >= 0.0:
                continue
            else:
                discriminant = closing**2 - speed_squared * c
                if discriminant < 0.0:
                    continue
                time = (-closing - float(np.sqrt(discriminant))) / speed_squared
            if 0.0 <= time <= clip_seconds:
                result[first] = min(result[first], time)
                result[second] = min(result[second], time)
    return result.astype(np.float32)


def _risk_labels(
    env: CaptureRadiusPursuit3DEnv,
    velocities: np.ndarray,
    *,
    obstacle_margin_m: float,
    boundary_margin_m: float,
    inter_agent_margin_m: float,
) -> dict[str, np.ndarray]:
    """Compute offline risk labels from one public state and a candidate velocity."""

    positions = np.asarray(env.defender_positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if positions.shape != velocities.shape:
        raise ValueError("risk label positions and velocities must have identical shapes")
    radius = float(env.agents["drone_radius"])
    max_acceleration = float(env.agents["defender_max_acceleration"])
    speeds = np.linalg.norm(velocities, axis=1)
    stopping_distance = speeds**2 / max(2.0 * max_acceleration, 1e-12)
    obstacle_ttc = np.full(env.n_defenders, TTC_CLIP_SECONDS, dtype=np.float64)
    boundary_ttc = np.full(env.n_defenders, TTC_CLIP_SECONDS, dtype=np.float64)
    for index, position in enumerate(positions):
        for obstacle in env.obstacles:
            clearance, normal = env._cylinder_clearance_and_normal(position, obstacle)
            gap = float(clearance) - radius - float(obstacle_margin_m)
            closing_speed = max(0.0, -float(np.dot(normal, velocities[index])))
            if gap <= 0.0:
                candidate = 0.0
            elif closing_speed <= 1e-9:
                continue
            else:
                candidate = gap / closing_speed
            obstacle_ttc[index] = min(obstacle_ttc[index], candidate)
        for axis in range(3):
            lower_gap = float(position[axis] - env.lower[axis] - radius - boundary_margin_m)
            upper_gap = float(env.upper[axis] - radius - boundary_margin_m - position[axis])
            speed = float(velocities[index, axis])
            if speed < -1e-9:
                candidate = 0.0 if lower_gap <= 0.0 else lower_gap / -speed
                boundary_ttc[index] = min(boundary_ttc[index], candidate)
            elif speed > 1e-9:
                candidate = 0.0 if upper_gap <= 0.0 else upper_gap / speed
                boundary_ttc[index] = min(boundary_ttc[index], candidate)
    return {
        "stopping_distance": stopping_distance.astype(np.float32),
        "obstacle_ttc": np.clip(obstacle_ttc, 0.0, TTC_CLIP_SECONDS).astype(np.float32),
        "boundary_ttc": np.clip(boundary_ttc, 0.0, TTC_CLIP_SECONDS).astype(np.float32),
        "pairwise_ttc": _pairwise_ttc_labels(
            positions,
            velocities,
            radius=radius,
            margin=inter_agent_margin_m,
        ),
    }


def _acceleration_slack_labels(diagnostics: Any, defender_count: int) -> np.ndarray:
    """Extract per-defender acceleration slack without inventing feasibility."""

    result = np.full(defender_count, -1.0, dtype=np.float32)
    slacks = getattr(diagnostics, "constraint_slacks", {}) if diagnostics is not None else {}
    if not isinstance(slacks, Mapping):
        return result
    for defender in range(defender_count):
        values = [
            float(value)
            for key, value in slacks.items()
            if str(key) == f"acceleration_defender_{defender}" and np.isfinite(float(value))
        ]
        if values:
            result[defender] = float(min(values))
    return result


def _boundary_shadow_rollout(
    env: CaptureRadiusPursuit3DEnv,
    *,
    horizon: int = max(HORIZON_STEPS),
) -> tuple[np.ndarray, np.ndarray]:
    """Create an offline-only outward-boundary stress label.

    The shadow is never sent to the environment or to the CBF executor.  It
    deliberately represents the unfiltered outward command so the boundary
    head sees negative clearance examples that cannot occur on a verified
    branch.  The action is still stored because the label is action-conditioned.
    """

    positions = np.asarray(env.defender_positions, dtype=np.float64).copy()
    lower = np.asarray(env.lower, dtype=np.float64)
    upper = np.asarray(env.upper, dtype=np.float64)
    radius = float(env.agents["drone_radius"])
    max_speed = float(env.agents["defender_max_speed"])
    dt = float(env.dt)
    if int(horizon) <= 0:
        raise ValueError("boundary shadow horizon must be positive")
    actions = np.zeros((int(horizon), env.n_defenders, 3), dtype=np.float32)
    for agent, position in enumerate(positions):
        lower_gaps = position - lower - radius
        upper_gaps = upper - position - radius
        axis = int(np.argmin(np.concatenate([lower_gaps, upper_gaps])))
        direction = np.zeros(3, dtype=np.float64)
        if axis < 3:
            direction[axis] = -1.0
        else:
            direction[axis - 3] = 1.0
        actions[:, agent] = (direction * max_speed).astype(np.float32)
    boundary = []
    for _step in range(int(horizon)):
        positions += actions[0].astype(np.float64) * dt
        values = np.min(
            np.concatenate(
                [positions - lower[None, :] - radius, upper[None, :] - positions - radius], axis=1
            ),
            axis=1,
        )
        boundary.append(values.astype(np.float32))
    return actions, np.stack(boundary, axis=0)


def _boundary_shadow_ttc(
    env: CaptureRadiusPursuit3DEnv,
    actions: np.ndarray,
    *,
    horizon: int = max(HORIZON_STEPS),
    clip_seconds: float = TTC_CLIP_SECONDS,
) -> np.ndarray:
    """Return non-negative time-to-boundary labels for the offline shadow."""

    positions = np.asarray(env.defender_positions, dtype=np.float64).copy()
    actions = np.asarray(actions, dtype=np.float64)
    radius = float(env.agents["drone_radius"])
    dt = float(env.dt)
    result: list[np.ndarray] = []
    for _step in range(int(horizon)):
        positions += actions[0] * dt
        per_agent = np.full(env.n_defenders, float(clip_seconds), dtype=np.float64)
        for agent, position in enumerate(positions):
            for axis in range(3):
                lower_gap = float(position[axis] - env.lower[axis] - radius)
                upper_gap = float(env.upper[axis] - position[axis] - radius)
                speed = float(actions[0, agent, axis])
                if speed < -1e-9:
                    candidate = 0.0 if lower_gap <= 0.0 else lower_gap / -speed
                elif speed > 1e-9:
                    candidate = 0.0 if upper_gap <= 0.0 else upper_gap / speed
                else:
                    continue
                per_agent[agent] = min(per_agent[agent], candidate)
        result.append(np.clip(per_agent, 0.0, clip_seconds).astype(np.float32))
    return np.stack(result, axis=0)


def _copy_cbf_filter(source: JointCBFQPSafetyFilter, env: CaptureRadiusPursuit3DEnv) -> JointCBFQPSafetyFilter:
    contract = source.contract
    return JointCBFQPSafetyFilter(
        env,
        gamma=float(contract["gamma"]),
        obstacle_margin_m=float(contract["obstacle_margin_m"]),
        inter_agent_margin_m=float(contract["inter_agent_margin_m"]),
        boundary_margin_m=float(contract["boundary_margin_m"]),
        max_correction_norm_mps=float(contract["max_correction_norm_mps"]),
        max_latency_ms=float(contract["max_latency_ms"]),
        solver_maxiter=int(contract["solver_maxiter"]),
        tolerance=float(contract["tolerance"]),
        active_tolerance=float(contract["active_tolerance"]),
        anticipatory_horizon_steps=int(contract["anticipatory_horizon_steps"]),
        barrier_mode=str(contract["barrier_mode"]),
    )


def _archive_cbf_contract(archive_config: Mapping[str, Any]) -> dict[str, Any]:
    """Read the archive's explicit CBF contract before collecting samples."""

    value = archive_config.get("cbf_contract", {})
    if not isinstance(value, Mapping):
        raise ValueError("archive config cbf_contract must be a mapping")
    horizon = int(value.get("anticipatory_horizon_steps", 0))
    barrier_mode = str(value.get("barrier_mode", "")).strip()
    if horizon <= 0:
        raise ValueError("archive cbf_contract anticipatory_horizon_steps must be positive")
    if barrier_mode not in {"strict_buffer", "physical_feasibility"}:
        raise ValueError("archive cbf_contract barrier_mode is invalid")
    return {
        "anticipatory_horizon_steps": horizon,
        "barrier_mode": barrier_mode,
    }


def _route_rollout(
    env: CaptureRadiusPursuit3DEnv,
    observation: Mapping[str, Any],
    controller: DynamicEncirclementController,
    route: Any,
    safety_filter: JointCBFQPSafetyFilter,
    *,
    extent: float,
) -> dict[str, np.ndarray | int | bool]:
    """Roll one route branch and preserve the first failed step as a label."""

    clone = copy.deepcopy(env)
    clone_observation = copy.deepcopy(dict(observation))
    continuation = DynamicEncirclementController(clone, horizon_seconds=controller.horizon_seconds)
    continuation.interceptor_id = controller.interceptor_id
    clone_filter = _copy_cbf_filter(safety_filter, clone)
    initial_distances = np.linalg.norm(clone.target_position[None, :] - clone.defender_positions, axis=1)
    maximum_horizon = max(HORIZON_STEPS)
    defender_count = clone.n_defenders
    labels: dict[str, list[np.ndarray]] = {
        "relative": [],
        "obstacle_clearance": [],
        "inter_agent_clearance": [],
        "boundary_clearance": [],
        "stopping_distance": [],
        "obstacle_ttc": [],
        "boundary_ttc": [],
        "pairwise_ttc": [],
        "acceleration_slack": [],
        "target_visible": [],
        "cbf_correction": [],
        "cbf_intervention": [],
        "cbf_feasible": [],
        "cbf_min_slack": [],
        "route_progress": [],
        "rollout_valid": [],
    }
    first_failure = maximum_horizon + 1
    failed = False
    cbf_failed = False
    for step in range(1, maximum_horizon + 1):
        if failed:
            obstacle, pairwise, boundary = _clearance_labels(clone)
            risk = _risk_labels(
                clone,
                np.asarray(clone.defender_velocities, dtype=np.float64),
                obstacle_margin_m=float(safety_filter.obstacle_margin_m),
                boundary_margin_m=float(safety_filter.boundary_margin_m),
                inter_agent_margin_m=float(safety_filter.inter_agent_margin_m),
            )
            labels["relative"].append(_target_relative(clone, extent))
            labels["obstacle_clearance"].append(obstacle.astype(np.float32))
            labels["inter_agent_clearance"].append(pairwise.astype(np.float32))
            labels["boundary_clearance"].append(boundary.astype(np.float32))
            labels["stopping_distance"].append(risk["stopping_distance"])
            labels["obstacle_ttc"].append(risk["obstacle_ttc"])
            labels["boundary_ttc"].append(risk["boundary_ttc"])
            labels["pairwise_ttc"].append(risk["pairwise_ttc"])
            labels["acceleration_slack"].append(np.full(defender_count, -1.0, dtype=np.float32))
            labels["target_visible"].append(np.asarray(clone.target_visible, dtype=np.float32))
            labels["cbf_correction"].append(np.zeros(defender_count, dtype=np.float32))
            labels["cbf_intervention"].append(np.zeros(defender_count, dtype=np.float32))
            labels["cbf_feasible"].append(np.zeros(defender_count, dtype=np.float32))
            labels["cbf_min_slack"].append(np.full(defender_count, -1.0, dtype=np.float32))
            labels["route_progress"].append(_target_progress(clone, initial_distances, extent))
            labels["rollout_valid"].append(np.zeros(defender_count, dtype=np.float32))
            continue
        if step <= int(route.action_chunk.shape[0]):
            requested = np.asarray(route.action_chunk[step - 1], dtype=np.float64)
        else:
            requested = np.asarray(continuation.act(clone_observation), dtype=np.float64)
        diagnostics = clone_filter.verify_requested_action(requested, clone_observation)
        feasible = bool(
            diagnostics.verified_feasible
            and not diagnostics.infeasible
            and not diagnostics.timed_out
            and diagnostics.fallback_mode == "none"
        )
        minimum = float(diagnostics.minimum_constraint_value)
        correction = float(diagnostics.action_correction_norm)
        def append_failure_labels(failure_minimum: float, failure_correction: float) -> None:
            nonlocal first_failure, failed, cbf_failed
            first_failure = step
            failed = True
            cbf_failed = True
            obstacle, pairwise, boundary = _clearance_labels(clone)
            risk = _risk_labels(
                clone,
                np.asarray(clone.defender_velocities, dtype=np.float64),
                obstacle_margin_m=float(safety_filter.obstacle_margin_m),
                boundary_margin_m=float(safety_filter.boundary_margin_m),
                inter_agent_margin_m=float(safety_filter.inter_agent_margin_m),
            )
            labels["relative"].append(_target_relative(clone, extent))
            labels["obstacle_clearance"].append(obstacle.astype(np.float32))
            labels["inter_agent_clearance"].append(pairwise.astype(np.float32))
            labels["boundary_clearance"].append(boundary.astype(np.float32))
            labels["stopping_distance"].append(risk["stopping_distance"])
            labels["obstacle_ttc"].append(risk["obstacle_ttc"])
            labels["boundary_ttc"].append(risk["boundary_ttc"])
            labels["pairwise_ttc"].append(risk["pairwise_ttc"])
            labels["acceleration_slack"].append(_acceleration_slack_labels(diagnostics, defender_count))
            labels["target_visible"].append(np.asarray(clone.target_visible, dtype=np.float32))
            labels["cbf_correction"].append(np.full(defender_count, failure_correction, dtype=np.float32))
            labels["cbf_intervention"].append(np.zeros(defender_count, dtype=np.float32))
            labels["cbf_feasible"].append(np.zeros(defender_count, dtype=np.float32))
            labels["cbf_min_slack"].append(np.full(defender_count, failure_minimum, dtype=np.float32))
            labels["route_progress"].append(_target_progress(clone, initial_distances, extent))
            labels["rollout_valid"].append(np.zeros(defender_count, dtype=np.float32))

        if not feasible:
            append_failure_labels(minimum, correction)
            continue
        executed, final_diagnostics = clone_filter.filter(requested, clone_observation)
        if not final_diagnostics.verified_feasible or final_diagnostics.fallback_mode != "none":
            # A primary probe can pass while the full filter rejects the same
            # request (for example when the correction bound is exceeded).
            # Preserve this as an execution-contract negative instead of
            # advancing the branch or silently treating it as safe.
            append_failure_labels(
                float(final_diagnostics.minimum_constraint_value),
                float(final_diagnostics.action_correction_norm),
            )
            continue
        correction = float(final_diagnostics.action_correction_norm)
        clone_observation, _reward, terminated, truncated, _info = clone.step(executed)
        obstacle, pairwise, boundary = _clearance_labels(clone)
        risk = _risk_labels(
            clone,
            np.asarray(clone.defender_velocities, dtype=np.float64),
            obstacle_margin_m=float(safety_filter.obstacle_margin_m),
            boundary_margin_m=float(safety_filter.boundary_margin_m),
            inter_agent_margin_m=float(safety_filter.inter_agent_margin_m),
        )
        labels["relative"].append(_target_relative(clone, extent))
        labels["obstacle_clearance"].append(obstacle.astype(np.float32))
        labels["inter_agent_clearance"].append(pairwise.astype(np.float32))
        labels["boundary_clearance"].append(boundary.astype(np.float32))
        labels["stopping_distance"].append(risk["stopping_distance"])
        labels["obstacle_ttc"].append(risk["obstacle_ttc"])
        labels["boundary_ttc"].append(risk["boundary_ttc"])
        labels["pairwise_ttc"].append(risk["pairwise_ttc"])
        labels["acceleration_slack"].append(_acceleration_slack_labels(final_diagnostics, defender_count))
        labels["target_visible"].append(np.asarray(clone.target_visible, dtype=np.float32))
        labels["cbf_correction"].append(np.full(defender_count, correction, dtype=np.float32))
        labels["cbf_intervention"].append(
            np.full(defender_count, float(correction > 1e-6), dtype=np.float32)
        )
        labels["cbf_feasible"].append(np.ones(defender_count, dtype=np.float32))
        labels["cbf_min_slack"].append(np.full(defender_count, minimum, dtype=np.float32))
        labels["route_progress"].append(_target_progress(clone, initial_distances, extent))
        labels["rollout_valid"].append(np.ones(defender_count, dtype=np.float32))
        if terminated or truncated:
            failed = True
            if first_failure == maximum_horizon + 1:
                first_failure = step + 1
    return {
        key: np.stack(value, axis=0).astype(np.float32)
        for key, value in labels.items()
    } | {
        "earliest_failure_step": int(first_failure),
        "branch_terminated": bool(failed),
        "cbf_failed": bool(cbf_failed),
    }


def _empty_samples() -> dict[str, list[Any]]:
    return {
        "inputs": [],
        "action_history": [],
        "route_action_chunk": [],
        "route_relative_action_chunk": [],
        "labels_relative": [],
        "labels_obstacle_clearance": [],
        "labels_boundary_clearance": [],
        "labels_inter_agent_clearance": [],
        "labels_stopping_distance": [],
        "labels_obstacle_ttc": [],
        "labels_boundary_ttc": [],
        "labels_pairwise_ttc": [],
        "labels_acceleration_slack": [],
        "labels_target_visible": [],
        "labels_cbf_correction": [],
        "labels_cbf_intervention": [],
        "labels_cbf_feasible": [],
        "labels_cbf_min_slack": [],
        "labels_route_progress": [],
        "labels_rollout_valid": [],
        "route_length_m": [],
        "route_geometric_clearance_m": [],
        "route_geometry_valid": [],
        "route_candidate_index": [],
        "route_side_index": [],
        "route_obstacle_id": [],
        "time_index": [],
        "episode_seed": [],
        "scenario_index": [],
        "earliest_failure_step": [],
        "branch_terminated": [],
        "sample_type": [],
    }


def _append_samples(
    samples: dict[str, list[Any]],
    *,
    observation_history: list[np.ndarray],
    executed_action_history: list[np.ndarray],
    route: Any,
    route_index: int,
    labels: Mapping[str, Any],
    episode_seed: int,
    scenario_index: int,
    time_index: int,
    action_scale: float,
    sample_type: int = 0,
) -> None:
    if len(observation_history) < 8 or len(executed_action_history) != len(observation_history) - 1:
        raise ValueError("Route archive histories are not causally aligned.")
    inputs = np.stack(observation_history[-8:], axis=0)
    past_actions = np.stack(executed_action_history[-7:], axis=0)
    candidate_chunk = np.asarray(route.action_chunk, dtype=np.float32)
    if candidate_chunk.ndim != 3 or candidate_chunk.shape[0] not in (3, 5):
        raise ValueError("Route archive requires a three- or five-step candidate chunk.")
    for agent in range(inputs.shape[1]):
        samples["inputs"].append(inputs[:, agent].copy())
        samples["action_history"].append(
            np.concatenate([past_actions[:, agent], candidate_chunk[0, agent][None, :]], axis=0) / action_scale
        )
        samples["route_action_chunk"].append(candidate_chunk[:, agent].copy())
        if candidate_chunk.shape[1] > 1:
            teammate_mean = (
                candidate_chunk.sum(axis=1) - candidate_chunk[:, agent]
            ) / float(candidate_chunk.shape[1] - 1)
        else:
            teammate_mean = np.zeros_like(candidate_chunk[:, agent])
        samples["route_relative_action_chunk"].append(
            (candidate_chunk[:, agent] - teammate_mean).copy()
        )
        for source, target in (
            ("relative", "labels_relative"),
            ("obstacle_clearance", "labels_obstacle_clearance"),
            ("boundary_clearance", "labels_boundary_clearance"),
            ("inter_agent_clearance", "labels_inter_agent_clearance"),
            ("stopping_distance", "labels_stopping_distance"),
            ("obstacle_ttc", "labels_obstacle_ttc"),
            ("boundary_ttc", "labels_boundary_ttc"),
            ("pairwise_ttc", "labels_pairwise_ttc"),
            ("acceleration_slack", "labels_acceleration_slack"),
            ("target_visible", "labels_target_visible"),
            ("cbf_correction", "labels_cbf_correction"),
            ("cbf_intervention", "labels_cbf_intervention"),
            ("cbf_feasible", "labels_cbf_feasible"),
            ("cbf_min_slack", "labels_cbf_min_slack"),
            ("route_progress", "labels_route_progress"),
            ("rollout_valid", "labels_rollout_valid"),
        ):
            samples[target].append(np.asarray(labels[source])[:, agent].copy())
        samples["route_length_m"].append(float(route.route_length_m))
        samples["route_geometric_clearance_m"].append(float(route.minimum_geometric_clearance_m))
        samples["route_geometry_valid"].append(float(route.valid))
        samples["route_candidate_index"].append(int(route_index) if int(sample_type) == 0 else -1)
        samples["route_side_index"].append(ROUTE_SIDES.index(str(route.side)))
        samples["route_obstacle_id"].append(-1 if route.obstacle_id is None else int(route.obstacle_id))
        samples["time_index"].append(int(time_index))
        samples["episode_seed"].append(int(episode_seed))
        samples["scenario_index"].append(int(scenario_index))
        samples["earliest_failure_step"].append(int(labels["earliest_failure_step"]))
        samples["branch_terminated"].append(float(labels["branch_terminated"]))
        samples["sample_type"].append(int(sample_type))


def _append_boundary_shadow_samples(
    samples: dict[str, list[Any]],
    *,
    observation_history: list[np.ndarray],
    executed_action_history: list[np.ndarray],
    env: CaptureRadiusPursuit3DEnv,
    episode_seed: int,
    scenario_index: int,
    time_index: int,
    action_scale: float,
    chunk_length_steps: int = max(HORIZON_STEPS),
) -> float:
    """Append one data-only boundary stress sample per defender."""

    if len(observation_history) < 8 or len(executed_action_history) != len(observation_history) - 1:
        raise ValueError("Boundary shadow histories are not causally aligned.")
    inputs = np.stack(observation_history[-8:], axis=0)
    past_actions = np.stack(executed_action_history[-7:], axis=0)
    if int(chunk_length_steps) <= 0:
        raise ValueError("boundary shadow chunk length must be positive")
    action_chunk, boundary = _boundary_shadow_rollout(env, horizon=int(chunk_length_steps))
    boundary_ttc = _boundary_shadow_ttc(env, action_chunk, horizon=int(chunk_length_steps))
    for agent in range(inputs.shape[1]):
        samples["inputs"].append(inputs[:, agent].copy())
        samples["action_history"].append(
            np.concatenate([past_actions[:, agent], action_chunk[0, agent][None, :]], axis=0) / action_scale
        )
        samples["route_action_chunk"].append(action_chunk[:, agent].copy())
        if action_chunk.shape[1] > 1:
            teammate_mean = (
                action_chunk.sum(axis=1) - action_chunk[:, agent]
            ) / float(action_chunk.shape[1] - 1)
        else:
            teammate_mean = np.zeros_like(action_chunk[:, agent])
        samples["route_relative_action_chunk"].append(
            (action_chunk[:, agent] - teammate_mean).copy()
        )
        zeros = np.zeros(max(HORIZON_STEPS), dtype=np.float32)
        for key, value in (
            ("labels_relative", np.zeros((max(HORIZON_STEPS), 3), dtype=np.float32)),
            ("labels_obstacle_clearance", zeros.copy()),
            ("labels_boundary_clearance", boundary[:, agent]),
            ("labels_inter_agent_clearance", zeros.copy()),
            ("labels_stopping_distance", np.full(max(HORIZON_STEPS), float(np.linalg.norm(action_chunk[0, agent]) ** 2 / max(2.0 * float(env.agents["defender_max_acceleration"]), 1e-12)), dtype=np.float32)),
            ("labels_obstacle_ttc", zeros.copy()),
            ("labels_boundary_ttc", boundary_ttc[:, agent]),
            ("labels_pairwise_ttc", zeros.copy()),
            ("labels_acceleration_slack", zeros.copy()),
            ("labels_target_visible", zeros.copy()),
            ("labels_cbf_correction", zeros.copy()),
            ("labels_cbf_intervention", zeros.copy()),
            ("labels_cbf_feasible", zeros.copy()),
            ("labels_cbf_min_slack", zeros.copy()),
            ("labels_route_progress", zeros.copy()),
            ("labels_rollout_valid", np.ones(max(HORIZON_STEPS), dtype=np.float32)),
        ):
            samples[key].append(value.copy())
        samples["route_length_m"].append(0.0)
        samples["route_geometric_clearance_m"].append(float(np.min(boundary[:, agent])))
        samples["route_geometry_valid"].append(0.0)
        samples["route_candidate_index"].append(-1)
        samples["route_side_index"].append(ROUTE_SIDES.index("boundary_shadow"))
        samples["route_obstacle_id"].append(-1)
        samples["time_index"].append(int(time_index))
        samples["episode_seed"].append(int(episode_seed))
        samples["scenario_index"].append(int(scenario_index))
        samples["earliest_failure_step"].append(0)
        samples["branch_terminated"].append(0.0)
        samples["sample_type"].append(1)
    return float(np.min(boundary))


def _arrayize(samples: Mapping[str, list[Any]]) -> dict[str, np.ndarray]:
    integer = {
        "route_candidate_index",
        "route_side_index",
        "route_obstacle_id",
        "time_index",
        "episode_seed",
        "scenario_index",
        "earliest_failure_step",
    }
    arrays: dict[str, np.ndarray] = {}
    for key, value in samples.items():
        try:
            arrays[key] = np.asarray(value, dtype=np.int64 if key in integer else np.float32)
        except ValueError as error:
            shapes = sorted({tuple(np.asarray(item).shape) for item in value})
            raise ValueError(f"Archive field {key!r} has inconsistent item shapes: {shapes}") from error
    return arrays


def _runtime_sample_mask(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return rows that correspond to branches actually eligible to run.

    Boundary-shadow rows are deliberately action-conditioned negatives, but
    they are offline labels and must not affect runtime feasibility metrics.
    """

    return np.asarray(arrays["sample_type"]) == 0


def _class_counts(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Summarize runtime and offline classes without mixing their contracts."""

    runtime = _runtime_sample_mask(arrays)
    boundary_negative = np.asarray(arrays["labels_boundary_clearance"]).min(axis=1) < 0.0
    return {
        "route_geometry_valid": int(np.sum(runtime & (arrays["route_geometry_valid"] > 0.5))),
        "route_geometry_invalid": int(np.sum(runtime & (arrays["route_geometry_valid"] <= 0.5))),
        "cbf_first_step_feasible": int(np.sum(runtime & (arrays["labels_cbf_feasible"][:, 0] > 0.5))),
        "cbf_first_step_infeasible": int(np.sum(runtime & (arrays["labels_cbf_feasible"][:, 0] <= 0.5))),
        "branch_failure_within_horizon": int(
            np.sum(runtime & (arrays["earliest_failure_step"] <= max(HORIZON_STEPS)))
        ),
        # Shadow rows are intentionally included so the auxiliary boundary
        # head receives the negative class it cannot observe on safe branches.
        "boundary_clearance_negative": int(np.sum(boundary_negative)),
        "boundary_shadow_samples": int(np.sum(~runtime)),
    }


def collect(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    # The protocol loader imports the full Torch-backed evaluation stack. Keep
    # it local so pure archive-contract helpers remain testable in lightweight
    # environments where training dependencies are intentionally absent.
    from evaluate_random_central_mixed_obstacles import (  # noqa: E402
        config_for_spec,
        episode_spec,
        load_protocol,
    )

    protocol = load_protocol(args.protocol.resolve())
    if args.episodes <= 0 or args.sample_stride <= 0 or args.history_length != 8 or args.chunk_length_steps not in (3, 5):
        raise ValueError("episodes/sample-stride must be positive; route archive supports history=8 and chunk=3 or 5")
    env_config_path = args.environment_config.resolve()
    archive_config_path = args.archive_config.resolve()
    archive_config = yaml.safe_load(archive_config_path.read_text(encoding="utf-8"))
    if not isinstance(archive_config, dict) or archive_config.get("locked_test_opened") is not False:
        raise ValueError("archive config must be a closed development protocol")
    contract = archive_config.get("data_contract", {})
    if contract.get("dataset_version") != args.dataset_version or contract.get("candidate_profile") != "obstacle_route_v1":
        raise ValueError("archive config does not match the route-identity collector")
    interaction_contract = contract.get("interaction_hard_negatives", {})
    if interaction_contract and not isinstance(interaction_contract, Mapping):
        raise ValueError("data_contract interaction_hard_negatives must be a mapping")
    if interaction_contract and bool(interaction_contract.get("enabled", False)) != bool(args.interaction_hard_negatives):
        raise ValueError("archive config interaction_hard_negatives.enabled does not match the collector flag")
    if interaction_contract and dict(interaction_contract.get("sample_type_mapping", {})) != INTERACTION_SAMPLE_TYPES:
        raise ValueError("archive config interaction sample_type_mapping does not match the collector")
    if int(contract.get("chunk_length_steps", -1)) != int(args.chunk_length_steps):
        raise ValueError("archive config chunk_length_steps does not match --chunk-length-steps")
    state_distribution = archive_config.get("state_distribution", {})
    if not isinstance(state_distribution, dict):
        raise ValueError("archive config state_distribution must be a mapping")
    if bool(state_distribution.get("actor_checkpoint_required", False)) and args.actor_checkpoint is None:
        raise ValueError("This archive contract requires --actor-checkpoint")
    if str(state_distribution.get("mode", "")).strip() == "frozen_runtime_actor" and args.actor_checkpoint is None:
        raise ValueError("frozen_runtime_actor archives require --actor-checkpoint")
    configured_split = str(archive_config.get("split", "")).strip()
    if configured_split and configured_split != args.split:
        raise ValueError(
            f"archive config split {configured_split!r} does not match --split {args.split!r}"
        )
    configured_source = str(archive_config.get("source_protocol", ""))
    if configured_source and Path(configured_source).name != args.protocol.resolve().name:
        raise ValueError("archive config source_protocol does not match --protocol")
    cbf_contract = _archive_cbf_contract(archive_config)
    base_config = yaml.safe_load(env_config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError("environment config must be a mapping")
    base_config = copy.deepcopy(base_config)
    base_config.setdefault("task", {}).setdefault("pursuit", {})["terminate_on_capture"] = False
    base_config["task"]["pursuit"]["obstacle_profile"] = "mixed"
    samples = _empty_samples()
    scenes: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    geometry_valid = 0
    geometry_total = 0
    cbf_feasible = 0
    cbf_total = 0
    branch_failures = 0
    interaction_counts: Counter[str] = Counter()
    interaction_branch_failures: Counter[str] = Counter()
    actor_policy = None
    actor_device = None
    actor_action_scale = 5.0
    actor_metadata: dict[str, Any] = {}
    actor_checkpoint: Path | None = None
    if args.actor_checkpoint is not None:
        actor_checkpoint = args.actor_checkpoint.resolve()
        if not actor_checkpoint.is_file():
            raise FileNotFoundError(f"actor checkpoint does not exist: {actor_checkpoint}")
        if torch is None:
            raise RuntimeError("--actor-checkpoint requires torch")
        from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402

        actor_device = select_device(args.actor_device)
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
        if actor_checkpoint is not None and actor_policy is None:
            # Loading against the first constructed environment validates the
            # checkpoint observation/action contract before any samples are
            # written.  All subsequent episodes reuse the same frozen actor.
            actor_policy, actor_action_scale, actor_metadata = load_policy(
                actor_checkpoint,
                env,
                observation,
                actor_device,
            )
        safety_filter = JointCBFQPSafetyFilter(
            env,
            anticipatory_horizon_steps=int(cbf_contract["anticipatory_horizon_steps"]),
            barrier_mode=str(cbf_contract["barrier_mode"]),
        )
        route_config = _route_config(env, safety_filter, args.chunk_length_steps)
        extent = float(config["world"]["half_extent_xy"])
        observation_history = [policy_observations(env, observation).copy()]
        executed_actions: list[np.ndarray] = []
        previous_action = np.asarray(env.defender_velocities, dtype=np.float64).copy()
        hidden = (
            actor_policy.initial_actor_hidden(env.n_defenders, device=actor_device)
            if actor_policy is not None and hasattr(actor_policy, "initial_actor_hidden")
            else None
        )
        recurrent_reset_interval = (
            int(actor_metadata["recurrent_reset_interval_steps"])
            if actor_policy is not None and actor_metadata.get("recurrent_reset_interval_steps") is not None
            else None
        )
        sampled_states = 0
        for time_index in range(int(env.max_steps)):
            if (
                actor_policy is not None
                and hidden is not None
                and recurrent_reset_interval is not None
                and time_index > 0
                and time_index % recurrent_reset_interval == 0
            ):
                hidden = actor_policy.initial_actor_hidden(env.n_defenders, device=actor_device)
            if actor_policy is not None:
                desired, hidden = _actor_action(
                    actor_policy,
                    policy_observations(env, observation),
                    actor_device,
                    actor_action_scale,
                    hidden,
                )
            else:
                desired = np.asarray(controller.act(observation), dtype=np.float64)
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
                for route_index, route in enumerate(route_batch.candidates):
                    route_counts[route.label] += 1
                    geometry_total += 1
                    geometry_valid += int(route.valid)
                    labels = _route_rollout(
                        env,
                        observation,
                        controller,
                        route,
                        safety_filter,
                        extent=extent,
                    )
                    labels["earliest_failure_step"] = int(labels["earliest_failure_step"])
                    labels["branch_terminated"] = bool(labels["branch_terminated"])
                    cbf_feasible += int(np.asarray(labels["cbf_feasible"])[0].mean() > 0.5)
                    cbf_total += 1
                    branch_failures += int(bool(labels["cbf_failed"]))
                    _append_samples(
                        samples,
                        observation_history=observation_history,
                        executed_action_history=executed_actions,
                        route=route,
                        route_index=route_index,
                        labels=labels,
                        episode_seed=int(spec["episode_seed"]),
                        scenario_index=scenario_index,
                        time_index=time_index,
                        action_scale=5.0,
                    )
                if args.interaction_hard_negatives:
                    # These branches are deliberately offline-only.  They use
                    # the same first candidate metadata for a stable archive
                    # schema, but their action chunks are projected synthetic
                    # interaction transitions and never reach env.step().
                    if not route_batch.candidates:
                        raise RuntimeError("Route generator returned no template for interaction branches")
                    template = route_batch.candidates[0]
                    interaction_chunks = _interaction_hard_negative_chunks(
                        env,
                        reachable_nominal,
                        previous_action,
                        args.chunk_length_steps,
                    )
                    for mode in INTERACTION_HARD_NEGATIVE_MODES:
                        chunk = interaction_chunks.get(mode)
                        if chunk is None:
                            continue
                        offline_route = _offline_interaction_route(template, mode, chunk)
                        interaction_labels = _route_rollout(
                            env,
                            observation,
                            controller,
                            offline_route,
                            safety_filter,
                            extent=extent,
                        )
                        interaction_counts[mode] += int(env.n_defenders)
                        interaction_branch_failures[mode] += int(bool(interaction_labels["cbf_failed"]))
                        _append_samples(
                            samples,
                            observation_history=observation_history,
                            executed_action_history=executed_actions,
                            route=offline_route,
                            route_index=-1,
                            labels=interaction_labels,
                            episode_seed=int(spec["episode_seed"]),
                            scenario_index=scenario_index,
                            time_index=time_index,
                            action_scale=5.0,
                            sample_type=INTERACTION_SAMPLE_TYPES[mode],
                        )
                _append_boundary_shadow_samples(
                    samples,
                    observation_history=observation_history,
                    executed_action_history=executed_actions,
                    env=env,
                    episode_seed=int(spec["episode_seed"]),
                    scenario_index=scenario_index,
                    time_index=time_index,
                    action_scale=5.0,
                    chunk_length_steps=args.chunk_length_steps,
                )
            action, diagnostics = safety_filter.filter(
                desired,
                observation,
                nominal_actions=reachable_nominal,
            )
            if not diagnostics.verified_feasible or diagnostics.fallback_mode == "controlled_abort":
                break
            observation, _reward, terminated, truncated, _info = env.step(action, record_history=False)
            executed_actions.append(np.asarray(action, dtype=np.float32).copy())
            observation_history.append(policy_observations(env, observation).copy())
            previous_action = np.asarray(action, dtype=np.float64).copy()
            if terminated or truncated:
                break
        scene = scenario_metadata(scenario)
        scene_hash = hashlib.sha256(
            json.dumps(_jsonable(scene), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        scenes.append(
            {
                "scenario_index": scenario_index,
                "episode_seed": int(spec["episode_seed"]),
                "layout_seed": int(spec["layout_seed"]),
                "scene_hash": scene_hash,
                "spec": spec,
                "scenario": scene,
                "sampled_states": sampled_states,
            }
        )
    arrays = _arrayize(samples)
    if not arrays["inputs"].size:
        raise RuntimeError("No route-identity samples were collected")
    runtime_mask = _runtime_sample_mask(arrays)
    class_counts = _class_counts(arrays)
    metadata = {
        "dataset_version": str(args.dataset_version),
        "task": "action_conditioned_interaction_aware_jepa_route_identity_counterfactual",
        "split": str(args.split),
        "development_only": True,
        "locked_test_opened": False,
        "episodes": int(args.episodes),
        "history_length": 8,
        "horizon_steps": list(HORIZON_STEPS),
        "chunk_length_steps": int(args.chunk_length_steps),
        "interaction_action_conditioned_route_chunk": True,
        "candidate_profile": "obstacle_route_v1",
        "candidate_count": len(ROUTE_LABELS),
        "route_labels": list(ROUTE_LABELS),
        "route_side_vocab": list(ROUTE_SIDES),
        "runtime_route_sample_type": 0,
        "boundary_shadow_sample_type": 1,
        "boundary_shadow_is_offline_only": True,
        "candidate_semantics": "geometry_conditioned_route_chunk_execute_first_step_then_replan",
        "action_history_alignment": "past_executed_actions_then_route_first_action",
        "action_scale": 5.0,
        "state_distribution_source": {
            "mode": "frozen_runtime_actor" if actor_checkpoint is not None else "dynamic_rule_controller",
            "actor_checkpoint": str(actor_checkpoint) if actor_checkpoint is not None else None,
            "actor_checkpoint_sha256": _sha256(actor_checkpoint) if actor_checkpoint is not None else None,
            "actor_action_scale": float(actor_action_scale) if actor_checkpoint is not None else None,
            "actor_recurrent_reset_interval_steps": (
                int(actor_metadata["recurrent_reset_interval_steps"])
                if actor_checkpoint is not None and actor_metadata.get("recurrent_reset_interval_steps") is not None
                else None
            ),
            "runtime_match_required": actor_checkpoint is not None,
        },
        "sample_stride": int(args.sample_stride),
        "sample_count_per_defender": int(arrays["inputs"].shape[0]),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "class_counts": class_counts,
        "interaction_hard_negatives": {
            "enabled": bool(args.interaction_hard_negatives),
            "modes": list(INTERACTION_HARD_NEGATIVE_MODES),
            "sample_type_mapping": dict(INTERACTION_SAMPLE_TYPES),
            "offline_only": True,
            "sample_counts": dict(sorted(interaction_counts.items())),
            "branch_failures": dict(sorted(interaction_branch_failures.items())),
        },
        "information_boundary": {
            "target_truth_used_only_for_offline_labels": True,
            "online_route_generation_uses_public_obstacles_and_target_beliefs": True,
            "locked_split_used": False,
            "validation_or_development_used_for_training": False,
            "raw_unverified_action_executed": False,
        },
        "source": {
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": _sha256(args.protocol.resolve()),
            "environment_config": str(args.environment_config.resolve()),
            "environment_config_sha256": _sha256(args.environment_config.resolve()),
            "archive_config": str(archive_config_path),
            "archive_config_sha256": _sha256(archive_config_path),
            "collector_git_revision": _git_revision(),
        },
        "cbf_contract": {
            "solver": "scipy_slsqp_joint_cbf_qp",
            "anticipatory_horizon_steps": int(cbf_contract["anticipatory_horizon_steps"]),
            "barrier_mode": str(cbf_contract["barrier_mode"]),
            "routes_advance_only_after_primary_verified_feasible": True,
            "controlled_abort_preserved": True,
        },
        "route_counts": dict(sorted(route_counts.items())),
        "cbf_branch_counts": {
            "first_step_feasible_branches": int(cbf_feasible),
            "total_branches": int(cbf_total),
            "branches_failed_within_horizon": int(branch_failures),
        },
    }
    return arrays, metadata, scenes


def main() -> int:
    args = parse_args()
    if not args.development_only:
        raise ValueError("route archive collection requires --development-only")
    if torch is None or SummaryWriter is None:
        raise RuntimeError(
            "route archive collection requires torch and tensorboard; install environment.yml first"
        )
    output_dir = _fresh(args.output_dir, "route archive output")
    tensorboard_dir = _fresh(args.tensorboard_logdir, "route archive TensorBoard logdir")
    arrays, metadata, scenes = collect(args)
    runtime_mask = _runtime_sample_mask(arrays)
    class_counts = _class_counts(arrays)
    dataset_path = output_dir / "route_identity_counterfactual.npz"
    np.savez_compressed(dataset_path, **arrays)
    _write_json(output_dir / "metadata.json", metadata)
    (output_dir / "scenes.jsonl").write_text(
        "".join(json.dumps(_jsonable(scene), sort_keys=True) + "\n" for scene in scenes), encoding="utf-8"
    )
    _write_json(
        output_dir / "provenance.json",
        {
            "dataset_sha256": _sha256(dataset_path),
            "metadata_sha256": _sha256(output_dir / "metadata.json"),
            "scenes_sha256": _sha256(output_dir / "scenes.jsonl"),
            "git_revision": _git_revision(),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tensorboard_logdir": str(tensorboard_dir),
            "actor_checkpoint": metadata.get("state_distribution_source", {}).get("actor_checkpoint"),
            "actor_checkpoint_sha256": metadata.get("state_distribution_source", {}).get("actor_checkpoint_sha256"),
            "development_only": True,
            "locked_test_opened": False,
        },
    )
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_scalar("Archive/sample_count", float(arrays["inputs"].shape[0]), 0)
        writer.add_scalar("Archive/runtime_sample_count", float(np.sum(runtime_mask)), 0)
        writer.add_scalar(
            "Archive/route_geometry_valid_fraction",
            float(np.mean(arrays["route_geometry_valid"][runtime_mask])),
            0,
        )
        writer.add_scalar(
            "Archive/cbf_first_step_feasible_fraction",
            float(np.mean(arrays["labels_cbf_feasible"][runtime_mask, 0])),
            0,
        )
        writer.add_scalar(
            "Archive/branch_failure_within_horizon_fraction",
            float(np.mean(arrays["earliest_failure_step"][runtime_mask] <= max(HORIZON_STEPS))),
            0,
        )
        writer.add_scalar(
            "Archive/negative_boundary_ttc_fraction",
            float(np.mean(arrays["labels_boundary_ttc"].min(axis=1) < TTC_CLIP_SECONDS)),
            0,
        )
        writer.add_scalar(
            "Archive/negative_pairwise_ttc_fraction",
            float(np.mean(arrays["labels_pairwise_ttc"].min(axis=1) < TTC_CLIP_SECONDS)),
            0,
        )
        writer.add_scalar(
            "Archive/negative_acceleration_slack_fraction",
            float(np.mean(arrays["labels_acceleration_slack"].min(axis=1) < 0.0)),
            0,
        )
        writer.add_scalar(
            "Archive/boundary_clearance_negative_fraction",
            float(class_counts["boundary_clearance_negative"] / arrays["inputs"].shape[0]),
            0,
        )
        for index, label in enumerate(metadata["route_labels"]):
            writer.add_scalar(
                f"Archive/route_samples/{label}",
                float(metadata["route_counts"].get(label, 0)),
                index,
            )
        for index, mode in enumerate(INTERACTION_HARD_NEGATIVE_MODES):
            writer.add_scalar(
                f"Archive/interaction_hard_negative_samples/{mode}",
                float(metadata["interaction_hard_negatives"]["sample_counts"].get(mode, 0)),
                index,
            )
            writer.add_scalar(
                f"Archive/interaction_hard_negative_branch_failures/{mode}",
                float(metadata["interaction_hard_negatives"]["branch_failures"].get(mode, 0)),
                index,
            )
        writer.add_text(
            "Archive/interaction_hard_negative_contract",
            json.dumps(metadata["interaction_hard_negatives"], sort_keys=True),
            0,
        )
        writer.add_text("Archive/dataset_version", DATASET_VERSION, 0)
        writer.add_text("Archive/protocol_sha256", metadata["source"]["protocol_sha256"], 0)
        writer.add_text("Archive/dataset_sha256", _sha256(dataset_path), 0)
        writer.add_text("Archive/information_boundary", json.dumps(metadata["information_boundary"], sort_keys=True), 0)
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "metadata": str(output_dir / "metadata.json"),
                "sample_count": int(arrays["inputs"].shape[0]),
                "runtime_sample_count": int(np.sum(runtime_mask)),
                "geometry_valid_fraction": float(np.mean(arrays["route_geometry_valid"][runtime_mask])),
                "cbf_first_step_feasible_fraction": float(
                    np.mean(arrays["labels_cbf_feasible"][runtime_mask, 0])
                ),
                "branch_failure_within_horizon": class_counts["branch_failure_within_horizon"],
                "class_counts": class_counts,
                "tensorboard": str(tensorboard_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
