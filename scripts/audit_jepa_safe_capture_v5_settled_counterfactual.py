"""Audit candidate ranking against offline settled counterfactual outcomes.

The paired evaluator only executes the selected first action.  This audit
reconstructs each frozen run with the recorded executed actions and, at every
JEPA decision, branches the environment for each eligible candidate.  Each
branch executes the constant three-step candidate chunk through the same
Joint CBF-QP and uses simulator ground truth *offline* to settle local target
progress, capture and safety labels.  No branch is used to alter the source
run or an online decision.

The result is deliberately a local counterfactual diagnostic, not a claim that
the unexecuted candidate would have produced a full-episode policy outcome.
It reports selected-not-best, rank correlations, safety/capture alignment,
CBF cost and a documented score-softmax Brier/ECE proxy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.jepa_safe_capture_candidates import (  # noqa: E402
    SafeCaptureCandidateConfig,
    make_safe_capture_candidate_chunks,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import prepare_showcase_episode, scenario_from_metadata  # noqa: E402
from evaluate_random_central_mixed_obstacles import config_for_spec, episode_spec, load_protocol  # noqa: E402


EXPECTED_VARIANTS = ("m3", "a1", "a2")
LABELS = ("nominal", "intercept", "lateral_clearance", "formation_clearance", "visibility_hold")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _scene_hash(metadata: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _jsonable(value: Any) -> Any:
    """Convert numpy values and non-finite diagnostics to JSON-safe values."""

    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _p95(values: Sequence[float]) -> float | None:
    return float(np.quantile(values, 0.95)) if values else None


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks with deterministic ties, without adding scipy dependency."""

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and array[order[end]] == array[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + 1 + end)
        index = end
    return ranks


def _spearman(predicted: Sequence[float], settled: Sequence[float]) -> float | None:
    if len(predicted) < 2 or len(predicted) != len(settled):
        return None
    left = _rankdata(predicted)
    right = _rankdata(settled)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _kendall(predicted: Sequence[float], settled: Sequence[float]) -> float | None:
    if len(predicted) < 2 or len(predicted) != len(settled):
        return None
    concordant = discordant = tied = 0
    for first in range(len(predicted)):
        for second in range(first + 1, len(predicted)):
            left = np.sign(float(predicted[first]) - float(predicted[second]))
            right = np.sign(float(settled[first]) - float(settled[second]))
            if left == 0 or right == 0:
                tied += 1
            elif left == right:
                concordant += 1
            else:
                discordant += 1
    denominator = concordant + discordant
    return float((concordant - discordant) / denominator) if denominator else 0.0


def _softmax_probabilities(scores: Sequence[float], eligible: Sequence[bool]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    result = np.zeros(len(values), dtype=np.float64)
    if not np.any(mask):
        return result
    logits = -values[mask]
    logits -= np.max(logits)
    probabilities = np.exp(np.clip(logits, -40.0, 40.0))
    probabilities /= max(float(np.sum(probabilities)), 1e-12)
    result[mask] = probabilities
    return result


def _ece(probabilities: Sequence[float], labels: Sequence[float], bins: int = 10) -> float | None:
    if not probabilities:
        return None
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    total = len(probabilities_array)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (probabilities_array >= lower) & (
            probabilities_array < upper if index < bins - 1 else probabilities_array <= upper
        )
        if np.any(selected):
            error += float(np.sum(selected)) / total * abs(
                float(np.mean(probabilities_array[selected])) - float(np.mean(labels_array[selected]))
            )
    return float(error)


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


def _target_distance(env: CaptureRadiusPursuit3DEnv) -> float:
    return float(np.min(np.linalg.norm(env.defender_positions - env.target_position[None, :], axis=1)))


def _manifest_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.resolve().read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid scene manifest row: {path}")
        index = int(value.get("episode_index", -1))
        if index in rows:
            raise ValueError(f"Duplicate scene manifest episode: {path}:{index}")
        rows[index] = value
    if not rows:
        raise ValueError(f"Empty scene manifest: {path}")
    return rows


def _trace_rows(run_dir: Path, episode_index: int) -> list[dict[str, Any]]:
    path = run_dir.resolve() / "step_traces" / f"episode_{episode_index:04d}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid trace row: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    return rows


def _episode_outcomes(run_dir: Path) -> dict[int, dict[str, Any]]:
    with (run_dir / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        return {int(row["episode_index"]): dict(row) for row in csv.DictReader(handle)}


def _validate_run(run_dir: Path, expected_manifest_hash: str | None = None) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    summary = _json(run_dir / "summary.json")
    metadata = summary.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run is outside the development boundary: {run_dir}")
    variant = str(metadata.get("variant", {}).get("variant", ""))
    if variant not in EXPECTED_VARIANTS:
        raise ValueError(f"Settled audit expects one of {EXPECTED_VARIANTS}, got {variant!r}")
    manifest_hash = str(metadata.get("inputs", {}).get("scene_manifest_sha256", ""))
    if len(manifest_hash) != 64:
        raise ValueError(f"Run has no valid scene manifest hash: {run_dir}")
    if expected_manifest_hash is not None and manifest_hash != expected_manifest_hash:
        raise ValueError(f"Scene manifest mismatch: {run_dir}")
    traces = sorted((run_dir / "step_traces").glob("episode_*.jsonl"))
    if not traces:
        raise FileNotFoundError(f"No step traces: {run_dir}")
    overall = summary.get("overall", {})
    if not isinstance(overall, dict):
        raise ValueError(f"Run summary has no overall metrics: {run_dir}")
    return {
        "run_dir": str(run_dir.resolve()),
        "variant": variant,
        "training_seed": int(metadata.get("training_seed", -1)),
        "episodes": int(summary.get("overall", {}).get("episodes", len(traces))),
        "scene_manifest": str(metadata.get("inputs", {}).get("scene_manifest", "")),
        "scene_manifest_sha256": manifest_hash,
        "protocol_sha256": str(metadata.get("inputs", {}).get("protocol_sha256", "")),
        "environment_config": str(metadata.get("inputs", {}).get("environment_config", "")),
        "actor_checkpoint_sha256": str(metadata.get("inputs", {}).get("actor_checkpoint_sha256", "")),
        "jepa_checkpoint_sha256": str(metadata.get("inputs", {}).get("jepa_checkpoint_sha256", "")),
        "ledger_sha256": metadata.get("inputs", {}).get("reliability_ledger_sha256"),
        "source_raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        "source_cbf_unverified_steps": int(overall.get("cbf_unverified_steps", 0)),
    }, _episode_outcomes(run_dir)


def _pair_labels(baseline_dir: Path, candidate_dir: Path) -> dict[int, str]:
    baseline = _episode_outcomes(baseline_dir)
    candidate = _episode_outcomes(candidate_dir)
    baseline_summary = _json(baseline_dir / "summary.json")
    candidate_summary = _json(candidate_dir / "summary.json")
    baseline_metadata = baseline_summary.get("metadata", {})
    candidate_metadata = candidate_summary.get("metadata", {})
    baseline_hash = str(baseline_metadata.get("inputs", {}).get("scene_manifest_sha256", ""))
    candidate_hash = str(candidate_metadata.get("inputs", {}).get("scene_manifest_sha256", ""))
    if baseline_hash != candidate_hash:
        raise ValueError("Baseline and candidate scene manifest hashes differ.")
    if int(baseline_metadata.get("training_seed", -1)) != int(candidate_metadata.get("training_seed", -2)):
        raise ValueError("Baseline and candidate training seeds differ.")
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate episode identities differ.")
    labels: dict[int, str] = {}
    for index in sorted(baseline):
        base = _bool(baseline[index].get("safe_capture_success"))
        method = _bool(candidate[index].get("safe_capture_success"))
        labels[index] = "improved" if method and not base else "degraded" if base and not method else "tied"
    return labels


def _candidate_config(env: CaptureRadiusPursuit3DEnv) -> SafeCaptureCandidateConfig:
    return SafeCaptureCandidateConfig(
        candidate_count=5,
        chunk_length_steps=3,
        perturbation_mps=0.10,
        max_speed_mps=float(env.agents["defender_max_speed"]),
        max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
        dt_seconds=float(env.dt),
        max_action_change_mps=float(env.agents["defender_max_acceleration"]) * float(env.dt),
        project_to_reachable_dynamics=True,
    )


def _branch_candidate(
    env: CaptureRadiusPursuit3DEnv,
    observation: Mapping[str, Any],
    candidate_action: np.ndarray,
    nominal_action: np.ndarray,
    chunk_length: int,
) -> dict[str, Any]:
    """Settle one candidate for one local chunk using simulator truth offline."""

    branch = copy.deepcopy(env)
    safety_filter = JointCBFQPSafetyFilter(branch)
    before_distance = _target_distance(branch)
    minimum_obstacle = float("inf")
    minimum_pairwise = float("inf")
    minimum_boundary = float("inf")
    corrections: list[float] = []
    latencies: list[float] = []
    unverified_steps = 0
    capture_event = False
    termination_reason = "chunk_complete"
    for _ in range(chunk_length):
        filtered, diagnostics = safety_filter.filter(
            np.asarray(candidate_action, dtype=np.float64),
            observation if branch.step_count == env.step_count else branch.observe(),
            nominal_actions=np.asarray(nominal_action, dtype=np.float64),
            execution_mode="normal",
        )
        filtered = np.asarray(filtered, dtype=np.float64)
        corrections.append(float(diagnostics.action_correction_norm))
        latencies.append(float(diagnostics.solve_latency_ms))
        if not diagnostics.verified_feasible:
            unverified_steps += 1
            termination_reason = "cbf_unverified"
            break
        next_observation, _reward, terminated, truncated, info = branch.step(filtered, record_history=True)
        values = _safety_observables(branch)
        minimum_obstacle = min(minimum_obstacle, values["minimum_obstacle_clearance_m"])
        minimum_pairwise = min(minimum_pairwise, values["minimum_pairwise_clearance_m"])
        minimum_boundary = min(minimum_boundary, values["minimum_boundary_clearance_m"])
        capture_event = capture_event or bool(info.get("capture_event", False))
        if bool(info.get("collision", False)) or bool(info.get("defender_boundary_violation", False)):
            termination_reason = "environment_safety_failure"
        if capture_event:
            termination_reason = "capture_event"
            break
        if terminated or truncated:
            termination_reason = str(info.get("termination_reason", "truncated"))
            break
        observation = next_observation
    after_distance = _target_distance(branch)
    safety_ok = bool(
        unverified_steps == 0
        and minimum_obstacle >= -1e-9
        and minimum_pairwise >= -1e-9
        and minimum_boundary >= -1e-9
        and not bool(branch.defender_boundary_violation)
    )
    return {
        "settled_capture_event": bool(capture_event),
        "settled_safe_capture": bool(capture_event and safety_ok),
        "settled_safety_ok": safety_ok,
        "settled_progress_m": float(before_distance - after_distance),
        "settled_before_distance_m": before_distance,
        "settled_after_distance_m": after_distance,
        "settled_min_obstacle_clearance_m": minimum_obstacle,
        "settled_min_pairwise_clearance_m": minimum_pairwise,
        "settled_min_boundary_clearance_m": minimum_boundary,
        "settled_cbf_correction_norm_mps": float(np.mean(corrections)) if corrections else float("inf"),
        "settled_cbf_correction_p95_mps": _p95(corrections),
        "settled_cbf_latency_p95_ms": _p95(latencies),
        "settled_cbf_unverified_steps": int(unverified_steps),
        "settled_termination_reason": termination_reason,
    }


def _best_candidate(outcomes: Sequence[Mapping[str, Any]], eligible: Sequence[bool]) -> int:
    indices = [index for index, valid in enumerate(eligible) if valid]
    if not indices:
        return 0
    # Safety/capture is lexicographically primary. Progress is the tie-breaker;
    # correction cost is the final deterministic tie-breaker.
    return min(
        indices,
        key=lambda index: (
            -int(bool(outcomes[index]["settled_safe_capture"])),
            -int(bool(outcomes[index]["settled_safety_ok"])),
            -float(outcomes[index]["settled_progress_m"]),
            float(outcomes[index]["settled_cbf_correction_norm_mps"]),
            index,
        ),
    )


def _settle_run(
    run_dir: Path,
    *,
    protocol_path: Path,
    environment_config_path: Path,
    baseline_dir: Path,
    eligibility_floor_override_m: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata, outcomes = _validate_run(run_dir)
    manifest_path = run_dir / "scene_manifest.jsonl"
    if sha256(manifest_path) != metadata["scene_manifest_sha256"]:
        raise ValueError(f"Scene manifest provenance hash mismatch: {run_dir}")
    manifest = _manifest_rows(manifest_path)
    protocol = load_protocol(protocol_path)
    environment_config = environment_config_path.resolve()
    expected_indices = set(outcomes)
    if set(manifest) != expected_indices:
        raise ValueError(f"Manifest/outcome episode identities differ: {run_dir}")
    for index, item in manifest.items():
        if item.get("spec") != episode_spec(protocol, "validation", index):
            raise ValueError(f"Manifest spec differs from supplied protocol: {run_dir} episode {index}")
        scenario_metadata = item.get("scenario")
        if not isinstance(scenario_metadata, dict) or str(item.get("scene_hash")) != _scene_hash(scenario_metadata):
            raise ValueError(f"Manifest scene hash is invalid: {run_dir} episode {index}")
        if int(item.get("training_seed", metadata["training_seed"])) != int(metadata["training_seed"]):
            raise ValueError(f"Manifest training seed differs from run metadata: {run_dir} episode {index}")
    episode_rows: list[dict[str, Any]] = []
    pair_labels = _pair_labels(baseline_dir, run_dir)
    for episode_index in sorted(manifest):
        item = manifest[episode_index]
        spec = dict(item["spec"])
        config = config_for_spec("f2", spec, environment_config)
        scenario = scenario_from_metadata(dict(item["scenario"]))
        env = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=len(scenario.obstacles),
            target_speed_scale=float(spec["target_speed_scale"]),
        )
        observation = prepare_showcase_episode(env, scenario, seed=int(spec["episode_seed"]), record_history=True, validate_scenario=False)
        previous_action = np.asarray(env.defender_velocities, dtype=np.float64).copy()
        candidate_config = _candidate_config(env)
        trace = _trace_rows(run_dir, episode_index)
        for record in trace:
            ranking = record.get("candidate_ranking")
            if not isinstance(ranking, dict):
                # M0 has no candidate ranking; this function is intentionally
                # restricted to JEPA variants.
                raise ValueError(f"Missing candidate ranking in {run_dir} episode {episode_index}")
            reachable_nominal = np.asarray(record.get("reachable_nominal_action"), dtype=np.float64)
            executed = np.asarray(record.get("executed_action"), dtype=np.float64)
            if reachable_nominal.shape != (env.n_defenders, 3) or executed.shape != reachable_nominal.shape:
                raise ValueError(f"Malformed action trace in {run_dir} episode {episode_index}")
            batch = make_safe_capture_candidate_chunks(
                reachable_nominal,
                observation,
                config=candidate_config,
                previous_action=previous_action,
            )
            recorded_eligible = np.asarray(ranking.get("eligible_mask", batch.valid_mask.tolist()), dtype=bool)
            if recorded_eligible.shape != (5,):
                raise ValueError(f"Malformed eligible_mask in {run_dir} episode {episode_index}")
            scores = np.asarray(ranking.get("scores", [float("inf")] * 5), dtype=np.float64)
            if scores.shape != (5,) or not np.isfinite(scores[recorded_eligible]).all():
                raise ValueError(f"Malformed finite scores in {run_dir} episode {episode_index}")
            if eligibility_floor_override_m is None:
                eligible = recorded_eligible
            else:
                predicted_clearance = np.asarray(
                    ranking.get("predicted_min_clearance_m", [float("nan")] * 5),
                    dtype=np.float64,
                )
                ledger_states = np.asarray(
                    ranking.get("ledger_states", ["safe_hold"] * 5),
                    dtype=object,
                )
                valid_mask = np.asarray(
                    ranking.get("valid_mask", batch.valid_mask.tolist()),
                    dtype=bool,
                )
                if (
                    predicted_clearance.shape != (5,)
                    or ledger_states.shape != (5,)
                    or valid_mask.shape != (5,)
                    or not np.isfinite(predicted_clearance).all()
                ):
                    raise ValueError(f"Malformed ranking safety fields in {run_dir} episode {episode_index}")
                # Recompute only the prediction-floor part of eligibility. The
                # finite/reachability and trusted-ledger gates remain fixed;
                # this is an offline sensitivity audit, not an online bypass.
                eligible = (
                    valid_mask
                    & (ledger_states == "trusted")
                    & np.isfinite(scores)
                    & (predicted_clearance >= float(eligibility_floor_override_m))
                )
            local_outcomes = [
                _branch_candidate(
                    env,
                    observation,
                    batch.chunks[index, 0],
                    reachable_nominal,
                    int(batch.chunks.shape[1]),
                )
                if bool(eligible[index])
                else {
                    "settled_capture_event": False,
                    "settled_safe_capture": False,
                    "settled_safety_ok": False,
                    "settled_progress_m": float("-inf"),
                    "settled_before_distance_m": None,
                    "settled_after_distance_m": None,
                    "settled_min_obstacle_clearance_m": None,
                    "settled_min_pairwise_clearance_m": None,
                    "settled_min_boundary_clearance_m": None,
                    "settled_cbf_correction_norm_mps": float("inf"),
                    "settled_cbf_correction_p95_mps": None,
                    "settled_cbf_latency_p95_ms": None,
                    "settled_cbf_unverified_steps": 0,
                    "settled_termination_reason": "ineligible",
                }
                for index in range(5)
            ]
            selected_index = int(ranking.get("selected_index", 0))
            if not 0 <= selected_index < 5:
                raise ValueError(f"Invalid selected_index in {run_dir} episode {episode_index}")
            best_index = _best_candidate(local_outcomes, eligible)
            selected_outcome = local_outcomes[selected_index]
            probabilities = _softmax_probabilities(scores, eligible)
            settled_labels = np.asarray(
                [float(bool(item["settled_safe_capture"])) for item in local_outcomes], dtype=np.float64
            )
            selected_probability = float(probabilities[selected_index])
            row = {
                "training_seed": int(metadata["training_seed"]),
                "variant": str(metadata["variant"]),
                "episode_index": int(episode_index),
                "episode_seed": int(spec["episode_seed"]),
                "step": int(record.get("step", env.step_count + 1)),
                "pair_label": pair_labels[episode_index],
                "selected_index": selected_index,
                "best_settled_index": int(best_index),
                "selected_not_best": bool(selected_index != best_index),
                "selected_settled_safe_capture": bool(selected_outcome["settled_safe_capture"]),
                "best_settled_safe_capture": bool(local_outcomes[best_index]["settled_safe_capture"]),
                "selected_settled_safety_ok": bool(selected_outcome["settled_safety_ok"]),
                "selected_settled_progress_m": float(selected_outcome["settled_progress_m"]),
                "best_settled_progress_m": float(local_outcomes[best_index]["settled_progress_m"]),
                "selected_cbf_correction_norm_mps": float(selected_outcome["settled_cbf_correction_norm_mps"]),
                "selected_cbf_unverified_steps": int(selected_outcome["settled_cbf_unverified_steps"]),
                "selected_settled_termination_reason": str(selected_outcome["settled_termination_reason"]),
                "score_softmax_selected_probability": selected_probability,
                "score_softmax_brier": float(np.mean((probabilities - settled_labels) ** 2)),
                "score_softmax_ece_label": float(bool(selected_outcome["settled_safe_capture"])),
                "scores": scores.tolist(),
                "eligible_mask": eligible.tolist(),
                "settled_candidates": local_outcomes,
                "predicted_min_clearance_m": ranking.get("predicted_min_clearance_m"),
                "predicted_min_ttc_s": ranking.get("predicted_min_ttc_s"),
                "predicted_uncertainty": ranking.get("predicted_uncertainty"),
                "predicted_visibility": ranking.get("predicted_visibility"),
                "predicted_cbf_risk": ranking.get("predicted_cbf_risk"),
                "ledger_states": ranking.get("ledger_states"),
                "ledger_credits": ranking.get("ledger_credits"),
                "top_two_margin_m": ranking.get("top_two_margin_m"),
            }
            row["predicted_rank_spearman"] = _spearman(
                [float(scores[index]) for index in range(5) if bool(eligible[index])],
                [float(local_outcomes[index]["settled_progress_m"]) for index in range(5) if bool(eligible[index])],
            )
            row["predicted_rank_kendall"] = _kendall(
                [float(scores[index]) for index in range(5) if bool(eligible[index])],
                [float(local_outcomes[index]["settled_progress_m"]) for index in range(5) if bool(eligible[index])],
            )
            episode_rows.append(row)
            # Advance the source replay by exactly the recorded executed action.
            next_observation, _reward, terminated, truncated, _info = env.step(executed, record_history=True)
            observation = next_observation
            previous_action = executed.copy()
            if terminated or truncated:
                break
        outcome = outcomes.get(episode_index)
        if outcome is None:
            raise ValueError(f"Missing episode outcome for {run_dir} episode {episode_index}")
        expected_steps = int(outcome.get("steps", -1))
        if env.step_count != expected_steps:
            raise ValueError(
                f"Source replay step mismatch for {run_dir} episode {episode_index}: "
                f"replayed={env.step_count}, recorded={expected_steps}"
            )
    result = {
        **metadata,
        "decision_count": len(episode_rows),
        "pair_counts": dict(Counter(row["pair_label"] for row in episode_rows)),
    }
    return result, episode_rows


def _group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"decisions": 0}
    def values(name: str) -> list[float]:
        return [number for row in rows if (number := _finite(row.get(name))) is not None]
    probabilities = values("score_softmax_selected_probability")
    labels = [float(bool(row.get("selected_settled_safe_capture"))) for row in rows]
    return {
        "decisions": len(rows),
        "episodes": len({(row.get("training_seed"), row.get("variant"), row.get("episode_index")) for row in rows}),
        "selected_not_best_count": int(sum(bool(row.get("selected_not_best")) for row in rows)),
        "selected_not_best_rate": float(np.mean([bool(row.get("selected_not_best")) for row in rows])),
        "selected_settled_safe_capture_rate": float(np.mean([bool(row.get("selected_settled_safe_capture")) for row in rows])),
        "best_settled_safe_capture_rate": float(np.mean([bool(row.get("best_settled_safe_capture")) for row in rows])),
        "selected_settled_safety_rate": float(np.mean([bool(row.get("selected_settled_safety_ok")) for row in rows])),
        "mean_selected_progress_m": _mean(values("selected_settled_progress_m")),
        "mean_best_progress_m": _mean(values("best_settled_progress_m")),
        "mean_selected_cbf_correction_norm_mps": _mean(values("selected_cbf_correction_norm_mps")),
        "selected_cbf_unverified_steps": int(sum(int(row.get("selected_cbf_unverified_steps", 0)) for row in rows)),
        "spearman_mean": _mean(values("predicted_rank_spearman")),
        "kendall_mean": _mean(values("predicted_rank_kendall")),
        "brier_proxy_mean": _mean(values("score_softmax_brier")),
        "ece_proxy": _ece(probabilities, labels),
    }


def _write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/settled_counterfactual", json.dumps(report["policy"], indent=2, sort_keys=True), 0)
        writer.add_text("Provenance/inputs", json.dumps(report["inputs"], indent=2, sort_keys=True), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2, sort_keys=True), 0)
        for variant, stats in report["by_variant"].items():
            writer.add_scalar(f"Ranking/{variant}/selected_not_best_rate", float(stats["selected_not_best_rate"]), 0)
            writer.add_scalar(f"Ranking/{variant}/spearman_mean", float(stats["spearman_mean"] or 0.0), 0)
            writer.add_scalar(f"Ranking/{variant}/kendall_mean", float(stats["kendall_mean"] or 0.0), 0)
            writer.add_scalar(f"Settled/{variant}/selected_safe_capture_rate", float(stats["selected_settled_safe_capture_rate"]), 0)
            writer.add_scalar(f"Settled/{variant}/best_safe_capture_rate", float(stats["best_settled_safe_capture_rate"]), 0)
            writer.add_scalar(f"Settled/{variant}/safety_rate", float(stats["selected_settled_safety_rate"]), 0)
            writer.add_scalar(f"CBF/{variant}/selected_unverified_steps", float(stats["selected_cbf_unverified_steps"]), 0)
            writer.add_scalar(f"Calibration/{variant}/brier_proxy", float(stats["brier_proxy_mean"] or 0.0), 0)
            writer.add_scalar(f"Calibration/{variant}/ece_proxy", float(stats["ece_proxy"] or 0.0), 0)
        for label, stats in report["by_pair_label"].items():
            writer.add_scalar(f"Pairs/{label}/selected_not_best_rate", float(stats.get("selected_not_best_rate", 0.0)), 0)
            writer.add_scalar(f"Pairs/{label}/selected_safe_capture_rate", float(stats.get("selected_settled_safe_capture_rate", 0.0)), 0)
            writer.add_scalar(f"Pairs/{label}/best_safe_capture_rate", float(stats.get("best_settled_safe_capture_rate", 0.0)), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/settled_counterfactual/text_summary",
        "Provenance/inputs/text_summary",
        "Gates/status/text_summary",
    }
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    missing = sorted(required.difference(tags.get("tensors", [])))
    if missing or not events:
        raise ValueError(f"Settled counterfactual TensorBoard validation failed: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", []))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument(
        "--eligibility-floor",
        type=float,
        default=None,
        help="Offline-only predicted clearance floor override; does not change the source replay or CBF margin.",
    )
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("Settled counterfactual audit requires --development-only.")
    protocol = load_protocol(args.protocol.resolve())
    protocol_payload = yaml.safe_load(args.protocol.resolve().read_text(encoding="utf-8"))
    if not isinstance(protocol_payload, dict) or protocol_payload.get("phase") != "development_only" or protocol_payload.get("locked_test_opened") is not False:
        raise ValueError("Settled counterfactual audit requires a closed development protocol.")
    if not args.baseline_run.resolve().is_dir() or not args.environment_config.resolve().is_file():
        raise FileNotFoundError("Baseline run and environment config are required.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    if len(args.run) < 1:
        raise ValueError("At least one JEPA run is required.")
    run_metadata: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    manifest_hash: str | None = None
    protocol_hash = sha256(args.protocol.resolve())
    for run_path in args.run:
        metadata, rows = _settle_run(
            run_path.resolve(),
            protocol_path=args.protocol,
            environment_config_path=args.environment_config,
            baseline_dir=args.baseline_run.resolve(),
            eligibility_floor_override_m=args.eligibility_floor,
        )
        if manifest_hash is None:
            manifest_hash = metadata["scene_manifest_sha256"]
        elif metadata["scene_manifest_sha256"] != manifest_hash:
            raise ValueError("JEPA runs do not share one scene manifest.")
        if metadata["protocol_sha256"] and metadata["protocol_sha256"] != protocol_hash:
            raise ValueError(f"Run protocol hash differs from supplied protocol: {run_path}")
        run_metadata.append(metadata)
        all_rows.extend(rows)
    by_variant_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_variant_rows[str(row["variant"])].append(row)
    by_pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_pair_rows[str(row["pair_label"])].append(row)
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "scene_manifest_shared": len({metadata["scene_manifest_sha256"] for metadata in run_metadata}) == 1,
        "protocol_hash_matches": all(not metadata["protocol_sha256"] or metadata["protocol_sha256"] == protocol_hash for metadata in run_metadata),
        "finite_scores": all(np.isfinite(np.asarray(row["scores"], dtype=np.float64)[np.asarray(row["eligible_mask"], dtype=bool)]).all() for row in all_rows),
        "selected_indices_valid": all(0 <= int(row["selected_index"]) < 5 for row in all_rows),
        # Unexecuted branches are allowed to be rejected/unverified; that is
        # precisely the risk signal this audit exposes. Only the source run's
        # actually executed path is subject to the raw-action hard gate.
        "source_run_raw_unverified_zero": all(
            int(metadata["source_raw_unverified_executed_steps"]) == 0 for metadata in run_metadata
        ),
        "counterfactual_unverified_observable": all(
            isinstance(row.get("selected_cbf_unverified_steps"), int) for row in all_rows
        ),
    }
    report: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_v5_settled_counterfactual",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "development_only": True,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "policy": {
            "settlement": "offline simulator ground truth, constant 3-step candidate chunk, every step through Joint CBF-QP",
            "counterfactual_scope": "local chunk, not full-episode policy outcome",
            "score_softmax_brier_ece": "proxy derived from -score among eligible candidates; not a calibrated capture probability",
            "protocol": str(args.protocol.resolve()),
            "environment_config": str(args.environment_config.resolve()),
            "eligibility_floor_override_m": args.eligibility_floor,
        },
        "inputs": {
            "baseline_run": str(args.baseline_run.resolve()),
            "runs": [str(path.resolve()) for path in args.run],
            "protocol_sha256": protocol_hash,
            "environment_config_sha256": sha256(args.environment_config.resolve()),
            "scene_manifest_sha256": manifest_hash,
            "training_seeds": sorted({int(metadata["training_seed"]) for metadata in run_metadata}),
        },
        "decision_count": len(all_rows),
        "runs": run_metadata,
        "by_variant": {variant: _group_stats(rows) for variant, rows in sorted(by_variant_rows.items())},
        "by_pair_label": {label: _group_stats(rows) for label, rows in sorted(by_pair_rows.items())},
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "source_hashes": {"scripts/audit_jepa_safe_capture_v5_settled_counterfactual.py": sha256(Path(__file__).resolve())},
        },
    }
    report["tensorboard"] = _write_tensorboard(args.tensorboard_logdir, report)
    (args.output_dir.resolve() / "settled_counterfactual.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir.resolve() / "decision_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(_jsonable(row), allow_nan=False, sort_keys=True) + "\n")
    lines = [
        "# Settled Counterfactual Ranking Audit",
        "",
        "> Offline local-chunk settlement; target ground truth is never used by the online evaluator.",
        "",
        f"All gates pass: `{report['all_gates_pass']}`.",
        "",
        "| Variant | Decisions | Selected-not-best | Selected settled safe | Best settled safe | Selected safety | Spearman | Kendall | Brier proxy | ECE proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, stats in sorted(report["by_variant"].items()):
        lines.append(
            f"| {variant} | {stats['decisions']} | {stats['selected_not_best_rate']:.3f} | "
            f"{stats['selected_settled_safe_capture_rate']:.3f} | {stats['best_settled_safe_capture_rate']:.3f} | "
            f"{stats['selected_settled_safety_rate']:.3f} | {stats['spearman_mean'] if stats['spearman_mean'] is not None else float('nan'):.3f} | "
            f"{stats['kendall_mean'] if stats['kendall_mean'] is not None else float('nan'):.3f} | "
            f"{stats['brier_proxy_mean'] if stats['brier_proxy_mean'] is not None else float('nan'):.3f} | "
            f"{stats['ece_proxy'] if stats['ece_proxy'] is not None else float('nan'):.3f} |"
        )
    lines.extend(
        [
            "",
            "Pair labels are inherited from the baseline-vs-candidate episode outcome; they are diagnostic strata, not independent samples.",
            "",
            "`locked_test_opened=false`; a positive local settled ranking signal does not establish full-episode safe-capture improvement.",
        ]
    )
    (args.output_dir.resolve() / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "decision_count": report["decision_count"], "by_variant": report["by_variant"], "by_pair_label": report["by_pair_label"], "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
