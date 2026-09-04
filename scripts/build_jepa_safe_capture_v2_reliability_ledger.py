"""Build a checkpoint-bound reliability ledger from calibration rollouts.

Only the independent P1 calibration split is read.  The target and safety
labels are settled offline; no development or locked scene is loaded.  The
resulting ledger is immutable at runtime and only gates JEPA ranking features.
CBF remains the final safety filter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
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

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    build_action_conditioned_predictor,
)
from encirclement3d.reliability import (  # noqa: E402
    SafeCaptureReliabilityLedger,
    _safe_capture_clearance_bucket,
    _safe_capture_observation_age_bucket,
    _safe_capture_risk_bucket,
    _safe_capture_separation_bucket,
    _safe_capture_ttc_bucket,
    _safe_capture_uncertainty_bucket,
    _safe_capture_visibility_bucket,
    make_safe_capture_coarse_context_key,
    make_safe_capture_context_key,
    make_safe_capture_global_key,
)


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_safe_capture_v2"
REQUIRED_ARRAYS = {
    "inputs",
    "action_history",
    "labels_relative",
    "labels_target_velocity",
    "labels_target_acceleration",
    "labels_obstacle_clearance",
    "labels_inter_agent_clearance",
    "labels_pairwise_ttc",
    "labels_target_visible",
    "labels_observation_age",
    "labels_cbf_correction",
    "labels_cbf_intervention",
    "labels_cbf_qp_feasible",
    "labels_collision",
    "labels_boundary",
    "agent_id",
    "time_index",
    "episode_seed",
    "scenario_index",
    "candidate_index",
    "candidate_action_norm_mps",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--minimum-sample-count", type=int, default=128)
    parser.add_argument("--minimum-credit", type=float, default=0.65)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def _load_metadata(path: Path, dataset: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Calibration metadata must be a JSON object.")
    if metadata.get("dataset_version") != "jepa_safe_capture_v2_p1" or metadata.get("split") != "calibration":
        raise ValueError("P3 requires the independent jepa_safe_capture_v2 calibration split.")
    boundary = metadata.get("information_boundary", {})
    if boundary.get("target_truth_used_only_for_offline_labels") is not True:
        raise ValueError("Calibration metadata does not prove target truth is offline-only.")
    if boundary.get("development_or_locked_data_used_for_training") is not False:
        raise ValueError("Calibration metadata does not prove development/locked separation.")
    if boundary.get("locked_test_opened") is not False:
        raise ValueError("Calibration metadata opened a locked test.")
    if sha256(dataset) != str(metadata.get("dataset_sha256", sha256(dataset))):
        raise ValueError("Calibration dataset SHA-256 does not match metadata.")
    return metadata


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    missing = sorted(REQUIRED_ARRAYS.difference(arrays))
    if missing:
        raise ValueError(f"Calibration dataset is missing arrays: {missing}")
    if not arrays["inputs"].ndim == 3 or arrays["inputs"].shape[1:] != (8, 63):
        raise ValueError(f"Unexpected calibration input shape: {arrays['inputs'].shape}")
    sample_count = arrays["inputs"].shape[0]
    for name, value in arrays.items():
        if value.shape[0] != sample_count:
            raise ValueError(f"Calibration array {name} has inconsistent sample count.")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Calibration array {name} contains non-finite values.")
    return arrays


def _predict(
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError("P3 requires a safe-capture v2 checkpoint.")
    model = build_action_conditioned_predictor(str(checkpoint["model_type"]), checkpoint["model"])
    if not isinstance(model, InteractionAwareActionConditionedSafeCaptureJEPAPredictor):
        raise RuntimeError("Checkpoint factory returned an unexpected safe-capture predictor.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    inputs = arrays["inputs"].astype(np.float32, copy=False)
    actions = arrays["action_history"].astype(np.float32, copy=False)
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            mean, log_variance, _latent, auxiliary = model.forward_multitask(
                torch.as_tensor(inputs[start:stop], device=device),
                torch.as_tensor(actions[start:stop], device=device),
            )
            parts["target_relative"].append(mean.cpu().numpy())
            parts["target_std"].append(torch.exp(0.5 * log_variance).cpu().numpy())
            parts["target_velocity"].append(auxiliary["target_velocity"].cpu().numpy())
            parts["target_acceleration"].append(auxiliary["target_acceleration"].cpu().numpy())
            parts["obstacle_clearance"].append(auxiliary["obstacle_clearance_lower_quantile"].cpu().numpy())
            parts["inter_agent_clearance"].append(auxiliary["inter_agent_clearance_lower_quantile"].cpu().numpy())
            parts["pairwise_ttc"].append(auxiliary["pairwise_ttc"].cpu().numpy())
            parts["visibility_probability"].append(torch.sigmoid(auxiliary["target_visibility_logit"]).cpu().numpy())
            parts["observation_age"].append(auxiliary["observation_age"].cpu().numpy())
            parts["cbf_correction"].append(auxiliary["cbf_correction"].cpu().numpy())
            parts["intervention_probability"].append(torch.sigmoid(auxiliary["cbf_intervention_logit"]).cpu().numpy())
            parts["qp_feasibility_probability"].append(torch.sigmoid(auxiliary["cbf_qp_feasibility_logit"]).cpu().numpy())
    predictions = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("P3 calibration prediction emitted non-finite values.")
    return predictions, checkpoint


def _group_ranking(
    arrays: dict[str, np.ndarray],
    predicted_relative: np.ndarray,
    horizon_index: int,
    horizon_seconds: float,
    extent: float,
    action_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack([arrays["episode_seed"], arrays["time_index"], arrays["agent_id"]], axis=1)
    _groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    if counts.size == 0 or not np.all(counts == 5):
        raise ValueError("Each calibration state-agent group must contain exactly five candidates.")
    candidate_action = arrays["action_history"][:, -1] * action_scale
    predicted_cost = np.linalg.norm(predicted_relative * extent - candidate_action * horizon_seconds, axis=1)
    settled_cost = np.linalg.norm(
        arrays["labels_relative"][:, horizon_index] * extent
        - candidate_action * horizon_seconds,
        axis=1,
    )
    ranking_credit = np.empty(predicted_cost.shape[0], dtype=np.float32)
    ranking_win = np.empty(predicted_cost.shape[0], dtype=np.float32)
    separation = np.empty(predicted_cost.shape[0], dtype=np.float32)
    for group_index in range(len(_groups)):
        indices = np.flatnonzero(inverse == group_index)
        predicted_order = indices[np.argsort(predicted_cost[indices], kind="stable")]
        settled = settled_cost[indices]
        selected = int(predicted_order[0])
        best = float(np.min(settled))
        worst = float(np.max(settled))
        regret = float(settled_cost[selected] - best)
        ranking_credit[selected] = 1.0 if worst - best <= 1e-9 else float(np.clip(1.0 - regret / (worst - best), 0.0, 1.0))
        ranking_credit[indices[indices != selected]] = ranking_credit[selected]
        ranking_win[indices] = float(regret <= 1e-8)
        separation[indices] = float(max(predicted_cost[predicted_order[1]] - predicted_cost[predicted_order[0]], 0.0))
    return ranking_credit, ranking_win, separation


def _scenario_lookup(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = metadata.get("scenario_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Calibration metadata is missing scenario_records.")
    lookup: dict[int, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Invalid scenario record in calibration metadata.")
        index = int(item["scenario_index"])
        lookup[index] = {
            "layout_signature": f"scenario_{index}",
            "obstacle_count": int(item["obstacle_count"]),
            "target_motion_mode": str(item["target_motion_mode"]),
            "target_speed_scale": float(item["target_speed_scale"]),
            "name": str(item["name"]),
        }
    return lookup


def _entry_template(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "context": context,
        "sample_count": 0,
        "credit_sum": 0.0,
        "target_error_sum_m": 0.0,
        "clearance_error_sum_m": 0.0,
        "clearance_overprediction_sum_m": 0.0,
        "visibility_brier_sum": 0.0,
        "intervention_brier_sum": 0.0,
        "qp_feasibility_brier_sum": 0.0,
        "ranking_win_sum": 0.0,
        "collision_sum": 0.0,
        "boundary_sum": 0.0,
        "cbf_intervention_sum": 0.0,
        "uncertainty_sum": 0.0,
    }


def _finalize_entries(entries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for key, value in entries.items():
        count = int(value["sample_count"])
        if count <= 0:
            continue
        finalized[key] = {
            "context": value["context"],
            "sample_count": count,
            "credit": float(value["credit_sum"] / count),
            "target_mae_m": float(value["target_error_sum_m"] / count),
            "clearance_mae_m": float(value["clearance_error_sum_m"] / count),
            "clearance_overprediction_m": float(value["clearance_overprediction_sum_m"] / count),
            "visibility_brier": float(value["visibility_brier_sum"] / count),
            "intervention_brier": float(value["intervention_brier_sum"] / count),
            "qp_feasibility_brier": float(value["qp_feasibility_brier_sum"] / count),
            "candidate_ranking_win_rate": float(value["ranking_win_sum"] / count),
            "collision_rate": float(value["collision_sum"] / count),
            "boundary_rate": float(value["boundary_sum"] / count),
            "cbf_intervention_rate": float(value["cbf_intervention_sum"] / count),
            "mean_uncertainty": float(value["uncertainty_sum"] / count),
        }
    return finalized


def build_payload(
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    metadata: dict[str, Any],
    checkpoint_path: Path,
    dataset_path: Path,
    metadata_path: Path,
    minimum_sample_count: int,
    minimum_credit: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    collection = yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))
    extent = float(collection["world"]["half_extent_xy"])
    maximum_observation_age = float(collection["task"]["pursuit"]["maximum_message_age_steps"])
    action_scale = float(metadata["action_scale"])
    horizon_seconds = [float(value) for value in metadata["horizon_seconds"]]
    scenario_lookup = _scenario_lookup(metadata)
    scenario_indices = arrays["scenario_index"].astype(np.int64)
    if not set(np.unique(scenario_indices)).issubset(scenario_lookup):
        raise ValueError("Calibration samples contain an unknown scenario index.")
    entries: dict[str, dict[str, Any]] = {}
    horizon_diagnostics: list[dict[str, Any]] = []
    for horizon_index, seconds in enumerate(horizon_seconds):
        target_error = np.linalg.norm(
            predictions["target_relative"][:, horizon_index] - arrays["labels_relative"][:, horizon_index], axis=1
        ) * extent
        predicted_clearance = np.minimum(
            predictions["obstacle_clearance"][:, horizon_index], predictions["inter_agent_clearance"][:, horizon_index]
        ) * extent
        settled_clearance = np.minimum(
            arrays["labels_obstacle_clearance"][:, horizon_index], arrays["labels_inter_agent_clearance"][:, horizon_index]
        ) * extent
        clearance_error = np.abs(predicted_clearance - settled_clearance)
        clearance_overprediction = np.maximum(predicted_clearance - settled_clearance, 0.0)
        visibility_probability = predictions["visibility_probability"][:, horizon_index]
        intervention_probability = predictions["intervention_probability"][:, horizon_index]
        qp_probability = predictions["qp_feasibility_probability"][:, horizon_index]
        visibility_label = arrays["labels_target_visible"][:, horizon_index]
        intervention_label = arrays["labels_cbf_intervention"][:, horizon_index]
        qp_label = arrays["labels_cbf_qp_feasible"][:, horizon_index]
        uncertainty = np.mean(predictions["target_std"][:, horizon_index], axis=1)
        ranking_credit, ranking_win, candidate_separation = _group_ranking(
            arrays,
            predictions["target_relative"][:, horizon_index],
            horizon_index,
            seconds,
            extent,
            action_scale,
        )
        target_credit = np.exp(-target_error / 0.50)
        clearance_credit = np.exp(-clearance_overprediction / 0.25)
        visibility_credit = 1.0 - np.abs(visibility_probability - visibility_label)
        intervention_credit = 1.0 - np.abs(intervention_probability - intervention_label)
        credit = (
            0.30 * ranking_credit
            + 0.25 * target_credit
            + 0.25 * clearance_credit
            + 0.10 * visibility_credit
            + 0.10 * intervention_credit
        )
        observed_age_steps = np.clip(
            arrays["inputs"][:, -1, 10] * maximum_observation_age,
            0.0,
            maximum_observation_age,
        )
        visible_fraction = np.clip(arrays["inputs"][:, -1, 9], 0.0, 1.0)
        predicted_ttc = predictions["pairwise_ttc"][:, horizon_index]
        cbf_risk = intervention_probability
        for index in range(credit.shape[0]):
            scenario = scenario_lookup[int(scenario_indices[index])]
            context = {
                "visibility_condition": float(visible_fraction[index]),
                "observation_age_steps": float(observed_age_steps[index]),
                "obstacle_count": scenario["obstacle_count"],
                "layout_signature": scenario["layout_signature"],
                "target_motion_mode": scenario["target_motion_mode"],
                "minimum_clearance_m": float(predicted_clearance[index]),
                "pairwise_ttc_s": float(predicted_ttc[index]),
                "uncertainty": float(uncertainty[index]),
                "cbf_risk": float(cbf_risk[index]),
                "candidate_separation_m": float(candidate_separation[index]),
            }
            visibility_bucket = _safe_capture_visibility_bucket(context["visibility_condition"])
            age_bucket = _safe_capture_observation_age_bucket(context["observation_age_steps"] / maximum_observation_age)
            clearance_bucket = _safe_capture_clearance_bucket(context["minimum_clearance_m"])
            ttc_bucket = _safe_capture_ttc_bucket(context["pairwise_ttc_s"])
            uncertainty_bucket = _safe_capture_uncertainty_bucket(context["uncertainty"])
            risk_bucket = _safe_capture_risk_bucket(context["cbf_risk"])
            separation_bucket = _safe_capture_separation_bucket(context["candidate_separation_m"])
            full_key = make_safe_capture_context_key(
                horizon_index,
                visibility_bucket,
                age_bucket,
                scenario["obstacle_count"],
                scenario["layout_signature"],
                scenario["target_motion_mode"],
                clearance_bucket,
                ttc_bucket,
                uncertainty_bucket,
                risk_bucket,
                separation_bucket,
            )
            coarse_key = make_safe_capture_coarse_context_key(
                horizon_index,
                visibility_bucket,
                age_bucket,
                scenario["obstacle_count"],
                scenario["target_motion_mode"],
                clearance_bucket,
                uncertainty_bucket,
                risk_bucket,
            )
            global_key = make_safe_capture_global_key(horizon_index)
            stored_context = {
                "horizon_index": horizon_index,
                "visibility_condition": visibility_bucket,
                "observation_age_bucket": age_bucket,
                "obstacle_count": scenario["obstacle_count"],
                "layout_signature": scenario["layout_signature"],
                "target_motion_mode": scenario["target_motion_mode"],
                "minimum_clearance_bucket": clearance_bucket,
                "ttc_bucket": ttc_bucket,
                "uncertainty_bucket": uncertainty_bucket,
                "cbf_risk_bucket": risk_bucket,
                "candidate_separation_bucket": separation_bucket,
            }
            for key in (full_key, coarse_key, global_key):
                entry = entries.setdefault(key, _entry_template(stored_context))
                entry["sample_count"] += 1
                entry["credit_sum"] += float(credit[index])
                entry["target_error_sum_m"] += float(target_error[index])
                entry["clearance_error_sum_m"] += float(clearance_error[index])
                entry["clearance_overprediction_sum_m"] += float(clearance_overprediction[index])
                entry["visibility_brier_sum"] += float((visibility_probability[index] - visibility_label[index]) ** 2)
                entry["intervention_brier_sum"] += float((intervention_probability[index] - intervention_label[index]) ** 2)
                entry["qp_feasibility_brier_sum"] += float((qp_probability[index] - qp_label[index]) ** 2)
                entry["ranking_win_sum"] += float(ranking_win[index])
                entry["collision_sum"] += float(arrays["labels_collision"][index, horizon_index])
                entry["boundary_sum"] += float(arrays["labels_boundary"][index, horizon_index])
                entry["cbf_intervention_sum"] += float(intervention_label[index])
                entry["uncertainty_sum"] += float(uncertainty[index])
        global_entry = entries[global_key]
        horizon_diagnostics.append(
            {
                "horizon_index": horizon_index,
                "horizon_seconds": seconds,
                "global_sample_count": int(global_entry["sample_count"]),
                "global_credit": float(global_entry["credit_sum"] / global_entry["sample_count"]),
                "target_mae_m": float(global_entry["target_error_sum_m"] / global_entry["sample_count"]),
                "clearance_mae_m": float(global_entry["clearance_error_sum_m"] / global_entry["sample_count"]),
                "collision_rate": float(global_entry["collision_sum"] / global_entry["sample_count"]),
                "boundary_rate": float(global_entry["boundary_sum"] / global_entry["sample_count"]),
                "qp_label_unique_values": int(np.unique(qp_label).size),
                "full_context_key_count": int(sum(1 for key, value in entries.items() if key.startswith(f"h{horizon_index}|vis="))),
            }
        )
    payload = {
        "ledger_type": SafeCaptureReliabilityLedger.LEDGER_TYPE,
        "ledger_version": 2,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "immutable_after_calibration": True,
        "update_rule": "offline_calibration_settled_outcomes_only_no_online_update",
        "source": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "calibration_dataset": str(dataset_path.resolve()),
            "calibration_dataset_sha256": sha256(dataset_path),
            "calibration_metadata": str(metadata_path.resolve()),
            "calibration_metadata_sha256": sha256(metadata_path),
            "protocol": str(Path(metadata["protocol"]).resolve()),
            "protocol_sha256": sha256(Path(metadata["protocol"]).resolve()),
            "samples": int(arrays["inputs"].shape[0]),
            "horizon_seconds": horizon_seconds,
            "action_scale": action_scale,
            "model_type": MODEL_TYPE,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "tensorboard": version("tensorboard"),
            },
        },
        "bucket_definition": {
            "visibility": "visible if >= 0.5 else occluded",
            "observation_age": "fresh <= 0.10, delayed <= 0.35, stale > 0.35 after normalization",
            "clearance_m": "critical < 0.35, near < 0.75, clear otherwise",
            "ttc_s": "imminent < 0.50, near < 1.50, distant otherwise",
            "uncertainty": "low <= 0.10, medium <= 0.25, high otherwise",
            "cbf_risk": "low < 0.25, medium < 0.60, high otherwise",
            "candidate_separation_m": "low < 0.05, medium < 0.25, high otherwise",
        },
        "decision_policy": {
            "states": ["trusted", "fallback_nominal", "safe_hold"],
            "minimum_sample_count": int(minimum_sample_count),
            "minimum_credit": float(minimum_credit),
            "maximum_observation_age_steps": 45.0,
            "safe_hold_uncertainty_threshold": 0.40,
            "safe_hold_ttc_seconds": 0.30,
            "low_credit_action": "frozen_v5_nominal_then_cbf",
            "ood_action": "safe_hold_then_nominal_cbf",
            "ledger_is_not_a_safety_certificate": True,
        },
        "credit_definition": {
            "ranking_weight": 0.30,
            "target_error_weight": 0.25,
            "conservative_clearance_weight": 0.25,
            "visibility_weight": 0.10,
            "cbf_intervention_weight": 0.10,
            "target_error_decay_m": 0.50,
            "clearance_overprediction_decay_m": 0.25,
        },
        "context_dimensions": [
            "visibility_condition",
            "observation_age_bucket",
            "obstacle_count",
            "layout_signature",
            "target_motion_mode",
            "minimum_clearance_bucket",
            "ttc_bucket",
            "uncertainty_bucket",
            "cbf_risk_bucket",
            "candidate_separation_bucket",
        ],
        "entries": _finalize_entries(entries),
    }
    diagnostics = {
        "horizon_diagnostics": horizon_diagnostics,
        "entry_count": len(payload["entries"]),
        "global_entry_count": len(horizon_diagnostics),
        "full_context_entry_count": sum(1 for key in payload["entries"] if "|layout=" in key),
        "coarse_context_entry_count": sum(1 for key in payload["entries"] if "|vis=" in key and "|layout=" not in key),
        "all_predictions_finite": True,
        "qp_label_variation_by_horizon": [item["qp_label_unique_values"] > 1 for item in horizon_diagnostics],
    }
    return payload, diagnostics


def _context_for_row(
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    metadata: dict[str, Any],
    scenario_lookup: dict[int, dict[str, Any]],
    horizon_index: int,
    extent: float,
    maximum_observation_age: float,
    index: int,
    candidate_separation: np.ndarray,
) -> dict[str, Any]:
    scenario = scenario_lookup[int(arrays["scenario_index"][index])]
    return {
        "visibility_condition": float(np.clip(arrays["inputs"][index, -1, 9], 0.0, 1.0)),
        "observation_age_steps": float(np.clip(arrays["inputs"][index, -1, 10] * maximum_observation_age, 0.0, maximum_observation_age)),
        "obstacle_count": scenario["obstacle_count"],
        "layout_signature": scenario["layout_signature"],
        "target_motion_mode": scenario["target_motion_mode"],
        "minimum_clearance_m": float(np.minimum(predictions["obstacle_clearance"][index, horizon_index], predictions["inter_agent_clearance"][index, horizon_index]) * extent),
        "pairwise_ttc_s": float(predictions["pairwise_ttc"][index, horizon_index]),
        "uncertainty": float(np.mean(predictions["target_std"][index, horizon_index])),
        "cbf_risk": float(predictions["intervention_probability"][index, horizon_index]),
        "candidate_separation_m": float(candidate_separation[index]),
    }


def forecast(
    payload: dict[str, Any],
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    metadata: dict[str, Any],
    extent: float,
    maximum_observation_age: float,
) -> dict[str, Any]:
    ledger = SafeCaptureReliabilityLedger(payload)
    scenario_lookup = _scenario_lookup(metadata)
    state_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    fallback_path_counts: Counter[str] = Counter()
    unsafe_by_state: defaultdict[str, list[float]] = defaultdict(list)
    per_horizon: list[dict[str, Any]] = []
    for horizon_index, seconds in enumerate(metadata["horizon_seconds"]):
        _credit, _win, separation = _group_ranking(
            arrays,
            predictions["target_relative"][:, horizon_index],
            horizon_index,
            float(seconds),
            extent,
            float(metadata["action_scale"]),
        )
        local_count = coarse_count = global_count = 0
        for index in range(arrays["inputs"].shape[0]):
            context = _context_for_row(
                arrays,
                predictions,
                metadata,
                scenario_lookup,
                horizon_index,
                extent,
                maximum_observation_age,
                index,
                separation,
            )
            decision = ledger.decision(horizon_index, context)
            state_counts[decision.state] += 1
            if decision.fallback_reason:
                reason_counts[decision.fallback_reason] += 1
            if decision.used_global_fallback:
                global_count += 1
            elif decision.used_coarse_fallback:
                coarse_count += 1
            else:
                local_count += 1
            unsafe = float(max(arrays["labels_collision"][index, horizon_index], arrays["labels_boundary"][index, horizon_index]))
            unsafe_by_state[decision.state].append(unsafe)
        per_horizon.append(
            {
                "horizon_index": horizon_index,
                "horizon_seconds": float(seconds),
                "local_fraction": local_count / arrays["inputs"].shape[0],
                "coarse_fraction": coarse_count / arrays["inputs"].shape[0],
                "global_fraction": global_count / arrays["inputs"].shape[0],
            }
        )
    for reason in ("low_credit", "missing_bucket", "ood", "stale_observation", "uncertainty_high", "joint_ttc_cbf_risk"):
        fallback_path_counts[reason] = reason_counts[reason]
    failure_rates = {
        state: (float(np.mean(values)) if values else None)
        for state, values in unsafe_by_state.items()
    }
    trusted_rate = failure_rates.get("trusted")
    fallback_rates = [failure_rates[state] for state in ("fallback_nominal", "safe_hold") if failure_rates.get(state) is not None]
    high_credit_not_worse = trusted_rate is None or not fallback_rates or trusted_rate <= max(fallback_rates) + 1e-12
    return {
        "per_horizon": per_horizon,
        "state_counts": dict(state_counts),
        "fallback_reason_counts": dict(fallback_path_counts),
        "unsafe_rate_by_state": failure_rates,
        "high_credit_failure_rate_not_above_low_credit": bool(high_credit_not_worse),
        "ood_or_hard_contexts_trigger_safe_hold": bool(
            reason_counts["ood"] + reason_counts["stale_observation"] + reason_counts["uncertainty_high"] + reason_counts["joint_ttc_cbf_risk"] == state_counts["safe_hold"]
        ),
    }


def write_tensorboard(
    payload: dict[str, Any],
    diagnostics: dict[str, Any],
    forecast_diagnostics: dict[str, Any],
    logdir: Path,
) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/ledger", json.dumps(payload["decision_policy"], indent=2), 0)
        writer.add_text("Provenance/source", json.dumps(payload["source"], indent=2), 0)
        writer.add_text("Provenance/buckets", json.dumps(payload["bucket_definition"], indent=2), 0)
        writer.add_text("Provenance/forecast", json.dumps(forecast_diagnostics, indent=2), 0)
        for item in diagnostics["horizon_diagnostics"]:
            step = int(item["horizon_index"]) + 1
            writer.add_scalar("Calibration/global_credit", item["global_credit"], step)
            writer.add_scalar("Calibration/target_mae_m", item["target_mae_m"], step)
            writer.add_scalar("Calibration/clearance_mae_m", item["clearance_mae_m"], step)
            writer.add_scalar("Calibration/collision_rate", item["collision_rate"], step)
            writer.add_scalar("Calibration/boundary_rate", item["boundary_rate"], step)
            writer.add_scalar("Calibration/qp_label_unique_values", item["qp_label_unique_values"], step)
            row = forecast_diagnostics["per_horizon"][item["horizon_index"]]
            writer.add_scalar("Reliability/local_fraction", row["local_fraction"], step)
            writer.add_scalar("Reliability/coarse_fraction", row["coarse_fraction"], step)
            writer.add_scalar("Reliability/global_fraction", row["global_fraction"], step)
        for state, count in forecast_diagnostics["state_counts"].items():
            writer.add_scalar(f"Fallback/state_count/{state}", count, 0)
        for reason, count in forecast_diagnostics["fallback_reason_counts"].items():
            writer.add_scalar(f"Fallback/reason_count/{reason}", count, 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0, "histograms": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/ledger/text_summary",
        "Provenance/source/text_summary",
        "Provenance/buckets/text_summary",
        "Provenance/forecast/text_summary",
    }
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    if missing:
        raise ValueError(f"P3 TensorBoard is missing text provenance: {missing}")
    return {
        "logdir": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "scalar_tag_count": len(tags.get("scalars", [])),
        "histogram_tag_count": len(tags.get("histograms", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_text_complete": not missing,
    }


def render_report(payload: dict[str, Any], diagnostics: dict[str, Any], forecast_diagnostics: dict[str, Any]) -> str:
    lines = [
        "# JEPA Safe-Capture v2 P3 Reliability Ledger",
        "",
        "> Calibration-only, checkpoint-bound, immutable runtime artifact. The ledger gates ranking features; CBF remains the safety proof boundary.",
        "",
        f"Checkpoint SHA-256: `{payload['source']['checkpoint_sha256']}`",
        f"Calibration dataset SHA-256: `{payload['source']['calibration_dataset_sha256']}`",
        f"Minimum credit/sample count: `{payload['decision_policy']['minimum_credit']:.2f}` / `{payload['decision_policy']['minimum_sample_count']}`",
        "",
        "## Global Calibration Summary",
        "",
        "| Horizon (s) | Samples | Credit | Target MAE (m) | Clearance MAE (m) | Collision rate | Boundary rate | Local/coarse/global forecast |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item, forecast in zip(diagnostics["horizon_diagnostics"], forecast_diagnostics["per_horizon"]):
        lines.append(
            f"| {item['horizon_seconds']:.1f} | {item['global_sample_count']} | {item['global_credit']:.4f} | "
            f"{item['target_mae_m']:.4f} | {item['clearance_mae_m']:.4f} | {item['collision_rate']:.4%} | "
            f"{item['boundary_rate']:.4%} | {forecast['local_fraction']:.2%}/{forecast['coarse_fraction']:.2%}/{forecast['global_fraction']:.2%} |"
        )
    lines += [
        "",
        "## Runtime State Forecast",
        "",
        f"State counts: `{json.dumps(forecast_diagnostics['state_counts'], sort_keys=True)}`",
        f"Fallback reasons: `{json.dumps(forecast_diagnostics['fallback_reason_counts'], sort_keys=True)}`  ",
        f"Unsafe rate by state: `{json.dumps(forecast_diagnostics['unsafe_rate_by_state'], sort_keys=True)}`",
        "",
        "## Gates and Limits",
        "",
        f"- High-credit failure-rate gate: **{'PASS' if forecast_diagnostics['high_credit_failure_rate_not_above_low_credit'] else 'FAIL'}**.",
        f"- OOD/stale/hard-context safe-hold routing: **{'PASS' if forecast_diagnostics['ood_or_hard_contexts_trigger_safe_hold'] else 'FAIL'}**.",
        f"- Full context entries: `{diagnostics['full_context_entry_count']}`; total entries including coarse/global: `{diagnostics['entry_count']}`.",
        "- Current P1 QP-feasibility labels have no class variation. The QP head is retained for future data, but this ledger does not claim QP-feasibility calibration.",
        "- Low credit requests frozen V5 nominal action followed by CBF. Unknown/OOD contexts request safe-hold followed by the declared CBF fallback ladder.",
        "- This artifact is not a closed-loop safe-capture result and must not be tuned with S3 development outcomes.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    for value in (args.checkpoint, args.dataset, args.metadata):
        if not value.resolve().is_file():
            raise FileNotFoundError(value)
    if args.batch_size <= 0 or args.minimum_sample_count <= 0 or not 0.0 <= args.minimum_credit <= 1.0:
        raise ValueError("Invalid P3 batch size or ledger policy.")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite an existing ledger or report.")
    metadata = _load_metadata(args.metadata.resolve(), args.dataset.resolve())
    arrays = _load_arrays(args.dataset.resolve())
    device = choose_device(args.device)
    predictions, _checkpoint = _predict(args.checkpoint.resolve(), arrays, args.batch_size, device)
    payload, diagnostics = build_payload(
        arrays,
        predictions,
        metadata,
        args.checkpoint.resolve(),
        args.dataset.resolve(),
        args.metadata.resolve(),
        args.minimum_sample_count,
        args.minimum_credit,
    )
    collection = yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))
    extent = float(collection["world"]["half_extent_xy"])
    maximum_observation_age = float(collection["task"]["pursuit"]["maximum_message_age_steps"])
    forecast_diagnostics = forecast(payload, arrays, predictions, metadata, extent, maximum_observation_age)
    tensorboard = write_tensorboard(payload, diagnostics, forecast_diagnostics, args.tensorboard_logdir)
    payload["diagnostics"] = diagnostics
    payload["forecast"] = forecast_diagnostics
    payload["tensorboard"] = tensorboard
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.resolve().write_text(render_report(payload, diagnostics, forecast_diagnostics), encoding="utf-8")
    print(json.dumps({"checkpoint": payload["source"]["checkpoint_sha256"], "forecast": forecast_diagnostics, "tensorboard": tensorboard}, indent=2))


if __name__ == "__main__":
    main()
