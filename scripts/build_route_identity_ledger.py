"""Calibrate and hash-bind a route-identity JEPA reliability ledger.

The route archive is per-defender and contains one row per route candidate.
This builder uses only the independent calibration split, records route-head
diagnostics, and emits the existing read-only ledger format with global
horizon entries.  The ledger gates learned ranking features; Joint CBF remains
the only execution safety certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedRouteJEPAPredictor,
    build_action_conditioned_predictor,
)
from encirclement3d.reliability import SafeCaptureReliabilityLedger, make_safe_capture_global_key  # noqa: E402


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_route_identity_v1"
REQUIRED_ARRAYS = (
    "inputs",
    "action_history",
    "route_action_chunk",
    "labels_relative",
    "labels_obstacle_clearance",
    "labels_boundary_clearance",
    "labels_inter_agent_clearance",
    "labels_target_visible",
    "labels_cbf_correction",
    "labels_cbf_intervention",
    "labels_cbf_feasible",
    "labels_cbf_min_slack",
    "labels_route_progress",
    "route_candidate_index",
    "route_side_index",
    "route_geometry_valid",
    "branch_terminated",
    "sample_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--minimum-sample-count", type=int, default=128)
    parser.add_argument("--minimum-credit", type=float, default=0.65)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def _load_metadata(path: Path, dataset: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("split") != "calibration":
        raise ValueError("Route ledger calibration requires split=calibration.")
    if metadata.get("dataset_version") != "jepa_safe_capture_route_identity_v1":
        raise ValueError("Unexpected route calibration dataset version.")
    if metadata.get("candidate_profile") != "obstacle_route_v1" or int(metadata.get("candidate_count", 0)) != 12:
        raise ValueError("Route ledger requires the 12-candidate obstacle_route_v1 archive.")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError("Route ledger calibration must remain development-only.")
    boundary = metadata.get("information_boundary", {})
    if boundary.get("target_truth_used_only_for_offline_labels") is not True or boundary.get("validation_or_development_used_for_training") is not False:
        raise ValueError("Route calibration metadata does not prove split/information isolation.")
    if metadata.get("dataset_sha256") not in (None, _sha256(dataset)):
        raise ValueError("Route calibration dataset hash does not match metadata.")
    result = dict(metadata)
    result["_metadata_path"] = str(path.resolve())
    return result


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    missing = sorted(set(REQUIRED_ARRAYS).difference(arrays))
    if missing:
        raise ValueError(f"Calibration archive is missing arrays: {missing}")
    count = int(arrays["inputs"].shape[0])
    if arrays["inputs"].shape[1:] != (8, 63) or arrays["action_history"].shape[1:] != (8, 3) or arrays["route_action_chunk"].shape[1:] != (3, 3):
        raise ValueError("Calibration archive does not match the route model input contract.")
    for name, value in arrays.items():
        if value.shape[0] != count:
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
    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Expected {MODEL_TYPE}, got {checkpoint.get('model_type')!r}.")
    model = build_action_conditioned_predictor(MODEL_TYPE, checkpoint["model"])
    if not isinstance(model, InteractionAwareActionConditionedRouteJEPAPredictor):
        raise TypeError("Route checkpoint did not construct the route-aware predictor.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    outputs: dict[str, list[np.ndarray]] = {}
    count = int(arrays["inputs"].shape[0])
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            inputs = torch.as_tensor(arrays["inputs"][start:stop], dtype=torch.float32, device=device)
            actions = torch.as_tensor(arrays["action_history"][start:stop], dtype=torch.float32, device=device)
            chunks = torch.as_tensor(
                arrays["route_action_chunk"][start:stop] / float(5.0),
                dtype=torch.float32,
                device=device,
            )
            mean, log_variance, _latent, auxiliary = model.forward_multitask(
                inputs,
                actions,
                chunks,
                torch.as_tensor(arrays["route_candidate_index"][start:stop], dtype=torch.long, device=device),
                torch.as_tensor(arrays["route_side_index"][start:stop], dtype=torch.long, device=device),
            )
            values = {
                "target_relative": mean.detach().cpu().numpy(),
                "target_std": torch.exp(0.5 * log_variance).detach().cpu().numpy(),
                "obstacle_clearance": auxiliary["obstacle_clearance"].detach().cpu().numpy(),
                "inter_agent_clearance": auxiliary["inter_agent_clearance"].detach().cpu().numpy(),
                "visibility_probability": torch.sigmoid(auxiliary["target_visibility_logit"]).detach().cpu().numpy(),
                "intervention_probability": torch.sigmoid(auxiliary["cbf_intervention_logit"]).detach().cpu().numpy(),
                "qp_feasibility_probability": torch.sigmoid(auxiliary["cbf_qp_feasibility_logit"]).detach().cpu().numpy(),
                "boundary_clearance": auxiliary["boundary_clearance"].detach().cpu().numpy(),
                "cbf_min_slack": auxiliary["cbf_min_slack"].detach().cpu().numpy(),
                "route_progress": auxiliary["route_progress"].detach().cpu().numpy(),
                "cbf_feasibility_probability_route": torch.sigmoid(auxiliary["cbf_feasibility_logit"]).detach().cpu().numpy(),
                "route_identity_logits": auxiliary["route_identity_logits"].detach().cpu().numpy(),
                "route_side_logits": auxiliary["route_side_logits"].detach().cpu().numpy(),
                "route_geometry_probability": torch.sigmoid(auxiliary["route_geometry_logit"]).detach().cpu().numpy(),
                "route_termination_probability": torch.sigmoid(auxiliary["route_termination_logit"]).detach().cpu().numpy(),
            }
            for key, value in values.items():
                outputs.setdefault(key, []).append(value)
    predictions = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("Route checkpoint emitted non-finite calibration predictions.")
    return predictions, checkpoint


def _mean(value: np.ndarray) -> float:
    return float(np.mean(np.asarray(value, dtype=np.float64)))


def _safety_priority_credit(
    *,
    target_error_m: np.ndarray,
    clearance_error_m: np.ndarray,
    clearance_overprediction_m: np.ndarray,
    visibility_error: np.ndarray,
    intervention_error: np.ndarray,
    feasibility_brier: np.ndarray,
    route_geometry_brier: np.ndarray,
    route_termination_brier: np.ndarray,
    route_progress_error: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Score a route evaluator by safety calibration before target accuracy.

    The target displacement remains useful for ranking, but it must not make a
    safe route model untrusted merely because its absolute target forecast is
    imperfect.  Every component is bounded in [0, 1], and the weights are
    explicit so the resulting ledger is auditable and reproducible.
    """

    values = {
        "target_error_m": np.asarray(target_error_m, dtype=np.float64),
        "clearance_error_m": np.asarray(clearance_error_m, dtype=np.float64),
        "clearance_overprediction_m": np.asarray(clearance_overprediction_m, dtype=np.float64),
        "visibility_error": np.asarray(visibility_error, dtype=np.float64),
        "intervention_error": np.asarray(intervention_error, dtype=np.float64),
        "feasibility_brier": np.asarray(feasibility_brier, dtype=np.float64),
        "route_geometry_brier": np.asarray(route_geometry_brier, dtype=np.float64),
        "route_termination_brier": np.asarray(route_termination_brier, dtype=np.float64),
        "route_progress_error": np.asarray(route_progress_error, dtype=np.float64),
    }
    shape = next(iter(values.values())).shape
    if any(value.shape != shape for value in values.values()):
        raise ValueError("Safety-priority credit inputs must have identical shapes.")
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Safety-priority credit inputs must be finite.")

    components = {
        "feasibility": np.clip(1.0 - values["feasibility_brier"], 0.0, 1.0),
        "clearance": np.exp(-np.maximum(values["clearance_error_m"], 0.0) / 1.0),
        "clearance_conservatism": np.exp(-np.maximum(values["clearance_overprediction_m"], 0.0) / 0.5),
        "route_geometry": np.clip(1.0 - values["route_geometry_brier"], 0.0, 1.0),
        "route_termination": np.clip(1.0 - values["route_termination_brier"], 0.0, 1.0),
        "intervention": np.clip(1.0 - values["intervention_error"], 0.0, 1.0),
        "visibility": np.clip(1.0 - values["visibility_error"], 0.0, 1.0),
        "route_progress": np.exp(-np.maximum(values["route_progress_error"], 0.0) / 0.25),
        # Target displacement is deliberately the smallest term.
        "target": np.exp(-np.maximum(values["target_error_m"], 0.0) / 3.0),
    }
    weights = {
        "feasibility": 0.30,
        "clearance": 0.20,
        "clearance_conservatism": 0.10,
        "route_geometry": 0.12,
        "route_termination": 0.08,
        "intervention": 0.08,
        "visibility": 0.04,
        "route_progress": 0.05,
        "target": 0.03,
    }
    credit = sum(weights[name] * components[name] for name in weights)
    components["credit"] = np.clip(credit, 0.0, 1.0)
    return components["credit"], components


def _calibrate(
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    metadata: dict[str, Any],
    checkpoint_path: Path,
    dataset_path: Path,
    metadata_path: Path,
    protocol_path: Path,
    minimum_sample_count: int,
    minimum_credit: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    extent = 10.0
    horizon_count = int(arrays["labels_relative"].shape[1])
    entries: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    route_mask = arrays["sample_type"] < 0.5
    runtime_count = int(np.count_nonzero(route_mask))
    for horizon in range(horizon_count):
        target_error = np.linalg.norm(
            predictions["target_relative"][:, horizon] - arrays["labels_relative"][:, horizon], axis=1
        ) * extent
        predicted_clearance = np.minimum(
            predictions["obstacle_clearance"][:, horizon], predictions["inter_agent_clearance"][:, horizon]
        )
        observed_clearance = np.minimum(
            arrays["labels_obstacle_clearance"][:, horizon], arrays["labels_inter_agent_clearance"][:, horizon]
        )
        boundary_error = np.abs(
            predictions["boundary_clearance"][:, horizon] - arrays["labels_boundary_clearance"][:, horizon]
        )
        clearance_error = np.maximum(np.abs(predicted_clearance - observed_clearance), boundary_error)
        overprediction = np.maximum(
            np.maximum(predicted_clearance - observed_clearance, predictions["boundary_clearance"][:, horizon] - arrays["labels_boundary_clearance"][:, horizon]),
            0.0,
        )
        visibility_label = arrays["labels_target_visible"][:, horizon]
        intervention_label = arrays["labels_cbf_intervention"][:, horizon]
        feasibility_label = arrays["labels_cbf_feasible"][:, horizon]
        visibility_brier = (predictions["visibility_probability"][:, horizon] - visibility_label) ** 2
        intervention_brier = (predictions["intervention_probability"][:, horizon] - intervention_label) ** 2
        feasibility_brier = (
            predictions["cbf_feasibility_probability_route"][:, horizon] - feasibility_label
        ) ** 2
        route_geometry_brier = (
            predictions["route_geometry_probability"] - arrays["route_geometry_valid"]
        ) ** 2
        route_termination_brier = (
            predictions["route_termination_probability"] - arrays["branch_terminated"]
        ) ** 2
        route_progress_error = np.abs(
            predictions["route_progress"][:, horizon] - arrays["labels_route_progress"][:, horizon]
        )
        credit, credit_components = _safety_priority_credit(
            target_error_m=target_error,
            clearance_error_m=clearance_error,
            clearance_overprediction_m=overprediction,
            visibility_error=np.abs(predictions["visibility_probability"][:, horizon] - visibility_label),
            intervention_error=np.abs(predictions["intervention_probability"][:, horizon] - intervention_label),
            feasibility_brier=feasibility_brier,
            route_geometry_brier=route_geometry_brier,
            route_termination_brier=route_termination_brier,
            route_progress_error=route_progress_error,
        )
        key = make_safe_capture_global_key(horizon)
        entries[key] = {
            "context": {
                "horizon_index": horizon,
                "scope": "route_identity_global",
                "candidate_profile": "obstacle_route_v1",
            },
            "sample_count": int(arrays["inputs"].shape[0]),
            "credit": _mean(credit),
            "target_mae_m": _mean(target_error),
            "clearance_mae_m": _mean(clearance_error),
            "clearance_overprediction_m": _mean(overprediction),
            "visibility_brier": _mean(visibility_brier),
            "intervention_brier": _mean(intervention_brier),
            "qp_feasibility_brier": _mean(feasibility_brier),
            "route_geometry_brier": _mean(route_geometry_brier),
            "route_termination_brier": _mean(route_termination_brier),
            "route_progress_mae": _mean(route_progress_error),
            "candidate_ranking_win_rate": _mean(1.0 - feasibility_brier),
            "collision_rate": 0.0,
            "boundary_rate": _mean(arrays["labels_boundary_clearance"][:, horizon] < 0.0),
            "cbf_intervention_rate": _mean(intervention_label),
            "mean_uncertainty": _mean(predictions["target_std"][:, horizon]),
        }
        diagnostics.append(
            {
                "horizon_index": horizon,
                "sample_count": int(arrays["inputs"].shape[0]),
                "runtime_sample_count": runtime_count,
                "global_credit": _mean(credit),
                "target_mae_m": _mean(target_error),
                "clearance_mae_m": _mean(clearance_error),
                "boundary_mae_m": _mean(np.abs(predictions["boundary_clearance"][:, horizon] - arrays["labels_boundary_clearance"][:, horizon])),
                "cbf_min_slack_mae": _mean(np.abs(predictions["cbf_min_slack"][:, horizon] - arrays["labels_cbf_min_slack"][:, horizon])),
                "cbf_feasibility_brier": _mean(feasibility_brier),
                "route_geometry_brier": _mean(route_geometry_brier),
                "route_termination_brier": _mean(route_termination_brier),
                "route_progress_mae": _mean(route_progress_error),
                "boundary_negative_rate": _mean(arrays["labels_boundary_clearance"][:, horizon] < 0.0),
                "cbf_infeasible_rate": _mean(feasibility_label < 0.5),
                "credit_components": {
                    name: _mean(value) for name, value in credit_components.items() if name != "credit"
                },
            }
        )
    runtime_identity = arrays["route_candidate_index"][route_mask]
    identity_accuracy = _mean(
        predictions["route_identity_logits"][route_mask].argmax(axis=1) == runtime_identity
    ) if runtime_count else 0.0
    runtime_side = arrays["route_side_index"][route_mask]
    side_accuracy = _mean(
        predictions["route_side_logits"][route_mask].argmax(axis=1) == runtime_side
    ) if runtime_count else 0.0
    geometry_accuracy = _mean(
        (predictions["route_geometry_probability"] >= 0.5) == (arrays["route_geometry_valid"] >= 0.5)
    )
    termination_accuracy = _mean(
        (predictions["route_termination_probability"] >= 0.5) == (arrays["branch_terminated"] >= 0.5)
    )
    payload = {
        "ledger_type": SafeCaptureReliabilityLedger.LEDGER_TYPE,
        "ledger_version": 2,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "immutable_after_calibration": True,
        "update_rule": "route_identity_calibration_split_only_no_online_update",
        "source": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "calibration_dataset": str(dataset_path.resolve()),
            "calibration_dataset_sha256": _sha256(dataset_path),
            "calibration_metadata": str(metadata_path.resolve()),
            "calibration_metadata_sha256": _sha256(metadata_path),
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": _sha256(protocol_path),
            "evaluation_protocol": str(protocol_path.resolve()),
            "evaluation_protocol_sha256": _sha256(protocol_path),
            "samples": int(arrays["inputs"].shape[0]),
            "candidate_profile": "obstacle_route_v1",
            "candidate_count": 12,
            "model_type": MODEL_TYPE,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
        },
        "bucket_definition": {
            "scope": "global horizon entries; route-specific heads are audited separately",
            "ledger_is_not_a_safety_certificate": True,
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
            "version": "route_safety_priority_v2",
            "target_weight": 0.03,
            "clearance_weight": 0.20,
            "clearance_conservatism_weight": 0.10,
            "route_geometry_weight": 0.12,
            "route_termination_weight": 0.08,
            "intervention_weight": 0.08,
            "visibility_weight": 0.04,
            "route_progress_weight": 0.05,
            "feasibility_weight": 0.30,
            "target_error_decay_m": 3.0,
            "clearance_error_decay_m": 1.0,
            "clearance_overprediction_decay_m": 0.5,
            "route_progress_error_decay": 0.25,
            "boundary_error_included_in_clearance": True,
        },
        "context_dimensions": ["horizon_index", "scope", "candidate_profile"],
        "entries": entries,
        "route_head_calibration": {
            "route_identity_accuracy": identity_accuracy,
            "route_side_accuracy": side_accuracy,
            "route_geometry_accuracy": geometry_accuracy,
            "route_termination_accuracy": termination_accuracy,
            "candidate_chance_accuracy": 1.0 / 12.0,
        },
    }
    report = {
        "calibration_type": "jepa_route_identity_global_ledger_v2_safety_priority",
        "development_only": True,
        "locked_test_opened": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "metadata_sha256": _sha256(metadata_path),
        "protocol_sha256": _sha256(protocol_path),
        "route_head_calibration": payload["route_head_calibration"],
        "horizon_diagnostics": diagnostics,
        "all_predictions_finite": True,
        "minimum_credit": float(minimum_credit),
        "global_credit_meets_threshold": [item["global_credit"] >= minimum_credit for item in diagnostics],
    }
    return payload, report


def _write_tensorboard(logdir: Path, payload: dict[str, Any], report: dict[str, Any]) -> None:
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/ledger", json.dumps(payload["decision_policy"], indent=2), 0)
        writer.add_text("Provenance/source", json.dumps(payload["source"], indent=2), 0)
        writer.add_text("Calibration/route_heads", json.dumps(report["route_head_calibration"], indent=2), 0)
        for row in report["horizon_diagnostics"]:
            h = int(row["horizon_index"])
            writer.add_scalar("Calibration/global_credit", row["global_credit"], h)
            writer.add_scalar("Calibration/target_mae_m", row["target_mae_m"], h)
            writer.add_scalar("Calibration/clearance_mae_m", row["clearance_mae_m"], h)
            writer.add_scalar("Calibration/boundary_mae_m", row["boundary_mae_m"], h)
            writer.add_scalar("Calibration/cbf_min_slack_mae", row["cbf_min_slack_mae"], h)
            writer.add_scalar("Calibration/cbf_feasibility_brier", row["cbf_feasibility_brier"], h)
            writer.add_scalar("Calibration/route_geometry_brier", row["route_geometry_brier"], h)
            writer.add_scalar("Calibration/route_termination_brier", row["route_termination_brier"], h)
            writer.add_scalar("Calibration/route_progress_mae", row["route_progress_mae"], h)
            writer.add_scalar("Calibration/boundary_negative_rate", row["boundary_negative_rate"], h)
            writer.add_scalar("Calibration/cbf_infeasible_rate", row["cbf_infeasible_rate"], h)
            for name, value in row["credit_components"].items():
                writer.add_scalar(f"Calibration/credit_component/{name}", value, h)
        writer.add_scalar("Calibration/route_identity_accuracy", report["route_head_calibration"]["route_identity_accuracy"], 0)
        writer.add_scalar("Calibration/route_side_accuracy", report["route_head_calibration"]["route_side_accuracy"], 0)
        writer.add_scalar("Calibration/route_geometry_accuracy", report["route_head_calibration"]["route_geometry_accuracy"], 0)
        writer.add_scalar("Calibration/route_termination_accuracy", report["route_head_calibration"]["route_termination_accuracy"], 0)


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.dataset, args.metadata, args.protocol):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite an existing route ledger/report.")
    if args.minimum_sample_count <= 0 or not 0.0 <= args.minimum_credit <= 1.0 or args.batch_size <= 0:
        raise ValueError("minimum sample count/batch size must be positive and credit in [0,1].")
    metadata = _load_metadata(args.metadata.resolve(), args.dataset.resolve())
    arrays = _load_arrays(args.dataset.resolve())
    device = _device(args.device)
    predictions, checkpoint = _predict(args.checkpoint.resolve(), arrays, args.batch_size, device)
    payload, report = _calibrate(
        arrays,
        predictions,
        metadata,
        args.checkpoint.resolve(),
        args.dataset.resolve(),
        args.metadata.resolve(),
        args.protocol.resolve(),
        args.minimum_sample_count,
        args.minimum_credit,
    )
    payload["source"]["training_variant"] = checkpoint.get("training_variant")
    report["device"] = str(device)
    report["tensorboard_logdir"] = str(args.tensorboard_logdir.resolve())
    _write_tensorboard(args.tensorboard_logdir.resolve(), payload, report)
    payload["tensorboard"] = {"logdir": str(args.tensorboard_logdir.resolve()), "event_files": sorted(path.name for path in args.tensorboard_logdir.glob("events.out.tfevents.*"))}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.report.resolve().write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"ledger": str(args.output.resolve()), "report": str(args.report.resolve()), "route_head_calibration": report["route_head_calibration"], "global_credit": [row["global_credit"] for row in report["horizon_diagnostics"]], "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
