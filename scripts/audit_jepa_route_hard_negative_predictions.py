"""Audit v2 route-JEPA risk heads on train/validation/calibration archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_route_identity_jepa import load_dataset
from encirclement3d.prediction import build_action_conditioned_predictor


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2"
RISK_HEADS = {
    "stopping_distance": "labels_stopping_distance",
    "obstacle_ttc": "labels_obstacle_ttc",
    "boundary_ttc": "labels_boundary_ttc",
    "pairwise_ttc_risk": "labels_pairwise_ttc",
    "acceleration_slack": "labels_acceleration_slack",
}
HAZARD_HEADS = {
    "obstacle_ttc": "labels_obstacle_ttc",
    "boundary_ttc": "labels_boundary_ttc",
    "pairwise_ttc": "labels_pairwise_ttc",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _correlation(prediction: np.ndarray, target: np.ndarray) -> float | None:
    if prediction.size < 2 or float(np.std(prediction)) <= 1e-12 or float(np.std(target)) <= 1e-12:
        return None
    return float(np.corrcoef(prediction, target)[0, 1])


def _head_metrics(prediction: np.ndarray, target: np.ndarray, *, sentinel: float | None = None) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(prediction) & np.isfinite(target)
    if sentinel is not None:
        mask &= target != float(sentinel)
    if not mask.any():
        raise ValueError("Risk head has no measured finite labels after masking.")
    prediction = prediction[mask]
    target = target[mask]
    return {
        "count": int(mask.sum()),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "bias": float(np.mean(prediction - target)),
        "correlation": _correlation(prediction, target),
    }


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unexpected checkpoint model type: {checkpoint.get('model_type')!r}")
    model = build_action_conditioned_predictor(MODEL_TYPE, checkpoint["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def audit_split(model: torch.nn.Module, dataset: Path, metadata: Path, split: str, device: torch.device) -> dict[str, Any]:
    tensors, archive_metadata = load_dataset(dataset.resolve(), metadata.resolve(), split)
    route_interaction_chunks = (
        tensors["route_relative_action_chunk"].to(device)
        if int(getattr(model, "route_interaction_chunk_dim", 0)) > 0
        else None
    )
    with torch.no_grad():
        _mean, _log_variance, _latent, auxiliary = model.forward_multitask(
            tensors["inputs"].to(device),
            tensors["action_history"].to(device),
            tensors["route_action_chunk"].to(device),
            tensors["route_candidate_index"].to(device),
            tensors["route_side_index"].to(device),
            route_interaction_chunks,
        )
    report: dict[str, Any] = {
        "split": split,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset.resolve()),
        "sample_count": int(tensors["inputs"].shape[0]),
        "heads": {},
    }
    for head, label in RISK_HEADS.items():
        prediction = auxiliary[head].detach().cpu().numpy()
        target = tensors[label].numpy()
        metrics = _head_metrics(
            prediction,
            target,
            sentinel=-1.0 if head == "acceleration_slack" else None,
        )
        report["heads"][head] = metrics
        if head in {"obstacle_ttc", "boundary_ttc", "pairwise_ttc_risk"}:
            for threshold in (1.0, 2.0):
                true_risk = target < threshold
                predicted_risk = prediction < threshold
                measured = target != -1.0
                true_risk = true_risk & measured
                predicted_risk = predicted_risk & measured
                positives = int(true_risk.sum())
                report["heads"][head][f"risk_lt_{threshold:g}_recall"] = (
                    float((predicted_risk & true_risk).sum() / positives) if positives else None
                )
                report["heads"][head][f"risk_lt_{threshold:g}_brier"] = float(
                    np.mean((predicted_risk.astype(np.float64) - true_risk.astype(np.float64)) ** 2)
                )
    for head, label in HAZARD_HEADS.items():
        target = tensors[label].numpy()
        logits = auxiliary[f"{head}_hazard_logits"].detach().cpu().numpy()
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        head_report: dict[str, Any] = {}
        for band_index, threshold in enumerate((0.5, 1.0, 2.0)):
            measured = np.isfinite(target)
            positive = measured & (target <= threshold)
            predicted_positive = measured & (probabilities[..., band_index] >= 0.5)
            true_positive = predicted_positive & positive
            positives = int(positive.sum())
            predicted_count = int(predicted_positive.sum())
            head_report[f"hazard_lt_{threshold:g}_recall"] = (
                float(true_positive.sum() / positives) if positives else None
            )
            head_report[f"hazard_lt_{threshold:g}_precision"] = (
                float(true_positive.sum() / predicted_count) if predicted_count else None
            )
            head_report[f"hazard_lt_{threshold:g}_brier"] = float(
                np.mean((probabilities[..., band_index] - positive.astype(np.float64)) ** 2)
            )
        sweep: dict[str, dict[str, float | None]] = {}
        for cutoff in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            cutoff_report: dict[str, float | None] = {}
            for band_index, threshold in enumerate((0.5, 1.0, 2.0)):
                measured = np.isfinite(target)
                positive = measured & (target <= threshold)
                predicted_positive = measured & (probabilities[..., band_index] >= cutoff)
                true_positive = predicted_positive & positive
                positives = int(positive.sum())
                predicted_count = int(predicted_positive.sum())
                cutoff_report[f"recall_lt_{threshold:g}"] = (
                    float(true_positive.sum() / positives) if positives else None
                )
                cutoff_report[f"precision_lt_{threshold:g}"] = (
                    float(true_positive.sum() / predicted_count) if predicted_count else None
                )
            sweep[f"{cutoff:g}"] = cutoff_report
        head_report["threshold_sweep"] = sweep
        report["heads"][f"{head}_hazard"] = head_report
        quantile_prediction = auxiliary[f"{head}_lower_quantile"].detach().cpu().numpy()
        report["heads"][f"{head}_lower_quantile"] = _head_metrics(quantile_prediction, target)
    report["archive_metadata_dataset_version"] = archive_metadata.get("dataset_version")
    report["archive_metadata_split"] = archive_metadata.get("split")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "validation", "calibration"), action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if not (len(args.archive) == len(args.metadata) == len(args.split) == 3):
        raise ValueError("Provide exactly three --archive, --metadata and --split values.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    if args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard logdir: {args.tensorboard_logdir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_model(args.checkpoint.resolve(), device)
    reports = [
        audit_split(model, dataset, metadata, split, device)
        for dataset, metadata, split in zip(args.archive, args.metadata, args.split)
    ]
    if {report["split"] for report in reports} != {"train", "validation", "calibration"}:
        raise ValueError("Prediction audit requires train, validation and calibration splits.")
    result = {
        "model_type": MODEL_TYPE,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "base_checkpoint_sha256": checkpoint.get("base_checkpoint_sha256"),
        "device": str(device),
        "head_only": bool(checkpoint.get("head_only", False)),
        "archives": reports,
        "development_only": True,
        "locked_test_opened": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tensorboard = args.tensorboard_logdir.resolve()
    tensorboard.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("Provenance/checkpoint", json.dumps({"path": str(args.checkpoint.resolve()), "sha256": result["checkpoint_sha256"]}), 0)
        writer.add_text("Provenance/archives", json.dumps([{item: report[item] for item in ("split", "dataset_sha256")} for report in reports]), 0)
        for report in reports:
            split = report["split"]
            for head, metrics in report["heads"].items():
                if "mae" in metrics:
                    writer.add_scalar(f"Prediction/{head}/mae/{split}", metrics["mae"], 0)
                    writer.add_scalar(f"Prediction/{head}/rmse/{split}", metrics["rmse"], 0)
                    if metrics["correlation"] is not None:
                        writer.add_scalar(f"Prediction/{head}/correlation/{split}", metrics["correlation"], 0)
                for metric_name, metric_value in metrics.items():
                    if metric_name.startswith("hazard_") and metric_value is not None:
                        writer.add_scalar(f"Prediction/{head}/{metric_name}/{split}", metric_value, 0)
                if head.endswith("_hazard"):
                    for cutoff, cutoff_metrics in metrics.get("threshold_sweep", {}).items():
                        for metric_name, metric_value in cutoff_metrics.items():
                            if metric_value is not None:
                                writer.add_scalar(
                                    f"Prediction/{head}/{metric_name}/cutoff_{cutoff}/{split}",
                                    metric_value,
                                    0,
                                )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
