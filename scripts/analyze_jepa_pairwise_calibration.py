"""Stratify pairwise JEPA hazard calibration by archive sample type."""

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

from audit_jepa_route_hard_negative_predictions import _load_model
from train_route_identity_jepa import load_dataset


SAMPLE_TYPES = {
    "runtime": 0,
    "boundary_shadow": 1,
    "near_pass": 2,
    "formation_crossing": 3,
    "split_merge": 4,
}
CUTOFFS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HAZARD_BANDS = (0.5, 1.0, 2.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _threshold_metrics(probability: np.ndarray, positive: np.ndarray, cutoff: float) -> dict[str, float | None]:
    predicted = probability >= cutoff
    true_positive = predicted & positive
    false_positive = predicted & ~positive
    negatives = ~positive
    return {
        "recall": _ratio(int(true_positive.sum()), int(positive.sum())),
        "precision": _ratio(int(true_positive.sum()), int(predicted.sum())),
        "false_positive_rate": _ratio(int(false_positive.sum()), int(negatives.sum())),
        "predicted_positive_rate": float(np.mean(predicted)),
        "positive_rate": float(np.mean(positive)),
        "predicted_positive_count": int(predicted.sum()),
        "positive_count": int(positive.sum()),
        "false_positive_count": int(false_positive.sum()),
    }


def _mode_report(probabilities: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    measured = np.isfinite(probabilities) & np.isfinite(target)
    if not measured.any():
        raise ValueError("Calibration mode has no finite pairwise labels")
    probabilities = probabilities[measured]
    target = target[measured]
    positive_by_band = {
        f"lt_{threshold:g}": target <= threshold for threshold in HAZARD_BANDS
    }
    report: dict[str, Any] = {
        "count": int(probabilities.size),
        "mean_probability": float(np.mean(probabilities)),
        "bands": {},
    }
    for band_name, positive in positive_by_band.items():
        report["bands"][band_name] = {
            "brier": float(np.mean((probabilities - positive.astype(np.float64)) ** 2)),
            "threshold_sweep": {
                str(cutoff): _threshold_metrics(probabilities, positive, cutoff)
                for cutoff in CUTOFFS
            },
        }
    return report


def analyze(
    checkpoint_path: Path,
    archive_path: Path,
    metadata_path: Path,
    output_path: Path,
    tensorboard_logdir: Path,
    device_name: str,
) -> dict[str, Any]:
    tensors, metadata = load_dataset(archive_path.resolve(), metadata_path.resolve(), "calibration")
    if metadata.get("split") != "calibration":
        raise ValueError("Pairwise calibration analysis requires the calibration split")
    device = torch.device(
        "cuda" if device_name == "cuda" or device_name == "auto" and torch.cuda.is_available() else "cpu"
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, checkpoint = _load_model(checkpoint_path.resolve(), device)
    if int(getattr(model, "route_interaction_chunk_dim", 0)) != 9:
        raise ValueError("Calibration analysis requires a pairwise relational checkpoint")
    with torch.no_grad():
        _mean, _log_variance, _latent, auxiliary = model.forward_multitask(
            tensors["inputs"].to(device),
            tensors["action_history"].to(device),
            tensors["route_action_chunk"].to(device),
            tensors["route_candidate_index"].to(device),
            tensors["route_side_index"].to(device),
            tensors["route_pairwise_relative_action_chunk"].to(device),
        )
    logits = auxiliary["pairwise_ttc_hazard_logits"].detach().cpu().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits[..., 1]))
    target = tensors["labels_pairwise_ttc"].numpy()
    sample_type = tensors["sample_type"].numpy()
    reports: dict[str, Any] = {}
    for name, sample_type_id in SAMPLE_TYPES.items():
        mask = sample_type == sample_type_id
        if not mask.any():
            raise ValueError(f"Calibration archive has no rows for sample type {name}")
        reports[name] = _mode_report(probabilities[mask], target[mask])
    reports["all"] = _mode_report(probabilities, target)
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path.resolve()),
        "archive": str(archive_path.resolve()),
        "archive_sha256": _sha256(archive_path.resolve()),
        "dataset_version": metadata.get("dataset_version"),
        "split": metadata.get("split"),
        "device": str(device),
        "sample_type_mapping": SAMPLE_TYPES,
        "target_band_seconds": HAZARD_BANDS,
        "reports": reports,
        "development_only": True,
        "locked_test_opened": False,
        "checkpoint_training_metadata": {
            "seed": checkpoint.get("seed"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_validation_loss": checkpoint.get("best_validation_loss"),
        },
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tensorboard_logdir = tensorboard_logdir.resolve()
    if tensorboard_logdir.exists() and any(tensorboard_logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard logdir: {tensorboard_logdir}")
    with SummaryWriter(log_dir=str(tensorboard_logdir), flush_secs=1) as writer:
        writer.add_text(
            "Provenance/checkpoint",
            json.dumps({"path": str(checkpoint_path.resolve()), "sha256": result["checkpoint_sha256"]}),
            0,
        )
        writer.add_text("Provenance/archive", json.dumps({"path": str(archive_path.resolve()), "sha256": result["archive_sha256"]}), 0)
        for mode, mode_report in reports.items():
            for band_name, band_report in mode_report["bands"].items():
                prefix = f"Calibration/{mode}/pairwise_ttc_{band_name}"
                writer.add_scalar(f"{prefix}/brier", band_report["brier"], 0)
                for cutoff, metrics in band_report["threshold_sweep"].items():
                    for metric in ("recall", "precision", "false_positive_rate", "positive_rate", "predicted_positive_rate"):
                        value = metrics[metric]
                        if value is not None:
                            writer.add_scalar(f"{prefix}/{metric}/cutoff_{cutoff}", value, 0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    result = analyze(
        args.checkpoint,
        args.archive,
        args.metadata,
        args.output,
        args.tensorboard_logdir,
        args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
