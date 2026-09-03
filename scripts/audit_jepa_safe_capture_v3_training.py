"""Audit hard-context weighted JEPA v3 training runs and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPE = "interaction_aware_action_conditioned_jepa_safe_capture_v2"
TRAINING_VARIANT = "hard_context_weighted_v1"

REQUIRED_SCALARS = {
    "Data/train_samples",
    "Data/validation_samples",
    "Data/calibration_episodes",
    "Loss/total/train",
    "Loss/total/validation",
    "Loss/target/train",
    "Loss/target/validation",
    "Loss/clearance/train",
    "Loss/clearance/validation",
    "Loss/visibility/train",
    "Loss/visibility/validation",
    "Loss/cbf_risk/train",
    "Loss/cbf_risk/validation",
    "Loss/velocity/train",
    "Loss/velocity/validation",
    "Loss/acceleration/train",
    "Loss/acceleration/validation",
    "Loss/action_consistency/train",
    "Loss/action_consistency/validation",
    "Calibration/target_one_std_coverage/train",
    "Calibration/target_one_std_coverage/validation",
    "Calibration/qp_feasibility_brier/train",
    "Calibration/qp_feasibility_brier/validation",
    "Uncertainty/target_mean_std/train",
    "Uncertainty/target_mean_std/validation",
    "HardContext/train_weight_mean",
    "HardContext/train_weight_p95",
    "HardContext/validation_weight_mean",
    "HardContext/validation_weight_p95",
    "Optimization/learning_rate",
}
REQUIRED_TEXT = {
    "Config/protocol/text_summary",
    "Config/training_config/text_summary",
    "Config/model/text_summary",
    "Config/hard_context_weights/text_summary",
    "Config/optimization/text_summary",
    "Dataset/train_metadata/text_summary",
    "Dataset/validation_metadata/text_summary",
    "Dataset/calibration_metadata/text_summary",
    "Provenance/source_hashes/text_summary",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_history(history: list[dict[str, Any]]) -> bool:
    for record in history:
        for key, value in record.items():
            if key == "epoch":
                continue
            if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
                return False
    return True


def audit_run(directory: Path, minimum_epochs: int = 40) -> dict[str, Any]:
    directory = directory.resolve()
    metadata_path = directory / "run_metadata.json"
    history_path = directory / "history.json"
    checkpoint_path = directory / "checkpoint.pt"
    for path in (metadata_path, history_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required v3 training artifact: {path}")
    metadata = load_json(metadata_path)
    if metadata.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unexpected model type in {metadata_path}")
    if metadata.get("training_variant") != TRAINING_VARIANT:
        raise ValueError(f"Unexpected training variant in {metadata_path}")
    if metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Training run opened locked test: {directory}")
    epochs = int(metadata.get("epochs", 0))
    if epochs < minimum_epochs:
        raise ValueError(f"Run has {epochs} configured epochs; expected at least {minimum_epochs}.")
    history = load_json(history_path)
    if not isinstance(history, list) or len(history) < minimum_epochs:
        raise ValueError(f"History has {len(history) if isinstance(history, list) else 'invalid'} epochs.")
    if not history or not all(isinstance(item, dict) for item in history):
        raise ValueError("History must contain epoch mappings.")
    epoch_numbers = [int(item.get("epoch", -1)) for item in history]
    if epoch_numbers != list(range(1, len(history) + 1)):
        raise ValueError(f"History epoch sequence is not contiguous: {epoch_numbers[:3]} ...")
    if not _finite_history(history):
        raise ValueError("History contains non-finite or non-numeric values.")
    required_history = {
        "train_loss",
        "validation_loss",
        "train_clearance_quantile",
        "validation_clearance_quantile",
        "train_velocity_mse",
        "validation_velocity_mse",
        "train_qp_feasibility_bce",
        "validation_qp_feasibility_bce",
        "train_hard_weight_mean",
        "validation_hard_weight_p95",
    }
    missing_history = sorted(required_history.difference(history[-1]))
    if missing_history:
        raise ValueError(f"History is missing v3 metrics: {missing_history}")
    hard_cap = float(metadata.get("hard_weight_kwargs", {}).get("cap", 0.0))
    if hard_cap < 1.0:
        raise ValueError("Hard-context weight cap is missing or invalid.")
    for record in history:
        for key in ("train_hard_weight_mean", "train_hard_weight_p95", "validation_hard_weight_mean", "validation_hard_weight_p95"):
            value = float(record[key])
            if value < 1.0 or value > hard_cap + 1e-6:
                raise ValueError(f"{key}={value} is outside [1, {hard_cap}].")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != MODEL_TYPE or checkpoint.get("training_variant") != TRAINING_VARIANT:
        raise ValueError("Checkpoint model or training variant does not match metadata.")
    if int(checkpoint.get("seed", -1)) != int(metadata.get("seed", -2)):
        raise ValueError("Checkpoint seed does not match run metadata.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint state is empty.")
    tensor_values = [value for value in state.values() if torch.is_tensor(value)]
    if not tensor_values or not all(bool(torch.isfinite(value).all()) for value in tensor_values):
        raise ValueError("Checkpoint contains non-finite parameters.")
    if checkpoint.get("hard_weight_kwargs") != metadata.get("hard_weight_kwargs"):
        raise ValueError("Checkpoint hard-context weighting does not match metadata.")
    tensorboard_path = Path(metadata.get("tensorboard_logdir", "")).resolve()
    if not tensorboard_path.is_dir():
        raise FileNotFoundError(f"Missing TensorBoard directory: {tensorboard_path}")
    accumulator = EventAccumulator(str(tensorboard_path), size_guidance={"scalars": 0, "tensors": 0, "histograms": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    missing_scalars = sorted(REQUIRED_SCALARS.difference(tags.get("scalars", [])))
    missing_text = sorted(REQUIRED_TEXT.difference(tags.get("tensors", [])))
    if missing_scalars:
        raise ValueError(f"TensorBoard is missing scalar tags: {missing_scalars}")
    if missing_text:
        raise ValueError(f"TensorBoard is missing text tags: {missing_text}")
    histogram_tags = tags.get("histograms", [])
    if not histogram_tags or not any(tag.startswith("Parameters/") for tag in histogram_tags) or not any(tag.startswith("Gradients/") for tag in histogram_tags):
        raise ValueError("TensorBoard is missing parameter and gradient histograms.")
    training_config = Path(metadata.get("training_config", "")).resolve()
    if not training_config.is_file() or sha256(training_config) != metadata.get("training_config_sha256"):
        raise ValueError("Training config provenance is missing or has a hash mismatch.")
    source_hashes = metadata.get("source_hashes", {})
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Source hashes are missing.")
    for relative, expected in source_hashes.items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Source hash mismatch: {relative}")
    return {
        "directory": str(directory),
        "seed": int(metadata["seed"]),
        "model_type": MODEL_TYPE,
        "training_variant": TRAINING_VARIANT,
        "epochs": len(history),
        "configured_epochs": epochs,
        "best_epoch": int(metadata["best_epoch"]),
        "best_validation_loss": float(metadata["best_validation_loss"]),
        "checkpoint_sha256": sha256(checkpoint_path),
        "train_dataset_sha256": metadata.get("train_dataset_sha256"),
        "validation_dataset_sha256": metadata.get("validation_dataset_sha256"),
        "calibration_metadata_sha256": metadata.get("calibration_metadata_sha256"),
        "training_config_sha256": metadata.get("training_config_sha256"),
        "hard_weight_kwargs": metadata["hard_weight_kwargs"],
        "device": metadata.get("device"),
        "torch": metadata.get("torch"),
        "tensorboard": {
            "path": str(tensorboard_path),
            "scalar_tag_count": len(tags.get("scalars", [])),
            "text_tag_count": len(tags.get("tensors", [])),
            "histogram_tag_count": len(histogram_tags),
            "required_tags_complete": True,
        },
        "locked_test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-epochs", type=int, default=40)
    args = parser.parse_args()
    if len(args.run) < 3:
        raise ValueError("The v3 audit requires at least three independent training runs.")
    reports = [audit_run(path, minimum_epochs=args.minimum_epochs) for path in args.run]
    seeds = [item["seed"] for item in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training runs contain duplicate seeds.")
    result = {
        "audit_type": "jepa_safe_capture_v3_hard_context_training",
        "model_type": MODEL_TYPE,
        "training_variant": TRAINING_VARIANT,
        "seed_count": len(reports),
        "runs": reports,
        "all_runs_pass": True,
        "locked_test_opened": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
