"""Offline calibration of a public route-utility label.

This audit evaluates route utility components that are available before action
execution: progress, geometric route length, predicted target-relative escape
distance, and multi-step CBF feasibility.  Weights are selected on the
independent calibration split and then applied unchanged to train/validation.
No action is executed and no CBF, Ledger, or online evaluator contract is
modified.  The current archive has no previous-route field, so route-switch
penalty is reported as unavailable instead of being inferred from the current
selected candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.prediction import build_action_conditioned_predictor  # noqa: E402
from train_route_identity_jepa import load_dataset  # noqa: E402


RANKING_HORIZON_INDEX = 2
ROUTE_COUNT = 12
GROUP_TIE_MARGIN = 0.005
MODEL_TYPE = "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2"
WEIGHT_GRID = {
    "length": (0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
    "escape": (0.0, 0.1, 0.2, 0.5),
    "cbf": (0.0, 0.1, 0.2, 0.5, 1.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def _predict(model: torch.nn.Module, tensors: dict[str, torch.Tensor], device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {"route_progress": [], "cbf_feasibility_logit": []}
    means: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(tensors["inputs"].shape[0]), batch_size):
            end = min(start + batch_size, int(tensors["inputs"].shape[0]))
            args = (
                tensors["inputs"][start:end].to(device),
                tensors["action_history"][start:end].to(device),
                tensors["route_action_chunk"][start:end].to(device),
                tensors["route_candidate_index"][start:end].to(device),
                tensors["route_side_index"][start:end].to(device),
                tensors["route_pairwise_relative_action_chunk"][start:end].to(device),
            )
            mean, _log_variance, _latent, auxiliary = model.forward_multitask(*args)
            means.append(mean.detach().cpu().numpy())
            for name in collected:
                collected[name].append(auxiliary[name].detach().cpu().numpy())
    result = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    result["mean"] = np.concatenate(means, axis=0)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Model produced non-finite utility inputs")
    return result


def _records(tensors: dict[str, torch.Tensor], predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    arrays = {name: value.numpy() for name, value in tensors.items()}
    runtime = arrays["sample_type"] == 0
    scenario = arrays["scenario_index"]
    time_index = arrays["time_index"]
    candidate = arrays["route_candidate_index"]
    groups: list[dict[str, Any]] = []
    keys = sorted({(int(scenario[i]), int(time_index[i])) for i in np.flatnonzero(runtime)})
    for key in keys:
        group = runtime & (scenario == key[0]) & (time_index == key[1])
        candidates: list[dict[str, Any]] = []
        for candidate_id in range(ROUTE_COUNT):
            rows = group & (candidate == candidate_id)
            if not np.any(rows):
                continue
            truth_progress = float(np.mean(arrays["labels_route_progress"][rows, RANKING_HORIZON_INDEX]))
            predicted_progress = float(np.mean(predictions["route_progress"][rows, RANKING_HORIZON_INDEX]))
            truth_escape = float(np.mean(np.linalg.norm(arrays["labels_relative"][rows, RANKING_HORIZON_INDEX], axis=-1)))
            predicted_escape = float(np.mean(np.linalg.norm(predictions["mean"][rows, RANKING_HORIZON_INDEX], axis=-1)))
            truth_cbf = float(np.mean(arrays["labels_cbf_feasible"][rows, : RANKING_HORIZON_INDEX + 1]))
            predicted_cbf = float(np.mean(_sigmoid(predictions["cbf_feasibility_logit"][rows, : RANKING_HORIZON_INDEX + 1])))
            geometry_valid = bool(np.all(arrays["route_geometry_valid"][rows] >= 0.5))
            first_step_cbf = bool(np.all(arrays["labels_cbf_feasible"][rows, 0] >= 0.5))
            candidates.append(
                {
                    "candidate": candidate_id,
                    "eligible": bool(geometry_valid and first_step_cbf),
                    "truth_progress": truth_progress,
                    "predicted_progress": predicted_progress,
                    "truth_escape": truth_escape,
                    "predicted_escape": predicted_escape,
                    "truth_cbf": truth_cbf,
                    "predicted_cbf": predicted_cbf,
                    "route_length": float(np.mean(arrays["route_length_m"][rows])),
                }
            )
        selected = arrays["selected_candidate_index"][group]
        selected_candidate = int(selected[0]) if selected.size else -1
        groups.append({"scenario_index": key[0], "time_index": key[1], "selected": selected_candidate, "candidates": candidates})
    return groups


def _utility(item: dict[str, Any], weights: dict[str, float], *, predicted: bool) -> float:
    prefix = "predicted" if predicted else "truth"
    progress = float(item[f"{prefix}_progress"]) / 0.3
    escape = float(item[f"{prefix}_escape"]) / 2.0
    cbf = float(item[f"{prefix}_cbf"])
    length = float(item["route_length"]) / 10.0
    return progress - weights["length"] * length - weights["escape"] * escape + weights["cbf"] * cbf


def _evaluate(groups: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    exact = tie_aware = informative = informative_exact = 0
    pair_correct = pair_total = selected_matches = selected_total = 0
    group_count = informative_count = 0
    for group in groups:
        candidates = [item for item in group["candidates"] if item["eligible"]]
        if len(candidates) < 2:
            continue
        truth_values = [(item, _utility(item, weights, predicted=False)) for item in candidates]
        predicted_values = [(item, _utility(item, weights, predicted=True)) for item in candidates]
        truth_best = max(truth_values, key=lambda pair: (pair[1], -pair[0]["candidate"]))[0]["candidate"]
        predicted_best = max(predicted_values, key=lambda pair: (pair[1], -pair[0]["candidate"]))[0]["candidate"]
        ordered = sorted((value for _item, value in truth_values), reverse=True)
        gap = ordered[0] - ordered[1]
        group_count += 1
        exact += int(predicted_best == truth_best)
        predicted_truth_value = next(value for item, value in truth_values if item["candidate"] == predicted_best)
        tie_aware += int(gap <= GROUP_TIE_MARGIN or ordered[0] - predicted_truth_value <= GROUP_TIE_MARGIN)
        if gap > GROUP_TIE_MARGIN:
            informative_count += 1
            informative_exact += int(predicted_best == truth_best)
        if 0 <= int(group["selected"]) < ROUTE_COUNT:
            selected_total += 1
            selected_matches += int(predicted_best == int(group["selected"]))
        for left_index, (_left_item, left_value) in enumerate(truth_values):
            for _right_item, right_value in truth_values[left_index + 1 :]:
                if abs(left_value - right_value) <= GROUP_TIE_MARGIN:
                    continue
                pair_total += 1
                prediction_left = _utility(truth_values[left_index][0], weights, predicted=True)
                prediction_right = _utility(_right_item, weights, predicted=True)
                pair_correct += int((prediction_left - prediction_right) * (left_value - right_value) > 0.0)
    return {
        "group_count": group_count,
        "exact_top1": exact / group_count if group_count else None,
        "tie_aware_top1": tie_aware / group_count if group_count else None,
        "informative_group_count": informative_count,
        "informative_exact_top1": informative_exact / informative_count if informative_count else None,
        "pairwise_total": pair_total,
        "pairwise_agreement": pair_correct / pair_total if pair_total else None,
        "selected_comparison_group_count": selected_total,
        "selected_agreement": selected_matches / selected_total if selected_total else None,
    }


def _grid() -> list[dict[str, float]]:
    return [
        {"length": float(length), "escape": float(escape), "cbf": float(cbf)}
        for length, escape, cbf in itertools.product(WEIGHT_GRID["length"], WEIGHT_GRID["escape"], WEIGHT_GRID["cbf"])
    ]


def _select(calibration: list[tuple[dict[str, float], dict[str, Any]]]) -> dict[str, float]:
    valid = [item for item in calibration if item[1]["informative_exact_top1"] is not None]
    if not valid:
        raise ValueError("Calibration has no informative groups")
    return max(
        valid,
        key=lambda item: (
            item[1]["informative_exact_top1"],
            item[1]["pairwise_agreement"] or -1.0,
            item[1]["exact_top1"] or -1.0,
            -sum(item[0].values()),
        ),
    )[0]


def _load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unexpected model type in {path}")
    model = build_action_conditioned_predictor(checkpoint["model_type"], checkpoint["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# DN-MPC P31 Route-Utility Label Calibration",
        "",
        "**Status:** development-only; offline-only; no action executed.",
        "",
        "The label combines progress, public route length, target-relative escape distance, and multi-step CBF feasibility. Route-switch penalty is unavailable in the current archive and is not inferred.",
        "",
        "| Model | Selected weights (length, escape, cbf) | Validation exact | Validation informative | Validation pairwise | Validation selected |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, report in result["models"].items():
        selected = report["selected_weights"]
        validation = report["splits"]["validation"]["selected"]
        lines.append(
            f"| {name} | `({selected['length']:g}, {selected['escape']:g}, {selected['cbf']:g})` | "
            f"{validation['exact_top1']:.2%} | {validation['informative_exact_top1']:.2%} | "
            f"{validation['pairwise_agreement']:.2%} | {validation['selected_agreement']:.2%} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "This is a label/score diagnostic only. It does not authorize online JEPA ranking, Ledger-Lite, or a relaxed CBF. A future archive must add the previous executed route identity before a route-switch penalty can be trained or audited.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    for split in ("train", "calibration", "validation"):
        parser.add_argument(f"--{split}-dataset", type=Path, required=True)
        parser.add_argument(f"--{split}-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    output_paths = [args.output.resolve(), args.markdown_output.resolve(), args.details_output.resolve()]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Refusing to overwrite P31 outputs")
    if args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir()):
        raise FileExistsError("Refusing to overwrite P31 TensorBoard directory")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    model_paths: dict[str, Path] = {}
    for value in args.model:
        if "=" not in value:
            raise ValueError("--model must be NAME=CHECKPOINT")
        name, path = value.split("=", 1)
        if not name or name in model_paths:
            raise ValueError(f"Duplicate or empty model name: {name!r}")
        model_paths[name] = Path(path).resolve()
    device = _choose_device(args.device)
    split_paths = {
        split: (getattr(args, f"{split}_dataset").resolve(), getattr(args, f"{split}_metadata").resolve())
        for split in ("train", "calibration", "validation")
    }
    tensors_by_split: dict[str, dict[str, torch.Tensor]] = {}
    for split, (dataset, metadata) in split_paths.items():
        tensors_by_split[split], _ = load_dataset(dataset, metadata, split)
        # ``route_length_m`` is a public geometry feature used only by this
        # audit and is intentionally not part of the trainer's model tensor
        # contract.  Read it separately from the immutable NPZ archive.
        with np.load(dataset, allow_pickle=False) as archive:
            if "route_length_m" not in archive.files:
                raise ValueError(f"{dataset} is missing route_length_m")
            tensors_by_split[split]["route_length_m"] = torch.from_numpy(
                np.asarray(archive["route_length_m"], dtype=np.float32)
            )
    result: dict[str, Any] = {
        "audit_type": "dn_mpc_p31_route_utility_label_calibration",
        "development_only": True,
        "locked_test_opened": False,
        "device": str(device),
        "route_switch_penalty": {"available": False, "reason": "archive lacks previous executed route identity"},
        "archives": {
            split: {"dataset": str(dataset), "dataset_sha256": _sha256(dataset), "metadata": str(metadata), "metadata_sha256": _sha256(metadata)}
            for split, (dataset, metadata) in split_paths.items()
        },
        "models": {},
    }
    detail_rows: list[dict[str, Any]] = []
    with SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=1) as writer:
        writer.add_text("Config/utility_label", json.dumps({"normalization": {"progress": 0.3, "escape": 2.0, "length": 10.0}, "grid": WEIGHT_GRID}, sort_keys=True), 0)
        writer.add_text("Provenance/archives", json.dumps(result["archives"], sort_keys=True), 0)
        writer.add_text("Contract/route_switch_penalty", json.dumps(result["route_switch_penalty"], sort_keys=True), 0)
        for name, checkpoint in model_paths.items():
            model = _load_model(checkpoint, device)
            split_groups: dict[str, list[dict[str, Any]]] = {}
            for split, tensors in tensors_by_split.items():
                predictions = _predict(model, tensors, device, args.batch_size)
                split_groups[split] = _records(tensors, predictions)
            calibration_reports = [(weights, _evaluate(split_groups["calibration"], weights)) for weights in _grid()]
            selected_weights = _select(calibration_reports)
            splits_report: dict[str, Any] = {}
            for split, groups in split_groups.items():
                baseline = _evaluate(groups, {"length": 0.0, "escape": 0.0, "cbf": 0.0})
                selected = _evaluate(groups, selected_weights)
                splits_report[split] = {"baseline": baseline, "selected": selected}
                writer.add_scalar(f"Utility/{name}/baseline_exact_top1/{split}", float(baseline["exact_top1"] or 0.0), 0)
                writer.add_scalar(f"Utility/{name}/selected_exact_top1/{split}", float(selected["exact_top1"] or 0.0), 0)
                writer.add_scalar(f"Utility/{name}/selected_informative_top1/{split}", float(selected["informative_exact_top1"] or 0.0), 0)
                writer.add_scalar(f"Utility/{name}/selected_pairwise/{split}", float(selected["pairwise_agreement"] or 0.0), 0)
                writer.add_scalar(f"Utility/{name}/selected_agreement/{split}", float(selected["selected_agreement"] or 0.0), 0)
            result["models"][name] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "selected_weights": selected_weights,
                "calibration_selected": _evaluate(split_groups["calibration"], selected_weights),
                "splits": splits_report,
            }
            for split, groups in split_groups.items():
                for group in groups:
                    for candidate in group["candidates"]:
                        detail_rows.append({"model": name, "split": split, "scenario_index": group["scenario_index"], "time_index": group["time_index"], "selected": group["selected"], **candidate})
    output_paths[0].parent.mkdir(parents=True, exist_ok=True)
    output_paths[0].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths[1].parent.mkdir(parents=True, exist_ok=True)
    output_paths[1].write_text(_markdown(result), encoding="utf-8")
    output_paths[2].parent.mkdir(parents=True, exist_ok=True)
    with output_paths[2].open("w", newline="", encoding="utf-8") as handle:
        if detail_rows:
            writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
            writer.writeheader()
            writer.writerows(detail_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
