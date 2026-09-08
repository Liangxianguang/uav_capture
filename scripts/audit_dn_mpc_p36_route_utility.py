"""Calibrate route utility with the P36 executed-route identity contract.

This is an offline-only extension of the P31 diagnostic.  For P39 archives it
uses the analytic planner's previous selected candidate for the route-switch
penalty; legacy P36 archives fall back to the previous executed route and are
marked as such.  It never executes an action, changes the planner, creates a
Ledger, or opens a locked test.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from audit_dn_mpc_p31_route_utility_label import (  # noqa: E402
    GROUP_TIE_MARGIN,
    RANKING_HORIZON_INDEX,
    ROUTE_COUNT,
    _choose_device,
    _load_model,
    _predict,
    _sha256,
    load_dataset,
)

DEFAULT_WEIGHT_GRID = {
    "length": (0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
    "escape": (0.0, 0.1, 0.2, 0.5),
    "cbf": (0.0, 0.1, 0.2, 0.5, 1.0),
    "switch": (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0),
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _records(tensors: dict[str, torch.Tensor], predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    arrays = {name: value.numpy() for name, value in tensors.items()}
    runtime = arrays["sample_type"] == 0
    scenario = arrays["scenario_index"]
    time_index = arrays["time_index"]
    candidate = arrays["route_candidate_index"]
    planner_fields = (
        "planner_selected_candidate_index",
        "previous_selected_candidate_index",
        "planner_route_switch_outcome",
    )
    present_planner_fields = [field in arrays for field in planner_fields]
    if any(present_planner_fields) and not all(present_planner_fields):
        raise ValueError("Planner selection identity contract is only partially present")
    planner_identity_available = all(present_planner_fields)
    previous = (
        arrays["previous_selected_candidate_index"]
        if planner_identity_available
        else arrays["previous_executed_route_index"]
    )
    selected_field = (
        "planner_selected_candidate_index"
        if planner_identity_available
        else "selected_candidate_index"
    )
    groups: list[dict[str, Any]] = []
    keys = sorted({(int(scenario[i]), int(time_index[i])) for i in np.flatnonzero(runtime)})
    for key in keys:
        group = runtime & (scenario == key[0]) & (time_index == key[1])
        previous_values = previous[group]
        previous_route = int(previous_values[0]) if previous_values.size else -1
        if previous_values.size and not np.all(previous_values == previous_route):
            raise ValueError(f"previous route is not state-consistent for group {key}")
        candidates: list[dict[str, Any]] = []
        for candidate_id in range(ROUTE_COUNT):
            rows = group & (candidate == candidate_id)
            if not np.any(rows):
                continue
            truth_progress = float(np.mean(arrays["labels_route_progress"][rows, RANKING_HORIZON_INDEX]))
            predicted_progress = float(np.mean(predictions["route_progress"][rows, RANKING_HORIZON_INDEX]))
            truth_escape = float(np.mean(arrays["labels_target_escape_cost"][rows, RANKING_HORIZON_INDEX]))
            predicted_escape = float(np.mean(np.linalg.norm(predictions["mean"][rows, RANKING_HORIZON_INDEX], axis=-1)))
            truth_cbf = float(np.mean(arrays["labels_cbf_feasible"][rows, : RANKING_HORIZON_INDEX + 1]))
            predicted_cbf = float(np.mean(_sigmoid(predictions["cbf_feasibility_logit"][rows, : RANKING_HORIZON_INDEX + 1])))
            candidates.append(
                {
                    "candidate": candidate_id,
                    "eligible": bool(
                        np.all(arrays["route_geometry_valid"][rows] >= 0.5)
                        and np.all(arrays["labels_cbf_feasible"][rows, 0] >= 0.5)
                    ),
                    "truth_progress": truth_progress,
                    "predicted_progress": predicted_progress,
                    "truth_escape": truth_escape,
                    "predicted_escape": predicted_escape,
                    "truth_cbf": truth_cbf,
                    "predicted_cbf": predicted_cbf,
                    "route_length": float(np.mean(arrays["route_length_m"][rows])),
                    "switch_penalty": float(previous_route >= 0 and candidate_id != previous_route),
                }
            )
        selected = arrays[selected_field][group]
        selected_candidate = int(selected[0]) if selected.size else -1
        groups.append(
            {
                "scenario_index": key[0],
                "time_index": key[1],
                "previous_route": previous_route,
                "selected": selected_candidate,
                "planner_identity_available": planner_identity_available,
                "candidates": candidates,
            }
        )
    return groups


def _utility(item: dict[str, Any], weights: dict[str, float], *, predicted: bool) -> float:
    prefix = "predicted" if predicted else "truth"
    return (
        float(item[f"{prefix}_progress"]) / 0.3
        - weights["length"] * float(item["route_length"]) / 10.0
        - weights["escape"] * float(item[f"{prefix}_escape"]) / 2.0
        + weights["cbf"] * float(item[f"{prefix}_cbf"])
        - weights["switch"] * float(item["switch_penalty"])
    )


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
            for right_item, right_value in truth_values[left_index + 1 :]:
                if abs(left_value - right_value) <= GROUP_TIE_MARGIN:
                    continue
                pair_total += 1
                left_prediction = _utility(truth_values[left_index][0], weights, predicted=True)
                right_prediction = _utility(right_item, weights, predicted=True)
                pair_correct += int((left_prediction - right_prediction) * (left_value - right_value) > 0.0)
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


def _grid(weight_grid: dict[str, tuple[float, ...]]) -> list[dict[str, float]]:
    return [dict(zip(weight_grid, values)) for values in itertools.product(*weight_grid.values())]


def _select(reports: list[tuple[dict[str, float], dict[str, Any]]]) -> dict[str, float]:
    valid = [item for item in reports if item[1]["informative_exact_top1"] is not None]
    if not valid:
        raise ValueError("calibration has no informative groups")
    return max(
        valid,
        key=lambda item: (
            item[1]["informative_exact_top1"],
            item[1]["pairwise_agreement"] or -1.0,
            item[1]["exact_top1"] or -1.0,
            -sum(item[0].values()),
        ),
    )[0]


def _at_grid_boundary(weights: dict[str, float], weight_grid: dict[str, tuple[float, ...]]) -> dict[str, bool]:
    return {
        name: float(weights[name]) == max(float(value) for value in values)
        for name, values in weight_grid.items()
    }


def _parse_switch_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise ValueError("--switch-grid must be a comma-separated list of finite non-negative numbers") from exc
    if not values or any(not np.isfinite(item) or item < 0.0 for item in values):
        raise ValueError("--switch-grid must be a comma-separated list of finite non-negative numbers")
    return values


def _parse_nonnegative_grid(value: str, *, name: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of finite non-negative numbers") from exc
    if not values or any(not np.isfinite(item) or item < 0.0 for item in values):
        raise ValueError(f"{name} must be a comma-separated list of finite non-negative numbers")
    return values


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
    parser.add_argument("--dataset-version", default="dn_mpc_route_identity_chunk5_p36_route_switch_v1")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--switch-grid",
        default=",".join(str(value) for value in DEFAULT_WEIGHT_GRID["switch"]),
        help="comma-separated non-negative switch-penalty weights",
    )
    parser.add_argument(
        "--length-grid",
        default=",".join(str(value) for value in DEFAULT_WEIGHT_GRID["length"]),
        help="comma-separated non-negative route-length weights",
    )
    parser.add_argument(
        "--cbf-grid",
        default=",".join(str(value) for value in DEFAULT_WEIGHT_GRID["cbf"]),
        help="comma-separated non-negative CBF-feasibility weights",
    )
    args = parser.parse_args()
    weight_grid = dict(DEFAULT_WEIGHT_GRID)
    weight_grid["switch"] = _parse_nonnegative_grid(args.switch_grid, name="--switch-grid")
    weight_grid["length"] = _parse_nonnegative_grid(args.length_grid, name="--length-grid")
    weight_grid["cbf"] = _parse_nonnegative_grid(args.cbf_grid, name="--cbf-grid")
    outputs = [args.output.resolve(), args.markdown_output.resolve(), args.details_output.resolve()]
    if any(path.exists() for path in outputs) or (args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir())):
        raise FileExistsError("refusing to overwrite P36 utility outputs")
    device = _choose_device(args.device)
    models: dict[str, Path] = {}
    for value in args.model:
        name, separator, path = value.partition("=")
        if not separator or not name or name in models:
            raise ValueError("--model must be unique NAME=CHECKPOINT")
        models[name] = Path(path).resolve()
    split_paths = {
        split: (getattr(args, f"{split}_dataset").resolve(), getattr(args, f"{split}_metadata").resolve())
        for split in ("train", "calibration", "validation")
    }
    tensors_by_split: dict[str, dict[str, torch.Tensor]] = {}
    for split, (dataset, metadata) in split_paths.items():
        tensors_by_split[split], _ = load_dataset(dataset, metadata, split)
        with np.load(dataset, allow_pickle=False) as archive:
            for field in ("route_length_m", "previous_executed_route_index", "labels_target_escape_cost"):
                if field not in archive.files:
                    raise ValueError(f"{dataset} is missing {field}")
                tensors_by_split[split][field] = torch.from_numpy(np.asarray(archive[field], dtype=np.float32))
    planner_identity_by_split = {
        split: all(
            field in tensors_by_split[split]
            for field in (
                "planner_selected_candidate_index",
                "previous_selected_candidate_index",
                "planner_route_switch_outcome",
            )
        )
        for split in tensors_by_split
    }
    if len(set(planner_identity_by_split.values())) > 1:
        raise ValueError(
            f"Planner selection identity contract differs across splits: {planner_identity_by_split}"
        )
    planner_identity_available = next(iter(planner_identity_by_split.values()))
    result: dict[str, Any] = {
        "audit_type": "dn_mpc_route_utility_calibration",
        "dataset_version": str(args.dataset_version),
        "development_only": True,
        "locked_test_opened": False,
        "device": str(device),
        "route_switch_penalty": {
            "available": True,
            "definition": (
                "candidate_id != previous_selected_candidate_index"
                if planner_identity_available
                else "candidate_id != previous_executed_route_index"
            ),
            "planner_selection_identity_used": planner_identity_available,
            "legacy_fallback": not planner_identity_available,
        },
        "weight_grid": weight_grid,
        "archives": {
            split: {"dataset": str(dataset), "dataset_sha256": _sha256(dataset), "metadata": str(metadata), "metadata_sha256": _sha256(metadata)}
            for split, (dataset, metadata) in split_paths.items()
        },
        "models": {},
    }
    details: list[dict[str, Any]] = []
    with SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=1) as writer:
        writer.add_text("Config/utility_label", json.dumps(weight_grid, sort_keys=True), 0)
        writer.add_text("Contract/route_switch_penalty", json.dumps(result["route_switch_penalty"], sort_keys=True), 0)
        for name, checkpoint in models.items():
            model = _load_model(checkpoint, device)
            groups_by_split: dict[str, list[dict[str, Any]]] = {}
            for split, tensors in tensors_by_split.items():
                groups_by_split[split] = _records(tensors, _predict(model, tensors, device, args.batch_size))
            selected = _select([ (weights, _evaluate(groups_by_split["calibration"], weights)) for weights in _grid(weight_grid) ])
            split_report: dict[str, Any] = {}
            for split, groups in groups_by_split.items():
                baseline_weights = {key: 0.0 for key in weight_grid}
                baseline = _evaluate(groups, baseline_weights)
                calibrated = _evaluate(groups, selected)
                split_report[split] = {"baseline": baseline, "selected": calibrated}
                for metric, value in (("exact_top1", calibrated["exact_top1"]), ("informative_top1", calibrated["informative_exact_top1"]), ("pairwise", calibrated["pairwise_agreement"]), ("selected_agreement", calibrated["selected_agreement"])):
                    writer.add_scalar(f"Utility/{name}/{metric}/{split}", float(value or 0.0), 0)
                for group in groups:
                    for candidate in group["candidates"]:
                        details.append({"model": name, "split": split, "scenario_index": group["scenario_index"], "time_index": group["time_index"], "previous_route": group["previous_route"], "selected": group["selected"], **candidate})
            result["models"][name] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "selected_weights": selected,
                "selected_at_grid_boundary": _at_grid_boundary(selected, weight_grid),
                "splits": split_report,
            }
            writer.add_text(
                f"Utility/{name}/selected_weights",
                json.dumps({"weights": selected, "at_grid_boundary": _at_grid_boundary(selected, weight_grid)}, sort_keys=True),
                0,
            )
    lines = [
        f"# DN-MPC Route Utility Calibration ({args.dataset_version})",
        "",
        "**Status:** development-only; offline-only; no action executed.",
        "",
        "The utility adds a measured penalty against the planner's previous selected candidate when the P39 identity contract is present; legacy archives use the previous executed route. Unknown rows are explicit CBF abstentions and are excluded from candidate ranking.",
        "",
        "| Model | Selected (length, escape, cbf, switch) | Validation exact | Validation informative | Validation pairwise | Validation selected |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, model_report in result["models"].items():
        weights = model_report["selected_weights"]
        metrics = model_report["splits"]["validation"]["selected"]
        lines.append(f"| {name} | `({weights['length']:g}, {weights['escape']:g}, {weights['cbf']:g}, {weights['switch']:g})` | {metrics['exact_top1']:.2%} | {metrics['informative_exact_top1']:.2%} | {metrics['pairwise_agreement']:.2%} | {metrics['selected_agreement']:.2%} |")
    lines += ["", "This is a calibration diagnostic only. It does not authorize JEPA route override, Ledger-Lite, or a locked benchmark.", ""]
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].write_text("\n".join(lines), encoding="utf-8")
    outputs[2].parent.mkdir(parents=True, exist_ok=True)
    with outputs[2].open("w", newline="", encoding="utf-8") as handle:
        if details:
            writer = csv.DictWriter(handle, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
