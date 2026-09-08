"""Audit a safety-first hierarchical route selector offline.

The selector treats geometry and verified first-step CBF feasibility as hard
eligibility gates.  It then ranks eligible candidates by progress minus target
escape cost.  Route length and candidate-relative switching are tie-breakers,
not freely tuned additive rewards.  Calibration chooses only the escape weight
and primary-score tie band; no action is executed.
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
from audit_dn_mpc_p36_route_utility import (  # noqa: E402
    _choose_device,
    _load_model,
    _predict,
    _records,
    _sha256,
    load_dataset,
)


ROUTE_COUNT = 12
RANKING_HORIZON_INDEX = 2
ESCAPE_WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
TIE_BAND_GRID = (0.0, 0.005, 0.01, 0.02, 0.05)


def _load_split(dataset: Path, metadata: Path, split: str) -> dict[str, torch.Tensor]:
    tensors, _ = load_dataset(dataset, metadata, split)
    with np.load(dataset, allow_pickle=False) as archive:
        for field in (
            "route_length_m",
            "previous_executed_route_index",
            "labels_target_escape_cost",
            "planner_selected_candidate_index",
            "previous_selected_candidate_index",
            "planner_route_switch_outcome",
        ):
            if field not in archive.files:
                raise ValueError(f"{dataset} is missing {field}")
            value = np.asarray(archive[field], dtype=np.float32)
            tensors[field] = torch.from_numpy(value)
    return tensors


def _scales(groups: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {name: [] for name in ("progress", "escape", "route_length")}
    for group in groups:
        for item in group["candidates"]:
            values["progress"].append(abs(float(item["truth_progress"])))
            values["escape"].append(abs(float(item["truth_escape"])))
            values["route_length"].append(abs(float(item["route_length"])))
    if not values["progress"]:
        raise ValueError("Calibration has no eligible candidates")
    result = {
        name: max(float(np.quantile(np.asarray(items), 0.95)), 1.0e-6)
        for name, items in values.items()
    }
    result["progress_p95_abs"] = result["progress"]
    result["escape_p95_abs"] = result["escape"]
    result["route_length_p95"] = result["route_length"]
    return result


def _primary(item: dict[str, Any], scales: dict[str, float], *, predicted: bool, escape_weight: float) -> float:
    prefix = "predicted" if predicted else "truth"
    return float(item[f"{prefix}_progress"]) / scales["progress"] - float(escape_weight) * float(item[f"{prefix}_escape"]) / scales["escape"]


def _select_candidate(
    candidates: list[dict[str, Any]],
    scales: dict[str, float],
    *,
    predicted: bool,
    escape_weight: float,
    tie_band: float,
) -> int:
    scored = [(item, _primary(item, scales, predicted=predicted, escape_weight=escape_weight)) for item in candidates]
    best_primary = max(value for _item, value in scored)
    tied = [item for item, value in scored if best_primary - value <= float(tie_band)]
    # Secondary terms are deliberately lexicographic and only operate within
    # the primary tie band.  This prevents a large route-length/switch weight
    # from overpowering capture progress.
    tied.sort(key=lambda item: (float(item["route_length"]), float(item["switch_penalty"]), int(item["candidate"])))
    return int(tied[0]["candidate"])


def _evaluate(groups: list[dict[str, Any]], scales: dict[str, float], *, escape_weight: float, tie_band: float) -> dict[str, Any]:
    group_count = exact = selected_matches = selected_total = 0
    for group in groups:
        candidates = [item for item in group["candidates"] if item["eligible"]]
        if len(candidates) < 2:
            continue
        truth_best = _select_candidate(candidates, scales, predicted=False, escape_weight=escape_weight, tie_band=tie_band)
        predicted_best = _select_candidate(candidates, scales, predicted=True, escape_weight=escape_weight, tie_band=tie_band)
        group_count += 1
        exact += int(predicted_best == truth_best)
        selected = int(group["selected"])
        if 0 <= selected < ROUTE_COUNT:
            selected_total += 1
            selected_matches += int(predicted_best == selected)
    return {
        "group_count": group_count,
        "exact_top1": exact / group_count if group_count else None,
        "selected_comparison_group_count": selected_total,
        "selected_agreement": selected_matches / selected_total if selected_total else None,
        "escape_weight": float(escape_weight),
        "tie_band": float(tie_band),
    }


def _select_parameters(groups: list[dict[str, Any]], scales: dict[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
    reports = [
        _evaluate(groups, scales, escape_weight=escape_weight, tie_band=tie_band)
        for escape_weight, tie_band in itertools.product(ESCAPE_WEIGHT_GRID, TIE_BAND_GRID)
    ]
    valid = [item for item in reports if item["exact_top1"] is not None]
    selected = max(
        valid,
        key=lambda item: (
            item["exact_top1"],
            item["selected_agreement"] or -1.0,
            -item["tie_band"],
            -item["escape_weight"],
        ),
    )
    return {"escape_weight": selected["escape_weight"], "tie_band": selected["tie_band"]}, selected


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# DN-MPC P45 Safety-First Hierarchy Audit",
        "",
        "**Phase:** development-only, offline-only",
        "",
        "Eligibility is a hard geometry/verified-first-step-CBF gate. Progress minus escape is the primary score; route length and candidate-relative switch are lexicographic tie-breakers inside a fixed primary tie band.",
        "",
        "## Calibration scales",
        "",
        f"- Progress P95 absolute scale: `{result['scales']['progress']:.6f}`",
        f"- Escape P95 absolute scale: `{result['scales']['escape']:.6f}`",
        f"- Route-length P95 scale: `{result['scales']['route_length']:.6f}`",
        "",
        "## Checkpoint comparison",
        "",
        "| Model | Calibration `(escape weight, tie band)` | Validation exact | Validation vs planner | Calibration exact |",
        "|---|---|---:|---:|---:|",
    ]
    for name, report in result["models"].items():
        params = report["selected_parameters"]
        validation = report["validation"]
        calibration = report["calibration"]
        lines.append(
            f"| {name} | `({params['escape_weight']:g}, {params['tie_band']:g})` | "
            f"{validation['exact_top1']:.2%} | {validation['selected_agreement']:.2%} | {calibration['exact_top1']:.2%} |"
        )
    lines += [
        "",
        "## Gate decision",
        "",
        "- No action was executed; CBF margins and stale/OOD/non-finite gates were unchanged.",
        "- This is an offline route-selection audit, not a safe-capture result.",
        "- Online JEPA override, Ledger-Lite, paired replay and locked testing remain closed until parameter stability and independent safety traces pass.",
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
    outputs = [args.output.resolve(), args.markdown_output.resolve(), args.details_output.resolve()]
    if any(path.exists() for path in outputs) or (args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir())):
        raise FileExistsError("Refusing to overwrite P45 outputs")
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
    tensors_by_split = {
        split: _load_split(dataset, metadata, split)
        for split, (dataset, metadata) in split_paths.items()
    }
    result: dict[str, Any] = {
        "audit_type": "dn_mpc_p45_safety_first_hierarchy",
        "development_only": True,
        "locked_test_opened": False,
        "device": str(device),
        "eligibility_contract": "route_geometry_valid AND labels_cbf_feasible[first_step]",
        "calibration_grids": {"escape_weight": ESCAPE_WEIGHT_GRID, "tie_band": TIE_BAND_GRID},
        "archives": {
            split: {
                "dataset": str(dataset),
                "dataset_sha256": _sha256(dataset),
                "metadata": str(metadata),
                "metadata_sha256": _sha256(metadata),
            }
            for split, (dataset, metadata) in split_paths.items()
        },
        "models": {},
    }
    details: list[dict[str, Any]] = []
    with SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=1) as writer:
        writer.add_text("Config/hierarchy", json.dumps(result["calibration_grids"], sort_keys=True), 0)
        writer.add_text("Contract/eligibility", result["eligibility_contract"], 0)
        for name, checkpoint in models.items():
            model = _load_model(checkpoint, device)
            groups_by_split: dict[str, list[dict[str, Any]]] = {}
            for split, tensors in tensors_by_split.items():
                groups_by_split[split] = _records(tensors, _predict(model, tensors, device, args.batch_size))
            scales = _scales(groups_by_split["calibration"])
            parameters, calibration_selection = _select_parameters(groups_by_split["calibration"], scales)
            split_reports = {
                split: _evaluate(groups, scales, escape_weight=parameters["escape_weight"], tie_band=parameters["tie_band"])
                for split, groups in groups_by_split.items()
            }
            result["models"][name] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "selected_parameters": parameters,
                "calibration_selection": calibration_selection,
                "scales": scales,
                "calibration": split_reports["calibration"],
                "validation": split_reports["validation"],
                "train": split_reports["train"],
            }
            if "scales" not in result:
                result["scales"] = scales
            for split, report in split_reports.items():
                writer.add_scalar(f"Hierarchy/{name}/exact_top1/{split}", float(report["exact_top1"] or 0.0), 0)
                writer.add_scalar(f"Hierarchy/{name}/selected_agreement/{split}", float(report["selected_agreement"] or 0.0), 0)
            writer.add_text(f"Hierarchy/{name}/parameters", json.dumps({"parameters": parameters, "scales": scales}, sort_keys=True), 0)
            for group in groups_by_split["validation"]:
                for candidate in group["candidates"]:
                    details.append({"model": name, "split": "validation", "scenario_index": group["scenario_index"], "time_index": group["time_index"], "selected": group["selected"], **candidate})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    with args.details_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]) if details else ["model", "split"])
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
