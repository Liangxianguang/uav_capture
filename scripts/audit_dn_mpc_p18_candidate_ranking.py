"""Offline P19 audit for the P18 route-JEPA candidate ranker.

The audit is deliberately development-only.  It checks the serialized
five-step route contract, runs the checkpoint on train/validation/calibration
archives, measures route-progress ranking direction, and records which online
promotion prerequisites are still absent.  It does not execute an action and
does not alter any CBF or Ledger policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.prediction import build_action_conditioned_predictor  # noqa: E402
from train_route_identity_jepa import load_dataset  # noqa: E402


AUDIT_TYPE = "dn_mpc_p19_p18_candidate_ranking_audit"
MODEL_TYPE = "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2"
REQUIRED_AUXILIARY = (
    "route_progress",
    "route_identity_logits",
    "route_side_logits",
    "route_geometry_logit",
    "route_termination_logit",
    "stopping_distance",
    "obstacle_ttc",
    "boundary_ttc",
    "pairwise_ttc_risk",
    "acceleration_slack",
)
CONTRACT_FIELDS = (
    "dataset_version",
    "candidate_profile",
    "candidate_count",
    "history_length",
    "chunk_length_steps",
    "horizon_steps",
    "action_scale",
    "interaction_action_conditioned_route_chunk",
    "pairwise_action_conditioned_route_chunk",
)
ROUTE_LABELS = (
    "nominal",
    "left_detour",
    "right_detour",
    "upper_detour",
    "lower_detour",
    "radial_out",
    "formation_split",
    "formation_contract",
    "braking",
    "safe_intercept",
    "visibility_hold",
    "verified_safe_hold",
)
DEFAULT_RANKING_HORIZON = 2
DEFAULT_RANKING_MARGIN = 0.005
PROMOTION_TOP1_MIN = 0.50
PROMOTION_PAIRWISE_MIN = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--calibration-dataset", type=Path, required=True)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--ranking-horizon-index", type=int, default=None)
    parser.add_argument("--ranking-margin", type=float, default=DEFAULT_RANKING_MARGIN)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line[3:] for line in result.stdout.splitlines() if len(line) >= 4]


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if name == "cuda" or name == "auto" and torch.cuda.is_available() else "cpu")


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _safe_mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def _model_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unexpected checkpoint model_type: {checkpoint.get('model_type')!r}")
    config = checkpoint.get("model")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing model configuration")
    expected = {
        "route_chunk_length": 5,
        "route_interaction_chunk_dim": 9,
        "route_candidate_count": 12,
        "route_side_count": 12,
    }
    mismatch = {key: {"actual": config.get(key), "expected": value} for key, value in expected.items() if config.get(key) != value}
    if mismatch:
        raise ValueError(f"P18 model contract mismatch: {mismatch}")
    return {key: config.get(key) for key in sorted(config)}


def _paired_contract(metadata_by_split: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = metadata_by_split["train"]
    mismatches: dict[str, dict[str, Any]] = {}
    for split, metadata in metadata_by_split.items():
        for field in CONTRACT_FIELDS:
            # Calibration is intentionally an independently generated archive;
            # its dataset version may differ while the tensor/action contract
            # must remain identical.
            if field == "dataset_version":
                continue
            if metadata.get(field) != reference.get(field):
                mismatches.setdefault(field, {})[split] = metadata.get(field)
    if mismatches:
        raise ValueError(f"Archive contracts differ: {mismatches}")
    required = {
        "candidate_profile": "obstacle_route_v1",
        "candidate_count": 12,
        "history_length": 8,
        "chunk_length_steps": 5,
        "interaction_action_conditioned_route_chunk": True,
        "pairwise_action_conditioned_route_chunk": True,
    }
    errors = {field: {"actual": reference.get(field), "expected": value} for field, value in required.items() if reference.get(field) != value}
    if errors:
        raise ValueError(f"Archive contract does not satisfy P18 requirements: {errors}")
    return {
        "structural_fields": {field: reference.get(field) for field in CONTRACT_FIELDS if field != "dataset_version"},
        "dataset_versions": {split: metadata.get("dataset_version") for split, metadata in metadata_by_split.items()},
    }


def _predict(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = int(tensors["inputs"].shape[0])
    collected: dict[str, list[np.ndarray]] = {name: [] for name in REQUIRED_AUXILIARY}
    finite = True
    max_abs = 0.0
    route_interaction_dim = int(getattr(model, "route_interaction_chunk_dim", 0))
    if route_interaction_dim != 9:
        raise ValueError(f"P18 audit requires route_interaction_chunk_dim=9, got {route_interaction_dim}")
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            args = (
                tensors["inputs"][start:end].to(device),
                tensors["action_history"][start:end].to(device),
                tensors["route_action_chunk"][start:end].to(device),
                tensors["route_candidate_index"][start:end].to(device),
                tensors["route_side_index"][start:end].to(device),
                tensors["route_pairwise_relative_action_chunk"][start:end].to(device),
            )
            _mean, _log_variance, _latent, auxiliary = model.forward_multitask(*args)
            for name in REQUIRED_AUXILIARY:
                value = auxiliary[name].detach().cpu().numpy()
                if not np.isfinite(value).all():
                    finite = False
                if value.size:
                    max_abs = max(max_abs, float(np.max(np.abs(value))))
                collected[name].append(value)
    outputs = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    return outputs, {"finite": finite, "max_abs_output": max_abs}


def _group_keys(tensors: dict[str, torch.Tensor], runtime: np.ndarray) -> list[tuple[int, int]]:
    scenarios = tensors["scenario_index"].numpy()
    times = tensors["time_index"].numpy()
    return sorted({(int(scenarios[index]), int(times[index])) for index in np.flatnonzero(runtime)})


def _rank_split(
    split: str,
    tensors: dict[str, torch.Tensor],
    outputs: dict[str, np.ndarray],
    ranking_horizon_index: int,
    ranking_margin: float,
    detail_writer: csv.DictWriter,
) -> dict[str, Any]:
    sample_type = tensors["sample_type"].numpy()
    candidate = tensors["route_candidate_index"].numpy()
    scenario = tensors["scenario_index"].numpy()
    time_index = tensors["time_index"].numpy()
    geometry = tensors["route_geometry_valid"].numpy() >= 0.5
    first_step_cbf = tensors["labels_cbf_feasible"].numpy()[:, 0] >= 0.5
    progress_truth = tensors["labels_route_progress"].numpy()[:, ranking_horizon_index]
    progress_pred = outputs["route_progress"][:, ranking_horizon_index]
    runtime = sample_type == 0
    keys = _group_keys(tensors, runtime)
    top1_correct = 0
    top1_total = 0
    pair_correct = 0
    pair_total = 0
    all_group_correlations: list[float] = []
    detail_rows = 0
    candidate_counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    group_records: list[dict[str, Any]] = []
    for key in keys:
        base_mask = runtime & (scenario == key[0]) & (time_index == key[1])
        ids = sorted(int(value) for value in np.unique(candidate[base_mask]) if 0 <= int(value) < 12)
        records: list[dict[str, Any]] = []
        for candidate_id in ids:
            row_mask = base_mask & (candidate == candidate_id)
            candidate_label = ROUTE_LABELS[candidate_id]
            branch_eligible = bool(np.all(geometry[row_mask]) and np.all(first_step_cbf[row_mask]))
            truth = float(np.mean(progress_truth[row_mask]))
            prediction = float(np.mean(progress_pred[row_mask]))
            candidate_counts["total"] += 1
            if branch_eligible:
                eligible_counts["eligible"] += 1
            else:
                if not np.all(geometry[row_mask]):
                    eligible_counts["geometry_invalid"] += 1
                if not np.all(first_step_cbf[row_mask]):
                    eligible_counts["first_step_cbf_infeasible"] += 1
            records.append({"candidate": candidate_id, "label": candidate_label, "eligible": branch_eligible, "truth": truth, "prediction": prediction})
        eligible_records = [record for record in records if record["eligible"] and math.isfinite(record["truth"]) and math.isfinite(record["prediction"])]
        if len(eligible_records) >= 2:
            truth_best = max(eligible_records, key=lambda item: (item["truth"], -item["candidate"]))
            prediction_best = max(eligible_records, key=lambda item: (item["prediction"], -item["candidate"]))
            top1_total += 1
            top1_correct += int(truth_best["candidate"] == prediction_best["candidate"])
            truths = [float(item["truth"]) for item in eligible_records]
            predictions = [float(item["prediction"]) for item in eligible_records]
            if len(set(truths)) > 1 and len(set(predictions)) > 1:
                correlation = float(np.corrcoef(np.asarray(truths), np.asarray(predictions))[0, 1])
                if math.isfinite(correlation):
                    all_group_correlations.append(correlation)
            for left_index, left in enumerate(eligible_records):
                for right in eligible_records[left_index + 1 :]:
                    delta = float(left["truth"] - right["truth"])
                    if abs(delta) <= ranking_margin:
                        continue
                    pair_total += 1
                    pair_correct += int((left["prediction"] - right["prediction"]) * delta > 0.0)
        for rank_truth, truth_record in enumerate(sorted(records, key=lambda item: (-item["truth"], item["candidate"])), start=1):
            prediction_rank = 1 + sum(
                (other["prediction"] > truth_record["prediction"])
                or (other["prediction"] == truth_record["prediction"] and other["candidate"] < truth_record["candidate"])
                for other in records
            )
            detail_writer.writerow(
                {
                    "split": split,
                    "scenario_index": key[0],
                    "time_index": key[1],
                    "candidate_index": truth_record["candidate"],
                    "candidate_label": truth_record["label"],
                    "eligible": int(truth_record["eligible"]),
                    "truth_progress": truth_record["truth"],
                    "predicted_progress": truth_record["prediction"],
                    "truth_rank": rank_truth,
                    "predicted_rank": prediction_rank,
                }
            )
            detail_rows += 1
        group_records.append({"scenario_index": key[0], "time_index": key[1], "candidate_count": len(records), "eligible_count": len(eligible_records)})
    runtime_count = int(runtime.sum())
    route_identity_logits = outputs["route_identity_logits"]
    route_side_logits = outputs["route_side_logits"]
    runtime_indices = np.flatnonzero(runtime & (candidate >= 0) & (candidate < 12))
    route_identity_accuracy = float(np.mean(np.argmax(route_identity_logits[runtime_indices], axis=1) == candidate[runtime_indices])) if runtime_indices.size else None
    route_side_accuracy = float(np.mean(np.argmax(route_side_logits[runtime_indices], axis=1) == tensors["route_side_index"].numpy()[runtime_indices])) if runtime_indices.size else None
    geometry_truth = tensors["route_geometry_valid"].numpy()[runtime_indices] >= 0.5
    geometry_accuracy = float(np.mean((outputs["route_geometry_logit"][runtime_indices] >= 0.0) == geometry_truth)) if runtime_indices.size else None
    return {
        "split": split,
        "sample_count": int(len(sample_type)),
        "runtime_sample_count": int(runtime_count),
        "runtime_group_count": len(keys),
        "candidate_counts": dict(candidate_counts),
        "eligibility_counts": dict(eligible_counts),
        "group_min_eligible_candidates": int(min((record["eligible_count"] for record in group_records), default=0)),
        "group_zero_eligible_count": int(sum(record["eligible_count"] == 0 for record in group_records)),
        "top1_correct": int(top1_correct),
        "top1_total": int(top1_total),
        "top1_agreement": float(top1_correct / top1_total) if top1_total else None,
        "pairwise_correct": int(pair_correct),
        "pairwise_total": int(pair_total),
        "pairwise_agreement": float(pair_correct / pair_total) if pair_total else None,
        "mean_group_progress_correlation": _safe_mean(all_group_correlations),
        "route_identity_accuracy": route_identity_accuracy,
        "route_side_accuracy": route_side_accuracy,
        "route_geometry_accuracy": geometry_accuracy,
        "detail_rows": detail_rows,
    }


def _counterfactual_contract(tensors_by_split: dict[str, dict[str, torch.Tensor]], metadata_by_split: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required_arrays = {"labels_cbf_feasible", "labels_cbf_min_slack", "labels_cbf_correction", "labels_cbf_intervention"}
    per_split: dict[str, Any] = {}
    selected_present = False
    nominal_present = True
    safe_hold_present = True
    for split, tensors in tensors_by_split.items():
        candidate = tensors["route_candidate_index"].numpy()
        sample_type = tensors["sample_type"].numpy()
        runtime = sample_type == 0
        nominal_rows = int(np.sum(runtime & (candidate == 0)))
        safe_hold_rows = int(np.sum(runtime & (candidate == 11)))
        per_split[split] = {
            "required_cbf_arrays_present": required_arrays.issubset(tensors),
            "runtime_nominal_rows": nominal_rows,
            "runtime_verified_safe_hold_rows": safe_hold_rows,
            "selected_candidate_index_field_present": False,
            "independent_selected_counterfactual_present": False,
        }
        nominal_present &= nominal_rows > 0
        safe_hold_present &= safe_hold_rows > 0
    return {
        "nominal_counterfactual_present": nominal_present,
        "verified_safe_hold_counterfactual_present": safe_hold_present,
        "selected_counterfactual_present": selected_present,
        "independent_selected_nominal_safe_hold_contract": bool(selected_present and nominal_present and safe_hold_present),
        "per_split": per_split,
        "reason_selected_missing": "P17 archives contain candidate rows but no per-step selected candidate and independent selected-CBF trace.",
    }


def _markdown(result: dict[str, Any]) -> str:
    gates = result["gates"]
    decision = "PROMOTE TO ONLINE REPLAY" if result["promotion_eligible"] else "STOP BEFORE ONLINE INTEGRATION"
    lines = [
        "# DN-MPC P19 P18 Candidate-Ranking Audit",
        "",
        f"- Audit type: `{result['audit_type']}`",
        "- Protocol: development-only; no action was executed",
        f"- Decision: **{decision}**",
        f"- Checkpoint SHA-256: `{result['checkpoint_sha256']}`",
        f"- Git revision: `{result['provenance']['git_revision']}`",
        "",
        "## Contract",
        "",
        f"- Model type: `{result['model_type']}`",
        f"- Route chunk shape contract: `[N,5,3]`; pairwise interaction: `[N,5,9]`",
        f"- Contract gate: `{gates['contract']}`",
        f"- Finite output gate: `{gates['finite_outputs']}`",
        "",
        "## Ranking Results",
        "",
        "| Split | groups | top-1 agreement | pairwise agreement | route identity | geometry accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "calibration"):
        item = result["splits"][split]
        def fmt(name: str) -> str:
            value = item.get(name)
            return "n/a" if value is None else f"{float(value) * 100:.2f}%"
        lines.append(
            f"| {split} | {item['runtime_group_count']} | {fmt('top1_agreement')} | {fmt('pairwise_agreement')} | {fmt('route_identity_accuracy')} | {fmt('route_geometry_accuracy')} |"
        )
    validation = result["splits"]["validation"]
    lines += [
        "",
        f"The validation route-progress top-1 gate requires >= {PROMOTION_TOP1_MIN:.0%}; observed `{validation['top1_agreement']}`.",
        f"The validation informative pairwise gate requires >= {PROMOTION_PAIRWISE_MIN:.0%}; observed `{validation['pairwise_agreement']}`.",
        "",
        "## Counterfactual Traceability",
        "",
        f"- Nominal rows present: `{result['counterfactual']['nominal_counterfactual_present']}`",
        f"- Verified safe-hold rows present: `{result['counterfactual']['verified_safe_hold_counterfactual_present']}`",
        f"- Selected-candidate independent CBF trace present: `{result['counterfactual']['selected_counterfactual_present']}`",
        f"- Three-way counterfactual gate: `{gates['counterfactual_traceability']}`",
        "",
        "## Promotion Gates",
        "",
    ]
    for name, value in gates.items():
        lines.append(f"- `{name}`: `{value}`")
    lines += [
        "",
        "The checkpoint remains offline-only until selected/nominal/safe-hold are recorded as independent CBF counterfactuals and OOD/disagreement fields are bound to a fresh calibration ledger.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    paths = {
        "train": (args.train_dataset.resolve(), args.train_metadata.resolve()),
        "validation": (args.validation_dataset.resolve(), args.validation_metadata.resolve()),
        "calibration": (args.calibration_dataset.resolve(), args.calibration_metadata.resolve()),
    }
    outputs = [args.output.resolve(), args.markdown_output.resolve(), args.details_output.resolve(), args.tensorboard_logdir.resolve()]
    if args.ranking_margin < 0.0 or args.batch_size <= 0:
        raise ValueError("ranking margin must be non-negative and batch size must be positive")
    for output in outputs[:3]:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite output: {output}")
    if outputs[3].exists() and any(outputs[3].iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard logdir: {outputs[3]}")
    checkpoint = args.checkpoint.resolve()
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_config = _model_contract(checkpoint_data)
    metadata_by_split: dict[str, dict[str, Any]] = {}
    tensors_by_split: dict[str, dict[str, torch.Tensor]] = {}
    for split, (dataset, metadata) in paths.items():
        tensors, archive_metadata = load_dataset(dataset, metadata, split)
        metadata_by_split[split] = archive_metadata
        tensors_by_split[split] = tensors
        if tuple(tensors["route_action_chunk"].shape[1:]) != (5, 3):
            raise ValueError(f"{split} route_action_chunk is not [N,5,3]")
        if tuple(tensors["route_pairwise_relative_action_chunk"].shape[1:]) != (5, 9):
            raise ValueError(f"{split} pairwise route chunk is not [N,5,9]")
    contract = _paired_contract(metadata_by_split)
    ranking_horizon = checkpoint_data.get("route_ranking_horizon_index", DEFAULT_RANKING_HORIZON) if args.ranking_horizon_index is None else args.ranking_horizon_index
    if not 0 <= int(ranking_horizon) < 5:
        raise ValueError("ranking horizon index must be within [0,4]")
    device = choose_device(args.device)
    model = build_action_conditioned_predictor(checkpoint_data["model_type"], checkpoint_data["model"]).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    model.eval()
    prediction_diagnostics: dict[str, Any] = {}
    predictions_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split, tensors in tensors_by_split.items():
        predictions_by_split[split], prediction_diagnostics[split] = _predict(model, tensors, device, args.batch_size)
    details_path = args.details_output.resolve()
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", newline="", encoding="utf-8") as handle:
        detail_writer = csv.DictWriter(handle, fieldnames=("split", "scenario_index", "time_index", "candidate_index", "candidate_label", "eligible", "truth_progress", "predicted_progress", "truth_rank", "predicted_rank"))
        detail_writer.writeheader()
        split_reports = {
            split: _rank_split(split, tensors_by_split[split], predictions_by_split[split], int(ranking_horizon), args.ranking_margin, detail_writer)
            for split in ("train", "validation", "calibration")
        }
    counterfactual = _counterfactual_contract(tensors_by_split, metadata_by_split)
    validation = split_reports["validation"]
    ood_available = False
    disagreement_available = True
    gates = {
        "contract": True,
        "finite_outputs": all(bool(item["finite"]) for item in prediction_diagnostics.values()),
        "candidate_eligibility": all(item["group_zero_eligible_count"] == 0 for item in split_reports.values()),
        "score_direction": bool(validation["pairwise_agreement"] is not None and validation["pairwise_agreement"] >= PROMOTION_PAIRWISE_MIN),
        "top1_ranking": bool(validation["top1_agreement"] is not None and validation["top1_agreement"] >= PROMOTION_TOP1_MIN),
        "ood_disagreement_binding": bool(ood_available and disagreement_available),
        "counterfactual_traceability": bool(counterfactual["independent_selected_nominal_safe_hold_contract"]),
    }
    result: dict[str, Any] = {
        "audit_type": AUDIT_TYPE,
        "model_type": MODEL_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "promotion_eligible": bool(all(gates.values())),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_model_config": model_config,
        "archive_contract": contract,
        "ranking_horizon_index": int(ranking_horizon),
        "ranking_margin": float(args.ranking_margin),
        "device": str(device),
        "prediction_diagnostics": prediction_diagnostics,
        "splits": split_reports,
        "counterfactual": counterfactual,
        "ood_distance_available": ood_available,
        "rollout_disagreement_available": disagreement_available,
        "gates": gates,
        "provenance": {
            "git_revision": git_revision(),
            "git_dirty_files": git_dirty_files(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "checkpoint_sha256": sha256(checkpoint),
            "archives": {
                split: {"dataset": str(dataset), "dataset_sha256": sha256(dataset), "metadata": str(metadata), "metadata_sha256": sha256(metadata)}
                for split, (dataset, metadata) in paths.items()
            },
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_value) + "\n", encoding="utf-8")
    markdown = args.markdown_output.resolve()
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(result), encoding="utf-8")
    tensorboard = args.tensorboard_logdir.resolve()
    tensorboard.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("P19/Provenance", json.dumps(result["provenance"], sort_keys=True), 0)
        writer.add_text("P19/Gates", json.dumps(gates, sort_keys=True), 0)
        writer.add_scalar("P19/PromotionEligible", float(result["promotion_eligible"]), 0)
        writer.add_scalar("P19/Counterfactual/SelectedTracePresent", float(counterfactual["selected_counterfactual_present"]), 0)
        for split, report in split_reports.items():
            for name in ("top1_agreement", "pairwise_agreement", "mean_group_progress_correlation", "route_identity_accuracy", "route_side_accuracy", "route_geometry_accuracy"):
                value = report.get(name)
                if value is not None:
                    writer.add_scalar(f"P19/Ranking/{name}/{split}", float(value), 0)
            writer.add_scalar(f"P19/Eligibility/zero_eligible_groups/{split}", report["group_zero_eligible_count"], 0)
            writer.add_scalar(f"P19/Eligibility/min_eligible_candidates/{split}", report["group_min_eligible_candidates"], 0)
            writer.add_scalar(f"P19/Prediction/max_abs_output/{split}", prediction_diagnostics[split]["max_abs_output"], 0)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
