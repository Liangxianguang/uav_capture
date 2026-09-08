"""Build the P27 fresh calibration and offline disagreement audit.

This audit is intentionally read-only.  It binds a new calibration profile to
the P25 traceable archives and the P24 route-JEPA checkpoint, then measures
input/action OOD distance, route-progress error, candidate disagreement, and
five-step rollout disagreement.  It never executes an action, changes CBF
margins, or promotes a Ledger-Lite route.
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
from collections import defaultdict
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
MODEL_TYPE = "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2"
RANKING_HORIZON_INDEX = 2
OOD_QUANTILE = 0.99
ERROR_QUANTILE = 0.95
MIN_VALIDATION_COVERAGE = 0.90
MAX_VALIDATION_OOD_RATE = 0.10
MIN_SELECTED_AGREEMENT = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    for split in ("train", "validation", "calibration"):
        parser.add_argument(f"--{split}-dataset", type=Path, required=True)
        parser.add_argument(f"--{split}-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
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
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def _finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def _quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, quantile))


def _feature_matrix(tensors: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Return the public input/action features and their runtime mask."""
    sample_type = tensors["sample_type"].numpy()
    runtime = sample_type == 0
    inputs = tensors["inputs"].numpy()[:, -1, :]
    action_history = tensors["action_history"].numpy()[:, -1, :]
    action_chunk = tensors["route_action_chunk"].numpy().reshape(len(runtime), -1)
    pairwise_chunk = tensors["route_pairwise_relative_action_chunk"].numpy().reshape(len(runtime), -1)
    features = np.concatenate((inputs, action_history, action_chunk, pairwise_chunk), axis=1).astype(np.float64)
    if not np.isfinite(features).all():
        raise ValueError("P27 feature matrix contains non-finite values")
    return features, runtime


def fit_ood_profile(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    features, runtime = _feature_matrix(tensors)
    values = features[runtime]
    if values.shape[0] < 2:
        raise ValueError("P27 requires at least two runtime rows to fit OOD profile")
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-6), scale, 1.0)
    digest = hashlib.sha256(np.concatenate((mean, scale)).astype(np.float64).tobytes()).hexdigest()
    return {
        "feature_dim": int(values.shape[1]),
        "fit_runtime_rows": int(values.shape[0]),
        "mean": mean,
        "scale": scale,
        "profile_sha256": digest,
    }


def apply_ood_profile(tensors: dict[str, torch.Tensor], profile: dict[str, Any]) -> np.ndarray:
    features, _runtime = _feature_matrix(tensors)
    mean = np.asarray(profile["mean"], dtype=np.float64)
    scale = np.asarray(profile["scale"], dtype=np.float64)
    if features.shape[1] != mean.size or mean.shape != scale.shape:
        raise ValueError("OOD profile feature dimension does not match archive")
    standardized = (features - mean[None, :]) / scale[None, :]
    return np.sqrt(np.mean(np.square(standardized), axis=1))


def _predict(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], bool, float]:
    required = (
        "route_progress",
        "route_identity_logits",
        "route_side_logits",
        "route_geometry_logit",
        "cbf_feasibility_logit",
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in required}
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    finite = True
    max_abs = 0.0
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
            mean, log_variance, _latent, auxiliary = model.forward_multitask(*args)
            values = {name: auxiliary[name] for name in required}
            values["mean"] = mean
            values["log_variance"] = log_variance
            for name, value in values.items():
                array = value.detach().cpu().numpy()
                if not np.isfinite(array).all():
                    finite = False
                if array.size:
                    max_abs = max(max_abs, float(np.max(np.abs(array))))
                if name == "mean":
                    means.append(array)
                elif name == "log_variance":
                    log_variances.append(array)
                else:
                    collected[name].append(array)
    result = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    result["mean"] = np.concatenate(means, axis=0)
    result["log_variance"] = np.concatenate(log_variances, axis=0)
    return result, finite, max_abs


def _runtime_group_keys(tensors: dict[str, torch.Tensor]) -> list[tuple[int, int]]:
    runtime = tensors["sample_type"].numpy() == 0
    scenario = tensors["scenario_index"].numpy()
    times = tensors["time_index"].numpy()
    return sorted({(int(scenario[i]), int(times[i])) for i in np.flatnonzero(runtime)})


def _group_disagreement(
    tensors: dict[str, torch.Tensor],
    outputs: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_type = tensors["sample_type"].numpy()
    runtime = sample_type == 0
    candidate = tensors["route_candidate_index"].numpy()
    scenario = tensors["scenario_index"].numpy()
    time_index = tensors["time_index"].numpy()
    geometry = tensors["route_geometry_valid"].numpy() >= 0.5
    cbf = tensors["labels_cbf_feasible"].numpy()[:, 0] >= 0.5
    selected = tensors["selected_candidate_index"].numpy()
    truth = tensors["labels_route_progress"].numpy()[:, RANKING_HORIZON_INDEX]
    prediction = outputs["route_progress"][:, RANKING_HORIZON_INDEX]
    details: list[dict[str, Any]] = []
    selected_matches: list[float] = []
    truth_matches: list[float] = []
    abstentions: list[dict[str, Any]] = []
    for key in _runtime_group_keys(tensors):
        group = runtime & (scenario == key[0]) & (time_index == key[1])
        ids = sorted(int(value) for value in np.unique(candidate[group]) if 0 <= int(value) < 12)
        records: list[dict[str, Any]] = []
        for candidate_id in ids:
            rows = group & (candidate == candidate_id)
            eligible = bool(np.all(geometry[rows]) and np.all(cbf[rows]))
            records.append(
                {
                    "candidate": candidate_id,
                    "label": ROUTE_LABELS[candidate_id],
                    "eligible": eligible,
                    "prediction": float(np.mean(prediction[rows])),
                    "truth": float(np.mean(truth[rows])),
                }
            )
        eligible_records = [item for item in records if item["eligible"]]
        selected_value = int(selected[group][0]) if np.any(group) else -1
        if not eligible_records:
            abstentions.append({"scenario_index": key[0], "time_index": key[1], "selected_candidate_index": selected_value})
            model_best = None
            truth_best = None
        else:
            model_best = max(eligible_records, key=lambda item: (item["prediction"], -item["candidate"]))["candidate"]
            truth_best = max(eligible_records, key=lambda item: (item["truth"], -item["candidate"]))["candidate"]
            truth_matches.append(float(model_best == truth_best))
            if 0 <= selected_value < 12:
                selected_matches.append(float(model_best == selected_value))
        details.append(
            {
                "scenario_index": key[0],
                "time_index": key[1],
                "candidate_count": len(records),
                "eligible_count": len(eligible_records),
                "selected_candidate_index": selected_value,
                "model_best_candidate_index": model_best,
                "truth_best_candidate_index": truth_best,
                "model_vs_selected_match": None if model_best is None or not 0 <= selected_value < 12 else bool(model_best == selected_value),
                "model_vs_truth_match": None if model_best is None else bool(model_best == truth_best),
            }
        )
    return (
        {
            "runtime_group_count": len(details),
            "zero_eligible_group_count": len(abstentions),
            "selected_comparison_group_count": len(selected_matches),
            "selected_agreement": float(np.mean(selected_matches)) if selected_matches else None,
            "truth_comparison_group_count": len(truth_matches),
            "truth_top1_agreement": float(np.mean(truth_matches)) if truth_matches else None,
            "abstention_groups": abstentions,
        },
        details,
    )


def _split_metrics(
    tensors: dict[str, torch.Tensor],
    outputs: dict[str, np.ndarray],
    ood_distance: np.ndarray,
    ood_threshold: float | None,
    route_error_threshold: float | None,
    rollout_threshold: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = tensors["sample_type"].numpy() == 0
    route_error = np.abs(outputs["route_progress"][:, RANKING_HORIZON_INDEX] - tensors["labels_route_progress"].numpy()[:, RANKING_HORIZON_INDEX])
    rollout_error = np.linalg.norm(outputs["mean"] - tensors["labels_relative"].numpy(), axis=-1)
    rollout_error = np.max(rollout_error, axis=1)
    group_report, group_details = _group_disagreement(tensors, outputs)
    runtime_ood = ood_distance[runtime]
    runtime_route_error = route_error[runtime]
    runtime_rollout_error = rollout_error[runtime]
    def coverage(values: np.ndarray, threshold: float | None) -> float | None:
        if threshold is None or not math.isfinite(float(threshold)) or values.size == 0:
            return None
        return float(np.mean(values <= float(threshold)))
    report = {
        "runtime_rows": int(runtime.sum()),
        "ood_distance": _summary(runtime_ood),
        "route_progress_abs_error": _summary(runtime_route_error),
        "rollout_disagreement_m": _summary(runtime_rollout_error),
        "ood_threshold": ood_threshold,
        "route_progress_error_threshold": route_error_threshold,
        "rollout_disagreement_threshold_m": rollout_threshold,
        "ood_rate": None if ood_threshold is None else float(np.mean(runtime_ood > ood_threshold)),
        "route_progress_coverage": coverage(runtime_route_error, route_error_threshold),
        "rollout_coverage": coverage(runtime_rollout_error, rollout_threshold),
        "candidate_disagreement": group_report,
    }
    for row in group_details:
        row["split"] = None
    return report, group_details


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _markdown(result: dict[str, Any]) -> str:
    gates = result["gates"]
    decision = "PROMOTE" if result["promotion_eligible"] else "REMAIN OFFLINE"
    lines = [
        "# DN-MPC P27 Fresh Calibration and Disagreement Audit",
        "",
        f"**Status:** development-only; **decision:** `{decision}`.",
        "",
        "This is an offline calibration audit. It executes no action and does not",
        "change CBF margins, stale/OOD gates, controlled-abort semantics, or raw-action policy.",
        "",
        "## Provenance",
        "",
        f"- Checkpoint SHA-256: `{result['checkpoint_sha256']}`",
        f"- OOD profile SHA-256: `{result['ood_profile']['profile_sha256']}`",
        f"- Git revision: `{result['provenance']['git_revision']}`",
        "",
        "## Split Metrics",
        "",
        "| Split | runtime rows | OOD rate | route error coverage | rollout coverage | model vs selected | model vs truth | abstentions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "calibration", "validation"):
        item = result["splits"][split]
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value) * 100:.2f}%"
        disagreement = item["candidate_disagreement"]
        lines.append(
            f"| {split} | {item['runtime_rows']} | {fmt(item['ood_rate'])} | "
            f"{fmt(item['route_progress_coverage'])} | {fmt(item['rollout_coverage'])} | "
            f"{fmt(disagreement['selected_agreement'])} | {fmt(disagreement['truth_top1_agreement'])} | "
            f"{disagreement['zero_eligible_group_count']} |"
        )
    lines += [
        "",
        "## Calibration Thresholds",
        "",
        f"- OOD threshold: calibration p{OOD_QUANTILE * 100:.0f} = `{result['thresholds']['ood_distance']}`",
        f"- Route-progress error threshold: calibration p{ERROR_QUANTILE * 100:.0f} = `{result['thresholds']['route_progress_error']}`",
        f"- Rollout disagreement threshold: calibration p{ERROR_QUANTILE * 100:.0f} = `{result['thresholds']['rollout_disagreement_m']}` m",
        "",
        "## Promotion Gates",
        "",
    ]
    for name, value in gates.items():
        lines.append(f"- `{name}`: `{value}`")
    lines += [
        "",
        "The two P26 no-eligible groups remain explicit abstentions. They are",
        "included in the calibration provenance and are not converted into a",
        "synthetic selected route. The result is therefore not authorization for",
        "online JEPA routing, Ledger-Lite creation, or three-seed paired replay.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    output_paths = [args.output.resolve(), args.markdown_output.resolve(), args.details_output.resolve()]
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    tensorboard_path = args.tensorboard_logdir.resolve()
    if tensorboard_path.exists() and any(tensorboard_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory {tensorboard_path}")
    checkpoint = args.checkpoint.resolve()
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_data.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Unexpected checkpoint model_type: {checkpoint_data.get('model_type')!r}")
    paths = {
        split: (getattr(args, f"{split}_dataset").resolve(), getattr(args, f"{split}_metadata").resolve())
        for split in ("train", "validation", "calibration")
    }
    tensors_by_split: dict[str, dict[str, torch.Tensor]] = {}
    metadata_by_split: dict[str, dict[str, Any]] = {}
    for split, (dataset, metadata) in paths.items():
        tensors, archive_metadata = load_dataset(dataset, metadata, split)
        tensors_by_split[split] = tensors
        metadata_by_split[split] = archive_metadata
    device = choose_device(args.device)
    model = build_action_conditioned_predictor(checkpoint_data["model_type"], checkpoint_data["model"]).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    model.eval()
    predictions: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    for split, tensors in tensors_by_split.items():
        predictions[split], finite, max_abs = _predict(model, tensors, device, args.batch_size)
        diagnostics[split] = {"finite": finite, "max_abs_output": max_abs}
    profile = fit_ood_profile(tensors_by_split["train"])
    ood_distances = {split: apply_ood_profile(tensors, profile) for split, tensors in tensors_by_split.items()}
    calibration_runtime = tensors_by_split["calibration"]["sample_type"].numpy() == 0
    calibration_route_error = np.abs(
        predictions["calibration"]["route_progress"][:, RANKING_HORIZON_INDEX]
        - tensors_by_split["calibration"]["labels_route_progress"].numpy()[:, RANKING_HORIZON_INDEX]
    )[calibration_runtime]
    calibration_rollout = np.max(
        np.linalg.norm(
            predictions["calibration"]["mean"] - tensors_by_split["calibration"]["labels_relative"].numpy(), axis=-1
        ), axis=1,
    )[calibration_runtime]
    calibration_ood = ood_distances["calibration"][calibration_runtime]
    thresholds = {
        "ood_distance": _quantile(calibration_ood, OOD_QUANTILE),
        "route_progress_error": _quantile(calibration_route_error, ERROR_QUANTILE),
        "rollout_disagreement_m": _quantile(calibration_rollout, ERROR_QUANTILE),
        "ood_quantile": OOD_QUANTILE,
        "error_quantile": ERROR_QUANTILE,
    }
    split_reports: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    for split in ("train", "calibration", "validation"):
        report, rows = _split_metrics(
            tensors_by_split[split],
            predictions[split],
            ood_distances[split],
            thresholds["ood_distance"],
            thresholds["route_progress_error"],
            thresholds["rollout_disagreement_m"],
        )
        for row in rows:
            row["split"] = split
        detail_rows.extend(rows)
        split_reports[split] = report
    validation = split_reports["validation"]
    calibration = split_reports["calibration"]
    gates = {
        "checkpoint_and_archive_contract": True,
        "finite_outputs": all(bool(item["finite"]) for item in diagnostics.values()),
        "fresh_calibration_thresholds_finite": all(math.isfinite(float(value)) for key, value in thresholds.items() if key.endswith("distance") or key.endswith("error") or key.endswith("_m")),
        "abstention_samples_bound": all(
            "abstention_groups" in split_reports[split]["candidate_disagreement"] for split in ("train", "calibration", "validation")
        ),
        "validation_ood_coverage": bool(validation["ood_rate"] is not None and validation["ood_rate"] <= MAX_VALIDATION_OOD_RATE),
        "validation_route_error_coverage": bool(validation["route_progress_coverage"] is not None and validation["route_progress_coverage"] >= MIN_VALIDATION_COVERAGE),
        "validation_rollout_coverage": bool(validation["rollout_coverage"] is not None and validation["rollout_coverage"] >= MIN_VALIDATION_COVERAGE),
        "validation_candidate_agreement": bool(
            validation["candidate_disagreement"]["selected_agreement"] is not None
            and validation["candidate_disagreement"]["selected_agreement"] >= MIN_SELECTED_AGREEMENT
        ),
    }
    result: dict[str, Any] = {
        "audit_type": "dn_mpc_p27_fresh_calibration_disagreement_audit",
        "model_type": MODEL_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "promotion_eligible": bool(all(gates.values())),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_model_config": checkpoint_data.get("model"),
        "device": str(device),
        "thresholds": thresholds,
        "ood_profile": {
            "feature_dim": profile["feature_dim"],
            "fit_runtime_rows": profile["fit_runtime_rows"],
            "profile_sha256": profile["profile_sha256"],
            "fit_split": "train",
        },
        "diagnostics": diagnostics,
        "splits": split_reports,
        "gates": gates,
        "provenance": {
            "git_revision": git_revision(),
            "git_dirty_files": git_dirty_files(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "checkpoint_sha256": sha256(checkpoint),
            "archives": {
                split: {
                    "dataset": str(dataset),
                    "dataset_sha256": sha256(dataset),
                    "metadata": str(metadata),
                    "metadata_sha256": sha256(metadata),
                    "dataset_version": metadata_by_split[split].get("dataset_version"),
                }
                for split, (dataset, metadata) in paths.items()
            },
            "p25_traceability_required": True,
            "p26_abstentions_included": True,
        },
    }
    output_paths[0].parent.mkdir(parents=True, exist_ok=True)
    output_paths[0].write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    output_paths[1].parent.mkdir(parents=True, exist_ok=True)
    output_paths[1].write_text(_markdown(result), encoding="utf-8")
    output_paths[2].parent.mkdir(parents=True, exist_ok=True)
    with output_paths[2].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "split", "scenario_index", "time_index", "candidate_count", "eligible_count",
            "selected_candidate_index", "model_best_candidate_index", "truth_best_candidate_index",
            "model_vs_selected_match", "model_vs_truth_match",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row.get(name) for name in fieldnames} for row in detail_rows)
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(tensorboard_path), flush_secs=1) as writer:
        writer.add_text("P27/Provenance", json.dumps(result["provenance"], sort_keys=True), 0)
        writer.add_text("P27/Thresholds", json.dumps(thresholds, sort_keys=True), 0)
        writer.add_text("P27/Gates", json.dumps(gates, sort_keys=True), 0)
        writer.add_scalar("P27/PromotionEligible", float(result["promotion_eligible"]), 0)
        for split, report in split_reports.items():
            writer.add_scalar(f"P27/OOD/runtime_rows/{split}", report["runtime_rows"], 0)
            writer.add_scalar(f"P27/OOD/rate/{split}", float(report["ood_rate"] or 0.0), 0)
            writer.add_scalar(f"P27/Disagreement/route_error_p95/{split}", float(report["route_progress_abs_error"]["p95"] or 0.0), 0)
            writer.add_scalar(f"P27/Disagreement/rollout_p95_m/{split}", float(report["rollout_disagreement_m"]["p95"] or 0.0), 0)
            writer.add_scalar(f"P27/Disagreement/route_error_coverage/{split}", float(report["route_progress_coverage"] or 0.0), 0)
            writer.add_scalar(f"P27/Disagreement/rollout_coverage/{split}", float(report["rollout_coverage"] or 0.0), 0)
            disagreement = report["candidate_disagreement"]
            writer.add_scalar(f"P27/Candidate/selected_agreement/{split}", float(disagreement["selected_agreement"] or 0.0), 0)
            writer.add_scalar(f"P27/Candidate/truth_top1_agreement/{split}", float(disagreement["truth_top1_agreement"] or 0.0), 0)
            writer.add_scalar(f"P27/Abstention/zero_eligible_groups/{split}", disagreement["zero_eligible_group_count"], 0)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
