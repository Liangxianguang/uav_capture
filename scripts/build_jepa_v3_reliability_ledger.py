"""Build a fixed JEPA-v3 reliability ledger from execution-settled validation rollouts.

The source data is the held-out counterfactual validation split. Every label is
the outcome of a completed cloned-environment rollout, never an online target
or a development-S3 control outcome. The output ledger is read-only at runtime
and requests the nominal V5 action when a context lacks enough reliable data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedMultitaskJEPAPredictor,
    build_action_conditioned_predictor,
)
from encirclement3d.reliability import ReliabilityLedger, make_context_key, make_global_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--minimum-sample-count", type=int, default=128)
    parser.add_argument("--minimum-credit", type=float, default=0.65)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def _ranking_credit(keys: np.ndarray, predicted_cost: np.ndarray, settled_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return group-level candidate ranking credit and exact winner rate.

    A group is one state-agent snapshot with its five counterfactual actions.
    Credit is one minus normalized settled-cost regret of the action selected
    by the predictor. A tie in settled cost is credited as a ranking win.
    """
    _groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    ends = np.concatenate((np.flatnonzero(np.diff(inverse[order])) + 1, [order.size]))
    starts = np.concatenate(([0], ends[:-1]))
    credit = np.empty(predicted_cost.shape[0], dtype=np.float32)
    win = np.empty(predicted_cost.shape[0], dtype=np.float32)
    for start, stop in zip(starts, ends):
        indices = order[start:stop]
        selected = indices[int(np.argmin(predicted_cost[indices]))]
        best = float(np.min(settled_cost[indices]))
        worst = float(np.max(settled_cost[indices]))
        regret = float(settled_cost[selected] - best)
        group_credit = 1.0 if worst - best <= 1e-9 else float(np.clip(1.0 - regret / (worst - best), 0.0, 1.0))
        group_win = float(regret <= 1e-8)
        credit[indices] = group_credit
        win[indices] = group_win
    return credit, win


def _predict(
    checkpoint_path: Path,
    inputs: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "interaction_aware_action_conditioned_jepa_multitask":
        raise ValueError("Reliability ledgers require an interaction-aware JEPA-v3 multitask checkpoint.")
    model = build_action_conditioned_predictor(str(checkpoint["model_type"]), checkpoint["model"])
    if not isinstance(model, InteractionAwareActionConditionedMultitaskJEPAPredictor):
        raise RuntimeError("Checkpoint factory did not return the JEPA-v3 multitask predictor.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
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
            parts["obstacle_clearance"].append(auxiliary["obstacle_clearance"].cpu().numpy())
            parts["inter_agent_clearance"].append(auxiliary["inter_agent_clearance"].cpu().numpy())
            parts["visibility_probability"].append(torch.sigmoid(auxiliary["target_visibility_logit"]).cpu().numpy())
            parts["intervention_probability"].append(torch.sigmoid(auxiliary["cbf_intervention_logit"]).cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in parts.items()}, checkpoint


def _entry_template() -> dict[str, float]:
    return {
        "sample_count": 0.0,
        "credit_sum": 0.0,
        "target_error_sum_m": 0.0,
        "clearance_error_sum_m": 0.0,
        "visibility_error_sum": 0.0,
        "intervention_error_sum": 0.0,
        "ranking_win_sum": 0.0,
    }


def build_payload(
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    metadata: dict[str, Any],
    checkpoint_path: Path,
    minimum_sample_count: int,
    minimum_credit: float,
) -> dict[str, Any]:
    if metadata.get("split") != "validation":
        raise ValueError("Reliability ledger must be built from the held-out validation split.")
    boundary = metadata.get("information_boundary", {})
    if boundary.get("development_s3_or_locked_data_used_for_training") is not False:
        raise ValueError("Validation metadata does not prove development/locked separation.")
    config = yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))
    extent = float(config["world"]["half_extent_xy"])
    horizon_count = int(predictions["target_relative"].shape[1])
    required = {"episode_seed", "time_index", "agent_id", "candidate_action_norm_mps"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"Counterfactual dataset is missing ledger fields: {sorted(missing)}")
    keys = np.stack([arrays["episode_seed"], arrays["time_index"], arrays["agent_id"]], axis=1)
    visible = np.clip(arrays["inputs"][:, -1, 9], 0.0, 1.0)
    message_age = np.clip(arrays["inputs"][:, -1, 10], 0.0, 1.0)
    action_magnitude = arrays["candidate_action_norm_mps"]
    entries: dict[str, dict[str, float]] = defaultdict(_entry_template)
    for horizon in range(horizon_count):
        target_error_m = np.linalg.norm(
            predictions["target_relative"][:, horizon] - arrays["labels_relative"][:, horizon], axis=1
        ) * extent
        clearance_error_m = 0.5 * (
            np.abs(predictions["obstacle_clearance"][:, horizon] - arrays["labels_obstacle_clearance"][:, horizon])
            + np.abs(predictions["inter_agent_clearance"][:, horizon] - arrays["labels_inter_agent_clearance"][:, horizon])
        ) * extent
        visibility_error = np.abs(
            predictions["visibility_probability"][:, horizon] - arrays["labels_target_visible"][:, horizon]
        )
        intervention_error = np.abs(
            predictions["intervention_probability"][:, horizon] - arrays["labels_cbf_intervention"][:, horizon]
        )
        seconds = float(metadata["horizon_seconds"][horizon])
        predicted_cost = np.linalg.norm(
            predictions["target_relative"][:, horizon] * extent - arrays["action_history"][:, -1] * float(metadata["action_scale"]) * seconds,
            axis=1,
        )
        settled_cost = np.linalg.norm(
            arrays["labels_relative"][:, horizon] * extent - arrays["action_history"][:, -1] * float(metadata["action_scale"]) * seconds,
            axis=1,
        )
        ranking_credit, ranking_win = _ranking_credit(keys, predicted_cost, settled_cost)
        target_credit = np.exp(-target_error_m / 0.50)
        clearance_credit = np.exp(-clearance_error_m / 1.00)
        # These terms reflect execution-settled labels only. They do not make
        # uncertainty or a risk probability a replacement for CBF.
        credit = (
            0.45 * ranking_credit
            + 0.20 * target_credit
            + 0.15 * clearance_credit
            + 0.10 * (1.0 - visibility_error)
            + 0.10 * (1.0 - intervention_error)
        )
        predicted_clearance_m = np.minimum(
            predictions["obstacle_clearance"][:, horizon], predictions["inter_agent_clearance"][:, horizon]
        ) * extent
        for index in range(credit.shape[0]):
            context = make_context_key(
                horizon,
                float(visible[index]),
                float(message_age[index]),
                float(predicted_clearance_m[index]),
                float(action_magnitude[index]),
            )
            for key in (context, make_global_key(horizon)):
                entry = entries[key]
                entry["sample_count"] += 1.0
                entry["credit_sum"] += float(credit[index])
                entry["target_error_sum_m"] += float(target_error_m[index])
                entry["clearance_error_sum_m"] += float(clearance_error_m[index])
                entry["visibility_error_sum"] += float(visibility_error[index])
                entry["intervention_error_sum"] += float(intervention_error[index])
                entry["ranking_win_sum"] += float(ranking_win[index])
    finalized: dict[str, dict[str, float | int]] = {}
    for key, value in entries.items():
        count = int(value["sample_count"])
        finalized[key] = {
            "sample_count": count,
            "credit": float(value["credit_sum"] / count),
            "target_mae_m": float(value["target_error_sum_m"] / count),
            "clearance_mae_m": float(value["clearance_error_sum_m"] / count),
            "visibility_mae": float(value["visibility_error_sum"] / count),
            "intervention_mae": float(value["intervention_error_sum"] / count),
            "candidate_ranking_win_rate": float(value["ranking_win_sum"] / count),
        }
    return {
        "ledger_type": "jepa_v3_execution_settled_reliability",
        "ledger_version": 1,
        "not_a_locked_test": True,
        "update_rule": "fixed_offline_execution_settled_validation_outcomes_only",
        "source": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "validation_dataset": str(Path(metadata["dataset_path"]).resolve()) if metadata.get("dataset_path") else None,
            "validation_dataset_sha256": metadata.get("dataset_sha256"),
            "validation_metadata_sha256": metadata.get("metadata_sha256"),
            "collection_split": metadata["split"],
            "samples": int(arrays["inputs"].shape[0]),
            "horizon_seconds": [float(value) for value in metadata["horizon_seconds"]],
            "action_scale": float(metadata["action_scale"]),
        },
        "credit_definition": {
            "candidate_ranking_weight": 0.45,
            "target_error_weight": 0.20,
            "clearance_error_weight": 0.15,
            "visibility_error_weight": 0.10,
            "intervention_error_weight": 0.10,
            "target_error_decay_m": 0.50,
            "clearance_error_decay_m": 1.00,
        },
        "decision_policy": {
            "minimum_sample_count": int(minimum_sample_count),
            "minimum_credit": float(minimum_credit),
            "low_credit_action": "fallback_to_frozen_v5_nominal_action_then_cbf",
        },
        "entries": finalized,
    }


def forecast(payload: dict[str, Any], arrays: dict[str, np.ndarray], predictions: dict[str, np.ndarray], metadata: dict[str, Any]) -> dict[str, Any]:
    ledger = ReliabilityLedger(payload)
    extent = float(yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))["world"]["half_extent_xy"])
    visible = np.clip(arrays["inputs"][:, -1, 9], 0.0, 1.0)
    message_age = np.clip(arrays["inputs"][:, -1, 10], 0.0, 1.0)
    action_magnitude = arrays["candidate_action_norm_mps"]
    per_horizon: list[dict[str, Any]] = []
    for horizon in range(predictions["target_relative"].shape[1]):
        predicted_clearance = np.minimum(
            predictions["obstacle_clearance"][:, horizon], predictions["inter_agent_clearance"][:, horizon]
        ) * extent
        decisions = [
            ledger.decision(horizon, float(visible[index]), float(message_age[index]), float(predicted_clearance[index]), float(action_magnitude[index]))
            for index in range(predicted_clearance.shape[0])
        ]
        per_horizon.append(
            {
                "horizon_index": horizon,
                "fallback_fraction": float(np.mean([item.fallback_to_nominal for item in decisions])),
                "global_fallback_fraction": float(np.mean([item.used_global_fallback for item in decisions])),
                "mean_credit": float(np.mean([item.credit for item in decisions])),
                "mean_sample_count": float(np.mean([item.sample_count for item in decisions])),
            }
        )
    return {"source_validation_policy_forecast": per_horizon}


def render(payload: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    policy = payload["decision_policy"]
    globals_ = [payload["entries"][make_global_key(index)] for index in range(len(payload["source"]["horizon_seconds"]))]
    lines = [
        "# JEPA-v3 Execution-Settled Reliability Ledger",
        "",
        "> Development-only artifact. The ledger is built from completed counterfactual validation rollouts, is read-only at runtime, and can only request the nominal frozen-V5 action before the existing CBF filter.",
        "",
        f"Minimum local credit: `{policy['minimum_credit']:.2f}`; minimum local samples: `{policy['minimum_sample_count']}`.",
        "",
        "| Horizon (s) | Global samples | Credit | Target MAE (m) | Clearance MAE (m) | Candidate ranking win rate | Validation fallback forecast |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, entry in enumerate(globals_):
        forecast_row = diagnostics["source_validation_policy_forecast"][index]
        lines.append(
            f"| {payload['source']['horizon_seconds'][index]:.1f} | {entry['sample_count']} | {entry['credit']:.4f} | "
            f"{entry['target_mae_m']:.4f} | {entry['clearance_mae_m']:.4f} | "
            f"{entry['candidate_ranking_win_rate']:.4f} | {forecast_row['fallback_fraction']:.2%} |"
        )
    lines += [
        "",
        "The forecast is an offline diagnostic on the ledger source. It is not a closed-loop performance estimate and must not be tuned with frozen S3 development outcomes.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.minimum_sample_count <= 0 or not 0.0 <= args.minimum_credit <= 1.0:
        raise ValueError("Invalid batch size or decision policy.")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite a reliability ledger or report.")
    metadata = json.loads(args.metadata.resolve().read_text(encoding="utf-8"))
    metadata["dataset_path"] = str(args.dataset.resolve())
    metadata["dataset_sha256"] = sha256(args.dataset.resolve())
    metadata["metadata_sha256"] = sha256(args.metadata.resolve())
    with np.load(args.dataset.resolve()) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    device = choose_device(args.device)
    predictions, _checkpoint = _predict(args.checkpoint.resolve(), arrays["inputs"].astype(np.float32), arrays["action_history"].astype(np.float32), args.batch_size, device)
    payload = build_payload(
        arrays,
        predictions,
        metadata,
        args.checkpoint.resolve(),
        args.minimum_sample_count,
        args.minimum_credit,
    )
    diagnostics = forecast(payload, arrays, predictions, metadata)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render(payload, diagnostics), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
