"""Train the route-aware JEPA evaluator on the obstacle-route archive.

This trainer is intentionally separate from the historical V3/V21 trainers.
The route archive has a 12-candidate, three- or five-step chunk contract and
contains explicit geometry/CBF negative branches. A fresh checkpoint is
therefore required before calibration or runtime use; old ledgers cannot be
reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedRouteHardNegativeJEPAPredictor,
    InteractionAwareActionConditionedRouteJEPAPredictor,
    build_action_conditioned_predictor,
    deterministic_mse,
    gaussian_nll,
)


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2"
REQUIRED_ARRAYS = (
    "inputs",
    "action_history",
    "route_action_chunk",
    "route_relative_action_chunk",
    "route_pairwise_relative_action_chunk",
    "labels_relative",
    "labels_obstacle_clearance",
    "labels_boundary_clearance",
    "labels_inter_agent_clearance",
    "labels_stopping_distance",
    "labels_obstacle_ttc",
    "labels_boundary_ttc",
    "labels_pairwise_ttc",
    "labels_acceleration_slack",
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
    "scenario_index",
    "time_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/central_random_mixed_obstacle_s3_route_v1_protocol.yaml",
    )
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        help="Optional frozen route-identity v1 checkpoint used to initialize the shared JEPA/backbone.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--clearance-loss-weight", type=float, default=0.5)
    parser.add_argument("--visibility-loss-weight", type=float, default=0.25)
    parser.add_argument("--cbf-correction-loss-weight", type=float, default=0.25)
    parser.add_argument("--cbf-intervention-loss-weight", type=float, default=0.25)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.75)
    parser.add_argument("--feasibility-loss-weight", type=float, default=1.0)
    parser.add_argument("--slack-loss-weight", type=float, default=0.75)
    parser.add_argument("--progress-loss-weight", type=float, default=0.50)
    parser.add_argument("--route-ranking-loss-weight", type=float, default=1.0)
    parser.add_argument("--route-ranking-horizon-index", type=int, default=2)
    parser.add_argument("--route-ranking-margin", type=float, default=0.005)
    parser.add_argument("--route-identity-loss-weight", type=float, default=0.35)
    parser.add_argument("--route-side-loss-weight", type=float, default=0.25)
    parser.add_argument("--geometry-loss-weight", type=float, default=0.50)
    parser.add_argument("--termination-loss-weight", type=float, default=0.50)
    parser.add_argument("--stopping-distance-loss-weight", type=float, default=0.50)
    parser.add_argument("--obstacle-ttc-loss-weight", type=float, default=0.50)
    parser.add_argument("--boundary-ttc-loss-weight", type=float, default=0.50)
    parser.add_argument("--pairwise-ttc-loss-weight", type=float, default=0.50)
    parser.add_argument("--acceleration-slack-loss-weight", type=float, default=0.50)
    parser.add_argument("--hazard-loss-weight", type=float, default=1.0)
    parser.add_argument("--hazard-positive-weight", type=float, default=4.0)
    parser.add_argument("--quantile-loss-weight", type=float, default=0.50)
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument(
        "--pairwise-pooling",
        action="store_true",
        help="Append an explicit latest-frame relative-teammate pooling block to risk heads.",
    )
    parser.add_argument(
        "--pairwise-relational",
        action="store_true",
        help="Encode each teammate pair with candidate-action-conditioned relative motion before pooling.",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze shared JEPA and legacy route heads; optimize only v2 hard-negative risk heads.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--histogram-interval", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def _load_metadata(path: Path, dataset: Path, expected_split: str) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    if metadata.get("split") != expected_split:
        raise ValueError(f"Expected {expected_split} metadata, got {metadata.get('split')!r}.")
    if metadata.get("candidate_profile") != "obstacle_route_v1" or int(metadata.get("candidate_count", 0)) != 12:
        raise ValueError("Route trainer requires the obstacle_route_v1 12-candidate contract.")
    if int(metadata.get("history_length", 0)) != 8 or int(metadata.get("chunk_length_steps", 0)) not in (3, 5):
        raise ValueError("Route trainer requires history_length=8 and chunk_length_steps in {3,5}.")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError("Route training is permitted only in the closed development protocol.")
    boundary = metadata.get("information_boundary", {})
    if not isinstance(boundary, dict) or boundary.get("target_truth_used_only_for_offline_labels") is not True:
        raise ValueError("Route archive does not prove the target-truth information boundary.")
    result = dict(metadata)
    result["_metadata_path"] = str(path.resolve())
    result["_dataset_path"] = str(dataset.resolve())
    return result


def _finite_tensor(name: str, values: np.ndarray, samples: int) -> torch.Tensor:
    if values.shape[0] != samples or not np.isfinite(values).all():
        raise ValueError(f"Dataset array {name} has invalid shape or non-finite values: {values.shape}")
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def load_dataset(path: Path, metadata_path: Path, expected_split: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    metadata = _load_metadata(metadata_path, path, expected_split)
    with np.load(path) as archive:
        archive_files = set(archive.files)
        # The interaction-conditioned route feature was introduced after the
        # v1/v2 archives.  Keep those archives loadable for audit/reproduction
        # by supplying an explicit zero feature, while requiring the field
        # whenever the metadata declares the new contract.
        interaction_field = "route_relative_action_chunk"
        declared_interaction = bool(metadata.get("interaction_action_conditioned_route_chunk", False))
        if declared_interaction and interaction_field not in archive_files:
            raise ValueError(f"{path} declares {interaction_field} but the array is missing")
        missing = (set(REQUIRED_ARRAYS) - {interaction_field, "route_pairwise_relative_action_chunk"}).difference(archive_files)
        if missing:
            raise ValueError(f"{path} is missing route arrays: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in REQUIRED_ARRAYS if name in archive_files}
        if interaction_field not in arrays:
            arrays[interaction_field] = np.zeros_like(arrays["route_action_chunk"], dtype=np.float32)
            metadata["interaction_action_conditioned_route_chunk"] = False
        else:
            metadata["interaction_action_conditioned_route_chunk"] = True
        pairwise_field = "route_pairwise_relative_action_chunk"
        pairwise_declared = bool(metadata.get("pairwise_action_conditioned_route_chunk", False))
        if pairwise_declared and pairwise_field not in archive_files:
            raise ValueError(f"{path} declares {pairwise_field} but the array is missing")
        if pairwise_field in archive_files:
            arrays[pairwise_field] = np.asarray(archive[pairwise_field])
        else:
            arrays[pairwise_field] = np.zeros(
                (*arrays["route_action_chunk"].shape[:2], 9), dtype=np.float32
            )
    samples = int(arrays["inputs"].shape[0])
    if arrays["inputs"].shape[1:] != (8, 63):
        raise ValueError(f"inputs must have shape [N,8,63], got {arrays['inputs'].shape}")
    if arrays["action_history"].shape[1:] != (8, 3):
        raise ValueError("action_history must have shape [N,8,3].")
    chunk_length = int(metadata["chunk_length_steps"])
    if arrays["route_action_chunk"].shape[1:] != (chunk_length, 3):
        raise ValueError(f"route_action_chunk must have shape [N,{chunk_length},3].")
    if arrays["route_relative_action_chunk"].shape[1:] != (chunk_length, 3):
        raise ValueError(f"route_relative_action_chunk must have shape [N,{chunk_length},3].")
    pairwise_declared = bool(metadata.get("pairwise_action_conditioned_route_chunk", False))
    if pairwise_declared:
        if "route_pairwise_relative_action_chunk" not in arrays:
            raise ValueError("Pairwise action-conditioned archive is missing its route feature.")
        if arrays["route_pairwise_relative_action_chunk"].shape[1:] != (chunk_length, 9):
            raise ValueError(
                "route_pairwise_relative_action_chunk must have shape "
                f"[N,{chunk_length},9]."
            )
        if not np.isfinite(arrays["route_pairwise_relative_action_chunk"]).all():
            raise ValueError("route_pairwise_relative_action_chunk contains non-finite values.")
    for name in ("labels_relative",):
        if arrays[name].shape[1:] != (5, 3):
            raise ValueError(f"{name} must have shape [N,5,3].")
    for name in (
        "labels_obstacle_clearance",
        "labels_boundary_clearance",
        "labels_inter_agent_clearance",
        "labels_stopping_distance",
        "labels_obstacle_ttc",
        "labels_boundary_ttc",
        "labels_pairwise_ttc",
        "labels_acceleration_slack",
        "labels_target_visible",
        "labels_cbf_correction",
        "labels_cbf_intervention",
        "labels_cbf_feasible",
        "labels_cbf_min_slack",
        "labels_route_progress",
    ):
        if arrays[name].shape[1:] != (5,):
            raise ValueError(f"{name} must have shape [N,5].")
    tensors: dict[str, torch.Tensor] = {
        name: _finite_tensor(name, value, samples)
        for name, value in arrays.items()
        if name not in {"route_candidate_index", "route_side_index", "scenario_index", "time_index"}
    }
    for name in ("route_candidate_index", "route_side_index", "scenario_index", "time_index"):
        values = arrays[name]
        if values.shape != (samples,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite with shape [N].")
        tensors[name] = torch.from_numpy(values.astype(np.int64, copy=False))
    candidate = tensors["route_candidate_index"]
    side = tensors["route_side_index"]
    sample_type = tensors["sample_type"]
    if torch.any((sample_type == 0) & ((candidate < 0) | (candidate >= 12))):
        raise ValueError("Runtime route rows must have candidate indices in [0,11].")
    if torch.any((sample_type == 0) & ((side < 0) | (side >= 12))):
        raise ValueError("Runtime route rows must have route-side indices in [0,11].")
    if torch.any((sample_type != 0) & (candidate != -1)):
        raise ValueError("Offline boundary-shadow rows must use candidate index -1.")
    action_scale = float(metadata.get("action_scale", 5.0))
    if not np.isfinite(action_scale) or action_scale <= 0.0:
        raise ValueError("Route archive action_scale must be positive and finite.")
    tensors["route_action_chunk"] = tensors["route_action_chunk"] / action_scale
    tensors["route_relative_action_chunk"] = tensors["route_relative_action_chunk"] / action_scale
    if pairwise_declared:
        tensors["route_pairwise_relative_action_chunk"] = (
            tensors["route_pairwise_relative_action_chunk"] / action_scale
        )
    return tensors, metadata


def _paired_contract(train: dict[str, Any], validation: dict[str, Any]) -> None:
    fields = (
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
    mismatch = {
        field: {"train": train.get(field), "validation": validation.get(field)}
        for field in fields
        if train.get(field) != validation.get(field)
    }
    if mismatch:
        raise ValueError(f"Train/validation route contracts differ: {mismatch}")


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    return value.sum() * 0.0


def _masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(mask.any()):
        return F.cross_entropy(logits[mask], labels[mask]), (logits[mask].argmax(dim=-1) == labels[mask]).float().mean()
    return _zero_like(logits), _zero_like(logits)


class _RouteGroupBatchSampler:
    """Keep all route candidates for a scenario/time group in one batch.

    The route-progress ranking loss compares alternatives from the same belief
    state.  A normal shuffled DataLoader would split those alternatives across
    batches and silently remove the supervision signal.
    """

    def __init__(self, tensors: dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        scenario = tensors["scenario_index"].detach().cpu().numpy().astype(np.int64, copy=False)
        time_index = tensors["time_index"].detach().cpu().numpy().astype(np.int64, copy=False)
        groups: dict[tuple[int, int], list[int]] = {}
        for index, key in enumerate(zip(scenario.tolist(), time_index.tolist())):
            groups.setdefault((int(key[0]), int(key[1])), []).append(index)
        self.groups = list(groups.values())
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def _batches(self, groups: list[list[int]]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        for group in groups:
            if current and len(current) + len(group) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(group)
        if current:
            batches.append(current)
        return batches

    def __iter__(self):
        groups = list(self.groups)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        self.epoch += 1
        yield from self._batches(groups)

    def __len__(self) -> int:
        return len(self._batches(list(self.groups)))


def _route_progress_ranking_metrics(
    auxiliary: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    horizon_index: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare route alternatives from the same belief state.

    Invalid geometry and first-step-infeasible branches are excluded from the
    task ranking target.  The CBF feasibility head remains trained on those
    negatives separately.
    """

    predicted = auxiliary["route_progress"][:, horizon_index]
    observed = batch["labels_route_progress"][:, horizon_index]
    runtime = (batch["sample_type"] < 0.5) & (batch["route_candidate_index"] >= 0)
    group_keys = torch.stack((batch["scenario_index"], batch["time_index"]), dim=1)
    losses: list[torch.Tensor] = []
    correct = 0
    pair_count = 0
    group_count = 0
    for key in torch.unique(group_keys, dim=0):
        group_mask = runtime & (group_keys == key).all(dim=1)
        candidate_ids = torch.unique(batch["route_candidate_index"][group_mask])
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for candidate in candidate_ids:
            rows = group_mask & (batch["route_candidate_index"] == candidate)
            rows &= batch["route_geometry_valid"] >= 0.5
            rows &= batch["labels_cbf_feasible"][:, 0] >= 0.5
            if bool(rows.any()):
                predictions.append(predicted[rows].mean())
                targets.append(observed[rows].mean())
        if len(predictions) < 2:
            continue
        group_count += 1
        predicted_values = torch.stack(predictions)
        target_values = torch.stack(targets)
        first, second = torch.triu_indices(len(predictions), len(predictions), offset=1, device=predicted.device)
        target_delta = target_values[first] - target_values[second]
        informative = torch.abs(target_delta) > float(margin)
        if not bool(informative.any()):
            continue
        predicted_delta = predicted_values[first] - predicted_values[second]
        signs = torch.sign(target_delta)
        pair_losses = torch.relu(float(margin) - signs * predicted_delta)[informative]
        losses.append(pair_losses.mean())
        correct += int((signs[informative] * predicted_delta[informative] > 0.0).sum().item())
        pair_count += int(informative.sum().item())
    if not losses:
        zero = _zero_like(predicted)
        return zero, zero, zero
    loss = torch.stack(losses).mean()
    accuracy = predicted.new_tensor(float(correct) / float(max(pair_count, 1)))
    groups = predicted.new_tensor(float(group_count))
    return loss, accuracy, groups


def _losses(
    model: InteractionAwareActionConditionedRouteJEPAPredictor,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    *,
    route_ranking_horizon_index: int,
    route_ranking_margin: float,
    hazard_positive_weight: float,
    quantile: float,
) -> dict[str, torch.Tensor]:
    if int(getattr(model, "route_interaction_chunk_dim", 0)) == 9:
        if "route_pairwise_relative_action_chunk" not in batch:
            raise ValueError("Pairwise relational model requires route_pairwise_relative_action_chunk")
        route_interaction_chunks = batch["route_pairwise_relative_action_chunk"]
    elif int(getattr(model, "route_interaction_chunk_dim", 0)) > 0:
        route_interaction_chunks = batch["route_relative_action_chunk"]
    else:
        route_interaction_chunks = None
    mean, log_variance, latent, auxiliary = model.forward_multitask(
        batch["inputs"],
        batch["action_history"],
        batch["route_action_chunk"],
        batch["route_candidate_index"],
        batch["route_side_index"],
        route_interaction_chunks,
    )
    target = batch["labels_relative"]
    target_nll = gaussian_nll(mean, log_variance, target)
    target_mse = deterministic_mse(mean, target)
    latent_mse = deterministic_mse(torch.tanh(latent), model.target_latent(target))
    obstacle_mse = F.smooth_l1_loss(auxiliary["obstacle_clearance"], batch["labels_obstacle_clearance"])
    inter_mse = F.smooth_l1_loss(auxiliary["inter_agent_clearance"], batch["labels_inter_agent_clearance"])
    clearance_mse = 0.5 * (obstacle_mse + inter_mse)
    visibility_bce = F.binary_cross_entropy_with_logits(
        auxiliary["target_visibility_logit"], batch["labels_target_visible"]
    )
    correction_mse = F.smooth_l1_loss(auxiliary["cbf_correction"], batch["labels_cbf_correction"])
    intervention_bce = F.binary_cross_entropy_with_logits(
        auxiliary["cbf_intervention_logit"], batch["labels_cbf_intervention"]
    )
    boundary_mse = F.smooth_l1_loss(auxiliary["boundary_clearance"], batch["labels_boundary_clearance"])
    slack_mse = F.smooth_l1_loss(auxiliary["cbf_min_slack"], batch["labels_cbf_min_slack"])
    stopping_distance_mse = F.smooth_l1_loss(
        auxiliary["stopping_distance"], batch["labels_stopping_distance"]
    )
    obstacle_ttc_mse = F.smooth_l1_loss(
        auxiliary["obstacle_ttc"] / 10.0, batch["labels_obstacle_ttc"] / 10.0
    )
    boundary_ttc_mse = F.smooth_l1_loss(
        auxiliary["boundary_ttc"] / 10.0, batch["labels_boundary_ttc"] / 10.0
    )
    pairwise_ttc_mse = F.smooth_l1_loss(
        auxiliary["pairwise_ttc_risk"] / 10.0, batch["labels_pairwise_ttc"] / 10.0
    )
    if hazard_positive_weight <= 0.0 or not 0.0 < quantile < 1.0:
        raise ValueError("hazard_positive_weight must be positive and quantile must be in (0,1)")
    hazard_loss_terms: list[torch.Tensor] = []
    quantile_loss_terms: list[torch.Tensor] = []
    hazard_metrics: dict[str, torch.Tensor] = {}
    for risk_name, label_name in (
        ("obstacle_ttc", "labels_obstacle_ttc"),
        ("boundary_ttc", "labels_boundary_ttc"),
        ("pairwise_ttc", "labels_pairwise_ttc"),
    ):
        target_ttc = batch[label_name]
        measured = torch.isfinite(target_ttc) & (target_ttc >= 0.0)
        target_bands = torch.stack(
            [(target_ttc <= threshold).float() for threshold in (0.5, 1.0, 2.0)], dim=-1
        )
        logits = auxiliary[f"{risk_name}_hazard_logits"]
        pos_weight = torch.full((3,), float(hazard_positive_weight), device=logits.device)
        hazard_element = F.binary_cross_entropy_with_logits(
            logits, target_bands, pos_weight=pos_weight, reduction="none"
        )
        hazard_element = hazard_element * measured.unsqueeze(-1)
        hazard_loss_terms.append(hazard_element.sum() / measured.sum().clamp_min(1.0) / 3.0)
        probability = torch.sigmoid(logits)
        for band_index, threshold in enumerate((0.5, 1.0, 2.0)):
            positive = measured & (target_ttc <= threshold)
            predicted_positive = measured & (probability[..., band_index] >= 0.5)
            true_positive = predicted_positive & positive
            hazard_metrics[f"{risk_name}_hazard_recall_{threshold:g}"] = (
                true_positive.sum().float() / positive.sum().clamp_min(1).float()
            )
        error = target_ttc / 10.0 - auxiliary[f"{risk_name}_lower_quantile"] / 10.0
        pinball = torch.maximum(float(quantile) * error, (float(quantile) - 1.0) * error)
        quantile_loss_terms.append((pinball * measured).sum() / measured.sum().clamp_min(1.0))
    hazard_loss = torch.stack(hazard_loss_terms).mean()
    quantile_loss = torch.stack(quantile_loss_terms).mean()
    slack_target = batch["labels_acceleration_slack"]
    slack_mask = slack_target != -1.0
    if bool(slack_mask.any()):
        acceleration_slack_mse = F.smooth_l1_loss(
            auxiliary["acceleration_slack"][slack_mask], slack_target[slack_mask]
        )
    else:
        acceleration_slack_mse = _zero_like(auxiliary["acceleration_slack"])
    progress_mse = F.smooth_l1_loss(auxiliary["route_progress"], batch["labels_route_progress"])
    feasibility_bce = F.binary_cross_entropy_with_logits(
        auxiliary["cbf_feasibility_logit"], batch["labels_cbf_feasible"]
    )
    runtime_mask = (batch["sample_type"] < 0.5) & (batch["route_candidate_index"] >= 0)
    identity_ce, identity_accuracy = _masked_cross_entropy(
        auxiliary["route_identity_logits"], batch["route_candidate_index"], runtime_mask
    )
    side_ce, side_accuracy = _masked_cross_entropy(
        auxiliary["route_side_logits"], batch["route_side_index"], runtime_mask
    )
    geometry_target = batch["route_geometry_valid"]
    geometry_bce = F.binary_cross_entropy_with_logits(auxiliary["route_geometry_logit"], geometry_target)
    termination_bce = F.binary_cross_entropy_with_logits(
        auxiliary["route_termination_logit"], batch["branch_terminated"]
    )
    route_ranking_loss, route_ranking_accuracy, route_ranking_groups = _route_progress_ranking_metrics(
        auxiliary,
        batch,
        horizon_index=route_ranking_horizon_index,
        margin=route_ranking_margin,
    )
    feasibility_probability = torch.sigmoid(auxiliary["cbf_feasibility_logit"])
    visibility_probability = torch.sigmoid(auxiliary["target_visibility_logit"])
    intervention_probability = torch.sigmoid(auxiliary["cbf_intervention_logit"])
    geometry_probability = torch.sigmoid(auxiliary["route_geometry_logit"])
    loss = (
        target_nll
        + weights["latent"] * latent_mse
        + weights["clearance"] * clearance_mse
        + weights["visibility"] * visibility_bce
        + weights["cbf_correction"] * correction_mse
        + weights["cbf_intervention"] * intervention_bce
        + weights["boundary"] * boundary_mse
        + weights["feasibility"] * feasibility_bce
        + weights["slack"] * slack_mse
        + weights["progress"] * progress_mse
        + weights["route_identity"] * identity_ce
        + weights["route_side"] * side_ce
        + weights["geometry"] * geometry_bce
        + weights["termination"] * termination_bce
        + weights["route_ranking"] * route_ranking_loss
        + weights["stopping_distance"] * stopping_distance_mse
        + weights["obstacle_ttc"] * obstacle_ttc_mse
        + weights["boundary_ttc"] * boundary_ttc_mse
        + weights["pairwise_ttc"] * pairwise_ttc_mse
        + weights["acceleration_slack"] * acceleration_slack_mse
        + weights["hazard"] * hazard_loss
        + weights["quantile"] * quantile_loss
    )
    return {
        "loss": loss,
        "target_nll": target_nll,
        "target_mse": target_mse,
        "latent_mse": latent_mse,
        "clearance_mse": clearance_mse,
        "obstacle_clearance_mse": obstacle_mse,
        "inter_agent_clearance_mse": inter_mse,
        "visibility_bce": visibility_bce,
        "visibility_brier": F.mse_loss(visibility_probability, batch["labels_target_visible"]),
        "cbf_correction_mse": correction_mse,
        "cbf_intervention_bce": intervention_bce,
        "cbf_intervention_brier": F.mse_loss(intervention_probability, batch["labels_cbf_intervention"]),
        "boundary_clearance_mse": boundary_mse,
        "cbf_min_slack_mse": slack_mse,
        "stopping_distance_mse": stopping_distance_mse,
        "obstacle_ttc_mse": obstacle_ttc_mse,
        "boundary_ttc_mse": boundary_ttc_mse,
        "pairwise_ttc_mse": pairwise_ttc_mse,
        "acceleration_slack_mse": acceleration_slack_mse,
        "hazard_loss": hazard_loss,
        "quantile_loss": quantile_loss,
        "route_progress_mse": progress_mse,
        "cbf_feasibility_bce": feasibility_bce,
        "cbf_feasibility_brier": F.mse_loss(feasibility_probability, batch["labels_cbf_feasible"]),
        "route_identity_ce": identity_ce,
        "route_identity_accuracy": identity_accuracy,
        "route_side_ce": side_ce,
        "route_side_accuracy": side_accuracy,
        "route_geometry_bce": geometry_bce,
        "route_geometry_brier": F.mse_loss(geometry_probability, geometry_target),
        "route_termination_bce": termination_bce,
        "route_progress_ranking_loss": route_ranking_loss,
        "route_progress_pairwise_accuracy": route_ranking_accuracy,
        "route_progress_ranking_groups": route_ranking_groups,
        "target_one_std_coverage": (torch.abs(mean - target) <= torch.exp(0.5 * log_variance)).float().mean(),
        **hazard_metrics,
    }


def _loader(tensors: dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int) -> DataLoader[Any]:
    ordered = tuple(tensors[name] for name in REQUIRED_ARRAYS)
    return DataLoader(
        TensorDataset(*ordered),
        batch_sampler=_RouteGroupBatchSampler(tensors, batch_size, shuffle, seed),
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _batch_dict(source: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in zip(REQUIRED_ARRAYS, source)}


def run_epoch(
    model: InteractionAwareActionConditionedRouteJEPAPredictor,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    weights: dict[str, float],
    *,
    route_ranking_horizon_index: int,
    route_ranking_margin: float,
    hazard_positive_weight: float,
    quantile: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for source in loader:
        batch = {name: value.to(device, non_blocking=True) for name, value in _batch_dict(source).items()}
        with torch.set_grad_enabled(training):
            metrics = _losses(
                model,
                batch,
                weights,
                route_ranking_horizon_index=route_ranking_horizon_index,
                route_ranking_margin=route_ranking_margin,
                hazard_positive_weight=hazard_positive_weight,
                quantile=quantile,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = int(batch["inputs"].shape[0])
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("Empty route JEPA data loader.")
    return {name: value / count for name, value in totals.items()}


def _source_hashes(protocol: Path) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "jepa_safe_capture_candidates.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "jepa_safe_capture_ranker.py",
        protocol.resolve(),
    )
    return {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in paths}


def _metric_group(name: str) -> str:
    if name.endswith("accuracy") or "brier" in name or "coverage" in name or "ranking_groups" in name:
        return "Calibration"
    if "ranking" in name:
        return "Route"
    if "loss" in name or name.endswith("mse") or name.endswith("nll") or name.endswith("ce"):
        return "Loss"
    if "feasibility" in name or "cbf" in name or "intervention" in name:
        return "Risk"
    if "route" in name or "geometry" in name:
        return "Route"
    if "visibility" in name:
        return "Visibility"
    return "Metrics"


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.hidden_dim, args.latent_dim, args.num_layers, args.patience) <= 0:
        raise ValueError("epochs, batch size, dimensions, layers, and patience must be positive.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or args.min_delta < 0.0:
        raise ValueError("learning rate must be positive; weight decay and min_delta non-negative.")
    if args.route_ranking_horizon_index < 0 or args.route_ranking_margin < 0.0:
        raise ValueError("Route ranking horizon must be non-negative and margin must be non-negative.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    if args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {args.tensorboard_logdir}")
    protocol = yaml.safe_load(args.protocol.resolve().read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("Route training requires the closed development protocol.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    train_tensors, train_metadata = load_dataset(args.train_dataset.resolve(), args.train_metadata.resolve(), "train")
    validation_tensors, validation_metadata = load_dataset(
        args.validation_dataset.resolve(), args.validation_metadata.resolve(), "validation"
    )
    _paired_contract(train_metadata, validation_metadata)
    if args.pairwise_relational:
        if not train_metadata.get("pairwise_action_conditioned_route_chunk", False):
            raise ValueError("--pairwise-relational requires a pairwise action-conditioned archive")
        if not validation_metadata.get("pairwise_action_conditioned_route_chunk", False):
            raise ValueError("Validation archive lacks the pairwise action-conditioned contract")
    model_config: dict[str, Any] = {
        "input_dim": 63,
        "horizon_count": int(train_tensors["labels_relative"].shape[1]),
        "action_dim": 3,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "num_layers": args.num_layers,
        "interaction_group_slices": protocol.get("model_contract", {}).get(
            "interaction_group_slices", [[0, 15], [15, 33], [33, 48], [48, 63]]
        ),
        "route_chunk_length": int(train_metadata["chunk_length_steps"]),
        "route_interaction_chunk_dim": (
            9 if train_metadata.get("pairwise_action_conditioned_route_chunk", False)
            else 3 if train_metadata.get("interaction_action_conditioned_route_chunk", False)
            else 0
        ),
        "route_candidate_count": 12,
        "route_side_count": 12,
        "pairwise_pooling": bool(args.pairwise_pooling),
        "pairwise_relational": bool(args.pairwise_relational),
    }
    model = build_action_conditioned_predictor(MODEL_TYPE, model_config).to(device)
    if not isinstance(model, InteractionAwareActionConditionedRouteHardNegativeJEPAPredictor):
        raise RuntimeError("Predictor factory did not create the route-aware JEPA model.")
    base_checkpoint_sha256 = None
    if args.base_checkpoint is not None:
        base_checkpoint = args.base_checkpoint.resolve()
        if not base_checkpoint.is_file():
            raise FileNotFoundError(f"Base checkpoint does not exist: {base_checkpoint}")
        base = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        if base.get("model_type") != "interaction_aware_action_conditioned_jepa_route_identity_v1":
            raise ValueError("--base-checkpoint must be a route-identity v1 checkpoint.")
        base_model = base.get("model", {})
        for field in ("input_dim", "horizon_count", "action_dim", "hidden_dim", "latent_dim", "num_layers"):
            if base_model.get(field) != model_config.get(field):
                raise ValueError(f"Base checkpoint model field {field!r} does not match the new model.")
        state = base.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Base checkpoint is missing model_state_dict.")
        target_state = model.state_dict()
        adapted_state: dict[str, torch.Tensor] = {}
        for name, value in state.items():
            if name not in target_state:
                continue
            if tuple(value.shape) == tuple(target_state[name].shape):
                adapted_state[name] = value
                continue
            # The interaction-conditioned route encoder appends a second
            # chunk to the first linear layer. Preserve the learned v1 route
            # columns and initialize only the new columns to zero so a zero
            # interaction feature reproduces the old model exactly.
            if name == "route_encoder.0.weight" and value.ndim == 2:
                target = target_state[name].detach().clone()
                if value.shape[0] != target.shape[0] or value.shape[1] > target.shape[1]:
                    raise ValueError(
                        "Base route encoder shape is incompatible with the interaction-conditioned model: "
                        f"base={tuple(value.shape)}, target={tuple(target.shape)}"
                    )
                target[:, : value.shape[1]] = value
                adapted_state[name] = target
                continue
            raise ValueError(
                f"Base checkpoint tensor {name!r} shape {tuple(value.shape)} does not match "
                f"target shape {tuple(target_state[name].shape)}."
            )
        missing, unexpected = model.load_state_dict(adapted_state, strict=False)
        expected_missing = {
            key for key in model.state_dict()
            if key.startswith((
                "stopping_distance_decoder.",
                "obstacle_ttc_decoder.",
                "boundary_ttc_decoder.",
                "pairwise_ttc_risk_decoder.",
                "acceleration_slack_decoder.",
                "risk_hazard_decoders.",
                "risk_quantile_decoders.",
                "pairwise_feature_encoder.",
                "pairwise_pair_encoder.",
                "pairwise_pool_encoder.",
            ))
        }
        if set(missing) != expected_missing or unexpected:
            raise ValueError(f"Base checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
        base_checkpoint_sha256 = _sha256(base_checkpoint)
    if args.head_only:
        trainable_prefixes = (
            "stopping_distance_decoder.",
            "obstacle_ttc_decoder.",
            "boundary_ttc_decoder.",
            "pairwise_ttc_risk_decoder.",
            "acceleration_slack_decoder.",
            "risk_hazard_decoders.",
            "risk_quantile_decoders.",
            "pairwise_feature_encoder.",
            "pairwise_pair_encoder.",
            "pairwise_pool_encoder.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)
    weights = {
        "latent": float(args.latent_loss_weight),
        "clearance": float(args.clearance_loss_weight),
        "visibility": float(args.visibility_loss_weight),
        "cbf_correction": float(args.cbf_correction_loss_weight),
        "cbf_intervention": float(args.cbf_intervention_loss_weight),
        "boundary": float(args.boundary_loss_weight),
        "feasibility": float(args.feasibility_loss_weight),
        "slack": float(args.slack_loss_weight),
        "progress": float(args.progress_loss_weight),
        "route_ranking": float(args.route_ranking_loss_weight),
        "route_identity": float(args.route_identity_loss_weight),
        "route_side": float(args.route_side_loss_weight),
        "geometry": float(args.geometry_loss_weight),
        "termination": float(args.termination_loss_weight),
        "stopping_distance": float(args.stopping_distance_loss_weight),
        "obstacle_ttc": float(args.obstacle_ttc_loss_weight),
        "boundary_ttc": float(args.boundary_ttc_loss_weight),
        "pairwise_ttc": float(args.pairwise_ttc_loss_weight),
        "acceleration_slack": float(args.acceleration_slack_loss_weight),
        "hazard": float(args.hazard_loss_weight),
        "quantile": float(args.quantile_loss_weight),
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("All task weights must be non-negative.")
    horizon_count = int(train_tensors["labels_route_progress"].shape[1])
    if args.route_ranking_horizon_index >= horizon_count:
        raise ValueError("--route-ranking-horizon-index is outside the archive horizon range.")
    train_loader = _loader(train_tensors, args.batch_size, True, args.seed)
    validation_loader = _loader(validation_tensors, args.batch_size, False, args.seed)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after --head-only filtering.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    args.tensorboard_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.tensorboard_logdir), flush_secs=5)
    source_hashes = _source_hashes(args.protocol)
    writer.add_text("Config/protocol", yaml.safe_dump(protocol, sort_keys=True), 0)
    writer.add_text("Config/model", json.dumps(model_config, indent=2), 0)
    writer.add_text("Config/optimization", json.dumps({**vars(args), "weights": weights}, default=str, indent=2), 0)
    writer.add_text("Dataset/train_metadata", json.dumps(train_metadata, indent=2), 0)
    writer.add_text("Dataset/validation_metadata", json.dumps(validation_metadata, indent=2), 0)
    writer.add_text("Provenance/source_hashes", json.dumps(source_hashes, indent=2), 0)
    writer.add_text("Provenance/base_checkpoint", json.dumps({"path": str(args.base_checkpoint.resolve()) if args.base_checkpoint else None, "sha256": base_checkpoint_sha256}, indent=2), 0)
    writer.add_scalar("Dataset/train_samples", len(train_tensors["inputs"]), 0)
    writer.add_scalar("Dataset/validation_samples", len(validation_tensors["inputs"]), 0)
    writer.add_scalar("Dataset/train_boundary_negative_fraction", float((train_tensors["labels_boundary_clearance"] < 0).any(dim=1).float().mean()), 0)
    writer.add_scalar("Dataset/train_cbf_infeasible_fraction", float((train_tensors["labels_cbf_feasible"] < 0.5).any(dim=1).float().mean()), 0)
    writer.add_scalar("Dataset/train_geometry_invalid_fraction", float((train_tensors["route_geometry_valid"] < 0.5).float().mean()), 0)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    stop_reason = "max_epochs"
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            weights,
            route_ranking_horizon_index=args.route_ranking_horizon_index,
            route_ranking_margin=args.route_ranking_margin,
            hazard_positive_weight=args.hazard_positive_weight,
            quantile=args.quantile,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                None,
                weights,
                route_ranking_horizon_index=args.route_ranking_horizon_index,
                route_ranking_margin=args.route_ranking_margin,
                hazard_positive_weight=args.hazard_positive_weight,
                quantile=args.quantile,
            )
        record: dict[str, float | int] = {"epoch": epoch}
        for name, value in train_metrics.items():
            record[f"train_{name}"] = value
            writer.add_scalar(f"{_metric_group(name)}/train_{name}", value, epoch)
        for name, value in validation_metrics.items():
            record[f"validation_{name}"] = value
            writer.add_scalar(f"{_metric_group(name)}/validation_{name}", value, epoch)
        history.append(record)
        if epoch % args.histogram_interval == 0 or epoch == 1:
            for name, parameter in model.named_parameters():
                writer.add_histogram(f"Parameters/{name}", parameter.detach(), epoch)
                if parameter.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", parameter.grad.detach(), epoch)
        improved = validation_metrics["loss"] < best_validation_loss - args.min_delta
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_type": MODEL_TYPE,
                    "model_state_dict": model.state_dict(),
                    "model": model_config,
                    "seed": args.seed,
                    "protocol": str(args.protocol.resolve()),
                    "protocol_sha256": _sha256(args.protocol.resolve()),
                    "train_metadata": train_metadata,
                    "validation_metadata": validation_metadata,
                    "task_weights": weights,
                    "source_hashes": source_hashes,
                    "training_variant": "route_identity_multitask_ranked_v2",
                    "route_ranking_horizon_index": args.route_ranking_horizon_index,
                    "route_ranking_margin": args.route_ranking_margin,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "base_checkpoint": str(args.base_checkpoint.resolve()) if args.base_checkpoint else None,
                    "base_checkpoint_sha256": base_checkpoint_sha256,
                    "head_only": bool(args.head_only),
                },
                args.output / "checkpoint.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                stop_reason = f"early_stop_no_validation_improvement_{args.patience}_epochs"
                writer.add_text("Training/stop_reason", stop_reason, epoch)
                break
        writer.add_scalar("Optimization/stale_epochs", stale_epochs, epoch)
        writer.flush()
    elapsed_seconds = time.perf_counter() - started
    hparams = {
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
    }
    hparam_metrics = {"hparam/best_validation_loss": best_validation_loss}
    try:
        writer.add_hparams(hparams, hparam_metrics)
        hparams_write_mode = "add_hparams"
    except (TypeError, AttributeError, ValueError) as exc:
        # TensorBoard 2.4.1 can fail to encode add_hparams with newer
        # protobuf containers.  Preserve the full configuration and metric as
        # ordinary TensorBoard text/scalars so the run remains auditable.
        hparams_write_mode = f"text_scalar_fallback:{type(exc).__name__}"
        writer.add_text("HParams/config", json.dumps(hparams, sort_keys=True), 0)
        writer.add_scalar("HParams/best_validation_loss", float(best_validation_loss), 0)
    writer.add_text("Training/summary", json.dumps({"stop_reason": stop_reason, "best_epoch": best_epoch}, indent=2), 0)
    writer.close()
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    run_metadata = {
        "model_type": MODEL_TYPE,
        "training_variant": "route_identity_multitask_ranked_v2",
        "device": str(device),
        "torch": version("torch"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "model": model_config,
        "task_weights": weights,
        "route_ranking_horizon_index": args.route_ranking_horizon_index,
        "route_ranking_margin": args.route_ranking_margin,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "base_checkpoint": str(args.base_checkpoint.resolve()) if args.base_checkpoint else None,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "head_only": bool(args.head_only),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "hparams_write_mode": hparams_write_mode,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed_seconds,
        "tensorboard_logdir": str(args.tensorboard_logdir.resolve()),
        "source_hashes": source_hashes,
        "train_dataset_sha256": _sha256(args.train_dataset.resolve()),
        "train_metadata_sha256": _sha256(args.train_metadata.resolve()),
        "validation_dataset_sha256": _sha256(args.validation_dataset.resolve()),
        "validation_metadata_sha256": _sha256(args.validation_metadata.resolve()),
        "development_only": True,
        "locked_test_opened": False,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_loss": best_validation_loss, "epochs_completed": len(history), "stop_reason": stop_reason, "device": str(device), "tensorboard_logdir": str(args.tensorboard_logdir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
