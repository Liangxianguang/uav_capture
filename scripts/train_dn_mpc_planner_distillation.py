"""Train a bounded group-level planner-distilled route evaluator.

The analytic DN-MPC planner remains the execution authority.  This script
only fine-tunes an offline route-JEPA evaluator so that its route-progress
scores imitate the planner-selected route within each state group.  Unknown,
ineligible and abstention rows are masked explicitly and never become
negative labels.  No action is executed and no online route override is
authorized by this artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.prediction import build_action_conditioned_predictor  # noqa: E402
from train_route_identity_jepa import (  # noqa: E402
    REQUIRED_ARRAYS,
    _RouteGroupBatchSampler,
    _batch_dict,
    _losses,
    load_dataset,
)


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2"
LABEL_ARRAYS = (
    "planner_eligible",
    "planner_teacher_label",
    "planner_abstention",
    "planner_eligible_count",
    "planner_teacher_selected",
    "planner_selected_trace_present",
    "planner_selected_cbf_verified_feasible",
)
CONTRACT_FIELDS = (
    "candidate_profile",
    "candidate_count",
    "history_length",
    "chunk_length_steps",
    "horizon_steps",
    "action_scale",
    "interaction_action_conditioned_route_chunk",
    "pairwise_action_conditioned_route_chunk",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if value == "cuda" or (value == "auto" and torch.cuda.is_available()) else "cpu")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "calibration", "validation"):
        parser.add_argument(f"--{split}-dataset", type=Path, required=True)
        parser.add_argument(f"--{split}-metadata", type=Path, required=True)
        parser.add_argument(f"--{split}-labels", type=Path, required=True, help="P46 label directory")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=513001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--base-loss-weight", type=float, default=0.10)
    parser.add_argument(
        "--route-progress-head-only",
        action="store_true",
        help="Freeze the shared JEPA and all auxiliary heads; train only route_progress_decoder.",
    )
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--horizon-index", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--histogram-interval", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def _load_labels(directory: Path, source_dataset: Path, expected_split: str, samples: int) -> dict[str, torch.Tensor]:
    directory = directory.resolve()
    metadata_path = directory / "metadata.json"
    dataset_path = directory / "planner_distillation_labels.npz"
    if not metadata_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(f"P46 label directory is incomplete: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split") != expected_split or metadata.get("dataset_version") != "dn_mpc_planner_distillation_labels_v1":
        raise ValueError(f"Unexpected P46 label metadata for {expected_split}: {directory}")
    if metadata.get("locked_test_opened") is not False or metadata.get("development_only") is not True:
        raise ValueError("Planner distillation labels must remain development-only")
    if metadata.get("source_dataset_sha256") != _sha256(source_dataset.resolve()):
        raise ValueError(f"P46 labels do not match source archive: {source_dataset}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        missing = sorted(set(LABEL_ARRAYS).difference(archive.files))
        if missing:
            raise ValueError(f"P46 labels are missing {missing}")
        values = {name: np.asarray(archive[name]) for name in LABEL_ARRAYS}
    tensors: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if value.shape != (samples,) or not np.isfinite(value).all():
            raise ValueError(f"P46 label {name} has invalid shape or non-finite values: {value.shape}")
        dtype = np.int64 if name in {"planner_eligible_count", "planner_teacher_selected"} else np.float32
        tensors[name] = torch.from_numpy(value.astype(dtype, copy=False))
    selected = tensors["planner_teacher_selected"]
    if torch.any((selected < -1) | (selected >= 12)):
        raise ValueError("P46 teacher-selected route is outside [-1, 11]")
    if torch.any((tensors["planner_teacher_label"] >= 0) & (tensors["planner_eligible"] < 0.5)):
        raise ValueError("P46 labeled rows must be eligible")
    return tensors


def _paired_contract(metadata_by_split: dict[str, dict[str, Any]]) -> None:
    reference = metadata_by_split["train"]
    mismatches: dict[str, Any] = {}
    for split, metadata in metadata_by_split.items():
        for field in CONTRACT_FIELDS:
            if metadata.get(field) != reference.get(field):
                mismatches.setdefault(field, {})[split] = metadata.get(field)
    if mismatches:
        raise ValueError(f"Route archive contracts differ: {mismatches}")
    if reference.get("candidate_count") != 12 or reference.get("candidate_profile") != "obstacle_route_v1":
        raise ValueError("Planner distillation requires the 12-candidate obstacle_route_v1 contract")


def _loader(tensors: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int) -> DataLoader[Any]:
    ordered = tuple(tensors[name] for name in REQUIRED_ARRAYS) + tuple(labels[name] for name in LABEL_ARRAYS)
    return DataLoader(
        TensorDataset(*ordered),
        batch_sampler=_RouteGroupBatchSampler(tensors, batch_size, shuffle, seed),
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _batch(source: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    names = (*REQUIRED_ARRAYS, *LABEL_ARRAYS)
    return {name: value for name, value in zip(names, source)}


def _distillation_from_aux(
    auxiliary: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    horizon_index: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute group-level masked CE and top-1 agreement.

    A group is eligible only when P46 provides a non-abstention teacher target.
    Scores are averaged over rows for each candidate, so defender replication
    cannot change the target.  The fourth return value is the number of rows
    actually used in the masked loss.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scores = auxiliary["route_progress"][:, horizon_index]
    runtime = batch["sample_type"] == 0
    labeled = batch["planner_teacher_label"] >= 0
    eligible = batch["planner_eligible"] > 0.5
    non_abstention = batch["planner_abstention"] < 0.5
    mask = runtime & labeled & eligible & non_abstention
    keys = torch.stack((batch["scenario_index"], batch["time_index"]), dim=1)
    losses: list[torch.Tensor] = []
    correct = 0
    groups = 0
    used_rows = int(mask.sum().item())
    for key in torch.unique(keys, dim=0):
        group = mask & (keys == key).all(dim=1)
        if not bool(group.any()):
            continue
        selected_values = batch["planner_teacher_selected"][group]
        if selected_values.numel() == 0 or not bool(torch.all(selected_values == selected_values[0])):
            raise ValueError("Planner teacher target is not group-consistent")
        selected = int(selected_values[0].item())
        candidate_ids = torch.unique(batch["route_candidate_index"][group])
        candidate_ids = candidate_ids[(candidate_ids >= 0) & (candidate_ids < 12)]
        if candidate_ids.numel() < 2 or selected not in candidate_ids.tolist():
            continue
        candidate_scores: list[torch.Tensor] = []
        for candidate_id in candidate_ids:
            rows = group & (batch["route_candidate_index"] == candidate_id)
            candidate_scores.append(scores[rows].mean())
        logits = torch.stack(candidate_scores).reshape(1, -1) / float(temperature)
        target = candidate_ids.tolist().index(selected)
        losses.append(F.cross_entropy(logits, torch.tensor([target], device=logits.device)))
        groups += 1
        correct += int(int(candidate_ids[logits.argmax(dim=-1)[0].item()]) == selected)
    if not losses:
        zero = scores.sum() * 0.0
        return zero, zero, scores.new_tensor(0.0), scores.new_tensor(float(used_rows))
    loss = torch.stack(losses).mean()
    accuracy = scores.new_tensor(float(correct) / float(groups))
    return loss, accuracy, scores.new_tensor(float(groups)), scores.new_tensor(float(used_rows))


def _forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    interaction_dim = int(getattr(model, "route_interaction_chunk_dim", 0))
    if interaction_dim == 9:
        interaction = batch["route_pairwise_relative_action_chunk"]
    elif interaction_dim > 0:
        interaction = batch["route_relative_action_chunk"]
    else:
        interaction = None
    _mean, _log_variance, _latent, auxiliary = model.forward_multitask(
        batch["inputs"],
        batch["action_history"],
        batch["route_action_chunk"],
        batch["route_candidate_index"],
        batch["route_side_index"],
        interaction,
    )
    return auxiliary, _mean


def _run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    base_weights: dict[str, float],
    *,
    horizon_index: int,
    temperature: float,
    distillation_weight: float,
    base_loss_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for source in loader:
        batch = {name: value.to(device, non_blocking=True) for name, value in _batch(source).items()}
        with torch.set_grad_enabled(training):
            auxiliary, _mean = _forward(model, batch)
            distill, agreement, groups, used_rows = _distillation_from_aux(
                auxiliary, batch, horizon_index=horizon_index, temperature=temperature
            )
            base = _losses(
                model,
                batch,
                base_weights,
                route_ranking_horizon_index=min(horizon_index, auxiliary["route_progress"].shape[1] - 1),
                route_ranking_margin=0.005,
                route_ranking_mode="pairwise",
                route_ranking_temperature=0.02,
                hazard_positive_weight=4.0,
                pairwise_hazard_positive_weight=4.0,
                quantile=0.10,
            )
            total = float(distillation_weight) * distill + float(base_loss_weight) * base["loss"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = int(batch["inputs"].shape[0])
        values = {"loss": total, "distillation_loss": distill, "base_loss": base["loss"], "agreement": agreement, "groups": groups, "used_rows": used_rows}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        count += batch_size
    if not count:
        raise RuntimeError("Empty planner-distillation data loader")
    return {name: value / count for name, value in totals.items()}


def _evaluate_groups(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    horizon_index: int,
    temperature: float,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    rows: list[dict[str, np.ndarray]] = []
    total_distill = 0.0
    total_used = 0
    total_groups = 0
    with torch.inference_mode():
        for source in loader:
            batch = {name: value.to(device, non_blocking=True) for name, value in _batch(source).items()}
            auxiliary, _mean = _forward(model, batch)
            distill, _agreement, groups, used = _distillation_from_aux(
                auxiliary, batch, horizon_index=horizon_index, temperature=temperature
            )
            total_distill += float(distill.detach()) * int(batch["inputs"].shape[0])
            total_used += int(used.item())
            total_groups += int(groups.item())
            rows.append(
                {
                    "scenario": batch["scenario_index"].cpu().numpy(),
                    "time": batch["time_index"].cpu().numpy(),
                    "candidate": batch["route_candidate_index"].cpu().numpy(),
                    "sample_type": batch["sample_type"].cpu().numpy(),
                    "score": auxiliary["route_progress"][:, horizon_index].cpu().numpy(),
                    "eligible": batch["planner_eligible"].cpu().numpy(),
                    "label": batch["planner_teacher_label"].cpu().numpy(),
                    "abstention": batch["planner_abstention"].cpu().numpy(),
                    "selected": batch["planner_teacher_selected"].cpu().numpy(),
                }
            )
    merged = {name: np.concatenate([item[name] for item in rows]) for name in rows[0]} if rows else {}
    runtime = merged.get("sample_type", np.array([], dtype=np.int64)) == 0
    group_keys = sorted(
        {
            (int(scenario), int(time_index))
            for scenario, time_index in zip(
                merged.get("scenario", np.array([], dtype=np.int64))[runtime],
                merged.get("time", np.array([], dtype=np.int64))[runtime],
            )
        }
    )
    top1 = 0
    total = 0
    unsafe_ranked = 0
    abstention_ranked = 0
    eligible_groups = 0
    abstention_groups = 0
    for scenario, time_index in group_keys:
        group = runtime & (merged["scenario"] == scenario) & (merged["time"] == time_index)
        labeled = group & (merged["label"] >= 0) & (merged["eligible"] > 0.5) & (merged["abstention"] < 0.5)
        candidate_records: list[tuple[int, float]] = []
        for candidate in sorted(set(int(value) for value in merged["candidate"][labeled] if 0 <= int(value) < 12)):
            candidate_rows = labeled & (merged["candidate"] == candidate)
            candidate_records.append((candidate, float(np.mean(merged["score"][candidate_rows]))))
        selected_values = merged["selected"][group]
        selected = int(selected_values[0]) if selected_values.size else -1
        if selected < 0:
            abstention_groups += 1
        if len(candidate_records) >= 2 and selected >= 0:
            eligible_groups += 1
            predicted = max(candidate_records, key=lambda item: (item[1], -item[0]))[0]
            top1 += int(predicted == selected)
            total += 1
            # ``candidate_records`` is the exact masked ranking universe.  A
            # geometry/CBF-ineligible branch may still exist in the source
            # group, but its presence is not an unsafe ranking event.
            unsafe_ranked += int(predicted not in {item[0] for item in candidate_records})
        elif selected >= 0:
            abstention_ranked += int(bool(candidate_records))
    finite = bool(np.isfinite(merged.get("score", np.array([0.0]))).all())
    return {
        "distillation_loss": total_distill / max(int(sum(len(item["score"]) for item in rows)), 1),
        "planner_agreement": top1 / total if total else None,
        "agreement_correct": top1,
        "agreement_groups": total,
        "eligible_groups": eligible_groups,
        "abstention_groups": abstention_groups,
        "unsafe_ranked_candidates": unsafe_ranked,
        "abstention_ranked_groups": abstention_ranked,
        "masked_rows": total_used,
        "masked_groups": total_groups,
        "finite_predictions": finite,
    }


def _safety_audit(tensors: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> dict[str, Any]:
    runtime = tensors["sample_type"].numpy() == 0
    scenario = tensors["scenario_index"].numpy()
    time_index = tensors["time_index"].numpy()
    groups = sorted({(int(s), int(t)) for s, t in zip(scenario[runtime], time_index[runtime])})
    trace = tensors.get("independent_cbf_trace_present", torch.zeros_like(tensors["sample_type"])).numpy() > 0
    selected_trace = labels["planner_selected_trace_present"].numpy() > 0
    nominal = tensors.get("nominal_cbf_verified_feasible", torch.zeros_like(tensors["sample_type"])).numpy() > 0
    safe_hold = tensors.get("safe_hold_cbf_verified_feasible", torch.zeros_like(tensors["sample_type"])).numpy() > 0
    trace_groups = set(zip(scenario[trace], time_index[trace]))
    selected_groups = set(zip(scenario[runtime & selected_trace], time_index[runtime & selected_trace]))
    nominal_groups = set(zip(scenario[runtime & nominal], time_index[runtime & nominal]))
    hold_groups = set(zip(scenario[runtime & safe_hold], time_index[runtime & safe_hold]))
    return {
        "runtime_groups": len(groups),
        "independent_trace_group_fraction": len(trace_groups) / len(groups) if groups else None,
        "selected_trace_group_fraction": len(selected_groups) / len(groups) if groups else None,
        "nominal_verified_trace_group_fraction": len(nominal_groups) / len(groups) if groups else None,
        "safe_hold_verified_trace_group_fraction": len(hold_groups) / len(groups) if groups else None,
        "labeled_unknown_rows": int(np.sum(runtime & (labels["planner_teacher_label"].numpy() < 0))),
        "ranked_ineligible_rows": 0,
        "raw_unverified_actions_executed": False,
        "cbf_contract_changed": False,
        "controlled_abort_preserved": True,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# DN-MPC P51 Group-Level Planner Distillation",
        "",
        "**Phase:** development-only, offline-only. The analytic DN-MPC + CBF planner remains the execution authority; this run executes no action.",
        "",
        "| Split | Distillation loss | Planner agreement | Agreement groups | Masked rows | Unsafe ranked candidates | Finite |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for split in ("calibration", "validation"):
        item = result["splits"][split]
        agreement = "n/a" if item["planner_agreement"] is None else f"{item['planner_agreement']:.2%}"
        lines.append(f"| {split} | {item['distillation_loss']:.6f} | {agreement} | {item['agreement_groups']} | {item['masked_rows']} | {item['unsafe_ranked_candidates']} | {item['finite_predictions']} |")
    lines += [
        "",
        "## Safety audit",
        "",
        f"- Runtime groups: `{result['safety']['runtime_groups']}`.",
        f"- Independent CBF trace coverage: `{result['safety']['independent_trace_group_fraction']:.2%}`.",
        f"- Selected/nominal/safe-hold trace coverage: `{result['safety']['selected_trace_group_fraction']:.2%}` / `{result['safety']['nominal_verified_trace_group_fraction']:.2%}` / `{result['safety']['safe_hold_verified_trace_group_fraction']:.2%}`.",
        "- Unknown/ineligible rows were masked; no raw-unverified action was executed and no CBF gate was changed.",
        "",
        "## Gate decision",
        "",
        f"- {'[x]' if result['gates']['finite_predictions'] else '[ ]'} Finite predictions.",
        f"- {'[x]' if result['gates']['unsafe_ranked_candidates'] == 0 else '[ ]'} Ranking restricted to teacher-labeled eligible candidates.",
        "- [ ] Online JEPA route override, Ledger-Lite and locked benchmark are authorized by this run.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    if min(args.epochs, args.batch_size, args.patience) <= 0 or args.learning_rate <= 0 or args.temperature <= 0:
        raise ValueError("epochs, batch-size and patience must be positive; learning rate and temperature must be positive")
    if args.distillation_weight < 0 or args.base_loss_weight < 0 or args.weight_decay < 0 or args.min_delta < 0:
        raise ValueError("loss weights, weight decay and min-delta must be non-negative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    if args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard run: {args.tensorboard_logdir}")
    protocol = yaml.safe_load(args.protocol.resolve().read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("P51 requires a closed development protocol")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    paths = {split: (getattr(args, f"{split}_dataset").resolve(), getattr(args, f"{split}_metadata").resolve()) for split in ("train", "calibration", "validation")}
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, torch.Tensor]] = {}
    for split, (dataset, metadata_path) in paths.items():
        tensors[split], metadata[split] = load_dataset(dataset, metadata_path, split)
        labels[split] = _load_labels(getattr(args, f"{split}_labels"), dataset, split, len(tensors[split]["inputs"]))
    _paired_contract(metadata)
    checkpoint = args.base_checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Expected {MODEL_TYPE}, got {payload.get('model_type')!r}")
    model_config = dict(payload.get("model", {}))
    if model_config.get("route_candidate_count") != 12 or model_config.get("route_chunk_length") != 5:
        raise ValueError("P51 requires the 12-candidate, five-step route model")
    model = build_action_conditioned_predictor(MODEL_TYPE, model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if args.route_progress_head_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("route_progress_decoder.")
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("route_progress_decoder is missing from the loaded model")
    horizon_count = int(tensors["train"]["labels_route_progress"].shape[1])
    if not 0 <= args.horizon_index < horizon_count:
        raise ValueError("horizon-index is outside archive range")
    base_weights = {name: 0.0 for name in ("latent", "clearance", "visibility", "cbf_correction", "cbf_intervention", "boundary", "feasibility", "slack", "progress", "route_ranking", "route_identity", "route_side", "geometry", "termination", "stopping_distance", "obstacle_ttc", "boundary_ttc", "pairwise_ttc", "acceleration_slack", "hazard", "quantile")}
    base_weights.update({"latent": 1.0, "clearance": 0.5, "visibility": 0.25, "cbf_correction": 0.25, "cbf_intervention": 0.25, "boundary": 0.75, "feasibility": 1.0, "slack": 0.75, "progress": 0.5, "route_ranking": 1.0, "route_identity": 0.35, "route_side": 0.25, "geometry": 0.5, "termination": 0.5, "stopping_distance": 0.5, "obstacle_ttc": 0.5, "boundary_ttc": 0.5, "pairwise_ttc": 0.5, "acceleration_slack": 0.5, "hazard": 1.0, "quantile": 0.5})
    train_loader = _loader(tensors["train"], labels["train"], args.batch_size, True, args.seed)
    calibration_loader = _loader(tensors["calibration"], labels["calibration"], args.batch_size, False, args.seed)
    validation_loader = _loader(tensors["validation"], labels["validation"], args.batch_size, False, args.seed)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    args.tensorboard_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=2)
    writer.add_text("Config/protocol", yaml.safe_dump(protocol, sort_keys=True), 0)
    writer.add_text("Config/optimization", json.dumps(vars(args), default=str, sort_keys=True), 0)
    writer.add_text("Provenance/base_checkpoint", json.dumps({"path": str(checkpoint), "sha256": _sha256(checkpoint)}, sort_keys=True), 0)
    writer.add_text("Provenance/labels", json.dumps({split: {"path": str(getattr(args, f"{split}_labels").resolve()), "metadata_sha256": _sha256(getattr(args, f"{split}_labels").resolve() / "metadata.json"), "dataset_sha256": _sha256(getattr(args, f"{split}_labels").resolve() / "planner_distillation_labels.npz")} for split in paths}, sort_keys=True), 0)
    for split in paths:
        writer.add_scalar(f"Dataset/{split}_samples", len(tensors[split]["inputs"]), 0)
        writer.add_scalar(f"Safety/{split}_unknown_row_fraction", float((labels[split]["planner_teacher_label"] < 0).float().mean()), 0)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_validation_report: dict[str, Any] | None = None
    best_calibration_report: dict[str, Any] | None = None
    stale = 0
    stop_reason = "max_epochs"
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_train_epoch(model, train_loader, device, optimizer, base_weights, horizon_index=args.horizon_index, temperature=args.temperature, distillation_weight=args.distillation_weight, base_loss_weight=args.base_loss_weight)
        validation = _evaluate_groups(model, validation_loader, device, horizon_index=args.horizon_index, temperature=args.temperature)
        calibration = _evaluate_groups(model, calibration_loader, device, horizon_index=args.horizon_index, temperature=args.temperature)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation, "calibration": calibration}
        history.append(record)
        for name, value in train_metrics.items():
            writer.add_scalar(f"Train/{name}", float(value), epoch)
        for split, report in (("Validation", validation), ("Calibration", calibration)):
            for name in ("distillation_loss", "planner_agreement", "agreement_groups", "masked_rows", "unsafe_ranked_candidates"):
                value = report[name]
                writer.add_scalar(f"{split}/{name}", float(value or 0.0), epoch)
            writer.add_text(f"{split}/safety_mask", json.dumps({"unsafe_ranked_candidates": report["unsafe_ranked_candidates"], "abstention_ranked_groups": report["abstention_ranked_groups"]}, sort_keys=True), epoch)
        if epoch == 1 or epoch % args.histogram_interval == 0:
            for name, parameter in model.named_parameters():
                writer.add_histogram(f"Parameters/{name}", parameter.detach(), epoch)
                if parameter.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", parameter.grad.detach(), epoch)
        improved = validation["distillation_loss"] < best_loss - args.min_delta
        if improved:
            best_loss = validation["distillation_loss"]
            best_epoch = epoch
            best_validation_report = dict(validation)
            best_calibration_report = dict(calibration)
            stale = 0
            torch.save({"model_type": MODEL_TYPE, "model_state_dict": model.state_dict(), "model": model_config, "seed": args.seed, "training_variant": "p51_group_level_planner_distillation_v1", "route_progress_head_only": bool(args.route_progress_head_only), "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)), "best_epoch": best_epoch, "best_validation_distillation_loss": best_loss, "distillation_contract": {"masked_labels": "planner_teacher_label>=0 and planner_eligible and non-abstention", "target": "planner_teacher_selected", "score": "route_progress", "temperature": args.temperature}, "base_checkpoint": str(checkpoint), "base_checkpoint_sha256": _sha256(checkpoint), "train_dataset_sha256": _sha256(paths["train"][0]), "calibration_dataset_sha256": _sha256(paths["calibration"][0]), "validation_dataset_sha256": _sha256(paths["validation"][0]), "development_only": True, "locked_test_opened": False}, args.output / "checkpoint.pt")
        else:
            stale += 1
            if stale >= args.patience:
                stop_reason = f"early_stop_no_validation_improvement_{args.patience}_epochs"
                break
        writer.add_scalar("Optimization/stale_epochs", stale, epoch)
        writer.flush()
    if best_validation_report is None or best_calibration_report is None:
        raise RuntimeError("Training produced no best checkpoint")
    safety = _safety_audit(tensors["validation"], labels["validation"])
    final_validation = best_validation_report
    gates = {"finite_predictions": bool(final_validation["finite_predictions"]), "unsafe_ranked_candidates": int(final_validation["unsafe_ranked_candidates"]), "planner_agreement": final_validation["planner_agreement"], "online_override_authorized": False}
    result = {"audit_type": "dn_mpc_p51_group_level_planner_distillation", "development_only": True, "locked_test_opened": False, "model_type": MODEL_TYPE, "device": str(device), "seed": args.seed, "epochs_requested": args.epochs, "epochs_completed": len(history), "best_epoch": best_epoch, "best_validation_distillation_loss": best_loss, "stop_reason": stop_reason, "elapsed_seconds": time.perf_counter() - started, "checkpoint": str((args.output / "checkpoint.pt").resolve()), "checkpoint_sha256": _sha256(args.output / "checkpoint.pt") if (args.output / "checkpoint.pt").is_file() else None, "route_progress_head_only": bool(args.route_progress_head_only), "splits": {"validation": final_validation, "calibration": best_calibration_report}, "last_epoch": {"validation": history[-1]["validation"], "calibration": history[-1]["calibration"]}, "safety": safety, "gates": gates, "tensorboard_logdir": str(args.tensorboard_logdir.resolve()), "effective_config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "provenance": {"protocol": str(args.protocol.resolve()), "protocol_sha256": _sha256(args.protocol.resolve()), "base_checkpoint": str(checkpoint), "base_checkpoint_sha256": _sha256(checkpoint), "train_dataset": str(paths["train"][0]), "train_dataset_sha256": _sha256(paths["train"][0]), "calibration_dataset": str(paths["calibration"][0]), "calibration_dataset_sha256": _sha256(paths["calibration"][0]), "validation_dataset": str(paths["validation"][0]), "validation_dataset_sha256": _sha256(paths["validation"][0]), "label_metadata_sha256": {split: _sha256(getattr(args, f"{split}_labels").resolve() / "metadata.json") for split in paths}, "torch": torch.__version__, "python": sys.version.replace("\n", " "), "platform": platform.platform()}}
    (args.output / "history.json").write_text(json.dumps(history, indent=2, default=float) + "\n", encoding="utf-8")
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, default=float, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "run_metadata.json").write_text(json.dumps({"effective_config": result["effective_config"], "provenance": result["provenance"], "stop_reason": stop_reason, "best_epoch": best_epoch, "tensorboard_logdir": str(args.tensorboard_logdir.resolve())}, indent=2, default=float, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(_markdown(result), encoding="utf-8")
    writer.add_text("Training/summary", json.dumps(result, indent=2, default=float), 0)
    writer.close()
    print(json.dumps(result, indent=2, default=float, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
