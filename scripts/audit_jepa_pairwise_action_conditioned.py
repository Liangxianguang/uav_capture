"""Audit candidate-action-conditioned pairwise interaction features.

This is an offline, development-only audit. It projects the focal defender's
candidate first-step velocity while holding teammate velocities at the public
belief value, then measures short-horizon relative-velocity and formation
topology features against the recorded pairwise TTC labels. It never executes
an action, changes a CBF contract, trains a model, or creates a Ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


DATASET_VERSION = "jepa_safe_capture_route_identity_hard_negative_v2"
SPLITS = ("train", "validation", "calibration")
N_TEAMMATES = 3
FOCAL_VELOCITY_SLICE = slice(0, 3)
RELATIVE_POSITION_SLICE = slice(15, 24)
RELATIVE_VELOCITY_SLICE = slice(24, 33)
WORLD_EXTENT_M = 10.0
MAX_DEFENDER_SPEED_MPS = 5.0
DRONE_RADIUS_M = 0.25
PAIRWISE_MARGIN_M = 0.35
TTC_CLIP_SECONDS = 10.0
DT_SECONDS = 0.1
PROJECTION_HORIZONS = (0.3, 0.5, 1.0)
HAZARD_THRESHOLD_S = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _fresh_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {label}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _pairwise_ttc(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    safe_distance = 2.0 * DRONE_RADIUS_M + PAIRWISE_MARGIN_M
    results = np.full(position.shape[:2], TTC_CLIP_SECONDS, dtype=np.float64)
    for teammate in range(N_TEAMMATES):
        relative_position = position[:, teammate]
        relative_velocity = velocity[:, teammate]
        a = np.sum(relative_velocity * relative_velocity, axis=1)
        b = np.sum(relative_position * relative_velocity, axis=1)
        c = np.sum(relative_position * relative_position, axis=1) - safe_distance**2
        overlap = c <= 0.0
        results[overlap, teammate] = 0.0
        approaching = (c > 0.0) & (b < 0.0) & (a > 1e-12)
        discriminant = b * b - a * c
        solvable = approaching & (discriminant >= 0.0)
        results[solvable, teammate] = np.maximum(
            0.0,
            (-b[solvable] - np.sqrt(np.maximum(discriminant[solvable], 0.0))) / a[solvable],
        )
    return np.clip(results, 0.0, TTC_CLIP_SECONDS)


def _candidate_chunk_distances(
    positions: np.ndarray,
    observed_velocity: np.ndarray,
    current_velocity: np.ndarray,
    route_action_chunk: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project pairwise geometry through the public, reachable action chunk.

    The teammate absolute velocity is reconstructed from the current public
    relative-velocity block and the focal current velocity.  The candidate
    focal velocity follows each projected chunk step and holds its final value
    after the chunk.  No simulator state or future target/teammate truth is
    consulted; this is an offline observability feature only.
    """

    if route_action_chunk.shape[1] <= 0:
        raise ValueError("route_action_chunk must contain at least one step")
    max_horizon = max(PROJECTION_HORIZONS)
    steps = int(math.ceil(max_horizon / DT_SECONDS))
    teammate_velocity = observed_velocity + current_velocity[:, None, :]
    projected = np.asarray(positions, dtype=np.float64).copy()
    traces: list[np.ndarray] = []
    for step in range(steps):
        chunk_index = min(step, route_action_chunk.shape[1] - 1)
        focal_velocity = np.asarray(route_action_chunk[:, chunk_index, :], dtype=np.float64)
        relative_velocity = teammate_velocity - focal_velocity[:, None, :]
        projected = projected + relative_velocity * DT_SECONDS
        traces.append(np.linalg.norm(projected, axis=2))
    trace = np.stack(traces, axis=1)
    features: dict[str, np.ndarray] = {}
    for horizon in PROJECTION_HORIZONS:
        end = min(int(math.ceil(horizon / DT_SECONDS)), trace.shape[1])
        window = trace[:, :end, :]
        features[f"candidate_chunk_min_distance_{horizon:g}s"] = np.min(window, axis=(1, 2))
        features[f"candidate_chunk_endpoint_distance_{horizon:g}s"] = np.min(trace[:, end - 1, :], axis=1)
    return features


def _features(inputs: np.ndarray, route_action_chunk: np.ndarray) -> dict[str, np.ndarray]:
    if inputs.ndim != 3 or inputs.shape[1:] != (8, 63):
        raise ValueError(f"Expected inputs [N,8,63], got {inputs.shape}")
    if route_action_chunk.ndim != 3 or route_action_chunk.shape[0] != inputs.shape[0]:
        raise ValueError("route_action_chunk must have shape [N,chunk,3]")
    current_velocity = inputs[:, -1, FOCAL_VELOCITY_SLICE] * MAX_DEFENDER_SPEED_MPS
    positions = inputs[:, -1, RELATIVE_POSITION_SLICE].reshape(-1, N_TEAMMATES, 3) * WORLD_EXTENT_M
    observed_velocity = inputs[:, -1, RELATIVE_VELOCITY_SLICE].reshape(-1, N_TEAMMATES, 3) * MAX_DEFENDER_SPEED_MPS
    candidate_velocity = route_action_chunk[:, 0, :].astype(np.float64)
    # Relative velocity is teammate velocity minus focal velocity. The
    # candidate replaces only the focal velocity; teammate public velocity is
    # held fixed to avoid using future simulator state.
    candidate_relative_velocity = observed_velocity - (candidate_velocity - current_velocity)[:, None, :]
    observed_ttc = np.min(_pairwise_ttc(positions, observed_velocity), axis=1)
    candidate_ttc = np.min(_pairwise_ttc(positions, candidate_relative_velocity), axis=1)
    closing_speed = np.maximum(
        0.0,
        -np.sum(positions * candidate_relative_velocity, axis=2)
        / np.maximum(np.linalg.norm(positions, axis=2), 1e-9),
    )
    chunk_distances = _candidate_chunk_distances(
        positions,
        observed_velocity,
        current_velocity,
        route_action_chunk,
    )
    distances = {}
    for horizon in PROJECTION_HORIZONS:
        projected = positions + candidate_relative_velocity * float(horizon)
        distances[f"candidate_min_distance_{horizon:g}s"] = np.min(np.linalg.norm(projected, axis=2), axis=1)
        distances[f"candidate_topology_edges_{horizon:g}s"] = np.sum(
            np.linalg.norm(projected, axis=2) <= (2.0 if horizon >= 0.5 else 1.5), axis=1
        ).astype(np.float64)
    current_edges = np.sum(np.linalg.norm(positions, axis=2) <= 2.0, axis=1).astype(np.float64)
    return {
        "observed_min_ttc_s": observed_ttc,
        "candidate_min_ttc_s": candidate_ttc,
        "candidate_max_closing_speed_mps": np.max(closing_speed, axis=1),
        "candidate_mean_closing_speed_mps": np.mean(closing_speed, axis=1),
        "candidate_topology_edge_delta": distances["candidate_topology_edges_1s"] - current_edges,
        **chunk_distances,
        **distances,
    }


def _metrics(labels: np.ndarray, values: np.ndarray, score_direction: str) -> dict[str, float | None]:
    score = values if score_direction == "high" else -values
    positive = labels <= HAZARD_THRESHOLD_S
    auc = _auc(positive, score)
    best: dict[str, float | None] = {"threshold": None, "precision": None, "recall": None, "f1": None}
    finite = np.isfinite(score)
    if np.any(finite):
        candidates = np.unique(np.quantile(score[finite], np.linspace(0.02, 0.98, 49)))
        for threshold in candidates:
            predicted = score >= threshold
            tp = int(np.sum(predicted & positive))
            fp = int(np.sum(predicted & ~positive))
            fn = int(np.sum(~predicted & positive))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            if best["f1"] is None or f1 > float(best["f1"]):
                best = {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1}
    return {"auc_le_1s": auc, **best}


def _load(path: Path, expected_split: str) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    directory = path.resolve()
    dataset = directory / "route_identity_counterfactual.npz"
    metadata_path = directory / "metadata.json"
    if not dataset.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing dataset or metadata: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_version") != DATASET_VERSION or metadata.get("split") != expected_split:
        raise ValueError(f"Unexpected archive contract: {directory}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Archive is not development-only: {directory}")
    arrays_npz = np.load(dataset, allow_pickle=False)
    required = {"inputs", "route_action_chunk", "labels_pairwise_ttc", "sample_type", "episode_seed"}
    missing = sorted(required.difference(arrays_npz.files))
    if missing:
        raise ValueError(f"Archive {directory} is missing {missing}")
    return {name: np.asarray(arrays_npz[name]) for name in arrays_npz.files}, metadata, dataset


def _split_result(split: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    inputs = arrays["inputs"]
    labels = np.min(np.asarray(arrays["labels_pairwise_ttc"], dtype=np.float64), axis=1)
    sample_type = np.asarray(arrays["sample_type"], dtype=np.int64)
    features = _features(inputs, arrays["route_action_chunk"])
    runtime = sample_type == 0
    metrics: dict[str, Any] = {}
    directions = {
        "observed_min_ttc_s": "low",
        "candidate_min_ttc_s": "low",
        "candidate_max_closing_speed_mps": "high",
        "candidate_mean_closing_speed_mps": "high",
        "candidate_topology_edge_delta": "high",
        "candidate_min_distance_0.3s": "low",
        "candidate_min_distance_0.5s": "low",
        "candidate_min_distance_1s": "low",
        "candidate_chunk_min_distance_0.3s": "low",
        "candidate_chunk_min_distance_0.5s": "low",
        "candidate_chunk_min_distance_1s": "low",
        "candidate_chunk_endpoint_distance_0.3s": "low",
        "candidate_chunk_endpoint_distance_0.5s": "low",
        "candidate_chunk_endpoint_distance_1s": "low",
        "candidate_topology_edges_0.3s": "high",
        "candidate_topology_edges_0.5s": "high",
        "candidate_topology_edges_1s": "high",
    }
    for name, direction in directions.items():
        metrics[name] = {
            "all": _metrics(labels, features[name], direction),
            "runtime": _metrics(labels[runtime], features[name][runtime], direction),
        }
    runtime_labels = labels[runtime]
    return {
        "split": split,
        "sample_count": int(labels.size),
        "runtime_sample_count": int(runtime.sum()),
        "episode_seeds": sorted({int(value) for value in arrays["episode_seed"].tolist()}),
        "pairwise_positive_runtime_count_le_1s": int(np.sum(runtime_labels <= HAZARD_THRESHOLD_S)),
        "pairwise_positive_runtime_fraction_le_1s": float(np.mean(runtime_labels <= HAZARD_THRESHOLD_S)),
        "features": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument(f"--{split}-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    tensorboard = _fresh_directory(args.tensorboard_logdir, "TensorBoard logdir")
    reports: dict[str, dict[str, Any]] = {}
    archives: dict[str, dict[str, str]] = {}
    seed_sets: dict[str, set[int]] = {}
    for split in SPLITS:
        arrays, metadata, dataset = _load(getattr(args, f"{split}_archive"), split)
        reports[split] = _split_result(split, arrays)
        seed_sets[split] = set(reports[split]["episode_seeds"])
        archives[split] = {
            "dataset": str(dataset),
            "dataset_sha256": _sha256(dataset),
            "metadata_sha256": _sha256(dataset.with_name("metadata.json")),
        }
    overlaps = {
        f"{first}_{second}": sorted(seed_sets[first] & seed_sets[second])
        for index, first in enumerate(SPLITS)
        for second in SPLITS[index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"Episode seed overlap detected: {overlaps}")
    calibration = reports["calibration"]
    candidate_auc = calibration["features"]["candidate_min_ttc_s"]["runtime"]["auc_le_1s"]
    observed_auc = calibration["features"]["observed_min_ttc_s"]["runtime"]["auc_le_1s"]
    topology_auc = calibration["features"]["candidate_topology_edge_delta"]["runtime"]["auc_le_1s"]
    candidate_features = {
        name: value
        for name, value in calibration["features"].items()
        if name != "observed_min_ttc_s" and value["runtime"]["auc_le_1s"] is not None
    }
    best = max(
        ((name, value["runtime"]["auc_le_1s"]) for name, value in candidate_features.items()),
        key=lambda item: float(item[1]),
    )
    best_values = calibration["features"][best[0]]["runtime"]
    gate_passed = any(
        values["runtime"]["precision"] is not None
        and values["runtime"]["recall"] is not None
        and float(values["runtime"]["precision"]) >= 0.50
        and float(values["runtime"]["recall"]) >= 0.80
        for values in candidate_features.values()
    )
    result: dict[str, Any] = {
        "audit": "jepa_pairwise_action_conditioned",
        "dataset_version": DATASET_VERSION,
        "archives": archives,
        "reports": reports,
        "episode_seed_disjoint": True,
        "decision": {
            "observed_auc_le_1s": observed_auc,
            "candidate_action_conditioned_auc_le_1s": candidate_auc,
            "candidate_topology_delta_auc_le_1s": topology_auc,
            "best_candidate_runtime_feature": best[0],
            "best_candidate_runtime_auc_le_1s": best[1],
            "best_candidate_runtime_precision": best_values["precision"],
            "best_candidate_runtime_recall": best_values["recall"],
            "candidate_improves_over_observed": bool(best[1] > observed_auc),
            "pairwise_gate_recall_target": 0.80,
            "pairwise_gate_precision_target": 0.50,
            "pairwise_gate_passed": bool(gate_passed),
            "jepa_training_authorized": bool(gate_passed),
            "stop_full_jepa_training": not bool(gate_passed),
            "stop_new_ledger": True,
            "stop_closed_loop": True,
            "interpretation": "diagnostic action-conditioned observability only; CBF remains the sole safety filter",
        },
        "contract": {
            "development_only": True,
            "locked_test_opened": False,
            "raw_unverified_execution_allowed": False,
            "cbf_margin_changed": False,
            "candidate_action_is_first_step_velocity": True,
            "teammate_future_state_used": False,
            "projection_horizons_s": list(PROJECTION_HORIZONS),
            "tensorboard_logdir": str(tensorboard),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("Provenance/contract", json.dumps(result["contract"], sort_keys=True), 0)
        writer.add_text("Provenance/archives", json.dumps(archives, sort_keys=True), 0)
        writer.add_text("Decision/summary", json.dumps(result["decision"], sort_keys=True), 0)
        for split, report in reports.items():
            writer.add_scalar(f"Coverage/{split}/sample_count", report["sample_count"], 0)
            writer.add_scalar(f"Coverage/{split}/runtime_positive_fraction_le_1s", report["pairwise_positive_runtime_fraction_le_1s"], 0)
            for name, values in report["features"].items():
                auc = values["runtime"]["auc_le_1s"]
                if auc is not None:
                    writer.add_scalar(f"AUC/{split}/{name}_runtime_le_1s", float(auc), 0)
        writer.add_scalar("Gate/calibration_candidate_improves_over_observed", float(result["decision"]["candidate_improves_over_observed"]), 0)
        writer.flush()
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
