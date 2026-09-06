"""Audit pairwise interaction-tail observability and split coverage.

This is a bounded, offline audit for the route-JEPA hard-negative archives. It
does not train a model, create a Reliability Ledger, execute a candidate, or
open the locked split. The audit reconstructs the relative teammate position
and velocity block from the public 63-D observation contract, derives simple
formation-topology features, and compares them with the offline pairwise TTC
labels. The resulting separability is diagnostic only; it is not a safety
certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from torch.utils.tensorboard import SummaryWriter


DATASET_VERSION = "jepa_safe_capture_route_identity_hard_negative_v2"
EXPECTED_SPLITS = ("train", "validation", "calibration")
ROUTE_SIDES = (
    "nominal",
    "left",
    "right",
    "upper",
    "lower",
    "radial_out",
    "split",
    "contract",
    "hold",
    "intercept",
    "visibility_hold",
    "boundary_shadow",
)

# Frozen shape-aware policy_observations layout for the current v4 environment:
# local/target/visibility/uncertainty features(15), three teammate-relative
# positions(9), three relative velocities(9), legacy obstacle records(15), and
# shape-aware obstacle geometry(15) = 63 features. Prediction features are off.
N_TEAMMATES = 3
RELATIVE_POSITION_SLICE = slice(15, 24)
RELATIVE_VELOCITY_SLICE = slice(24, 33)
WORLD_EXTENT_M = 10.0
MAX_DEFENDER_SPEED_MPS = 5.0
DRONE_RADIUS_M = 0.25
PAIRWISE_MARGIN_M = 0.35
TTC_CLIP_SECONDS = 10.0
TTC_THRESHOLDS = (0.5, 1.0, 2.0)


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
    if isinstance(value, Mapping):
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
    """Mann-Whitney AUC with average ranks for ties, without sklearn."""

    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
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
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: None for key in ("p01", "p05", "p25", "p50", "p75", "p95", "p99")}
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        key: float(value)
        for key, value in zip(("p01", "p05", "p25", "p50", "p75", "p95", "p99"), quantiles)
    }


def _ttc_from_relative(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Estimate pairwise TTC from the latest public relative state."""

    safe_distance = 2.0 * DRONE_RADIUS_M + PAIRWISE_MARGIN_M
    count = position.shape[0]
    values: list[np.ndarray] = []
    for teammate in range(N_TEAMMATES):
        relative_position = position[:, teammate]
        relative_velocity = velocity[:, teammate]
        a = np.sum(relative_velocity * relative_velocity, axis=1)
        b = np.sum(relative_position * relative_velocity, axis=1)
        c = np.sum(relative_position * relative_position, axis=1) - safe_distance**2
        result = np.full(count, TTC_CLIP_SECONDS, dtype=np.float64)
        overlapping = c <= 0.0
        result[overlapping] = 0.0
        moving_towards = (c > 0.0) & (b < 0.0) & (a > 1e-12)
        discriminant = b * b - a * c
        solvable = moving_towards & (discriminant >= 0.0)
        result[solvable] = np.maximum(
            0.0,
            (-b[solvable] - np.sqrt(np.maximum(discriminant[solvable], 0.0))) / a[solvable],
        )
        values.append(np.clip(result, 0.0, TTC_CLIP_SECONDS))
    return np.stack(values, axis=1)


def _observation_features(inputs: np.ndarray) -> dict[str, np.ndarray]:
    if inputs.ndim != 3 or inputs.shape[1:] != (8, 63):
        raise ValueError(f"Expected inputs with shape [N,8,63], got {inputs.shape}")
    history = inputs[:, :, RELATIVE_POSITION_SLICE]
    velocity_history = inputs[:, :, RELATIVE_VELOCITY_SLICE]
    if history.shape[-1] != N_TEAMMATES * 3 or velocity_history.shape[-1] != N_TEAMMATES * 3:
        raise ValueError("63-D observation does not match the three-teammate interaction block.")
    positions = history.reshape(inputs.shape[0], 8, N_TEAMMATES, 3) * WORLD_EXTENT_M
    velocities = velocity_history.reshape(inputs.shape[0], 8, N_TEAMMATES, 3) * MAX_DEFENDER_SPEED_MPS
    pair_distances: list[np.ndarray] = []
    pair_speeds: list[np.ndarray] = []
    pair_closing: list[np.ndarray] = []
    for teammate in range(N_TEAMMATES):
        relative_position = positions[:, -1, teammate]
        relative_velocity = velocities[:, -1, teammate]
        distance = np.linalg.norm(relative_position, axis=1)
        pair_distances.append(distance)
        pair_speeds.append(np.linalg.norm(relative_velocity, axis=1))
        pair_closing.append(
            np.maximum(0.0, -np.divide(
                np.sum(relative_position * relative_velocity, axis=1),
                np.maximum(distance, 1e-9),
            ))
        )
    distance_matrix = np.stack(pair_distances, axis=1)
    speed_matrix = np.stack(pair_speeds, axis=1)
    closing_matrix = np.stack(pair_closing, axis=1)
    topology_edges_15 = np.sum(distance_matrix <= 1.5, axis=1)
    topology_edges_20 = np.sum(distance_matrix <= 2.0, axis=1)
    return {
        "min_pairwise_distance_m": np.min(distance_matrix, axis=1),
        "mean_pairwise_distance_m": np.mean(distance_matrix, axis=1),
        "pairwise_distance_spread_m": np.std(distance_matrix, axis=1),
        "max_relative_speed_mps": np.max(speed_matrix, axis=1),
        "mean_relative_speed_mps": np.mean(speed_matrix, axis=1),
        "max_closing_speed_mps": np.max(closing_matrix, axis=1),
        "mean_closing_speed_mps": np.mean(closing_matrix, axis=1),
        "formation_edges_le_1_5m": topology_edges_15.astype(np.float64),
        "formation_edges_le_2m": topology_edges_20.astype(np.float64),
        "observation_pairwise_ttc_s": np.min(_ttc_from_relative(positions[:, -1], velocities[:, -1]), axis=1),
        "history_min_pairwise_distance_m": np.min(
            np.stack(
                [
                    np.min(
                        np.stack(
                            [np.linalg.norm(positions[:, step, teammate], axis=1) for teammate in range(N_TEAMMATES)],
                            axis=1,
                        ),
                        axis=1,
                    )
                    for step in range(8)
                ],
                axis=1,
            ),
            axis=1,
        ),
    }


def _split_report(
    split: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    inputs = np.asarray(arrays["inputs"], dtype=np.float64)
    features = _observation_features(inputs)
    pairwise_ttc = np.asarray(arrays["labels_pairwise_ttc"], dtype=np.float64)
    if pairwise_ttc.ndim != 2 or pairwise_ttc.shape[1] != 5:
        raise ValueError(f"labels_pairwise_ttc must have shape [N,5], got {pairwise_ttc.shape}")
    measured = np.isfinite(pairwise_ttc)
    minimum_label_ttc = np.min(np.where(measured, pairwise_ttc, TTC_CLIP_SECONDS), axis=1)
    sample_type = np.asarray(arrays["sample_type"], dtype=np.int64)
    episode_seed = np.asarray(arrays["episode_seed"], dtype=np.int64)
    route_side = np.asarray(arrays["route_side_index"], dtype=np.int64)
    if not (sample_type.shape == episode_seed.shape == route_side.shape == (inputs.shape[0],)):
        raise ValueError("Sample metadata arrays do not align with inputs.")
    runtime = sample_type == 0
    shadow = sample_type == 1
    report: dict[str, Any] = {
        "split": split,
        "sample_count": int(inputs.shape[0]),
        "runtime_sample_count": int(runtime.sum()),
        "offline_shadow_sample_count": int(shadow.sum()),
        "episode_count": int(np.unique(episode_seed).size),
        "episode_seeds": sorted({int(value) for value in episode_seed.tolist()}),
        "route_side_counts": {
            ROUTE_SIDES[int(index)] if 0 <= int(index) < len(ROUTE_SIDES) else str(int(index)): int(count)
            for index, count in zip(*np.unique(route_side, return_counts=True))
        },
        "features": {},
        "hard_tail": {},
        "separability_auc": {},
        "topology": {},
        "archive_sha256": None,
        "metadata_sha256": None,
    }
    for name, values in features.items():
        report["features"][name] = _quantiles(values)
    for name in ("formation_edges_le_1_5m", "formation_edges_le_2m"):
        values = features[name]
        unique, counts = np.unique(values.astype(np.int64), return_counts=True)
        report["topology"][name] = {str(int(key)): int(value) for key, value in zip(unique, counts)}
    for threshold in TTC_THRESHOLDS:
        positive = minimum_label_ttc <= threshold
        measured_rows = np.any(measured, axis=1)
        positive &= measured_rows
        by_episode = {
            int(seed): bool(np.any(positive[episode_seed == seed]))
            for seed in np.unique(episode_seed)
        }
        runtime_positive = positive & runtime
        shadow_positive = positive & shadow
        key = f"le_{threshold:g}s"
        report["hard_tail"][key] = {
            "positive_sample_count": int(positive.sum()),
            "positive_sample_fraction": float(np.mean(positive)),
            "positive_episode_count": int(sum(by_episode.values())),
            "positive_episode_fraction": float(np.mean(list(by_episode.values()))) if by_episode else 0.0,
            "runtime_positive_count": int(runtime_positive.sum()),
            "runtime_positive_fraction": float(runtime_positive.sum() / max(int(runtime.sum()), 1)),
            "shadow_positive_count": int(shadow_positive.sum()),
            "shadow_positive_fraction": float(shadow_positive.sum() / max(int(shadow.sum()), 1)),
            "route_side_positive_counts": {
                ROUTE_SIDES[int(index)] if 0 <= int(index) < len(ROUTE_SIDES) else str(int(index)): int(
                    np.sum(positive & (route_side == index))
                )
                for index in np.unique(route_side)
            },
        }
        for feature_name in (
            "observation_pairwise_ttc_s",
            "min_pairwise_distance_m",
            "max_relative_speed_mps",
            "max_closing_speed_mps",
            "history_min_pairwise_distance_m",
        ):
            score = -features[feature_name]
            report["separability_auc"].setdefault(key, {})[feature_name] = _auc(positive, score)
    return report


def _load_archive(directory: Path, expected_split: str) -> tuple[dict[str, np.ndarray], dict[str, Any], Path, Path]:
    directory = directory.resolve()
    dataset = directory / "route_identity_counterfactual.npz"
    metadata_path = directory / "metadata.json"
    if not dataset.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Archive must contain route_identity_counterfactual.npz and metadata.json: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_version") != DATASET_VERSION or metadata.get("split") != expected_split:
        raise ValueError(f"Unexpected archive contract for {directory}: {metadata.get('dataset_version')!r}/{metadata.get('split')!r}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Archive is not development-only: {directory}")
    arrays_npz = np.load(dataset, allow_pickle=False)
    required = {"inputs", "labels_pairwise_ttc", "sample_type", "episode_seed", "route_side_index"}
    missing = sorted(required.difference(arrays_npz.files))
    if missing:
        raise ValueError(f"Archive {directory} is missing arrays: {missing}")
    arrays = {name: np.asarray(arrays_npz[name]) for name in arrays_npz.files}
    return arrays, metadata, dataset, metadata_path


def _distribution_shift(reports: dict[str, dict[str, Any]], feature_names: tuple[str, ...]) -> dict[str, Any]:
    train = reports["train"]
    result: dict[str, Any] = {}
    for feature in feature_names:
        train_q = train["features"][feature]
        result[feature] = {}
        for split in ("validation", "calibration"):
            current = reports[split]["features"][feature]
            result[feature][split] = {
                key: None if train_q[key] is None or current[key] is None else float(current[key] - train_q[key])
                for key in train_q
            }
    return result


def _decision(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    calibration = reports["calibration"]
    one_second = calibration["hard_tail"]["le_1s"]
    aucs = calibration["separability_auc"]["le_1s"]
    best_feature, best_auc = max(
        ((name, value) for name, value in aucs.items() if value is not None),
        key=lambda item: item[1],
        default=(None, None),
    )
    if one_second["positive_episode_count"] < 3:
        diagnosis = "data_coverage_insufficient"
        next_step = "collect additional disjoint pairwise-tail episodes before any model training"
    elif best_auc is not None and best_auc >= 0.70:
        diagnosis = "pairwise_signal_observable_representation_or_calibration_limited"
        next_step = "add explicit pairwise pooling/topology features and recalibrate the hazard head"
    else:
        diagnosis = "pairwise_signal_weak_or_feature_missing"
        next_step = "inspect observation information loss and collect targeted relative-velocity/topology transitions"
    return {
        "diagnosis": diagnosis,
        "best_calibration_feature_for_le_1s": best_feature,
        "best_calibration_auc_for_le_1s": best_auc,
        "calibration_le_1s_positive_episode_count": one_second["positive_episode_count"],
        "stop_full_jepa_training": True,
        "stop_new_ledger": True,
        "stop_closed_loop": True,
        "next_bounded_action": next_step,
        "reason": "calibration pairwise hazard recall/precision gate was not passed by the preceding hazard-head smoke; this audit is diagnostic only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in EXPECTED_SPLITS:
        parser.add_argument(f"--{split}-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite pairwise audit output/report.")
    tensorboard = _fresh_directory(args.tensorboard_logdir, "TensorBoard logdir")

    reports: dict[str, dict[str, Any]] = {}
    archive_paths: dict[str, Path] = {}
    metadata_paths: dict[str, Path] = {}
    seed_sets: dict[str, set[int]] = {}
    for split in EXPECTED_SPLITS:
        arrays, metadata, dataset, metadata_path = _load_archive(getattr(args, f"{split}_archive"), split)
        split_report = _split_report(split, arrays, metadata)
        split_report["archive_sha256"] = _sha256(dataset)
        split_report["metadata_sha256"] = _sha256(metadata_path)
        reports[split] = split_report
        archive_paths[split] = dataset
        metadata_paths[split] = metadata_path
        seed_sets[split] = set(split_report["episode_seeds"])
    overlaps = {
        f"{first}_{second}": sorted(seed_sets[first] & seed_sets[second])
        for index, first in enumerate(EXPECTED_SPLITS)
        for second in EXPECTED_SPLITS[index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"Episode seed overlap detected: {overlaps}")
    feature_names = (
        "min_pairwise_distance_m",
        "mean_pairwise_distance_m",
        "pairwise_distance_spread_m",
        "max_relative_speed_mps",
        "mean_relative_speed_mps",
        "max_closing_speed_mps",
        "mean_closing_speed_mps",
        "observation_pairwise_ttc_s",
        "history_min_pairwise_distance_m",
    )
    result: dict[str, Any] = {
        "audit": "jepa_pairwise_interaction_tail",
        "dataset_version": DATASET_VERSION,
        "archives": reports,
        "episode_seed_disjoint": True,
        "distribution_shift_vs_train": _distribution_shift(reports, feature_names),
        "decision": _decision(reports),
        "contract": {
            "development_only": True,
            "locked_test_opened": False,
            "raw_unverified_execution_allowed": False,
            "cbf_margin_changed": False,
            "stale_ood_nonfinite_gates_changed": False,
            "relative_position_slice": [RELATIVE_POSITION_SLICE.start, RELATIVE_POSITION_SLICE.stop],
            "relative_velocity_slice": [RELATIVE_VELOCITY_SLICE.start, RELATIVE_VELOCITY_SLICE.stop],
            "tensorboard_logdir": str(tensorboard),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("Provenance/contract", json.dumps(result["contract"], sort_keys=True), 0)
        writer.add_text(
            "Provenance/archives",
            json.dumps({split: {"dataset": str(path), "sha256": _sha256(path)} for split, path in archive_paths.items()}, sort_keys=True),
            0,
        )
        writer.add_text("Decision/diagnosis", json.dumps(result["decision"], sort_keys=True), 0)
        for split, split_report in reports.items():
            writer.add_scalar(f"Coverage/{split}/sample_count", split_report["sample_count"], 0)
            writer.add_scalar(f"Coverage/{split}/episode_count", split_report["episode_count"], 0)
            writer.add_scalar(f"Coverage/{split}/runtime_sample_count", split_report["runtime_sample_count"], 0)
            for key, tail in split_report["hard_tail"].items():
                writer.add_scalar(f"Coverage/{split}/hard_tail/{key}/sample_fraction", tail["positive_sample_fraction"], 0)
                writer.add_scalar(f"Coverage/{split}/hard_tail/{key}/episode_fraction", tail["positive_episode_fraction"], 0)
                writer.add_scalar(f"Coverage/{split}/hard_tail/{key}/positive_count", tail["positive_sample_count"], 0)
            for name, quantiles in split_report["features"].items():
                for quantile, value in quantiles.items():
                    if value is not None:
                        writer.add_scalar(f"Observation/{split}/{name}/{quantile}", value, 0)
            for name, counts in split_report["topology"].items():
                for category, count in counts.items():
                    writer.add_scalar(f"Topology/{split}/{name}/edges_{category}", count, 0)
            for key, values in split_report["separability_auc"].items():
                for name, value in values.items():
                    if value is not None:
                        writer.add_scalar(f"Separability/{split}/{key}/{name}/auc", value, 0)
    report_lines = [
        "# JEPA Pairwise Interaction-Tail Audit",
        "",
        "**Phase:** development-only offline audit  ",
        "**Decision:** stop full JEPA training, new Ledger calibration, and closed-loop evaluation pending a pairwise gate  ",
        "**Locked test:** not opened  ",
        "",
        "## Scope",
        "",
        "This audit reconstructs relative teammate position/velocity from the public 63-D observation contract and measures pairwise hard-tail coverage and simple observability. It does not execute actions, change CBF margins, disable stale/OOD gates, or certify safety.",
        "",
        "## Coverage",
        "",
        "| split | samples | episodes | TTC <=0.5s samples | TTC <=1s samples | TTC <=2s samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in EXPECTED_SPLITS:
        item = reports[split]
        report_lines.append(
            f"| {split} | {item['sample_count']} | {item['episode_count']} | "
            f"{item['hard_tail']['le_0.5s']['positive_sample_count']} "
            f"({item['hard_tail']['le_0.5s']['positive_sample_fraction']:.1%}) | "
            f"{item['hard_tail']['le_1s']['positive_sample_count']} "
            f"({item['hard_tail']['le_1s']['positive_sample_fraction']:.1%}) | "
            f"{item['hard_tail']['le_2s']['positive_sample_count']} "
            f"({item['hard_tail']['le_2s']['positive_sample_fraction']:.1%}) |"
        )
    report_lines.extend([
        "",
        "## Calibration observability",
        "",
        "| feature score (higher means more risky) | AUC for pairwise TTC <=1s |",
        "|---|---:|",
    ])
    for name, value in reports["calibration"]["separability_auc"]["le_1s"].items():
        report_lines.append(f"| `{name}` | {'n/a' if value is None else f'{value:.3f}'} |")
    report_lines.extend([
        "",
        "## Diagnosis",
        "",
        f"- **Classification:** `{result['decision']['diagnosis']}`",
        f"- **Best calibration feature:** `{result['decision']['best_calibration_feature_for_le_1s']}` with AUC `{result['decision']['best_calibration_auc_for_le_1s']}`",
        f"- **Pairwise-positive calibration episodes:** `{result['decision']['calibration_le_1s_positive_episode_count']}`",
        f"- **Next bounded action:** {result['decision']['next_bounded_action']}",
        "",
        "The preceding hazard-head smoke did not pass the calibration pairwise recall/precision gate. This audit therefore stops further training and runtime integration; its AUC values are diagnostic evidence, not a safe-capture claim.",
        "",
        "## Provenance",
        "",
        f"- JSON: `{output}`",
        f"- TensorBoard: `{tensorboard}`",
        "- `locked_test_opened=false`",
        "- `raw_unverified_execution_allowed=false`",
        "- `cbf_margin_changed=false`",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
