"""Calibration-only transforms for JEPA clearance lower-bound predictions.
The learned clearance heads emit values in the archive's normalized coordinate
system.  This module keeps the conversion and quantile calibration explicit so
that a runtime ledger can apply exactly the transform that was fitted on an
independent calibration split.  It does not replace the geometric CBF safety
proof.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


CALIBRATION_QUANTILE = 0.10


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def normalized_to_meters(value: Any, world_extent_m: float, *, name: str = "value") -> np.ndarray:
    """Convert a normalized clearance array to metres using one frozen scale."""

    extent = float(world_extent_m)
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("world_extent_m must be finite and positive.")
    return _finite_array(value, name=name) * extent


def q10_residual_offset(
    raw_prediction_m: Any,
    label_m: Any,
    *,
    quantile: float = CALIBRATION_QUANTILE,
) -> dict[str, Any]:
    """Fit a lower-quantile residual offset on calibration-only samples.

    ``residual = label - raw_prediction`` and the fitted transform is
    ``calibrated = raw_prediction + offset_q10``.  The offset is never fitted
    from development or locked episodes and is intentionally not a CBF margin.
    """

    q = float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be strictly between zero and one.")
    raw = _finite_array(raw_prediction_m, name="raw_prediction_m")
    label = _finite_array(label_m, name="label_m")
    if raw.shape != label.shape:
        raise ValueError("raw prediction and label shapes must match.")
    residual = label - raw
    flat = residual.reshape(-1)
    if flat.size < 2:
        raise ValueError("At least two calibration residuals are required.")
    offset = float(np.quantile(flat, q, method="linear"))
    sorted_residual = np.sort(flat)
    # A deterministic order-statistic confidence interval avoids stochastic
    # bootstrap state in the published transform while still recording
    # uncertainty around the fitted quantile.
    standard_error = float(np.sqrt(flat.size * q * (1.0 - q)))
    lower_index = max(0, int(np.floor(flat.size * q - 1.96 * standard_error)))
    upper_index = min(flat.size - 1, int(np.ceil(flat.size * q + 1.96 * standard_error)))
    calibrated = raw + offset
    calibrated_residual = label - calibrated
    return {
        "quantile": q,
        "method": "numpy_linear_order_statistic_ci",
        "sample_count": int(flat.size),
        "offset_m": offset,
        "quantile_ci95_m": [float(sorted_residual[lower_index]), float(sorted_residual[upper_index])],
        "raw_prediction_mean_m": float(np.mean(raw)),
        "label_mean_m": float(np.mean(label)),
        "raw_residual_mean_m": float(np.mean(residual)),
        "calibrated_residual_mean_m": float(np.mean(calibrated_residual)),
        "raw_overprediction_rate": float(np.mean(residual < 0.0)),
        "calibrated_overprediction_rate": float(np.mean(calibrated_residual < 0.0)),
        "calibrated_residual_q10_m": float(np.quantile(calibrated_residual, q, method="linear")),
    }


def apply_head_offsets(
    raw_obstacle_m: Any,
    raw_inter_agent_m: Any,
    offsets_m: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply checkpoint-bound per-head metre offsets without changing CBF margins."""

    obstacle = _finite_array(raw_obstacle_m, name="raw_obstacle_m")
    inter_agent = _finite_array(raw_inter_agent_m, name="raw_inter_agent_m")
    if obstacle.shape != inter_agent.shape:
        raise ValueError("Obstacle and inter-agent clearance shapes must match.")
    try:
        obstacle_offset = float(offsets_m["obstacle_clearance"])
        inter_offset = float(offsets_m["inter_agent_clearance"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Clearance calibration requires both head offsets.") from error
    if not np.isfinite(obstacle_offset) or not np.isfinite(inter_offset):
        raise ValueError("Clearance calibration offsets must be finite.")
    return obstacle + obstacle_offset, inter_agent + inter_offset


def build_calibration_transform(
    raw_obstacle_normalized: Any,
    raw_inter_agent_normalized: Any,
    obstacle_labels_normalized: Any,
    inter_agent_labels_normalized: Any,
    *,
    world_extent_m: float,
    horizon_seconds: list[float],
    quantile: float = CALIBRATION_QUANTILE,
) -> dict[str, Any]:
    """Fit per-head/per-horizon transforms from normalized archive arrays."""

    raw_obstacle = normalized_to_meters(raw_obstacle_normalized, world_extent_m, name="raw_obstacle_normalized")
    raw_inter = normalized_to_meters(raw_inter_agent_normalized, world_extent_m, name="raw_inter_agent_normalized")
    labels_obstacle = normalized_to_meters(obstacle_labels_normalized, world_extent_m, name="obstacle_labels_normalized")
    labels_inter = normalized_to_meters(inter_agent_labels_normalized, world_extent_m, name="inter_agent_labels_normalized")
    arrays = (raw_obstacle, raw_inter, labels_obstacle, labels_inter)
    if any(array.ndim != 2 for array in arrays) or len({array.shape for array in arrays}) != 1:
        raise ValueError("Clearance calibration arrays must share shape [samples, horizon].")
    horizons = [float(value) for value in horizon_seconds]
    if len(horizons) != raw_obstacle.shape[1]:
        raise ValueError("horizon_seconds does not match clearance horizon count.")
    by_horizon: list[dict[str, Any]] = []
    for index, seconds in enumerate(horizons):
        obstacle = q10_residual_offset(raw_obstacle[:, index], labels_obstacle[:, index], quantile=quantile)
        inter = q10_residual_offset(raw_inter[:, index], labels_inter[:, index], quantile=quantile)
        by_horizon.append(
            {
                "horizon_index": int(index),
                "horizon_seconds": seconds,
                "obstacle_clearance": obstacle,
                "inter_agent_clearance": inter,
                "combined_raw_min_mean_m": float(np.mean(np.minimum(raw_obstacle[:, index], raw_inter[:, index]))),
                "combined_label_min_mean_m": float(np.mean(np.minimum(labels_obstacle[:, index], labels_inter[:, index]))),
                "combined_calibrated_min_mean_m": float(
                    np.mean(
                        np.minimum(
                            raw_obstacle[:, index] + obstacle["offset_m"],
                            raw_inter[:, index] + inter["offset_m"],
                        )
                    )
                ),
            }
        )
    return {
        "version": 1,
        "quantile": float(quantile),
        "residual_definition": "label_m_minus_raw_prediction_m",
        "normalized_label_unit": "world_extent_normalized",
        "world_extent_m": float(world_extent_m),
        "horizon_seconds": horizons,
        "heads": ["obstacle_clearance", "inter_agent_clearance"],
        "by_horizon": by_horizon,
    }


def offsets_for_horizon(transform: Mapping[str, Any], horizon_index: int) -> dict[str, float]:
    """Read and validate one immutable horizon transform."""

    if int(transform.get("version", -1)) != 1:
        raise ValueError("Unsupported clearance calibration transform version.")
    rows = transform.get("by_horizon")
    if not isinstance(rows, list) or not 0 <= int(horizon_index) < len(rows):
        raise ValueError("Clearance calibration horizon is unavailable.")
    row = rows[int(horizon_index)]
    if not isinstance(row, Mapping):
        raise ValueError("Malformed clearance calibration horizon row.")
    offsets: dict[str, float] = {}
    for head in ("obstacle_clearance", "inter_agent_clearance"):
        item = row.get(head)
        if not isinstance(item, Mapping):
            raise ValueError(f"Missing clearance calibration head: {head}")
        value = float(item.get("offset_m"))
        if not np.isfinite(value):
            raise ValueError("Clearance calibration offset must be finite.")
        offsets[head] = value
    return offsets
