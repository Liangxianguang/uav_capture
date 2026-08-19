"""Traceable fits for capture-net bench measurements.

The functions intentionally estimate only effective segment-level parameters.
They do not turn a bench result into a validated flight-net model; callers
must retain the raw data and review the fit quality before using any output.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StaticStiffnessFit:
    """Least-squares force-extension fit for one representative segment."""

    stiffness_n_per_m: float
    preload_n: float
    r_squared: float
    samples: int


@dataclass(frozen=True)
class DampingFit:
    """Effective viscous damping from decaying free-oscillation peaks."""

    damping_n_s_per_m: float
    damping_ratio: float
    natural_frequency_rad_s: float
    samples: int
    peaks: int


@dataclass(frozen=True)
class ImpactMetrics:
    """Force-time quantities measured in a low-speed impact test."""

    peak_force_n: float
    impulse_n_s: float
    samples: int


def load_measurement_csv(path: str | Path, *columns: str) -> tuple[np.ndarray, ...]:
    """Load required finite numeric columns from a raw bench-measurement CSV."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Measurement CSV does not exist: {source}")
    if not columns:
        raise ValueError("At least one measurement column is required.")

    values = {column: [] for column in columns}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(column not in reader.fieldnames for column in columns):
            expected = ", ".join(columns)
            raise ValueError(f"{source} must include CSV columns: {expected}")
        for row_index, row in enumerate(reader, start=2):
            for column in columns:
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{source}:{row_index} has an invalid {column} value.") from error
                if not np.isfinite(value):
                    raise ValueError(f"{source}:{row_index} has a non-finite {column} value.")
                values[column].append(value)
    return tuple(np.asarray(values[column], dtype=np.float64) for column in columns)


def _finite_vector(values: np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def fit_static_stiffness(
    extension_m: np.ndarray | list[float], force_n: np.ndarray | list[float]
) -> StaticStiffnessFit:
    """Fit ``force = stiffness * extension + preload`` for one segment."""
    extension = _finite_vector(extension_m, "extension_m")
    force = _finite_vector(force_n, "force_n")
    if len(extension) != len(force) or len(extension) < 3:
        raise ValueError("Static stiffness fitting requires at least three paired samples.")
    if np.ptp(extension) <= 1e-9:
        raise ValueError("Static extension samples must span a nonzero range.")
    # This is a two-parameter scalar fit. Keeping the normal-equation terms
    # elementwise avoids loading a second BLAS/OpenMP runtime in the Windows
    # training environment, while retaining the same least-squares solution.
    count = float(len(extension))
    sum_extension = float(np.sum(extension))
    sum_force = float(np.sum(force))
    sum_extension_squared = float(np.sum(extension * extension))
    sum_extension_force = float(np.sum(extension * force))
    denominator = count * sum_extension_squared - sum_extension * sum_extension
    stiffness = (count * sum_extension_force - sum_extension * sum_force) / denominator
    preload = (sum_force - stiffness * sum_extension) / count
    if stiffness <= 0.0:
        raise ValueError("Static stiffness fit must have a positive slope.")
    prediction = stiffness * extension + preload
    residual = float(np.sum((force - prediction) ** 2))
    total = float(np.sum((force - np.mean(force)) ** 2))
    r_squared = 1.0 if total <= 1e-12 else 1.0 - residual / total
    return StaticStiffnessFit(
        stiffness_n_per_m=float(stiffness),
        preload_n=float(preload),
        r_squared=float(r_squared),
        samples=len(extension),
    )


def fit_free_decay_damping(
    time_s: np.ndarray | list[float], displacement_m: np.ndarray | list[float], moving_mass_kg: float
) -> DampingFit:
    """Estimate effective viscous damping from same-sign half-cycle peaks.

    Absolute displacement peaks are separated by a half period.  The
    logarithmic decrement therefore uses ``pi`` rather than ``2*pi`` in the
    damping-ratio conversion.
    """
    time = _finite_vector(time_s, "time_s")
    displacement = _finite_vector(displacement_m, "displacement_m")
    if len(time) != len(displacement) or len(time) < 7:
        raise ValueError("Free-decay fitting requires at least seven paired samples.")
    if moving_mass_kg <= 0.0 or not np.isfinite(moving_mass_kg):
        raise ValueError("moving_mass_kg must be finite and positive.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Free-decay timestamps must be strictly increasing.")

    amplitude = np.abs(displacement)
    peak_indices = np.flatnonzero(
        (amplitude[1:-1] > amplitude[:-2]) & (amplitude[1:-1] >= amplitude[2:])
    ) + 1
    if len(peak_indices) < 3:
        raise ValueError("Free-decay data needs at least three resolved amplitude peaks.")
    peak_amplitudes = amplitude[peak_indices]
    peak_times = time[peak_indices]
    valid_pairs = peak_amplitudes[:-1] > peak_amplitudes[1:]
    if int(np.sum(valid_pairs)) < 2:
        raise ValueError("Free-decay peaks must decay across at least two half cycles.")
    logarithmic_decrements = np.log(peak_amplitudes[:-1][valid_pairs] / peak_amplitudes[1:][valid_pairs])
    half_periods = np.diff(peak_times)[valid_pairs]
    decrement = float(np.median(logarithmic_decrements))
    if decrement <= 0.0 or np.any(half_periods <= 0.0):
        raise ValueError("Free-decay data has no positive damping decrement.")
    damping_ratio = decrement / float(np.sqrt(np.pi**2 + decrement**2))
    damped_frequency = float(np.pi / np.median(half_periods))
    natural_frequency = damped_frequency / float(np.sqrt(1.0 - damping_ratio**2))
    damping = 2.0 * damping_ratio * natural_frequency * float(moving_mass_kg)
    return DampingFit(
        damping_n_s_per_m=float(damping),
        damping_ratio=float(damping_ratio),
        natural_frequency_rad_s=float(natural_frequency),
        samples=len(time),
        peaks=len(peak_indices),
    )


def measure_impact(time_s: np.ndarray | list[float], force_n: np.ndarray | list[float]) -> ImpactMetrics:
    """Compute peak force and signed force-time impulse from sampled contact."""
    time = _finite_vector(time_s, "time_s")
    force = _finite_vector(force_n, "force_n")
    if len(time) != len(force) or len(time) < 2:
        raise ValueError("Impact measurement requires at least two paired samples.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Impact timestamps must be strictly increasing.")
    return ImpactMetrics(
        peak_force_n=float(np.max(np.abs(force))),
        impulse_n_s=float(np.trapezoid(force, time)),
        samples=len(time),
    )
