"""Preflight raw flexible-net bench measurements before parameter fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.calibration import (  # noqa: E402
    fit_free_decay_damping,
    fit_static_stiffness,
    load_measurement_csv,
    measure_impact,
)


REQUIRED_TEXT_FIELDS = (
    "test_date_utc",
    "operator",
    "material_id",
    "attachment_description",
    "static_fixture",
    "decay_fixture",
    "impact_fixture",
    "load_cell_calibration",
    "ambient_conditions",
)
REQUIRED_POSITIVE_FIELDS = (
    "representative_segment_length_m",
    "total_moving_net_mass_kg",
    "safe_working_tension_n",
    "maximum_strain",
    "sampling_rate_hz",
    "impact_approach_speed_m_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-yaml", type=Path, required=True)
    parser.add_argument("--static-csv", type=Path, required=True)
    parser.add_argument("--decay-csv", type=Path, required=True)
    parser.add_argument("--impact-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Measurement metadata does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Measurement metadata must be a YAML mapping.")
    if data.get("schema_version") != 1:
        raise ValueError("Measurement metadata schema_version must be 1.")

    issues: list[str] = []
    for name in REQUIRED_TEXT_FIELDS:
        if not isinstance(data.get(name), str) or not data[name].strip():
            issues.append(f"{name} must be a non-empty string")
    for name in REQUIRED_POSITIVE_FIELDS:
        value = data.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            issues.append(f"{name} must be a positive number")
    pretension = data.get("measured_pretension_n")
    if (
        isinstance(pretension, bool)
        or not isinstance(pretension, (int, float))
        or not math.isfinite(pretension)
        or pretension < 0.0
    ):
        issues.append("measured_pretension_n must be a non-negative number")
    strain = data.get("maximum_strain")
    if isinstance(strain, (int, float)) and not isinstance(strain, bool) and math.isfinite(strain) and not 0.0 < strain < 1.0:
        issues.append("maximum_strain must lie in (0, 1)")
    if issues:
        raise ValueError("Invalid measurement metadata:\n- " + "\n- ".join(issues))
    return data


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight output: {args.output}")

    metadata = load_metadata(args.metadata_yaml)
    extension, static_force = load_measurement_csv(args.static_csv, "extension_m", "force_n")
    decay_time, decay_displacement = load_measurement_csv(args.decay_csv, "time_s", "displacement_m")
    impact_time, impact_force = load_measurement_csv(args.impact_csv, "time_s", "force_n")

    static_fit = fit_static_stiffness(extension, static_force)
    damping_fit = fit_free_decay_damping(
        decay_time,
        decay_displacement,
        moving_mass_kg=float(metadata["total_moving_net_mass_kg"]),
    )
    impact_metrics = measure_impact(impact_time, impact_force)
    sources = (args.metadata_yaml, args.static_csv, args.decay_csv, args.impact_csv)
    summary = {
        "preflight_status": "passed_requires_experimental_review",
        "scope": "raw_input_integrity_and_fit_prerequisites_only",
        "not_a_hardware_claim": True,
        "review_gates": [
            "Review raw traces, fixture equivalence, and rejected runs before parameter fitting.",
            "Set tension and strain limits from dedicated material and attachment-strength testing.",
            "Validate the selected model against a held-out physical sag and impact experiment.",
        ],
        "metadata": metadata,
        "static_stiffness_diagnostic": asdict(static_fit),
        "free_decay_damping_diagnostic": asdict(damping_fit),
        "low_speed_impact_diagnostic": asdict(impact_metrics),
        "inputs": {str(source.resolve()): sha256(source) for source in sources},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.joinpath("measurement_preflight.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
