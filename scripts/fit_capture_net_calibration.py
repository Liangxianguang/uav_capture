"""Fit traceable flexible-net candidates from bench-test CSV measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.calibration import (  # noqa: E402
    fit_free_decay_damping,
    fit_static_stiffness,
    load_measurement_csv,
    measure_impact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-csv", type=Path, required=True, help="CSV columns: extension_m,force_n")
    parser.add_argument("--decay-csv", type=Path, required=True, help="CSV columns: time_s,displacement_m")
    parser.add_argument("--impact-csv", type=Path, required=True, help="CSV columns: time_s,force_n")
    parser.add_argument("--total-moving-net-mass-kg", type=float, required=True)
    parser.add_argument("--measured-pretension-n", type=float, required=True)
    parser.add_argument("--safe-working-tension-n", type=float, required=True)
    parser.add_argument("--maximum-strain", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration output: {args.output}")
    if args.total_moving_net_mass_kg <= 0.0:
        raise ValueError("--total-moving-net-mass-kg must be positive.")
    if args.measured_pretension_n < 0.0:
        raise ValueError("--measured-pretension-n must be non-negative.")
    if args.safe_working_tension_n <= 0.0:
        raise ValueError("--safe-working-tension-n must be positive.")
    if not 0.0 < args.maximum_strain < 1.0:
        raise ValueError("--maximum-strain must lie in (0, 1).")

    extension, static_force = load_measurement_csv(args.static_csv, "extension_m", "force_n")
    decay_time, decay_displacement = load_measurement_csv(args.decay_csv, "time_s", "displacement_m")
    impact_time, impact_force = load_measurement_csv(args.impact_csv, "time_s", "force_n")
    static_fit = fit_static_stiffness(extension, static_force)
    damping_fit = fit_free_decay_damping(
        decay_time,
        decay_displacement,
        moving_mass_kg=float(args.total_moving_net_mass_kg),
    )
    impact_metrics = measure_impact(impact_time, impact_force)

    args.output.mkdir(parents=True, exist_ok=False)
    candidate = {
        "calibration_status": "candidate_requires_review",
        "model_scope": "representative_segment_effective_parameters",
        "review_gates": [
            "Confirm fixture geometry matches the simulated segment or mesh edge.",
            "Confirm static fit quality and inspect raw residuals.",
            "Confirm safe-working tension and maximum strain from material/attachment tests.",
            "Validate the resulting mesh against a separate physical sag and impact test.",
        ],
        "task": {
            "capture": {
                # The refined solver preserves a total moving mass of four
                # legacy nodes, so this is the explicit conversion required
                # by its current compatibility parameterization.
                "net_node_mass": float(args.total_moving_net_mass_kg / 4.0),
                "net_spring_stiffness": static_fit.stiffness_n_per_m,
                "net_spring_damping": damping_fit.damping_n_s_per_m,
                "net_spring_pretension": float(args.measured_pretension_n),
                "net_max_tension": float(args.safe_working_tension_n),
                "net_max_strain": float(args.maximum_strain),
            }
        },
    }
    summary = {
        "static_stiffness_fit": asdict(static_fit),
        "free_decay_damping_fit": asdict(damping_fit),
        "low_speed_impact": asdict(impact_metrics),
        "total_moving_net_mass_kg": float(args.total_moving_net_mass_kg),
        "measured_pretension_n": float(args.measured_pretension_n),
        "safe_working_tension_n": float(args.safe_working_tension_n),
        "maximum_strain": float(args.maximum_strain),
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in (args.static_csv, args.decay_csv, args.impact_csv)
        },
    }
    args.output.joinpath("calibration_candidate.yaml").write_text(
        yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
    )
    args.output.joinpath("calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
