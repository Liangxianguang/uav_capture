from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "validate_capture_net_measurements.py"


def _write_csv(path: Path, header: str, rows: np.ndarray) -> None:
    lines = [header, *(",".join(f"{value:.12g}" for value in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_measurement_preflight_cli_writes_traceable_diagnostic(tmp_path: Path) -> None:
    metadata = {
        "schema_version": 1,
        "test_date_utc": "2026-08-18T08:00:00Z",
        "operator": "test operator",
        "material_id": "net-lot-01",
        "attachment_description": "same cord and knot as the mesh edge",
        "representative_segment_length_m": 0.25,
        "total_moving_net_mass_kg": 0.08,
        "measured_pretension_n": 0.2,
        "safe_working_tension_n": 10.0,
        "maximum_strain": 0.15,
        "static_fixture": "linear pull fixture",
        "decay_fixture": "known moving mass",
        "impact_fixture": "instrumented low-speed contact fixture",
        "load_cell_calibration": "LC-2026-01",
        "sampling_rate_hz": 1000.0,
        "impact_approach_speed_m_s": 0.2,
        "ambient_conditions": "20 C, 45 percent RH, indoor",
    }
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    extension = np.linspace(0.0, 0.10, 11)
    _write_csv(tmp_path / "static.csv", "extension_m,force_n", np.column_stack((extension, 40.0 * extension + 0.2)))
    damping_ratio = 0.05
    natural_frequency = 20.0
    time = np.arange(0.0, 2.0, 0.001)
    decay = np.exp(-damping_ratio * natural_frequency * time) * np.cos(
        natural_frequency * np.sqrt(1.0 - damping_ratio**2) * time
    )
    _write_csv(tmp_path / "decay.csv", "time_s,displacement_m", np.column_stack((time, decay)))
    _write_csv(
        tmp_path / "impact.csv",
        "time_s,force_n",
        np.array([[0.0, 0.0], [0.1, 4.0], [0.2, 0.0]]),
    )

    output = tmp_path / "preflight"
    command = [
        sys.executable,
        str(SCRIPT),
        "--metadata-yaml",
        str(tmp_path / "metadata.yaml"),
        "--static-csv",
        str(tmp_path / "static.csv"),
        "--decay-csv",
        str(tmp_path / "decay.csv"),
        "--impact-csv",
        str(tmp_path / "impact.csv"),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    report = json.loads(output.joinpath("measurement_preflight.json").read_text(encoding="utf-8"))
    assert report["preflight_status"] == "passed_requires_experimental_review"
    assert report["not_a_hardware_claim"]
    assert report["static_stiffness_diagnostic"]["samples"] == 11
    assert report["free_decay_damping_diagnostic"]["peaks"] >= 3
    assert len(report["inputs"]) == 4


def test_measurement_preflight_rejects_nonfinite_metadata(tmp_path: Path) -> None:
    metadata = {
        "schema_version": 1,
        "test_date_utc": "2026-08-18T08:00:00Z",
        "operator": "test operator",
        "material_id": "net-lot-01",
        "attachment_description": "same cord and knot as the mesh edge",
        "representative_segment_length_m": 0.25,
        "total_moving_net_mass_kg": float("nan"),
        "measured_pretension_n": 0.2,
        "safe_working_tension_n": 10.0,
        "maximum_strain": 0.15,
        "static_fixture": "linear pull fixture",
        "decay_fixture": "known moving mass",
        "impact_fixture": "instrumented low-speed contact fixture",
        "load_cell_calibration": "LC-2026-01",
        "sampling_rate_hz": 1000.0,
        "impact_approach_speed_m_s": 0.2,
        "ambient_conditions": "20 C, 45 percent RH, indoor",
    }
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        "--metadata-yaml",
        str(metadata_path),
        "--static-csv",
        str(tmp_path / "missing_static.csv"),
        "--decay-csv",
        str(tmp_path / "missing_decay.csv"),
        "--impact-csv",
        str(tmp_path / "missing_impact.csv"),
        "--output",
        str(tmp_path / "preflight"),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "total_moving_net_mass_kg must be a positive number" in result.stderr
