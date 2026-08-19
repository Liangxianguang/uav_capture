from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "fit_capture_net_calibration.py"


def _write_csv(path: Path, header: str, rows: np.ndarray) -> None:
    lines = [header, *(",".join(f"{value:.12g}" for value in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_calibration_cli_writes_reviewable_candidate_and_manifest(tmp_path: Path) -> None:
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
    output = tmp_path / "candidate"
    command = [
        sys.executable,
        str(SCRIPT),
        "--static-csv",
        str(tmp_path / "static.csv"),
        "--decay-csv",
        str(tmp_path / "decay.csv"),
        "--impact-csv",
        str(tmp_path / "impact.csv"),
        "--total-moving-net-mass-kg",
        "0.08",
        "--measured-pretension-n",
        "0.25",
        "--safe-working-tension-n",
        "10.0",
        "--maximum-strain",
        "0.15",
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    candidate = yaml.safe_load(output.joinpath("calibration_candidate.yaml").read_text(encoding="utf-8"))
    summary = json.loads(output.joinpath("calibration_summary.json").read_text(encoding="utf-8"))
    assert candidate["calibration_status"] == "candidate_requires_review"
    np.testing.assert_allclose(candidate["task"]["capture"]["net_node_mass"], 0.02)
    np.testing.assert_allclose(candidate["task"]["capture"]["net_spring_stiffness"], 40.0)
    assert len(summary["inputs"]) == 3
