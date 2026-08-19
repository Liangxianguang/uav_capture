from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "pybullet_spatial_containment_rule_test100.yaml"
SCRIPT = PROJECT_ROOT / "scripts" / "run_experiments.py"


def test_spatial_containment_cli_writes_semantic_summary_without_softbody(tmp_path: Path) -> None:
    output = tmp_path / "spatial_containment"
    command = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(CONFIG),
        "--output",
        str(output),
        "--controller",
        "spatial_containment_rule",
        "--episodes",
        "2",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    legacy = json.loads(output.joinpath("summary.json").read_text(encoding="utf-8"))
    spatial = json.loads(output.joinpath("spatial_containment_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads(output.joinpath("run_metadata.json").read_text(encoding="utf-8"))
    scenario = "spatial_containment_locked_test"
    assert metadata["controller"] == "spatial_containment_rule"
    assert spatial["target_model"] == "kinematic_evasive_target"
    assert "physical net deployment" in spatial["not_claimed"]
    assert spatial["scenarios"][scenario]["episodes"] == 2
    assert spatial["scenarios"][scenario]["spatial_containment_success_rate"] == legacy[scenario][
        "capture_success_rate"
    ]
    assert spatial["scenarios"][scenario]["virtual_cage_closure_rate"] == legacy[scenario][
        "capture_closure_episode_rate"
    ]


def test_spatial_containment_cli_rejects_softbody_net_mode(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["dynamics"]["pybullet_softbody_net_enabled"] = True
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(invalid_config),
        "--output",
        str(tmp_path / "unused"),
        "--controller",
        "spatial_containment_rule",
        "--episodes",
        "1",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "cannot enable the PyBullet soft-body net diagnostic" in result.stderr
