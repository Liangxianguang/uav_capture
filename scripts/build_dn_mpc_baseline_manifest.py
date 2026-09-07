"""Freeze the development inputs for the DN-MPC direction.

This script only records inputs and provenance.  It never opens a locked test,
executes the simulator, or modifies historical manifests and result folders.
The generated JSON stays under ``results/``; the accompanying audit report is
the versioned record of what was frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty_files() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in output.splitlines() if line.strip()]


def _environment_metadata() -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "conda_prefix": sys.prefix,
    }
    for package in ("numpy", "torch", "PyYAML", "tensorboard"):
        try:
            values[package.lower().replace("-", "_")] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package.lower().replace("-", "_")] = "not-installed"
    try:
        import torch

        values["cuda_available"] = bool(torch.cuda.is_available())
        values["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            values["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        values["torch_runtime_error"] = type(exc).__name__
    return values


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved.relative_to(PROJECT_ROOT.resolve())),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _manifest_record(path: Path) -> dict[str, Any]:
    record = _file_record(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    record["record_count"] = len([line for line in lines if line.strip()])
    return record


def _summary_record(path: Path) -> dict[str, Any]:
    record = _file_record(path)
    record["summary"] = json.loads(path.read_text(encoding="utf-8"))
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "dn_mpc_jepa_safe_capture_dev" / "baseline_manifest.json")
    parser.add_argument("--tensorboard-dir", type=Path, default=PROJECT_ROOT / "results" / "dn_mpc_jepa_safe_capture_tensorboard" / "p0_baseline_freeze_seed20260907")
    parser.add_argument("--g5-manifest", type=Path, default=PROJECT_ROOT / "results" / "jepa_route_recovery_dev_g1b_seed20260911" / "scene_manifest.jsonl")
    parser.add_argument("--g5-summary", type=Path, default=PROJECT_ROOT / "results" / "dn_mpc_cbf_g5_dev_seed20260907_v2" / "summary.json")
    parser.add_argument("--m0-manifest", type=Path, default=PROJECT_ROOT / "results" / "jepa_safe_capture_l0_l3_paired_smoke_seed20260911_m0" / "scene_manifest.jsonl")
    parser.add_argument("--m3-manifest", type=Path, default=PROJECT_ROOT / "results" / "jepa_safe_capture_l0_l3_paired_smoke_seed20260911_m3" / "scene_manifest.jsonl")
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_route_v1_protocol.yaml")
    parser.add_argument("--environment-config", type=Path, default=PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml")
    parser.add_argument("--actor-checkpoint", type=Path, default=PROJECT_ROOT / "models" / "v5_development_exact_reactive_seed661606.pt")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--force", action="store_true", help="allow replacing only the manifest output file")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite baseline manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.tensorboard_dir.resolve()
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {tensorboard_dir}")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    environment_config = yaml.safe_load(args.environment_config.read_text(encoding="utf-8"))
    world = dict(environment_config.get("world", {}))
    agents = dict(environment_config.get("agents", {}))
    task = dict(environment_config.get("task", {}))
    pursuit = dict(task.get("pursuit", {}))
    manifest = {
        "schema_version": "dn_mpc_baseline_manifest_v1",
        "created_by": "scripts/build_dn_mpc_baseline_manifest.py",
        "development_only": True,
        "locked_test_opened": False,
        "seed": int(args.seed),
        "git": {"revision": _git_revision(), "dirty_files": _git_dirty_files()},
        "environment": _environment_metadata(),
        "contract": {
            "dt_seconds": float(world.get("dt", 0.1)),
            "episode_limit_steps": int(world.get("max_steps", 0)),
            "defenders": int(agents.get("defenders", 0)),
            "defender_max_speed_mps": float(agents.get("defender_max_speed", 0.0)),
            "defender_max_acceleration_mps2": float(agents.get("defender_max_acceleration", 0.0)),
            "drone_radius_m": float(agents.get("drone_radius", 0.0)),
            "capture_radius_m": float(pursuit.get("capture_radius", 0.0)),
            "cbf_gamma": float(task.get("cbf_gamma", 0.0)),
            "cbf_margin_m": float(pursuit.get("safety_margin", 0.0)),
            "barrier_mode": "strict_buffer",
            "raw_unverified_execution_allowed": False,
            "controlled_abort_preserved": True,
        },
        "inputs": {
            "g5_scene_manifest": _manifest_record(args.g5_manifest),
            "m0_scene_manifest": _manifest_record(args.m0_manifest),
            "m3_scene_manifest": _manifest_record(args.m3_manifest),
            "g5_summary": _summary_record(args.g5_summary),
            "protocol": _file_record(args.protocol),
            "environment_config": _file_record(args.environment_config),
            "actor_checkpoint": _file_record(args.actor_checkpoint),
        },
        "tensorboard": {
            "logdir": str(tensorboard_dir),
            "required": True,
            "tags": [
                "Baseline/manifest_created",
                "Baseline/g5_scene_records",
                "Baseline/m0_scene_records",
                "Baseline/m3_scene_records",
                "Baseline/g5_safe_capture_rate",
            ],
        },
        "replay_status": {
            "g5": "frozen_summary_referenced; v2 development replay",
            "m0": "scene_manifest_frozen; replay_parity_pending",
            "m3": "scene_manifest_frozen; replay_parity_pending",
        },
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    g5_rate = float(manifest["inputs"]["g5_summary"]["summary"]["overall"]["safe_capture_rate"])
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_scalar("Baseline/manifest_created", 1.0, 0)
    writer.add_scalar("Baseline/g5_scene_records", manifest["inputs"]["g5_scene_manifest"]["record_count"], 0)
    writer.add_scalar("Baseline/m0_scene_records", manifest["inputs"]["m0_scene_manifest"]["record_count"], 0)
    writer.add_scalar("Baseline/m3_scene_records", manifest["inputs"]["m3_scene_manifest"]["record_count"], 0)
    writer.add_scalar("Baseline/g5_safe_capture_rate", g5_rate, 0)
    writer.add_text("Provenance/metadata", json.dumps(manifest, sort_keys=True), 0)
    writer.flush()
    writer.close()
    print(json.dumps({"output": str(output), "tensorboard_dir": str(tensorboard_dir), "sha256": _sha256(output), "g5_safe_capture_rate": g5_rate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
