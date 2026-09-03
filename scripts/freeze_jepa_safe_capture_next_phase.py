"""Freeze the post-P7 development contract and provenance manifest.

The command is intentionally read-only with respect to existing experiment
directories. It validates the parent P7 aggregate, hashes the new protocol and
its declared inputs, and writes a small manifest plus a TensorBoard record.
This is a development artifact; it never opens or reads a locked-test block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_status() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in output.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/jepa_safe_capture_v3_next_phase.yaml")
    parser.add_argument("--p7-summary", type=Path, default=PROJECT_ROOT / "results/jepa_safe_capture_v2_p7_readiness_full_20260904_rerun/summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_parent_summary(summary: dict[str, Any], path: Path) -> None:
    if summary.get("decision", {}).get("classification") != "positive_development_evidence":
        raise ValueError(f"Unexpected parent P7 classification in {path}")
    if summary.get("tensorboard", {}).get("required_provenance") is not True:
        raise ValueError(f"Parent P7 TensorBoard provenance is incomplete: {path}")
    if summary.get("decision", {}).get("a3_excluded_from_safety_decision") is not True:
        raise ValueError("Parent P7 aggregate did not exclude raw/no-CBF diagnostic from safety decision.")
    if summary.get("decision", {}).get("safety_hard_gate") is not True:
        raise ValueError("Parent P7 safety hard gate is not recorded as passed.")
    for key in ("locked_test_opened",):
        if summary.get(key) is not False:
            raise ValueError(f"Parent P7 {key} must be false.")


def validate_protocol(protocol: dict[str, Any], path: Path) -> None:
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError(f"Next-phase protocol crosses the locked-test boundary: {path}")
    objective = protocol.get("objective", {})
    if objective.get("primary_endpoint") != "safe_capture":
        raise ValueError("Next-phase primary endpoint must be safe_capture.")
    invariants = protocol.get("immutable_invariants", {})
    required = {
        "online_target_ground_truth": False,
        "jepa_can_generate_final_action": False,
        "execute_only_first_step_then_replan": True,
        "cbf_is_final_execution_boundary": True,
    }
    for key, expected in required.items():
        if invariants.get(key) is not expected:
            raise ValueError(f"Protocol invariant {key} must be {expected!r}.")
    if int(invariants.get("candidate_count", -1)) != 5 or int(invariants.get("chunk_length_steps", -1)) != 3:
        raise ValueError("Next-phase candidate contract must remain five candidates and three-step chunks.")


def write_tensorboard(manifest: dict[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/next_phase_protocol", json.dumps(manifest["protocol_summary"], indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(manifest["inputs"], indent=2), 0)
        writer.add_text("Provenance/environment", json.dumps(manifest["environment"], indent=2), 0)
        writer.add_scalar("Safety/parent_p7_hard_gate", 1.0, 0)
        writer.add_scalar("Safety/locked_test_opened", 0.0, 0)
        writer.add_scalar("Contract/candidate_count", float(manifest["protocol_summary"]["candidate_count"]), 0)
        writer.add_scalar("Contract/chunk_length_steps", float(manifest["protocol_summary"]["chunk_length_steps"]), 0)
        writer.flush()
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/next_phase_protocol/text_summary",
        "Provenance/inputs/text_summary",
        "Provenance/environment/text_summary",
    }
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    if missing:
        raise ValueError(f"Freeze TensorBoard provenance is incomplete: {missing}")
    return {
        "logdir": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_provenance": not missing,
    }


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    p7_path = args.p7_summary.resolve()
    output_path = args.output.resolve()
    if not protocol_path.is_file() or not p7_path.is_file():
        raise FileNotFoundError("Protocol and parent P7 summary must both exist.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite freeze manifest: {output_path}")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Protocol YAML must contain a mapping.")
    parent_summary = load_mapping(p7_path)
    validate_protocol(protocol, protocol_path)
    validate_parent_summary(parent_summary, p7_path)
    declared_inputs = dict(protocol.get("frozen_inputs", {}))
    input_records: dict[str, Any] = {}
    for label, relative in sorted(declared_inputs.items()):
        path = (PROJECT_ROOT / str(relative)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Declared frozen input is missing: {label} -> {path}")
        input_records[label] = {"path": str(path), "sha256": sha256(path)}
    input_records["next_phase_protocol"] = {"path": str(protocol_path), "sha256": sha256(protocol_path)}
    input_records["parent_p7_summary"] = {"path": str(p7_path), "sha256": sha256(p7_path)}
    parent_report = p7_path.with_name("report.md")
    if parent_report.is_file():
        input_records["parent_p7_report"] = {"path": str(parent_report), "sha256": sha256(parent_report)}
    manifest: dict[str, Any] = {
        "freeze_type": "jepa_safe_capture_v3_next_phase_wp0_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "locked_test_opened": False,
        "development_only": True,
        "protocol_summary": {
            "protocol_name": protocol["protocol_name"],
            "protocol_version": protocol["protocol_version"],
            "primary_endpoint": protocol["objective"]["primary_endpoint"],
            "candidate_count": protocol["immutable_invariants"]["candidate_count"],
            "chunk_length_steps": protocol["immutable_invariants"]["chunk_length_steps"],
            "execute_only_first_step_then_replan": protocol["immutable_invariants"]["execute_only_first_step_then_replan"],
            "cbf_is_final_execution_boundary": protocol["immutable_invariants"]["cbf_is_final_execution_boundary"],
            "training_seeds": protocol["development_evaluation"]["training_seeds"],
            "final_episodes_per_seed": protocol["development_evaluation"]["final_episodes_per_seed"],
        },
        "parent_p7": {
            "summary_path": str(p7_path),
            "summary_sha256": sha256(p7_path),
            "classification": parent_summary["decision"]["classification"],
            "m3_mean_paired_delta_rate": parent_summary["decision"]["m3_mean_paired_delta_rate"],
            "m3_bootstrap_ci95": parent_summary["decision"]["m3_cross_seed_bootstrap"],
            "safety_hard_gate": parent_summary["decision"]["safety_hard_gate"],
            "locked_test_opened": parent_summary["locked_test_opened"],
        },
        "inputs": input_records,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tensorboard": version("tensorboard"),
            "platform": platform.platform(),
            "git_revision": git_revision(),
            "git_status_at_freeze": git_status(),
            "command": " ".join(sys.argv),
        },
    }
    tensorboard = write_tensorboard(manifest, args.tensorboard_logdir)
    manifest["tensorboard"] = tensorboard
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
