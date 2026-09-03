"""Audit target/defender boundary semantics for a frozen development rollout.

The historical pursuit environment exposed one ``world_violation_steps``
counter for both target and defenders.  This audit replays one existing M3
scene with the exact recorded inputs and instruments the boundary hook in
memory, so the original result directory is never modified.  It is a
diagnostic script only and cannot open a locked test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_jepa_safe_capture_v2_paired as evaluator  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Scene manifest is empty: {path}")
    return rows


def _metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("locked_test_opened") is not False:
        raise ValueError("Boundary audit requires locked_test_opened=false.")
    if payload.get("variant", {}).get("variant") != "m3":
        raise ValueError("Boundary audit currently requires an M3 run directory.")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=19)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(run_dir)
    manifest_path = run_dir / "scene_manifest.jsonl"
    manifest = _load_manifest(manifest_path)
    if args.episode_index < 0 or args.episode_index >= len(manifest):
        raise IndexError(args.episode_index)
    manifest_item = manifest[args.episode_index]
    spec = dict(manifest_item["spec"])
    protocol_path = Path(metadata["inputs"]["protocol"])
    environment_path = Path(metadata["inputs"]["environment_config"])
    actor_path = Path(metadata["inputs"]["actor_checkpoint"])
    jepa_path = Path(metadata["inputs"]["jepa_checkpoint"])
    ledger_path = Path(metadata["inputs"]["reliability_ledger"])
    for path in (protocol_path, environment_path, actor_path, jepa_path, ledger_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(manifest_path) != metadata["inputs"].get("scene_manifest_sha256"):
        raise ValueError("Scene manifest hash does not match the run provenance.")

    protocol = evaluator.load_protocol(protocol_path)
    config = evaluator.config_for_spec("f2", spec, environment_path)
    scenario = evaluator.scenario_from_metadata(dict(manifest_item["scenario"]))
    device = evaluator.select_device(args.device)

    prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(spec["target_speed_scale"]),
    )
    prototype_observation = prototype.reset(seed=int(spec["episode_seed"]))
    policy, action_scale, actor_metadata = evaluator.load_policy(
        actor_path,
        prototype,
        prototype_observation,
        device,
    )
    jepa = evaluator._load_jepa(jepa_path, device)
    ledger = evaluator._load_ledger(ledger_path, jepa_path)

    boundary_events: list[dict[str, Any]] = []
    original_enforce = CaptureRadiusPursuit3DEnv._enforce_world_bounds

    def instrumented_enforce(
        self: CaptureRadiusPursuit3DEnv,
        positions: np.ndarray,
        velocities: np.ndarray,
        entity_type: str | None = None,
    ) -> None:
        values = np.asarray(positions, dtype=np.float64)
        entity = entity_type or ("target" if values.shape[0] == 1 else "defender")
        for axis in range(3):
            below = values[:, axis] < self.lower[axis]
            above = values[:, axis] > self.upper[axis]
            for index in np.flatnonzero(below | above):
                boundary_events.append(
                    {
                        "entity": entity,
                        "entity_index": int(index),
                        "axis": int(axis),
                        "side": "lower" if bool(below[index]) else "upper",
                        "step_before_clamp": int(self.step_count),
                        "value_m": float(values[index, axis]),
                        "lower_m": float(self.lower[axis]),
                        "upper_m": float(self.upper[axis]),
                    }
                )
        original_enforce(self, positions, velocities, entity_type=entity_type)

    CaptureRadiusPursuit3DEnv._enforce_world_bounds = instrumented_enforce
    replay_dir = output_dir / "replay"
    try:
        row, scene = evaluator._run_episode(
            manifest_item=manifest_item,
            config=config,
            policy=policy,
            action_scale=action_scale,
            device=device,
            contract=evaluator._variant_contract("m3"),
            jepa=jepa,
            ledger=ledger,
            history_length=8,
            recurrent_reset_interval=actor_metadata.get("recurrent_reset_interval_steps"),
            output_dir=replay_dir,
        )
    finally:
        CaptureRadiusPursuit3DEnv._enforce_world_bounds = original_enforce

    target_events = [event for event in boundary_events if event["entity"] == "target"]
    defender_events = [event for event in boundary_events if event["entity"] == "defender"]
    audit = {
        "audit": "jepa_safe_capture_v3_boundary_semantics",
        "development_only": True,
        "locked_test_opened": False,
        "run_dir": str(run_dir),
        "run_provenance_sha256": _sha256(run_dir / "provenance.json"),
        "episode_index": int(args.episode_index),
        "episode_seed": int(spec["episode_seed"]),
        "scene_hash": manifest_item.get("scene_hash"),
        "inputs": {
            "protocol": str(protocol_path),
            "protocol_sha256": _sha256(protocol_path),
            "environment_config": str(environment_path),
            "environment_config_sha256": _sha256(environment_path),
            "actor_checkpoint": str(actor_path),
            "actor_checkpoint_sha256": _sha256(actor_path),
            "jepa_checkpoint": str(jepa_path),
            "jepa_checkpoint_sha256": _sha256(jepa_path),
            "reliability_ledger": str(ledger_path),
            "reliability_ledger_sha256": _sha256(ledger_path),
            "scene_manifest": str(manifest_path),
            "scene_manifest_sha256": _sha256(manifest_path),
        },
        "replay_row": row,
        "boundary_event_counts": {
            "all": len(boundary_events),
            "target": len(target_events),
            "defender": len(defender_events),
        },
        "target_boundary_events": target_events,
        "defender_boundary_events": defender_events,
        "legacy_world_violation_steps": int(row.get("world_violation_steps", 0)),
        "semantic_finding": (
            "target_only_boundary_violation_legacy_counter_mismatch"
            if target_events and not defender_events and int(row.get("world_violation_steps", 0)) > 0
            else "defender_boundary_violation_present"
            if defender_events
            else "no_boundary_event_reproduced"
        ),
    }
    tensorboard_dir = (
        args.tensorboard_dir.resolve()
        if args.tensorboard_dir is not None
        else output_dir / "tensorboard"
    )
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {tensorboard_dir}")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=2) as writer:
        writer.add_text("Provenance/audit", json.dumps(_jsonable(audit), indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(_jsonable(audit["inputs"]), indent=2), 0)
        writer.add_scalar("Boundary/legacy_world_violation_steps", float(audit["legacy_world_violation_steps"]), 0)
        writer.add_scalar("Boundary/target_events", float(len(target_events)), 0)
        writer.add_scalar("Boundary/defender_events", float(len(defender_events)), 0)
        writer.add_scalar("Replay/safe_capture", float(bool(row.get("safe_capture_success", False))), 0)
        writer.add_scalar("Replay/collision", float(bool(row.get("collision", False))), 0)
        writer.add_scalar("Replay/legacy_boundary_violation", float(bool(row.get("boundary_violation", False))), 0)
        writer.add_scalar("Replay/minimum_defender_boundary_clearance_m", float(row.get("minimum_boundary_clearance_m", 0.0)), 0)
        writer.flush()
    event_files = sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("TensorBoard writer did not create an event file.")
    audit["tensorboard"] = {
        "logdir": str(tensorboard_dir),
        "event_files": event_files,
        "required_provenance": True,
    }
    (output_dir / "boundary_audit.json").write_text(
        json.dumps(_jsonable(audit), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(audit), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
