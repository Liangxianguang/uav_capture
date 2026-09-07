"""Build a deterministic development scene manifest for DN-MPC replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    random_central_mixed_obstacle_scenario,
    scenario_metadata,
)
from evaluate_random_central_mixed_obstacles import config_for_spec, episode_spec  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _scene_hash(metadata: dict[str, Any]) -> str:
    payload = json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Protocol must be a YAML mapping.")
    if str(payload.get("phase")) != "development_only" or bool(payload.get("locked_test_opened")):
        raise ValueError("Manifest builder only accepts a closed development-only protocol.")
    return payload


def build_manifest(protocol: dict[str, Any], environment_config: Path, episodes: int, split: str) -> list[dict[str, Any]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if split not in protocol.get("seed_blocks", {}):
        raise ValueError(f"Unknown split {split!r}")
    records: list[dict[str, Any]] = []
    s3 = protocol["s3"]
    for episode_index in range(episodes):
        spec = episode_spec(protocol, split, episode_index)
        config = config_for_spec("f2", spec, environment_config)
        sampler_env = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=0,
            target_speed_scale=float(spec["target_speed_scale"]),
        )
        scenario = random_central_mixed_obstacle_scenario(
            sampler_env,
            layout_seed=int(spec["layout_seed"]),
            initial_side_distance=float(spec["initial_side_distance"]),
            defender_side=str(spec["defender_side"]),
            target_crossing_required=bool(spec["target_crossing_required"]),
            obstacle_count_range=(int(spec["obstacle_count"]), int(spec["obstacle_count"])),
            max_attempts=int(s3.get("max_sampling_attempts", 500)),
            required_defender_zone_entries=int(s3.get("required_defender_zone_entries", 1)),
        )
        metadata = scenario_metadata(scenario)
        records.append(
            {
                "episode_index": int(episode_index),
                "episode_seed": int(spec["episode_seed"]),
                "layout_seed": int(spec["layout_seed"]),
                "spec": spec,
                "scenario": metadata,
                "scene_hash": _scene_hash(metadata),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "calibration", "development"), default="validation")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = build_manifest(
        _load_protocol(args.protocol),
        args.environment_config,
        int(args.episodes),
        args.split,
    )
    output = args.output.resolve()
    if output.exists() and output.stat().st_size:
        raise FileExistsError(f"Refusing to overwrite non-empty manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(_jsonable(record), sort_keys=True, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "episodes": len(records), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
