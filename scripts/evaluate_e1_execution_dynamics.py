"""Evaluate a rule expert or frozen V4 policy on one pre-registered E1 profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.e1_protocol import (  # noqa: E402
    E1_PROFILES,
    E1_SPLITS,
    case_sha256,
    environment_config,
    episode_count,
    episode_spec,
    execution_config,
    load_e1_protocol,
)
from encirclement3d.execution_evaluation import rollout_execution_expert, rollout_execution_policy  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import random_central_mixed_obstacle_scenario, scenario_metadata  # noqa: E402
from evaluate_capture_radius_mappo import load_policy  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "e1_execution_dynamics_protocol.yaml"
EXECUTION_MODES = ("raw", "kinematic_cbf", "execution_aware_cbf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline", choices=("dynamic_encirclement",))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--profile", choices=E1_PROFILES, required=True)
    parser.add_argument("--split", choices=E1_SPLITS, required=True)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, required=True)
    parser.add_argument("--episodes", type=int, help="Optional smoke/development count override; locked count is frozen.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero E1 episodes.")
    capture_times = [float(row["capture_time_seconds"]) for row in rows if row["capture_time_seconds"] is not None]
    return {
        "episodes": len(rows),
        "capture_rate": _rate(rows, "capture_event"),
        "safe_capture_rate": _rate(rows, "safe_capture_success"),
        "cooperative_safe_capture_rate": _rate(rows, "cooperative_safe_capture"),
        "collision_rate": _rate(rows, "collision"),
        "boundary_violation_rate": float(np.mean([int(row["world_violation_steps"]) > 0 for row in rows])),
        "transit_success_rate": _rate(rows, "transit_success"),
        "mean_time_to_capture_seconds": float(np.mean(capture_times)) if capture_times else None,
        "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in rows])),
        "worst_min_clearance_m": float(min(float(row["min_clearance_m"]) for row in rows)),
        "mean_defender_path_length_m": float(np.mean([float(row["mean_defender_path_length_m"]) for row in rows])),
        "mean_command_execution_error_mps": float(
            np.mean([float(row["mean_command_execution_error_mps"]) for row in rows])
        ),
        "p95_command_execution_error_mps": float(
            np.quantile([float(row["p95_command_execution_error_mps"]) for row in rows], 0.95)
        ),
        "max_command_execution_error_mps": float(
            max(float(row["max_command_execution_error_mps"]) for row in rows)
        ),
        "mean_command_age_steps": float(np.mean([float(row["mean_command_age_steps"]) for row in rows])),
        "acceleration_saturation_rate": float(np.mean([float(row["acceleration_saturation_rate"]) for row in rows])),
        "speed_saturation_rate": float(np.mean([float(row["speed_saturation_rate"]) for row in rows])),
        "mean_cbf_action_correction_norm": float(
            np.mean([float(row["mean_cbf_action_correction_norm"]) for row in rows])
        ),
        "max_cbf_action_correction_norm": float(max(float(row["max_cbf_action_correction_norm"]) for row in rows)),
        "termination_reasons": dict(sorted(Counter(str(row["termination_reason"]) for row in rows).items())),
    }


def main() -> None:
    args = parse_args()
    if (args.checkpoint is None) == (args.baseline is None):
        raise ValueError("Provide exactly one of --checkpoint or --baseline.")
    protocol_path = args.protocol.resolve()
    protocol = load_e1_protocol(protocol_path)
    count = episode_count(protocol, args.split, args.episodes)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    policy: Any = None
    action_scale: float | None = None
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    profile = execution_config(protocol, args.profile)
    for index in range(count):
        spec = episode_spec(protocol, args.split, index)
        config = environment_config(protocol, protocol_path, spec)
        prototype = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=0,
            target_speed_scale=float(spec["target_speed_scale"]),
        )
        scenario = random_central_mixed_obstacle_scenario(
            prototype,
            layout_seed=int(spec["layout_seed"]),
            initial_side_distance=float(spec["initial_side_distance"]),
            defender_side=str(spec["defender_side"]),
            target_crossing_required=bool(spec["target_crossing_required"]),
            obstacle_count_range=(int(spec["obstacle_count"]), int(spec["obstacle_count"])),
            max_attempts=int(protocol["s3"]["max_sampling_attempts"]),
            required_defender_zone_entries=int(spec["required_defender_zone_entries"]),
        )
        metadata = scenario_metadata(scenario)
        signature = case_sha256(spec, metadata)
        if checkpoint is not None and policy is None:
            policy, action_scale, _checkpoint_metadata = load_policy(
                checkpoint,
                prototype,
                prototype.reset(seed=int(spec["episode_seed"])),
                device,
            )
        if checkpoint is None:
            row, _env, _wrapper = rollout_execution_expert(
                config,
                scenario,
                seed=int(spec["episode_seed"]),
                execution_config=profile,
                execution_mode=args.execution_mode,
            )
        else:
            assert policy is not None and action_scale is not None
            row, _env, _wrapper = rollout_execution_policy(
                policy,
                config,
                scenario,
                seed=int(spec["episode_seed"]),
                device=device,
                action_scale=action_scale,
                execution_config=profile,
                execution_mode=args.execution_mode,
            )
        row.update(
            {
                "split": args.split,
                "profile": args.profile,
                "episode_index": int(index),
                "episode_seed": int(spec["episode_seed"]),
                "layout_seed": int(spec["layout_seed"]),
                "execution_noise_seed": int(spec["execution_noise_seed"]),
                "case_sha256": signature,
                "method": str(args.baseline) if checkpoint is None else "frozen_v4_policy",
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "checkpoint_sha256": _sha256(checkpoint) if checkpoint is not None else None,
            }
        )
        rows.append(row)
        cases.append({"episode_index": int(index), "spec": spec, "scenario": metadata, "case_sha256": signature})

    with output.joinpath("episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.joinpath("cases.jsonl").write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8"
    )
    output.joinpath("summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    output.joinpath("effective_protocol.yaml").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")
    output.joinpath("evaluation_metadata.json").write_text(
        json.dumps(
            {
                "experiment": "E1_execution_dynamics",
                "locked_test": args.split == "locked_test",
                "profile": args.profile,
                "execution_mode": args.execution_mode,
                "method": str(args.baseline) if checkpoint is None else "frozen_v4_policy",
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "checkpoint_sha256": _sha256(checkpoint) if checkpoint is not None else None,
                "episodes": count,
                "seed_block": int(protocol["seed_blocks"][args.split]),
                "protocol_sha256": _sha256(protocol_path),
                "cases_sha256": _sha256(output / "cases.jsonl"),
                "raw_kinematic_execution_aware_case_pairing": True,
                "execution_noise_seed_pairing": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summarize(rows), indent=2), flush=True)


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()

