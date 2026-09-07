"""Development-only DN-MPC + strict CBF replay for a frozen scene manifest.

This entry point reuses the audited G5 episode implementation but removes the
G5-only four-episode restriction. It is intended for S2 L0-L1 smoke replays;
it never opens a locked test and never enables JEPA or Ledger-Lite.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from evaluate_dn_mpc_cbf_g5 import (  # noqa: E402
    _environment_metadata,
    _fresh,
    _jsonable,
    _planner_config,
    _run_episode,
    _sha256,
    _write_json,
)
from evaluate_jepa_safe_capture_v2_paired import _scene_hash  # noqa: E402
from evaluate_random_central_mixed_obstacles import config_for_spec, load_protocol  # noqa: E402
from encirclement3d.showcase import scenario_from_metadata, validate_showcase_scenario  # noqa: E402


def _load_manifest_records(
    path: Path,
    environment_config: Path,
    episodes: int,
    start_index: int,
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if start_index < 0 or episodes <= 0 or start_index + episodes > len(records):
        raise ValueError(
            f"Requested manifest slice [{start_index}, {start_index + episodes}) "
            f"from {len(records)} records."
        )
    records = records[start_index : start_index + episodes]
    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        expected_index = int(start_index) + index
        if int(record.get("episode_index", -1)) != expected_index:
            raise ValueError("Scene manifest episode indices are not contiguous in the requested slice.")
        scenario_metadata = record.get("scenario")
        if not isinstance(scenario_metadata, dict):
            raise ValueError("Scene manifest scenario metadata is missing.")
        if record.get("scene_hash") != _scene_hash(scenario_metadata):
            raise ValueError(f"Invalid scene_hash at episode {index}.")
        spec = record.get("spec")
        if not isinstance(spec, dict):
            raise ValueError(f"Missing spec at episode {index}.")
        scenario = scenario_from_metadata(dict(scenario_metadata))
        config = config_for_spec("f2", spec, environment_config)
        validation_env = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=len(scenario.obstacles),
            target_speed_scale=float(spec["target_speed_scale"]),
        )
        validate_showcase_scenario(validation_env, scenario)
        result.append(dict(record))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "jepa_safe_capture_l0_l3_collection_v1.yaml",
    )
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml",
    )
    parser.add_argument(
        "--actor-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "v5_development_exact_reactive_seed661606.pt",
    )
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cbf-horizon", type=int, default=3)
    parser.add_argument("--route-probe-horizon", type=int, default=3)
    parser.add_argument("--route-corridor-samples", type=int, default=65)
    parser.add_argument("--minimum-hold-steps", type=int, default=3)
    parser.add_argument("--switch-improvement-m", type=float, default=0.35)
    parser.add_argument("--tangent-route-hold-steps", type=int, default=6)
    parser.add_argument("--boundary-rescue-enabled", action="store_true")
    parser.add_argument("--boundary-rescue-trigger-m", type=float, default=3.0)
    parser.add_argument("--boundary-rescue-offset-m", type=float, default=2.0)
    parser.add_argument("--development-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.development_only:
        raise SystemExit("This evaluator is development-only; pass --development-only explicitly.")
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive.")
    if args.cbf_horizon <= 0 or args.route_probe_horizon <= 0:
        raise SystemExit("CBF and route probe horizons must be positive.")
    if args.minimum_hold_steps < 0 or args.tangent_route_hold_steps < 0:
        raise SystemExit("Route hold steps must be non-negative.")

    output_dir = _fresh(args.output_dir, "output directory")
    (output_dir / "step_traces").mkdir(parents=True, exist_ok=True)
    tensorboard_dir = _fresh(args.tensorboard_dir, "TensorBoard directory")
    device = select_device(args.device)
    # The manifest is authoritative for development replay. Validation here
    # checks its hashes and geometry without regenerating specs from a possibly
    # different collection protocol (for example the dedicated L0 recovery
    # protocol has a development-only split rather than S3 validation).
    manifest = _load_manifest_records(args.scene_manifest, args.environment_config, args.episodes, args.start_index)
    if not manifest:
        raise SystemExit("scene manifest produced no episodes")

    probe_spec = dict(manifest[0]["spec"])
    base_config = config_for_spec("f2", probe_spec, args.environment_config)
    probe_env = CaptureRadiusPursuit3DEnv(
        base_config,
        obstacle_count=len(manifest[0]["scenario"]["obstacles"]),
        target_speed_scale=float(probe_spec["target_speed_scale"]),
    )
    actor, action_scale, _checkpoint = load_policy(
        args.actor_checkpoint,
        probe_env,
        probe_env.reset(seed=int(probe_spec["episode_seed"]), record_history=True),
        device,
    )
    rows: list[dict[str, Any]] = []
    for item in manifest:
        spec = dict(item["spec"])
        config = config_for_spec("f2", spec, args.environment_config)
        rows.append(
            _run_episode(
                item=item,
                config=config,
                actor=actor,
                action_scale=action_scale,
                device=device,
                args=args,
                output_dir=output_dir,
            )
        )

    overall = {
        "episodes": len(rows),
        "safe_capture_count": int(sum(row["safe_capture_success"] for row in rows)),
        "safe_capture_rate": float(np.mean([row["safe_capture_success"] for row in rows])),
        "collision_count": int(sum(row["collision"] for row in rows)),
        "boundary_violation_count": int(sum(row["boundary_violation"] for row in rows)),
        "target_boundary_violation_count": int(sum(row["target_boundary_violation"] for row in rows)),
        "pairwise_violation_count": int(sum(row["pairwise_violation"] for row in rows)),
        "raw_unverified_executed_steps": int(sum(row["route_counters"]["raw_unverified"] for row in rows)),
        "controlled_abort_steps": int(sum(row["route_counters"]["controlled_abort"] for row in rows)),
        "cbf_fallback_steps": int(sum(row["route_counters"]["fallback"] for row in rows)),
        "route_switch_steps": int(sum(row["route_switch_steps"] for row in rows)),
        "mean_capture_time_seconds": (
            float(np.mean([row["capture_time_seconds"] for row in rows if row["capture_time_seconds"] is not None]))
            if any(row["capture_time_seconds"] is not None for row in rows)
            else None
        ),
        "worst_min_clearance_m": float(min(row["min_clearance_m"] for row in rows)),
        "route_probe_checks": int(sum(row["route_counters"]["cbf_checks"] for row in rows)),
        "route_probe_accepted": int(sum(row["route_counters"]["cbf_accepted"] for row in rows)),
        "route_probe_rejected": int(sum(row["route_counters"]["cbf_rejected"] for row in rows)),
    }
    metadata = {
        "evaluation_type": "dn_mpc_cbf_scene_manifest_development",
        "development_only": True,
        "locked_test_opened": False,
        "episodes": len(rows),
        "requested_episodes": int(args.episodes),
        "manifest_start_index": int(args.start_index),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "inputs": {
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "actor_checkpoint_sha256": _sha256(args.actor_checkpoint),
            "environment_config": str(args.environment_config.resolve()),
            "environment_config_sha256": _sha256(args.environment_config),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": _sha256(args.protocol),
            "scene_manifest": str(args.scene_manifest.resolve()),
            "scene_manifest_sha256": _sha256(args.scene_manifest),
        },
        "scene_conditions": sorted(
            {str(item["spec"].get("difficulty", item["spec"].get("observation_condition", "unknown"))) for item in manifest}
        ),
        "contract": {
            "planner": asdict(_planner_config(args, 0.1)),
            "cbf": JointCBFQPSafetyFilter(
                probe_env,
                anticipatory_horizon_steps=args.cbf_horizon,
                barrier_mode="strict_buffer",
            ).contract,
            "route_probe_horizon": int(args.route_probe_horizon),
            "route_chunk_length_steps": 5,
            "boundary_rescue_enabled": bool(args.boundary_rescue_enabled),
            "execute_first_step_then_replan": True,
            "jepa_enabled": False,
            "ledger_enabled": False,
        },
        "environment": _environment_metadata(device),
        "tensorboard_dir": str(tensorboard_dir),
    }
    _write_json(
        output_dir / "summary.json",
        {
            "overall": overall,
            "metadata": metadata,
            "episodes": [{key: value for key, value in row.items() if key != "traces"} for row in rows],
        },
    )
    _write_json(output_dir / "provenance.json", metadata)
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=2) as writer:
        writer.add_text("Provenance/metadata", json.dumps(_jsonable(metadata), indent=2), 0)
        for key, value in overall.items():
            if isinstance(value, (int, float)) and value is not None:
                writer.add_scalar(f"Aggregate/{key}", float(value), 0)
        for row in rows:
            step = int(row["episode_index"])
            writer.add_scalar("Episode/safe_capture", float(row["safe_capture_success"]), step)
            writer.add_scalar("Episode/route_switch_steps", float(row["route_switch_steps"]), step)
            writer.add_scalar("Episode/controlled_abort_steps", float(row["route_counters"]["controlled_abort"]), step)
            writer.add_scalar("Episode/worst_min_clearance_m", float(row["min_clearance_m"]), step)
    print(json.dumps(_jsonable({"overall": overall, "metadata": metadata}), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
