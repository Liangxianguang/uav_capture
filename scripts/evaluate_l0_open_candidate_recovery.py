"""Evaluate candidate-based recovery on the frozen eight-episode L0-open set.

This development diagnostic avoids rebuilding the L1-L3 random maps.  It is
used only to compare the legacy and geometry-conditioned candidate contracts
against the same actor, JEPA, ledger, and frozen L0 manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import scenario_from_metadata  # noqa: E402
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from evaluate_jepa_safe_capture_v2_paired import (  # noqa: E402
    _fresh,
    _jsonable,
    _load_jepa,
    _load_ledger,
    _metric_summary,
    _ranker_config,
    _run_episode,
    _variant_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--reliability-ledger", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--collection-config", type=Path, default=ROOT / "configs/jepa_safe_capture_l0_l3_collection_v2.yaml")
    parser.add_argument("--environment-config", type=Path, default=ROOT / "configs/capture_radius_pursuit_central_v4_flee.yaml")
    parser.add_argument("--candidate-profile", choices=("legacy", "extended_v1", "obstacle_route_v1"), default="extended_v1")
    parser.add_argument("--proactive-braking-clearance", type=float)
    parser.add_argument(
        "--cbf-horizon",
        type=int,
        default=5,
        help="Joint CBF anticipatory horizon; L0 recovery defaults to 5 after diagnosis.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = _fresh(args.output_dir, "L0 candidate recovery output")
    trace_dir = output / "step_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(line)
        for line in args.scene_manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if str(record["spec"].get("difficulty")) == "l0_open"]
    if len(records) != 8:
        raise ValueError(f"Expected exactly 8 frozen L0-open records, got {len(records)}.")
    config = yaml.safe_load(args.environment_config.resolve().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Environment configuration must be a mapping.")
    device = select_device(args.device)
    contract = _variant_contract("m3")
    jepa = _load_jepa(args.jepa_checkpoint.resolve(), device)
    ledger = _load_ledger(args.reliability_ledger.resolve(), args.jepa_checkpoint.resolve(), args.collection_config.resolve())
    ranker_config = _ranker_config("m3", {"ranking_device": "execution", "actor_device": "execution"})
    first_spec = records[0]["spec"]
    prototype_config = copy.deepcopy(config)
    prototype_config.setdefault("task", {}).setdefault("pursuit", {}).update(dict(first_spec["pursuit_overrides"]))
    prototype_config["task"]["pursuit"]["target_motion_mode"] = str(first_spec["target_motion_mode"])
    prototype = CaptureRadiusPursuit3DEnv(
        prototype_config,
        obstacle_count=int(first_spec["obstacle_count"]),
        target_speed_scale=float(first_spec["target_speed_scale"]),
    )
    prototype_observation = prototype.reset(seed=int(first_spec["episode_seed"]))
    policy, action_scale, actor_metadata = load_policy(
        args.actor_checkpoint.resolve(), prototype, prototype_observation, device
    )
    reset_interval = actor_metadata.get("recurrent_reset_interval_steps")
    rows = []
    scenes = []
    for item in records:
        spec = item["spec"]
        episode_config = copy.deepcopy(config)
        episode_config.setdefault("task", {}).setdefault("pursuit", {}).update(dict(spec["pursuit_overrides"]))
        episode_config["task"]["pursuit"]["target_motion_mode"] = str(spec["target_motion_mode"])
        row, scene = _run_episode(
            manifest_item=item,
            config=episode_config,
            policy=policy,
            action_scale=action_scale,
            device=device,
            contract=contract,
            jepa=jepa,
            ledger=ledger,
            history_length=8,
            jepa_perturbation_mps=0.1,
            recurrent_reset_interval=int(reset_interval) if reset_interval is not None else None,
            ranker_config=ranker_config,
            output_dir=output,
            candidate_profile=args.candidate_profile,
            candidate_cbf_prefilter=True,
            proactive_braking_clearance_m=args.proactive_braking_clearance,
            cbf_anticipatory_horizon_steps=args.cbf_horizon,
            ranking_device=device,
            actor_device=device,
        )
        rows.append(row)
        scenes.append(scene)
    (output / "scene_manifest.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), allow_nan=False) + "\n" for item in records),
        encoding="utf-8",
    )
    (output / "scenes.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), allow_nan=False) + "\n" for item in scenes),
        encoding="utf-8",
    )
    summary = _metric_summary(rows)
    for row in rows:
        row["training_seed"] = 20260911
        row["variant"] = "m3"
        row["route_profile"] = args.candidate_profile
    import csv

    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "evaluation_type": "l0_open_candidate_recovery_development",
        "development_only": True,
        "locked_test_opened": False,
        "variant": "m3",
        "candidate_profile": args.candidate_profile,
        "cbf_anticipatory_horizon_steps": args.cbf_horizon,
        "episodes": len(rows),
        "actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "jepa_checkpoint": str(args.jepa_checkpoint.resolve()),
        "reliability_ledger": str(args.reliability_ledger.resolve()),
        "scene_manifest_sha256": hashlib.sha256(args.scene_manifest.resolve().read_bytes()).hexdigest(),
        "recurrent_reset_interval_steps": reset_interval,
        "device": str(device),
    }
    (output / "summary.json").write_text(json.dumps({"overall": _jsonable(summary), "metadata": metadata}, indent=2), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"overall": _jsonable(summary), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
