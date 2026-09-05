"""Run paired development rollouts over the stratified L0-L3 collection.

This evaluator reuses the P6 runtime kernel but builds its manifest from the
stratified L0-L3 collection protocol. It is development-only and never opens
the locked split. Every variant for every training seed receives the same
scenario geometry and episode seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    ShowcaseScenario,
    _opposite_side_positions,
    _random_central_obstacle,
    random_central_mixed_obstacle_scenario,
    scenario_from_metadata,
    scenario_metadata,
    validate_showcase_scenario,
)
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from evaluate_jepa_safe_capture_v2_paired import (  # noqa: E402
    _environment_metadata,
    _fresh,
    _jsonable,
    _load_jepa,
    _load_ledger,
    _ranker_config,
    _run_episode,
    _scene_hash,
    _variant_contract,
    _write_episodes_csv,
    _write_json,
    _write_tensorboard,
)


VARIANTS = ("m0", "m1", "m2", "m3", "a1", "a2", "a3")
TRAINING_SEEDS = (20260911, 20260912, 20260913)
DEFAULT_COLLECTION = PROJECT_ROOT / "configs" / "jepa_safe_capture_l0_l3_collection_v2.yaml"
DEFAULT_ENVIRONMENT_CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"
DEFAULT_ACTOR_CHECKPOINT = PROJECT_ROOT / "models" / "v5_development_exact_reactive_seed661606.pt"


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _experiments(collection: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = collection.get("experiments")
    if not isinstance(values, list) or not values:
        raise ValueError("Collection protocol must declare experiments.")
    result: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Each collection experiment must be a mapping.")
        required = ("name", "difficulty", "obstacle_count", "target_speed_scale", "pursuit_overrides")
        if any(key not in value for key in required):
            raise ValueError(f"Collection experiment is missing a required field: {value!r}")
        overrides = value["pursuit_overrides"]
        if not isinstance(overrides, dict):
            raise ValueError("Collection pursuit_overrides must be a mapping.")
        result.append(dict(value))
    return result


def _spec_for_experiment(
    collection: Mapping[str, Any],
    experiment: Mapping[str, Any],
    experiment_index: int,
    episode_index: int,
    episodes_per_scenario: int,
) -> dict[str, Any]:
    blocks = collection.get("seed_blocks", {})
    if not isinstance(blocks, Mapping) or "development" not in blocks:
        raise ValueError("Collection must provide a development seed block.")
    base = int(blocks["development"])
    ordinal = experiment_index * 10_000 + episode_index
    overrides = dict(experiment["pursuit_overrides"])
    motion = str(overrides.get("target_motion_mode", "flee_persistence"))
    side = "left" if ordinal % 2 == 0 else "right"
    return {
        "episode_seed": base + ordinal,
        "layout_seed": base + 1_000_000 + ordinal,
        "defender_side": side,
        "initial_side_distance": 6.0 + 0.5 * (ordinal % 4),
        "target_speed_scale": float(experiment["target_speed_scale"]),
        "target_motion_mode": motion,
        "target_crossing_required": bool(experiment.get("target_crossing_required", False)),
        "observation_condition": str(experiment["name"]),
        "pursuit_overrides": overrides,
        "obstacle_count": int(experiment["obstacle_count"]),
        "condition_index": experiment_index,
        "condition_table_size": len(_experiments(collection)),
        "difficulty": str(experiment["difficulty"]),
        "scenario_name": str(experiment["name"]),
        "episodes_per_scenario": int(episodes_per_scenario),
    }


def _scenario_for_spec(
    spec: Mapping[str, Any],
    environment_config: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(environment_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Environment config must be a mapping.")
    config = copy.deepcopy(config)
    pursuit = config.setdefault("task", {}).setdefault("pursuit", {})
    pursuit.update(copy.deepcopy(dict(spec["pursuit_overrides"])))
    pursuit["target_motion_mode"] = str(spec["target_motion_mode"])
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=0,
        target_speed_scale=float(spec["target_speed_scale"]),
    )
    count = int(spec["obstacle_count"])
    side = str(spec["defender_side"])
    distance = float(spec["initial_side_distance"])
    layout_seed = int(spec["layout_seed"])
    if count >= 3:
        scenario = random_central_mixed_obstacle_scenario(
            env,
            layout_seed=layout_seed,
            initial_side_distance=distance,
            defender_side=side,
            target_crossing_required=bool(spec["target_crossing_required"]),
            obstacle_count_range=(count, count),
            max_attempts=500,
            required_defender_zone_entries=1,
        )
    else:
        defenders, target, escape = _opposite_side_positions(distance, side)
        obstacles = []
        if count == 1:
            rng = np.random.default_rng(layout_seed)
            for _ in range(500):
                candidate = _random_central_obstacle(rng, "cylinder", (-2.5, 3.0))
                try:
                    validate_showcase_scenario(
                        env,
                        ShowcaseScenario(
                            name=f"l0_single_{layout_seed}",
                            obstacles=(candidate,),
                            defender_positions=defenders,
                            target_position=target,
                            target_escape_direction=escape,
                            obstacle_zone_x=(-2.5, 3.0),
                            target_crossing_required=bool(spec["target_crossing_required"]),
                            defender_side=side,
                            layout_seed=layout_seed,
                            required_defender_zone_entries=1,
                        ),
                    )
                except ValueError:
                    continue
                obstacles = [candidate]
                break
            if not obstacles:
                raise RuntimeError(f"Unable to sample a valid single-obstacle map: seed={layout_seed}")
        scenario = ShowcaseScenario(
            name=f"l0_open_{layout_seed}" if count == 0 else f"l0_single_{layout_seed}",
            obstacles=tuple(obstacles),
            defender_positions=defenders,
            target_position=target,
            target_escape_direction=escape,
            obstacle_zone_x=(-2.5, 3.0),
            target_crossing_required=bool(spec["target_crossing_required"]),
            defender_side=side,
            layout_seed=layout_seed,
            required_defender_zone_entries=1,
        )
        validate_showcase_scenario(env, scenario)
    return scenario_metadata(scenario)


def _build_manifest(
    collection: Mapping[str, Any],
    environment_config: Path,
    episodes_per_scenario: int,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for experiment_index, experiment in enumerate(_experiments(collection)):
        for episode_index in range(episodes_per_scenario):
            spec = _spec_for_experiment(collection, experiment, experiment_index, episode_index, episodes_per_scenario)
            scenario = _scenario_for_spec(spec, environment_config)
            manifest.append(
                {
                    "episode_index": len(manifest),
                    "episode_seed": int(spec["episode_seed"]),
                    "layout_seed": int(spec["layout_seed"]),
                    "spec": spec,
                    "scenario": scenario,
                    "scene_hash": _scene_hash(scenario),
                }
            )
    return manifest


def _load_manifest(path: Path, expected: list[dict[str, Any]], environment_config: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != len(expected):
        raise ValueError(f"Manifest has {len(records)} records; expected {len(expected)}.")
    for index, (record, reference) in enumerate(zip(records, expected)):
        if int(record.get("episode_index", -1)) != index:
            raise ValueError("L0-L3 manifest episode indices are not contiguous.")
        if record.get("spec") != reference["spec"]:
            raise ValueError(f"Manifest specification mismatch at episode {index}.")
        scenario = record.get("scenario")
        if not isinstance(scenario, Mapping) or record.get("scene_hash") != _scene_hash(scenario):
            raise ValueError(f"Manifest scene hash is invalid at episode {index}.")
        # Validate geometry and reachability before the manifest enters a run.
        restored = scenario_from_metadata(dict(scenario))
        spec = reference["spec"]
        config = yaml.safe_load(environment_config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Environment config must be a mapping.")
        config = copy.deepcopy(config)
        config.setdefault("task", {}).setdefault("pursuit", {}).update(dict(spec["pursuit_overrides"]))
        validation_env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=float(spec["target_speed_scale"]))
        validate_showcase_scenario(validation_env, restored)
    return [dict(record) for record in records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--episodes-per-scenario", type=int, default=8)
    parser.add_argument("--split", choices=("development",), default="development")
    parser.add_argument("--collection-config", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR_CHECKPOINT)
    parser.add_argument("--jepa-checkpoint", type=Path)
    parser.add_argument("--reliability-ledger", type=Path)
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--jepa-history-length", type=int, default=8)
    parser.add_argument("--jepa-perturbation-mps", type=float, default=0.1)
    parser.add_argument("--recurrent-reset-interval", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("L0-L3 evaluator requires --development-only.")
    if args.episodes_per_scenario <= 0 or args.jepa_history_length <= 0:
        raise ValueError("episodes-per-scenario and history length must be positive.")
    collection_path = args.collection_config.resolve()
    environment_config = args.environment_config.resolve()
    actor_checkpoint = args.actor_checkpoint.resolve()
    for path in (collection_path, environment_config, actor_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    collection = _load_yaml(collection_path)
    if collection.get("phase") != "development_only" or collection.get("locked_test_opened") is not False:
        raise ValueError("Collection protocol is not a closed development-only contract.")
    expected = _build_manifest(collection, environment_config, args.episodes_per_scenario)
    manifest = expected if args.scene_manifest is None else _load_manifest(args.scene_manifest.resolve(), expected, environment_config)
    contract = _variant_contract(args.variant)
    ranking_contract = {"ranking_device": "execution", "actor_device": "execution"}
    ranker_config = _ranker_config(args.variant, ranking_contract)
    device = select_device(args.device)
    jepa_checkpoint = args.jepa_checkpoint.resolve() if args.jepa_checkpoint else None
    ledger_path = args.reliability_ledger.resolve() if args.reliability_ledger else None
    if contract["use_jepa"] and jepa_checkpoint is None:
        raise ValueError("JEPA variants require --jepa-checkpoint.")
    if contract["use_ledger"] and ledger_path is None:
        raise ValueError("Ledger variants require --reliability-ledger.")
    if jepa_checkpoint is not None and not jepa_checkpoint.is_file():
        raise FileNotFoundError(jepa_checkpoint)
    if ledger_path is not None and not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    output_dir = _fresh(args.output_dir, "L0-L3 output directory")
    tensorboard_dir = args.tensorboard_dir.resolve()
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {tensorboard_dir}")
    tensorboard_dir.parent.mkdir(parents=True, exist_ok=True)
    jepa = _load_jepa(jepa_checkpoint, device) if contract["use_jepa"] and jepa_checkpoint else None
    ledger = _load_ledger(ledger_path, jepa_checkpoint, collection_path) if contract["use_ledger"] and ledger_path and jepa_checkpoint else None
    prototype_spec = expected[0]["spec"]
    prototype_config = yaml.safe_load(environment_config.read_text(encoding="utf-8"))
    if not isinstance(prototype_config, dict):
        raise ValueError("Environment config must be a mapping.")
    prototype_config = copy.deepcopy(prototype_config)
    prototype_config.setdefault("task", {}).setdefault("pursuit", {}).update(dict(prototype_spec["pursuit_overrides"]))
    prototype = CaptureRadiusPursuit3DEnv(prototype_config, obstacle_count=0, target_speed_scale=float(prototype_spec["target_speed_scale"]))
    prototype_observation = prototype.reset(seed=int(prototype_spec["episode_seed"]))
    policy, action_scale, actor_metadata = load_policy(actor_checkpoint, prototype, prototype_observation, device)
    recurrent_reset_interval = args.recurrent_reset_interval
    if recurrent_reset_interval is None and actor_metadata.get("recurrent_reset_interval_steps") is not None:
        recurrent_reset_interval = int(actor_metadata["recurrent_reset_interval_steps"])
    for item in manifest:
        item["training_seed"] = int(args.training_seed)
    (output_dir / "scene_manifest.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), allow_nan=False) + "\n" for item in manifest), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    for item in manifest:
        spec = item["spec"]
        config = yaml.safe_load(environment_config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Environment config must be a mapping.")
        config = copy.deepcopy(config)
        config.setdefault("task", {}).setdefault("pursuit", {}).update(dict(spec["pursuit_overrides"]))
        config["task"]["pursuit"]["target_motion_mode"] = str(spec["target_motion_mode"])
        row, scene = _run_episode(
            manifest_item=item,
            config=config,
            policy=policy,
            action_scale=action_scale,
            device=device,
            contract=contract,
            jepa=jepa,
            ledger=ledger,
            history_length=args.jepa_history_length,
            jepa_perturbation_mps=args.jepa_perturbation_mps,
            recurrent_reset_interval=recurrent_reset_interval,
            ranker_config=ranker_config,
            action_comparison_quantum_mps=0.0,
            ranking_device=device,
            actor_device=device,
            output_dir=output_dir,
        )
        row.update(
            {
                "training_seed": int(args.training_seed),
                "scene_hash": item["scene_hash"],
                "episode_index": int(item["episode_index"]),
                "episode_seed": int(spec["episode_seed"]),
                "layout_seed": int(spec["layout_seed"]),
                "level": str(spec["difficulty"]),
                "scenario_name": str(spec["scenario_name"]),
                "obstacle_count": int(spec["obstacle_count"]),
                "target_motion_mode": str(spec["target_motion_mode"]),
                "observation_condition": str(spec["observation_condition"]),
            }
        )
        scene["outcome"] = row
        rows.append(row)
        scenes.append(scene)
    summary = __import__("evaluate_jepa_safe_capture_v2_paired", fromlist=["_metric_summary"])._metric_summary(rows)
    inputs = {
        "collection_config": str(collection_path),
        "collection_config_sha256": __import__("hashlib").sha256(collection_path.read_bytes()).hexdigest(),
        "environment_config": str(environment_config),
        "actor_checkpoint": str(actor_checkpoint),
        "jepa_checkpoint": str(jepa_checkpoint) if jepa_checkpoint else None,
        "reliability_ledger": str(ledger_path) if ledger_path else None,
        "scene_manifest_sha256": __import__("hashlib").sha256((output_dir / "scene_manifest.jsonl").read_bytes()).hexdigest(),
    }
    metadata = {
        "evaluation_type": "jepa_safe_capture_l0_l3_paired_development",
        "trace_schema_version": 2,
        "development_only": True,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "variant": contract,
        "training_seed": int(args.training_seed),
        "split": args.split,
        "episodes": len(rows),
        "episodes_per_scenario": int(args.episodes_per_scenario),
        "scenario_count": len(_experiments(collection)),
        "candidate_contract": {
            "candidate_count": 5,
            "chunk_length_steps": 3,
            "perturbation_mps": float(args.jepa_perturbation_mps),
            "execute_first_step_then_replan": True,
            "project_to_reachable_dynamics": True,
        },
        "recurrent_reset_interval_steps": recurrent_reset_interval,
        "action_scale": float(action_scale),
        "git_revision": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "inputs": inputs,
        "environment": _environment_metadata(device),
        "tensorboard_dir": str(tensorboard_dir),
    }
    _write_episodes_csv(output_dir / "episodes.csv", rows)
    (output_dir / "scenes.jsonl").write_text("".join(json.dumps(_jsonable(item), allow_nan=False) + "\n" for item in scenes), encoding="utf-8")
    _write_json(output_dir / "summary.json", {"overall": summary, "metadata": metadata})
    _write_json(output_dir / "provenance.json", metadata)
    metadata["tensorboard"] = _write_tensorboard(logdir=tensorboard_dir, metadata=metadata, rows=rows, summary=summary)
    _write_json(output_dir / "summary.json", {"overall": summary, "metadata": metadata})
    _write_json(output_dir / "provenance.json", metadata)
    print(json.dumps(_jsonable({"overall": summary, "metadata": metadata}), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
