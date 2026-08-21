"""Frozen case generation and validation for the E1 execution benchmark."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .execution_dynamics import ExecutionDynamicsConfig


E1_SPLITS = ("smoke", "development", "locked_test")
E1_PROFILES = tuple(f"E{index}" for index in range(7))


def load_e1_protocol(path: Path) -> dict[str, Any]:
    """Load and validate the complete pre-registration input contract."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("E1 protocol must be a mapping.")
    seed_blocks = document.get("seed_blocks")
    profiles = document.get("execution_profiles")
    budget = document.get("evaluation_budget")
    scenario = document.get("s3")
    if not isinstance(seed_blocks, dict) or not isinstance(profiles, dict) or not isinstance(budget, dict):
        raise ValueError("E1 protocol requires seed_blocks, execution_profiles, and evaluation_budget mappings.")
    if not isinstance(scenario, dict):
        raise ValueError("E1 protocol requires an s3 sampling mapping.")
    if set(profiles) != set(E1_PROFILES):
        raise ValueError("E1 protocol must define exactly E0 through E6.")
    if set(seed_blocks) != set(E1_SPLITS):
        raise ValueError("E1 seed_blocks must define smoke, development, and locked_test.")
    seed_values = [int(seed_blocks[name]) for name in E1_SPLITS]
    if len(set(seed_values)) != len(seed_values) or any(value < 0 for value in seed_values):
        raise ValueError("E1 seed blocks must be distinct non-negative integers.")
    for name in E1_PROFILES:
        ExecutionDynamicsConfig.from_mapping(profiles[name])
    if str(document.get("primary_profile")) != "E6":
        raise ValueError("E1 primary_profile must remain E6.")
    for name in (
        "smoke_episodes_per_profile",
        "development_s3_episodes_per_profile",
        "locked_s3_episodes_per_profile",
        "fixed_episodes_per_scene",
    ):
        if int(budget.get(name, 0)) <= 0:
            raise ValueError(f"E1 evaluation_budget.{name} must be positive.")
    for name in (
        "obstacle_count_range",
        "initial_side_distances",
        "defender_sides",
        "target_speed_scales",
        "target_motion_modes",
        "observation_conditions",
    ):
        if not isinstance(scenario.get(name), list) or not scenario[name]:
            raise ValueError(f"E1 s3.{name} must be a non-empty list.")
    if len(scenario["obstacle_count_range"]) != 2:
        raise ValueError("E1 s3.obstacle_count_range must contain two values.")
    if not all(isinstance(item, dict) for item in scenario["observation_conditions"]):
        raise ValueError("E1 observation_conditions must contain mappings.")
    for item in scenario["observation_conditions"]:
        if not isinstance(item.get("name"), str) or not isinstance(item.get("pursuit_overrides"), dict):
            raise ValueError("E1 observation condition requires name and pursuit_overrides.")
    if not isinstance(document.get("environment_config"), str):
        raise ValueError("E1 environment_config must be a relative YAML path string.")
    return document


def episode_count(protocol: dict[str, Any], split: str, override: int | None = None) -> int:
    if split not in E1_SPLITS:
        raise ValueError(f"Unknown E1 split: {split}")
    budget = protocol["evaluation_budget"]
    configured = {
        "smoke": int(budget["smoke_episodes_per_profile"]),
        "development": int(budget["development_s3_episodes_per_profile"]),
        "locked_test": int(budget["locked_s3_episodes_per_profile"]),
    }[split]
    if split == "locked_test" and override is not None and int(override) != configured:
        raise ValueError(f"E1 locked_test requires exactly {configured} episodes.")
    selected = configured if override is None else int(override)
    if selected <= 0:
        raise ValueError("E1 episode count must be positive.")
    return selected


def execution_config(protocol: dict[str, Any], profile: str) -> ExecutionDynamicsConfig:
    if profile not in E1_PROFILES:
        raise ValueError(f"Unknown E1 execution profile: {profile}")
    return ExecutionDynamicsConfig.from_mapping(protocol["execution_profiles"][profile])


def episode_spec(protocol: dict[str, Any], split: str, episode_index: int) -> dict[str, Any]:
    """Generate one deterministic case independent of policy and CBF mode."""
    if split not in E1_SPLITS:
        raise ValueError(f"Unknown E1 split: {split}")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative.")
    settings = protocol["s3"]
    base_seed = int(protocol["seed_blocks"][split])
    lower, upper = (int(value) for value in settings["obstacle_count_range"])
    if lower <= 0 or upper < lower:
        raise ValueError("E1 obstacle_count_range must be positive and ordered.")
    conditions = list(
        itertools.product(
            range(lower, upper + 1),
            settings["defender_sides"],
            settings["initial_side_distances"],
            settings["target_speed_scales"],
            settings["target_motion_modes"],
            settings["observation_conditions"],
        )
    )
    order = np.random.default_rng(base_seed + 2_000_000).permutation(len(conditions))
    obstacle_count, defender_side, initial_side_distance, target_speed_scale, target_motion_mode, observation = conditions[
        int(order[episode_index % len(order)])
    ]
    return {
        "episode_index": int(episode_index),
        "episode_seed": base_seed + int(episode_index),
        "layout_seed": base_seed + 1_000_000 + int(episode_index),
        "execution_noise_seed": _execution_noise_seed(base_seed + int(episode_index)),
        "obstacle_count": int(obstacle_count),
        "defender_side": str(defender_side),
        "initial_side_distance": float(initial_side_distance),
        "target_speed_scale": float(target_speed_scale),
        "target_motion_mode": str(target_motion_mode),
        "observation_condition": str(observation["name"]),
        "pursuit_overrides": copy.deepcopy(observation["pursuit_overrides"]),
        "required_defender_zone_entries": int(settings["required_defender_zone_entries"]),
        "target_crossing_required": bool(settings.get("target_crossing_required", False)),
        "condition_index": int(order[episode_index % len(order)]),
        "condition_table_size": int(len(conditions)),
    }


def environment_config(protocol: dict[str, Any], protocol_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Construct a case-specific environment while retaining V4 observation rules."""
    source = (protocol_path.parent / str(protocol["environment_config"])).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("task"), dict):
        raise ValueError("E1 environment_config must be a pursuit environment YAML.")
    config = copy.deepcopy(config)
    config["task"]["pursuit"].update(copy.deepcopy(spec["pursuit_overrides"]))
    config["task"]["pursuit"]["target_motion_mode"] = str(spec["target_motion_mode"])
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments or not isinstance(experiments[0], dict):
        raise ValueError("E1 environment config must contain experiments[0].")
    experiments[0]["target_speed_scale"] = float(spec["target_speed_scale"])
    return config


def case_sha256(spec: dict[str, Any], scenario_metadata: dict[str, Any]) -> str:
    """Stable signature used to enforce raw/K-CBF/E-CBF case pairing."""
    payload = {"spec": spec, "scenario": scenario_metadata}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _execution_noise_seed(episode_seed: int) -> int:
    sequence = np.random.SeedSequence([int(episode_seed), 0xE1_2026])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])

