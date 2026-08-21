"""Replay one fixed-wall development failure with timestep-level safety evidence.

The replay is diagnostic only.  It cannot be used to choose an evaluation
seed, to construct a training sample, or to open V5 locked block 647201.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_controllers import PursuitCBFSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    load_central_capture_protocol,
    prepare_showcase_episode,
)
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from run_mixed_obstacle_showcase import build_config, build_showcase_scenario  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safety_components(env: CaptureRadiusPursuit3DEnv) -> dict[str, float]:
    """Return signed defender-surface clearances for every safety constraint."""
    radius = float(env.agents["drone_radius"])
    obstacle = min(
        (
            float(env._obstacle_clearance(position, item) - radius)
            for position in env.defender_positions
            for item in env.obstacles
        ),
        default=float("inf"),
    )
    defender_pairs = [
        float(np.linalg.norm(env.defender_positions[first] - env.defender_positions[second]) - 2.0 * radius)
        for first in range(env.n_defenders)
        for second in range(first + 1, env.n_defenders)
    ]
    inter_agent = min(defender_pairs, default=float("inf"))
    lower = env.defender_positions - env.lower[None, :] - radius
    upper = env.upper[None, :] - env.defender_positions - radius
    boundary = float(min(np.min(lower), np.min(upper)))
    return {
        "obstacle_clearance_m": obstacle,
        "inter_agent_clearance_m": inter_agent,
        "boundary_clearance_m": boundary,
        "minimum_clearance_m": float(min(obstacle, inter_agent, boundary)),
    }


def collision_category(components: dict[str, float], world_violation_steps: int) -> str:
    categories: list[str] = []
    if components["obstacle_clearance_m"] < 0.0:
        categories.append("defender_obstacle")
    if components["inter_agent_clearance_m"] < 0.0:
        categories.append("defender_defender")
    if components["boundary_clearance_m"] < 0.0 or world_violation_steps > 0:
        categories.append("world_boundary")
    return "+".join(categories) if categories else "none"


def boundary_violation_labels(
    before: np.ndarray,
    after: np.ndarray,
    velocity_after: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    prefix: str,
) -> list[str]:
    """Identify axes clamped at a bound during the just-executed integration step."""
    labels: list[str] = []
    axis_names = ("x", "y", "z")
    for axis, axis_name in enumerate(axis_names):
        at_lower = np.isclose(after[axis], lower[axis], atol=1e-10)
        at_upper = np.isclose(after[axis], upper[axis], atol=1e-10)
        crossed_lower = at_lower and (before[axis] > lower[axis] + 1e-10 or velocity_after[axis] > 0.0)
        crossed_upper = at_upper and (before[axis] < upper[axis] - 1e-10 or velocity_after[axis] < 0.0)
        if crossed_lower:
            labels.append(f"{prefix}_{axis_name}_lower")
        if crossed_upper:
            labels.append(f"{prefix}_{axis_name}_upper")
    return labels


def _policy_action(
    policy: Any,
    local_observation: np.ndarray,
    hidden: Any,
    device: torch.device,
    action_scale: float,
) -> tuple[np.ndarray, Any]:
    local = torch.as_tensor(local_observation, device=device)
    if hidden is not None:
        distribution, hidden = policy.distribution_step(local, hidden)
    else:
        distribution = policy.distribution(local)
    return torch.tanh(distribution.mean).cpu().numpy() * action_scale, hidden


def replay_diagnostic(
    *,
    checkpoint: Path,
    seed: int,
    device_name: str,
    protocol_config: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], CaptureRadiusPursuit3DEnv]:
    """Reproduce the fixed S1 wall episode and retain diagnostic-only trace data."""
    protocol = load_central_capture_protocol(protocol_config)
    scenario = build_showcase_scenario("s1", 5.0, protocol=protocol, layout="wall")
    config = build_config(
        "f2",
        14.0,
        0.55,
        target_crossing_required=bool(scenario.target_crossing_required),
        protocol=protocol,
        obstacle_count=len(scenario.obstacles),
    )
    device = select_device(device_name)
    prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    policy, action_scale, _metadata = load_policy(checkpoint, prototype, prototype.reset(seed=seed), device)

    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(env, scenario, seed=seed, record_history=True, validate_scenario=True)
    safety_filter = PursuitCBFSafetyFilter(env)
    local_observation = policy_observations(env, observation)
    hidden = policy.initial_actor_hidden(env.n_defenders, device=device) if hasattr(policy, "initial_actor_hidden") else None
    frames: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    violation_events: list[dict[str, Any]] = []

    with torch.no_grad():
        while True:
            before_positions = env.defender_positions.copy()
            before_target = env.target_position.copy()
            before_components = safety_components(env)
            world_violations_before = int(env.world_violation_steps)
            raw_action, hidden = _policy_action(policy, local_observation, hidden, device, action_scale)
            safe_action, diagnostics = safety_filter.filter(raw_action, observation)
            observation, _reward, terminated, truncated, final_info = env.step(safe_action, record_history=True)
            after_components = safety_components(env)
            world_violations_after = int(final_info.get("world_violation_steps", 0))
            labels: list[str] = []
            for defender_index, (before, after, velocity) in enumerate(
                zip(before_positions, env.defender_positions, env.defender_velocities, strict=True)
            ):
                labels.extend(
                    boundary_violation_labels(
                        before,
                        after,
                        velocity,
                        env.lower,
                        env.upper,
                        f"pursuer_{defender_index + 1}",
                    )
                )
            labels.extend(
                boundary_violation_labels(
                    before_target,
                    env.target_position,
                    env.target_velocity,
                    env.lower,
                    env.upper,
                    "target",
                )
            )
            if world_violations_after > world_violations_before:
                violation_events.append(
                    {
                        "step": int(env.step_count),
                        "world_violation_increment": world_violations_after - world_violations_before,
                        "labels": labels,
                    }
                )
            frames.append(
                {
                    "step": int(env.step_count),
                    "defender_positions_before": before_positions,
                    "defender_positions_after": env.defender_positions.copy(),
                    "target_position_before": before_target,
                    "target_position_after": env.target_position.copy(),
                    "raw_action": raw_action,
                    "cbf_action": safe_action,
                    "cbf_action_correction_norm": float(diagnostics.action_correction_norm),
                    "cbf_minimum_barrier_value": float(diagnostics.minimum_barrier_value),
                    "before": before_components,
                    "after": after_components,
                    "message_age_steps": env.message_age_steps.copy(),
                    "observation_age_steps": np.maximum(
                        env.step_count - env.target_observation_timestamps, 0
                    ).astype(np.int64),
                    "world_violation_steps": world_violations_after,
                    "target_boundary_clearance_m": float(
                        min(np.min(env.target_position - env.lower), np.min(env.upper - env.target_position))
                    ),
                    "termination_reason": str(final_info.get("termination_reason", "running")),
                }
            )
            if terminated or truncated:
                break
            local_observation = policy_observations(env, observation)

    trace = {
        "step": np.asarray([frame["step"] for frame in frames], dtype=np.int64),
        "defender_positions_before": np.asarray([frame["defender_positions_before"] for frame in frames]),
        "defender_positions_after": np.asarray([frame["defender_positions_after"] for frame in frames]),
        "target_positions_before": np.asarray([frame["target_position_before"] for frame in frames]),
        "target_positions_after": np.asarray([frame["target_position_after"] for frame in frames]),
        "raw_actions": np.asarray([frame["raw_action"] for frame in frames]),
        "cbf_actions": np.asarray([frame["cbf_action"] for frame in frames]),
        "cbf_action_correction_norm": np.asarray([frame["cbf_action_correction_norm"] for frame in frames]),
        "cbf_minimum_barrier_value": np.asarray([frame["cbf_minimum_barrier_value"] for frame in frames]),
        "obstacle_clearance_m": np.asarray([frame["after"]["obstacle_clearance_m"] for frame in frames]),
        "inter_agent_clearance_m": np.asarray([frame["after"]["inter_agent_clearance_m"] for frame in frames]),
        "boundary_clearance_m": np.asarray([frame["after"]["boundary_clearance_m"] for frame in frames]),
        "minimum_clearance_m": np.asarray([frame["after"]["minimum_clearance_m"] for frame in frames]),
        "target_boundary_clearance_m": np.asarray([frame["target_boundary_clearance_m"] for frame in frames]),
        "message_age_steps": np.asarray([frame["message_age_steps"] for frame in frames]),
        "observation_age_steps": np.asarray([frame["observation_age_steps"] for frame in frames]),
    }
    final_components = safety_components(env)
    target_distances = np.linalg.norm(
        trace["defender_positions_after"] - trace["target_positions_after"][:, None, :], axis=2
    )
    closest_frame, closest_defender = np.unravel_index(int(np.argmin(target_distances)), target_distances.shape)
    summary = {
        "diagnostic_type": "central_v5_fixed_wall_timestep_replay",
        "not_a_training_or_locked_evaluation": True,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "seed": int(seed),
        "scenario": "s1",
        "layout": "wall",
        "use_cbf": True,
        "steps": int(env.step_count),
        "capture_event": bool(final_info.get("capture_event", False)),
        "safe_capture_success": bool(final_info.get("safe_capture_success", False)),
        "collision": bool(final_info.get("collision", False)),
        "world_violation_steps": int(final_info.get("world_violation_steps", 0)),
        "termination_reason": str(final_info.get("termination_reason", "running")),
        "world_boundary_violation_events": violation_events,
        "final_safety_components_m": final_components,
        "collision_category": collision_category(final_components, int(final_info.get("world_violation_steps", 0))),
        "minimum_obstacle_clearance_m": float(np.min(trace["obstacle_clearance_m"])),
        "minimum_inter_agent_clearance_m": float(np.min(trace["inter_agent_clearance_m"])),
        "minimum_boundary_clearance_m": float(np.min(trace["boundary_clearance_m"])),
        "minimum_target_boundary_clearance_m": float(np.min(trace["target_boundary_clearance_m"])),
        "minimum_target_distance_m": float(target_distances[closest_frame, closest_defender]),
        "minimum_target_distance_step": int(trace["step"][closest_frame]),
        "minimum_target_distance_defender_id": int(closest_defender + 1),
        "maximum_cbf_action_correction_norm": float(np.max(trace["cbf_action_correction_norm"])),
        "minimum_cbf_barrier_value": float(np.min(trace["cbf_minimum_barrier_value"])),
        "maximum_message_age_steps": int(np.max(trace["message_age_steps"])),
        "maximum_observation_age_steps": int(np.max(trace["observation_age_steps"])),
        "obstacle_geometry": [
            {
                "shape": item.shape,
                "center_xy": item.center_xy.tolist(),
                "radius": float(item.radius),
                "height": float(item.height),
                "half_extents_xy": None if item.half_extents_xy is None else item.half_extents_xy.tolist(),
            }
            for item in env.obstacles
        ],
    }
    return summary, trace, env


def _read_reference_row(path: Path, seed: int) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if int(row["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one reference row for seed {seed} in {path}")
    return matches[0]


def verify_reference(summary: dict[str, Any], reference: dict[str, str]) -> dict[str, bool]:
    checks = {
        "steps": int(reference["steps"]) == summary["steps"],
        "collision": (reference["collision"].strip().lower() == "true") == summary["collision"],
        "safe_capture_success": (
            reference["safe_capture_success"].strip().lower() == "true"
        ) == summary["safe_capture_success"],
        "termination_reason": reference["termination_reason"] == summary["termination_reason"],
    }
    if not all(checks.values()):
        raise ValueError(f"Diagnostic replay does not match fixed-regression reference: {checks}")
    return checks


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_topdown(trace: dict[str, np.ndarray], summary: dict[str, Any], path: Path) -> None:
    """Draw a dependency-light map view; PNG rendering must not need a GUI backend."""
    width, height = 1120, 760
    image = Image.new("RGB", (width, height), (11, 17, 27))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font, label_font = _font(26, True), _font(15)
    map_box = (54, 106, 790, 720)
    extent = 8.0
    scale = min(map_box[2] - map_box[0], map_box[3] - map_box[1]) / (2.0 * extent)

    def project(point: np.ndarray) -> tuple[int, int]:
        return (
            int(map_box[0] + (float(point[0]) + extent) * scale),
            int(map_box[3] - (float(point[1]) + extent) * scale),
        )

    draw.rectangle((0, 0, width, 78), fill=(16, 25, 39))
    draw.text((42, 19), "V5 FIXED-WALL FAILURE REPLAY", font=title_font, fill=(239, 246, 255))
    draw.text((43, 56), "Yellow markers show the timestep with minimum signed safety clearance.", font=label_font, fill=(155, 181, 208))
    draw.rounded_rectangle(map_box, radius=8, fill=(20, 30, 44), outline=(74, 100, 129), width=2)
    for coordinate in np.linspace(-extent, extent, 9):
        first, second = project(np.array([coordinate, -extent, 0.0])), project(np.array([coordinate, extent, 0.0]))
        third, fourth = project(np.array([-extent, coordinate, 0.0])), project(np.array([extent, coordinate, 0.0]))
        draw.line((first, second), fill=(88, 113, 141, 60), width=1)
        draw.line((third, fourth), fill=(88, 113, 141, 60), width=1)
    for item in summary["obstacle_geometry"]:
        center = project(np.array([item["center_xy"][0], item["center_xy"][1], 0.0]))
        if item["shape"] == "cylinder":
            radius = int(float(item["radius"]) * scale)
            draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=(85, 103, 127), outline=(192, 210, 229), width=2)
        else:
            half_x, half_y = item["half_extents_xy"]
            draw.rectangle((center[0] - int(half_x * scale), center[1] - int(half_y * scale), center[0] + int(half_x * scale), center[1] + int(half_y * scale)), fill=(85, 103, 127), outline=(192, 210, 229), width=2)
    colors = ((32, 205, 245), (255, 174, 54), (123, 231, 94), (192, 130, 255))
    positions = trace["defender_positions_after"]
    target = trace["target_positions_after"]
    for index, color in enumerate(colors):
        points = [project(point) for point in positions[:, index]]
        if len(points) > 1:
            draw.line(points, fill=color + (190,), width=3)
        draw.ellipse((points[-1][0] - 7, points[-1][1] - 7, points[-1][0] + 7, points[-1][1] + 7), fill=color + (255,), outline=(240, 248, 255), width=1)
        draw.text((points[-1][0] + 9, points[-1][1] + 6), f"P{index + 1}", font=label_font, fill=color + (255,))
    target_points = [project(point) for point in target]
    draw.line(target_points, fill=(255, 74, 94, 220), width=4)
    draw.ellipse((target_points[-1][0] - 8, target_points[-1][1] - 8, target_points[-1][0] + 8, target_points[-1][1] + 8), fill=(255, 74, 94), outline=(255, 235, 240), width=2)
    failure_index = int(np.argmin(trace["minimum_clearance_m"]))
    for point in positions[failure_index]:
        x, y = project(point)
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(255, 235, 59), width=3)
    failure_target = project(target[failure_index])
    draw.ellipse((failure_target[0] - 13, failure_target[1] - 13, failure_target[0] + 13, failure_target[1] + 13), outline=(255, 235, 59), width=3)
    panel_x = 830
    draw.text((panel_x, 122), "DIAGNOSTIC", font=_font(19, True), fill=(157, 181, 208))
    lines = (
        ("Outcome", summary["termination_reason"]),
        ("Collision", summary["collision_category"]),
        ("Min obstacle", f"{summary['minimum_obstacle_clearance_m']:.3f} m"),
        ("Min inter-agent", f"{summary['minimum_inter_agent_clearance_m']:.3f} m"),
        ("Max CBF change", f"{summary['maximum_cbf_action_correction_norm']:.3f}"),
        ("Failure step", str(int(trace["step"][failure_index]))),
    )
    for line_index, (label, value) in enumerate(lines):
        y = 172 + 68 * line_index
        draw.text((panel_x, y), label.upper(), font=_font(13, True), fill=(135, 160, 190))
        draw.text((panel_x, y + 22), value, font=_font(17, True), fill=(239, 246, 255))
    image.save(path)


def render_3d(trace: dict[str, np.ndarray], summary: dict[str, Any], path: Path) -> None:
    """Draw an oblique three-dimensional projection using Pillow only."""
    width, height = 1120, 760
    image = Image.new("RGB", (width, height), (11, 17, 27))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width, 78), fill=(16, 25, 39))
    draw.text((42, 19), "V5 FIXED-WALL FAILURE REPLAY — 3D VIEW", font=_font(26, True), fill=(239, 246, 255))
    draw.text((43, 56), "Oblique projection of altitude, wall geometry, and all pursuer / evader trajectories.", font=_font(15), fill=(155, 181, 208))
    extent, scale, origin = 8.0, 31.0, (530, 620)

    def project(point: np.ndarray) -> tuple[int, int]:
        x, y, z = (float(value) for value in point)
        return (int(origin[0] + (x - y) * scale), int(origin[1] - (x + y) * scale * 0.38 - z * scale * 1.12))

    for coordinate in np.linspace(-extent, extent, 9):
        draw.line((project(np.array([coordinate, -extent, 0.0])), project(np.array([coordinate, extent, 0.0]))), fill=(88, 113, 141, 60), width=1)
        draw.line((project(np.array([-extent, coordinate, 0.0])), project(np.array([extent, coordinate, 0.0]))), fill=(88, 113, 141, 60), width=1)
    for item in summary["obstacle_geometry"]:
        half_x, half_y = (item["radius"], item["radius"]) if item["shape"] == "cylinder" else item["half_extents_xy"]
        x0, x1 = item["center_xy"][0] - half_x, item["center_xy"][0] + half_x
        y0, y1 = item["center_xy"][1] - half_y, item["center_xy"][1] + half_y
        base = [project(np.array([x, y, 0.0])) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        top = [project(np.array([x, y, item["height"]])) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        draw.polygon(base, fill=(54, 67, 86), outline=(137, 158, 184))
        draw.polygon((base[1], base[2], top[2], top[1]), fill=(75, 92, 113), outline=(155, 177, 201))
        draw.polygon((base[2], base[3], top[3], top[2]), fill=(65, 82, 103), outline=(155, 177, 201))
        draw.polygon(top, fill=(105, 123, 146), outline=(200, 214, 229))
    colors = ((32, 205, 245), (255, 174, 54), (123, 231, 94), (192, 130, 255))
    positions = trace["defender_positions_after"]
    target = trace["target_positions_after"]
    for index, color in enumerate(colors):
        points = [project(point) for point in positions[:, index]]
        draw.line(points, fill=color + (205,), width=3)
        draw.ellipse((points[-1][0] - 7, points[-1][1] - 7, points[-1][0] + 7, points[-1][1] + 7), fill=color + (255,), outline=(240, 248, 255), width=1)
    target_points = [project(point) for point in target]
    draw.line(target_points, fill=(255, 74, 94, 230), width=4)
    draw.ellipse((target_points[-1][0] - 8, target_points[-1][1] - 8, target_points[-1][0] + 8, target_points[-1][1] + 8), fill=(255, 74, 94), outline=(255, 235, 240), width=2)
    failure_index = int(np.argmin(trace["minimum_clearance_m"]))
    failure_point = project(target[failure_index])
    draw.ellipse((failure_point[0] - 15, failure_point[1] - 15, failure_point[0] + 15, failure_point[1] + 15), outline=(255, 235, 59), width=3)
    draw.text((56, 700), f"Minimum-clearance step: {int(trace['step'][failure_index])}  |  collision: {summary['collision_category']}", font=_font(16, True), fill=(255, 235, 59))
    image.save(path)


def render_report(summary: dict[str, Any], checks: dict[str, bool], trace_sha256: str) -> str:
    return "\n".join(
        [
            "# V5 Fixed-Wall Regression Diagnostic",
            "",
            "This is one deterministic replay of an already observed development failure. It is not a new evaluation and its seed is excluded from training.",
            "",
            "## Reference Match",
            "",
            *[f"- `{name}`: `{passed}`" for name, passed in checks.items()],
            "",
            "## Replay Outcome",
            "",
            f"- Checkpoint SHA-256: `{summary['checkpoint_sha256']}`",
            f"- Fixed-wall evaluation seed: `{summary['seed']}`",
            f"- Steps / termination: `{summary['steps']}` / `{summary['termination_reason']}`",
            f"- Collision category: `{summary['collision_category']}`",
            f"- Boundary violation events: `{summary['world_boundary_violation_events']}`",
            f"- Capture / safe capture: `{summary['capture_event']}` / `{summary['safe_capture_success']}`",
            f"- Minimum obstacle / inter-agent / boundary clearance: `{summary['minimum_obstacle_clearance_m']:.6f}` / `{summary['minimum_inter_agent_clearance_m']:.6f}` / `{summary['minimum_boundary_clearance_m']:.6f}` m",
            f"- Minimum target-boundary clearance / target distance: `{summary['minimum_target_boundary_clearance_m']:.6f}` / `{summary['minimum_target_distance_m']:.6f}` m at step `{summary['minimum_target_distance_step']}` (P{summary['minimum_target_distance_defender_id']})",
            f"- Maximum CBF correction / minimum CBF barrier: `{summary['maximum_cbf_action_correction_norm']:.6f}` / `{summary['minimum_cbf_barrier_value']:.6f}`",
            f"- Maximum message / observation age: `{summary['maximum_message_age_steps']}` / `{summary['maximum_observation_age_steps']}` steps",
            f"- Timestep trace SHA-256: `{trace_sha256}`",
            "",
            "## Interpretation",
            "",
            "The next permitted intervention is data-only fixed-wall coverage recovery after pre-registration. This diagnostic must not be used to add this evaluation seed or its exact map to training, change CBF parameters, or open locked block 647201.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=660514)
    parser.add_argument("--protocol-config", type=Path, default=Path("configs/central_bidirectional_v4.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, trace, _env = replay_diagnostic(
        checkpoint=checkpoint,
        seed=args.seed,
        device_name=args.device,
        protocol_config=args.protocol_config,
    )
    reference = _read_reference_row(args.reference_episodes.resolve(), args.seed)
    checks = verify_reference(summary, reference)
    trace_path = output_dir / "timestep_trace.npz"
    np.savez_compressed(trace_path, **trace)
    trace_hash = _sha256(trace_path)
    summary["reference_episodes_csv"] = str(args.reference_episodes.resolve())
    summary["reference_episodes_csv_sha256"] = _sha256(args.reference_episodes.resolve())
    summary["reference_match"] = checks
    summary["timestep_trace"] = trace_path.name
    summary["timestep_trace_sha256"] = trace_hash
    render_topdown(trace, summary, output_dir / "topdown.png")
    render_3d(trace, summary, output_dir / "three_d.png")
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "diagnostic_report.md").write_text(render_report(summary, checks, trace_hash), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
