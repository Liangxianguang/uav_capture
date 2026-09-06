"""Development-only WP1 audit for obstacle-conditioned route proposals.

The audit uses four deterministic public-observation micro-scenes to verify
the causal chain required before JEPA retraining:

    obstacle geometry -> route proposal changes -> safe reachable option

It never opens a locked split, reads simulator target truth, lowers CBF
margins, or executes an action.  ``--with-cbf`` optionally probes every valid
first-step request through the existing Joint CBF implementation; the probe is
read-only and does not replace the execution filter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.obstacle_route_candidates import (  # noqa: E402
    ROUTE_LABELS,
    ObstacleRouteConfig,
    make_obstacle_route_candidates,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"
SCENE_NAMES = ("central_single", "left_blocked", "right_blocked", "wall_single_gap")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--with-cbf",
        action="store_true",
        help="Run read-only Joint CBF probes for valid candidate first steps.",
    )
    return parser.parse_args()


def _cylinder(center: tuple[float, float], radius: float = 1.0) -> dict[str, Any]:
    return {
        "center_xy": np.asarray(center, dtype=np.float64),
        "radius": float(radius),
        "height": 5.0,
        "shape": "cylinder",
        "half_extents_xy": None,
    }


def _wall(center: tuple[float, float], half: tuple[float, float]) -> dict[str, Any]:
    return {
        "center_xy": np.asarray(center, dtype=np.float64),
        "radius": float(min(half)),
        "height": 5.0,
        "shape": "wall",
        "half_extents_xy": np.asarray(half, dtype=np.float64),
    }


def _scene_obstacles(scene: str) -> list[dict[str, Any]]:
    if scene == "central_single":
        return [_cylinder((0.0, 0.0))]
    if scene == "left_blocked":
        return [_cylinder((0.0, 0.0)), _cylinder((0.0, 3.5))]
    if scene == "right_blocked":
        return [_cylinder((0.0, 0.0)), _cylinder((0.0, -3.5))]
    if scene == "wall_single_gap":
        return [_wall((0.0, -3.0), (0.35, 2.0)), _wall((0.0, 3.0), (0.35, 2.0))]
    raise ValueError(f"Unknown WP1 scene: {scene!r}.")


def _public_observation(scene: str, *, lower: np.ndarray, upper: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    positions = np.array(
        [[-6.0, -1.0, 4.0], [-6.0, 1.0, 4.0], [-6.0, 0.8, 5.0], [-6.0, -0.8, 5.0]],
        dtype=np.float64,
    )
    target = np.array([6.0, 0.0, 4.0], dtype=np.float64)
    beliefs = np.repeat(target[None, :], 4, axis=0)
    nominal = np.stack(
        [2.0 * (target - position) / max(float(np.linalg.norm(target - position)), 1e-12) for position in positions]
    )
    observation: dict[str, Any] = {
        "defender_positions": positions,
        "defender_velocities": np.zeros_like(positions),
        "target_belief_positions": beliefs,
        "target_belief_velocities": np.zeros_like(positions),
        "target_visible": np.ones(4, dtype=bool),
        "target_observation_age_steps": np.zeros(4, dtype=np.float64),
        "message_age_steps": np.zeros(4, dtype=np.float64),
        "obstacles": _scene_obstacles(scene),
        "world_lower": lower.copy(),
        "world_upper": upper.copy(),
    }
    return observation, nominal


def _make_cbf_probe(config: Mapping[str, Any]) -> Any:
    """Build a CBF object without resetting an environment or sampling truth."""

    from encirclement3d.cbf_qp import JointCBFQPSafetyFilter
    from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, pursuit_settings

    # Calling the environment constructor would sample/reset simulator state
    # and is unnecessary for a read-only public-observation probe.  Populate
    # only the static fields consumed by JointCBFQPSafetyFilter.
    env = CaptureRadiusPursuit3DEnv.__new__(CaptureRadiusPursuit3DEnv)
    env.config = copy.deepcopy(dict(config))
    env.world = env.config["world"]
    env.agents = env.config["agents"]
    env.task = env.config["task"]
    env.pursuit = pursuit_settings(env.task)
    env.n_defenders = int(env.agents["defenders"])
    env.dt = float(env.world["dt"])
    half_extent = float(env.world["half_extent_xy"])
    env.lower = np.array([-half_extent, -half_extent, float(env.world["minimum_altitude"])], dtype=np.float64)
    env.upper = np.array([half_extent, half_extent, float(env.world["height"])], dtype=np.float64)
    env.obstacles = []
    return JointCBFQPSafetyFilter(env)


def _cbf_row(probe: Any, action: np.ndarray, observation: Mapping[str, Any]) -> dict[str, Any]:
    try:
        diagnostics = probe.verify_requested_action(action, observation)
    except Exception as error:  # pragma: no cover - depends on local SciPy ABI
        return {
            "status": "probe_error",
            "verified_feasible": False,
            "infeasible": True,
            "timed_out": False,
            "fallback_mode": "none",
            "solver_status": "probe_exception",
            "minimum_constraint_value": None,
            "error": repr(error),
        }
    return {
        "status": "verified" if diagnostics.verified_feasible and not diagnostics.timed_out else "rejected",
        "verified_feasible": bool(diagnostics.verified_feasible),
        "infeasible": bool(diagnostics.infeasible),
        "timed_out": bool(diagnostics.timed_out),
        "fallback_mode": str(diagnostics.fallback_mode),
        "solver_status": str(diagnostics.solver_status),
        "minimum_constraint_value": float(diagnostics.minimum_constraint_value),
        "action_correction_norm": float(diagnostics.action_correction_norm),
    }


def _scene_result(
    scene: str,
    *,
    config: Mapping[str, Any],
    route_config: ObstacleRouteConfig,
    cbf_probe: Any | None,
) -> dict[str, Any]:
    lower = np.array(
        [-float(config["world"]["half_extent_xy"]), -float(config["world"]["half_extent_xy"]), float(config["world"]["minimum_altitude"])],
        dtype=np.float64,
    )
    upper = np.array(
        [float(config["world"]["half_extent_xy"]), float(config["world"]["half_extent_xy"]), float(config["world"]["height"])],
        dtype=np.float64,
    )
    observation, nominal = _public_observation(scene, lower=lower, upper=upper)
    batch = make_obstacle_route_candidates(
        nominal,
        observation,
        config=route_config,
        previous_action=np.zeros_like(nominal),
    )
    candidates: list[dict[str, Any]] = []
    for candidate in batch.candidates:
        row = candidate.as_dict()
        row["cbf_probe"] = None if cbf_probe is None or not candidate.valid else _cbf_row(
            cbf_probe,
            candidate.action_chunk[0],
            observation,
        )
        candidates.append(row)
    by_label = {row["label"]: row for row in candidates}
    left = by_label["left_detour"]
    right = by_label["right_detour"]
    valid_rows = [row for row in candidates if bool(row["valid"])]
    cbf_verified = [
        row
        for row in valid_rows
        if row["cbf_probe"] is not None and bool(row["cbf_probe"].get("verified_feasible"))
    ]
    return {
        "scene": scene,
        "obstacles": [
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in obstacle.items()
            }
            for obstacle in observation["obstacles"]
        ],
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid_rows),
        "cbf_verified_candidate_count": len(cbf_verified),
        "left_right_waypoints_distinct": not np.allclose(
            np.asarray(left["waypoints"], dtype=np.float64),
            np.asarray(right["waypoints"], dtype=np.float64),
        ),
        "left_route_valid": bool(left["valid"]),
        "right_route_valid": bool(right["valid"]),
        "candidates": candidates,
    }


def _acceptance(results: Mapping[str, Mapping[str, Any]], *, with_cbf: bool) -> dict[str, Any]:
    central = results["central_single"]
    left_blocked = results["left_blocked"]
    right_blocked = results["right_blocked"]
    gap = results["wall_single_gap"]
    checks = {
        "central_left_right_distinct": bool(central["left_right_waypoints_distinct"]),
        "central_left_valid": bool(central["left_route_valid"]),
        "central_right_valid": bool(central["right_route_valid"]),
        "left_blocked_rejects_left": not bool(left_blocked["left_route_valid"]),
        "left_blocked_keeps_right": bool(left_blocked["right_route_valid"]),
        "right_blocked_keeps_left": bool(right_blocked["left_route_valid"]),
        "right_blocked_rejects_right": not bool(right_blocked["right_route_valid"]),
        "wall_gap_nominal_valid": bool(
            next(row for row in gap["candidates"] if row["label"] == "nominal")["valid"]
        ),
        "every_scene_has_reachable_route": all(int(result["valid_candidate_count"]) > 0 for result in results.values()),
    }
    if with_cbf:
        checks["every_scene_has_cbf_verified_route"] = all(
            int(result["cbf_verified_candidate_count"]) > 0 for result in results.values()
        )
    return {"checks": checks, "passed": bool(all(checks.values()))}


def _markdown_report(report: Mapping[str, Any]) -> str:
    acceptance = report["acceptance"]
    lines = [
        "# WP1 Obstacle-Conditioned Route Audit",
        "",
        "Status: `development_only`; `locked_test_opened=false`.",
        "",
        "This audit checks the causal chain `public obstacle geometry -> distinct route proposals -> reachable option`.",
        "It does not train JEPA, change CBF margins, or execute an action.",
        "",
        "## Scene Summary",
        "",
        "| Scene | Valid routes | CBF verified routes | Left valid | Right valid | Distinct left/right |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scene in SCENE_NAMES:
        result = report["scenes"][scene]
        lines.append(
            f"| `{scene}` | {result['valid_candidate_count']} / {result['candidate_count']} | "
            f"{result['cbf_verified_candidate_count']} | {result['left_route_valid']} | "
            f"{result['right_route_valid']} | {result['left_right_waypoints_distinct']} |"
        )
    lines.extend(["", "## Acceptance Checks", ""])
    for name, passed in acceptance["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            f"Overall: **{'PASS' if acceptance['passed'] else 'FAIL'}**",
            "",
            "## Interpretation",
            "",
            "A PASS establishes that the route proposal layer reacts to public obstacle geometry and preserves a reachable alternative in the four WP1 micro-scenes.",
            "It is not evidence of safe-capture improvement and does not open a locked evaluation.",
            "",
            "## Provenance",
            "",
            f"- Git revision: `{report['provenance']['git_revision']}`",
            f"- Python: `{report['provenance']['python']}`; NumPy: `{report['provenance']['numpy']}`",
            f"- Route module SHA-256: `{report['provenance']['route_module_sha256']}`",
            f"- Audit script SHA-256: `{report['provenance']['audit_script_sha256']}`",
            f"- Environment config SHA-256: `{report['provenance']['config_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Environment config must contain a mapping.")
    route_config = ObstacleRouteConfig(
        dt_seconds=float(config["world"]["dt"]),
        max_speed_mps=float(config["agents"]["defender_max_speed"]),
        max_acceleration_mps2=float(config["agents"]["defender_max_acceleration"]),
        vehicle_radius_m=float(config["agents"]["drone_radius"]),
        obstacle_margin_m=float(config["task"]["pursuit"]["safety_margin"]),
        world_lower=(
            -float(config["world"]["half_extent_xy"]),
            -float(config["world"]["half_extent_xy"]),
            float(config["world"]["minimum_altitude"]),
        ),
        world_upper=(
            float(config["world"]["half_extent_xy"]),
            float(config["world"]["half_extent_xy"]),
            float(config["world"]["height"]),
        ),
    )
    cbf_probe = _make_cbf_probe(config) if args.with_cbf else None
    scenes = {
        scene: _scene_result(
            scene,
            config=config,
            route_config=route_config,
            cbf_probe=cbf_probe,
        )
        for scene in SCENE_NAMES
    }
    acceptance = _acceptance(scenes, with_cbf=args.with_cbf)
    report: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_wp1_obstacle_routes",
        "development_only": True,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "cbf_probe_requested": bool(args.with_cbf),
        "route_contract": route_config.contract(),
        "scenes": scenes,
        "acceptance": acceptance,
        "provenance": {
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "route_module_sha256": sha256(PROJECT_ROOT / "src" / "encirclement3d" / "obstacle_route_candidates.py"),
            "audit_script_sha256": sha256(Path(__file__).resolve()),
            "config_sha256": sha256(config_path),
            "command": " ".join(sys.argv),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "route_candidates.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "audit_type": report["audit_type"],
                "development_only": True,
                "locked_test_opened": False,
                "cbf_probe_requested": bool(args.with_cbf),
                "acceptance": acceptance,
                "scene_summary": {
                    scene: {
                        key: value
                        for key, value in result.items()
                        if key != "candidates" and key != "obstacles"
                    }
                    for scene, result in scenes.items()
                },
                "provenance": report["provenance"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "acceptance": acceptance}, indent=2), flush=True)
    if not acceptance["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
