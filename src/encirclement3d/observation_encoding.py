"""Optional policy-observation extensions for shape-aware obstacle tasks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .pursuit_env import CaptureRadiusPursuit3DEnv


SHAPE_AWARE_OBSTACLE_GEOMETRY = "shape_extents_and_type"


def policy_observations(
    env: CaptureRadiusPursuit3DEnv,
    observation: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return legacy features or append obstacle geometry requested by the task.

    The legacy encoder remains unchanged so historical checkpoints retain their
    input contract. V4 appends horizontal half-extents and a one-hot
    cylinder/box/wall type to every obstacle description.
    """
    current = env.observe() if observation is None else observation
    base = env.policy_observations(current)
    mode = str(env.task.get("policy_obstacle_geometry", "legacy"))
    if mode == "legacy":
        return base
    if mode != SHAPE_AWARE_OBSTACLE_GEOMETRY:
        raise ValueError(
            "task.policy_obstacle_geometry must be 'legacy' or "
            f"'{SHAPE_AWARE_OBSTACLE_GEOMETRY}'."
        )

    positions = np.asarray(current["defender_positions"], dtype=np.float32)
    obstacles = list(current["obstacles"])
    extent = float(env.world["half_extent_xy"])
    max_obstacles = int(env.pursuit["max_observation_obstacles"])
    geometry_rows: list[np.ndarray] = []
    for position in positions:
        nearest = sorted(
            obstacles,
            key=lambda obstacle: float(
                np.linalg.norm(np.asarray(obstacle["center_xy"], dtype=np.float32) - position[:2])
                - float(obstacle["radius"])
            ),
        )[:max_obstacles]
        geometry = np.zeros((max_obstacles, 5), dtype=np.float32)
        for index, obstacle in enumerate(nearest):
            shape = str(obstacle["shape"])
            if shape not in {"cylinder", "box", "wall"}:
                raise ValueError(f"Unsupported obstacle shape in policy observation: {shape}")
            half_extents = obstacle.get("half_extents_xy")
            if half_extents is None:
                half_x = half_y = float(obstacle["radius"])
            else:
                half_x, half_y = map(float, half_extents)
            geometry[index] = np.array(
                [
                    half_x / extent,
                    half_y / extent,
                    float(shape == "cylinder"),
                    float(shape == "box"),
                    float(shape == "wall"),
                ],
                dtype=np.float32,
            )
        geometry_rows.append(geometry.reshape(-1))
    values = np.concatenate([base, np.stack(geometry_rows)], axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Shape-aware policy observation emitted a non-finite value.")
    return values
