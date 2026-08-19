"""Rendering and metrics reporting for reproducible benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .environment import CylinderObstacle, Encirclement3DEnv


def plot_trajectory(env: Encirclement3DEnv, path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    if not env.history:
        raise ValueError("Trajectory history is empty.")

    defender_traces = np.array([frame["defender_positions"] for frame in env.history])
    target_trace = np.array([frame["target_position"] for frame in env.history])
    for index in range(defender_traces.shape[1]):
        axis.plot(
            defender_traces[:, index, 0],
            defender_traces[:, index, 1],
            defender_traces[:, index, 2],
            color=colors[index],
            label=f"defender {index + 1}",
        )
    axis.plot(target_trace[:, 0], target_trace[:, 1], target_trace[:, 2], color="black", linewidth=2.0, label="target")

    for obstacle in env.obstacles:
        _draw_cylinder(axis, obstacle)

    final_slots = env.history[-1]["slot_positions"]
    axis.scatter(final_slots[:, 0], final_slots[:, 1], final_slots[:, 2], marker="x", s=70, color="purple", label="final slots")
    extent = float(env.world["half_extent_xy"])
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(0.0, float(env.world["height"]))
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(title)
    axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary(rows: list[dict[str, Any]], path: Path, success_label: str = "containment success rate") -> None:
    scenario_names = sorted({str(row["scenario"]) for row in rows})
    success = [np.mean([row["success"] for row in rows if row["scenario"] == name]) for name in scenario_names]
    collision = [
        np.mean([int(row["collision_steps"]) > 0 for row in rows if row["scenario"] == name])
        for name in scenario_names
    ]
    clearance = [np.mean([row["min_clearance"] for row in rows if row["scenario"] == name]) for name in scenario_names]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(scenario_names, success, color="tab:blue")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel(success_label)
    axes[0].set_title("Controller baseline")
    axes[1].bar(scenario_names, collision, color="tab:red")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("collision episode rate")
    axes[1].set_title("Safety failures")
    axes[2].bar(scenario_names, clearance, color="tab:green")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("mean minimum clearance (m)")
    axes[2].set_title("Safety margin")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pursuit_trajectory(env: Any, path: Path, title: str) -> None:
    """Render a truthful 3D trajectory for the capture-radius task."""
    if not env.history:
        raise ValueError("Trajectory history is empty.")
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    defender_traces = np.asarray([frame["defender_positions"] for frame in env.history])
    target_trace = np.asarray([frame["target_position"] for frame in env.history])
    for index in range(defender_traces.shape[1]):
        axis.plot(
            defender_traces[:, index, 0],
            defender_traces[:, index, 1],
            defender_traces[:, index, 2],
            color=colors[index],
            label=f"defender {index + 1}",
        )
    axis.plot(target_trace[:, 0], target_trace[:, 1], target_trace[:, 2], color="black", linewidth=2.0, label="target")
    if getattr(env, "capturing_defender_id", None) is not None:
        index = int(env.capturing_defender_id)
        final = defender_traces[-1, index]
        axis.scatter(final[0], final[1], final[2], color=colors[index], s=60, marker="o", label="capturing defender")

    for obstacle in env.obstacles:
        _draw_cylinder(axis, obstacle)

    final_target = target_trace[-1]
    radius = float(env.pursuit["capture_radius"])
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 12)
    x = final_target[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = final_target[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = final_target[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_surface(x, y, z, color="gold", alpha=0.16, linewidth=0)

    extent = float(env.world["half_extent_xy"])
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(0.0, float(env.world["height"]))
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(title)
    axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_cylinder(axis: Any, obstacle: CylinderObstacle) -> None:
    if getattr(obstacle, "shape", "cylinder") != "cylinder":
        half_extents = getattr(obstacle, "half_extents_xy", None)
        if half_extents is None:
            return
        x0, x1 = obstacle.center_xy[0] - half_extents[0], obstacle.center_xy[0] + half_extents[0]
        y0, y1 = obstacle.center_xy[1] - half_extents[1], obstacle.center_xy[1] + half_extents[1]
        vertices = np.array(
            [
                [x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0],
                [x0, y0, obstacle.height], [x1, y0, obstacle.height],
                [x1, y1, obstacle.height], [x0, y1, obstacle.height],
            ]
        )
        faces = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
        polygons = [vertices[list(face)] for face in faces]
        axis.add_collection3d(Poly3DCollection(polygons, facecolor="gray", alpha=0.25, edgecolor="none"))
        return
    theta = np.linspace(0.0, 2.0 * np.pi, 32)
    z = np.linspace(0.0, obstacle.height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = obstacle.center_xy[0] + obstacle.radius * np.cos(theta_grid)
    y_grid = obstacle.center_xy[1] + obstacle.radius * np.sin(theta_grid)
    axis.plot_surface(x_grid, y_grid, z_grid, color="gray", alpha=0.25, linewidth=0)
