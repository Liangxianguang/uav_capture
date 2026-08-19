"""Render a saved pursuit trajectory without importing PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    return parser.parse_args()


def draw_cylinder(axis, center_xy: np.ndarray, radius: float, height: float) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 32)
    z = np.linspace(0.0, height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    axis.plot_surface(
        center_xy[0] + radius * np.cos(theta_grid),
        center_xy[1] + radius * np.sin(theta_grid),
        z_grid,
        color="gray",
        alpha=0.25,
        linewidth=0,
    )


def main() -> None:
    args = parse_args()
    data = np.load(args.trajectory)
    defenders = np.asarray(data["defender_positions"])
    target = np.asarray(data["target_positions"])
    centers = np.asarray(data["obstacle_centers_xy"])
    radii = np.asarray(data["obstacle_radii"])
    heights = np.asarray(data["obstacle_heights"])
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    for index in range(defenders.shape[1]):
        axis.plot(
            defenders[:, index, 0],
            defenders[:, index, 1],
            defenders[:, index, 2],
            color=colors[index],
            label=f"defender {index + 1}",
        )
    axis.plot(target[:, 0], target[:, 1], target[:, 2], color="black", linewidth=2.0, label="target")
    capturing_defender = int(data["capturing_defender_id"])
    if capturing_defender >= 0:
        position = defenders[-1, capturing_defender]
        axis.scatter(
            position[0], position[1], position[2], color=colors[capturing_defender], s=60, marker="o", label="capturing defender"
        )
    for center, radius, height in zip(centers, radii, heights, strict=True):
        draw_cylinder(axis, center, float(radius), float(height))
    final_target = target[-1]
    capture_radius = float(data["capture_radius"])
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 12)
    axis.plot_surface(
        final_target[0] + capture_radius * np.outer(np.cos(u), np.sin(v)),
        final_target[1] + capture_radius * np.outer(np.sin(u), np.sin(v)),
        final_target[2] + capture_radius * np.outer(np.ones_like(u), np.cos(v)),
        color="gold",
        alpha=0.16,
        linewidth=0,
    )
    extent = float(data["world_half_extent"])
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(0.0, float(data["world_height"]))
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(args.title)
    axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
