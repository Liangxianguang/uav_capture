"""Replay one frozen capture checkpoint and render its full 3D episode.

The command evaluates one episode with the same local-information rollout used
by the formal experiments, stores every simulator frame as a compressed NPZ,
and renders a GIF. If FFmpeg is installed, an MP4 is written as well.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from evaluate_capture_radius_mappo import load_policy, rollout_episode, save_trajectory, select_device  # noqa: E402
from run_stage4c_formal import CONDITIONS  # noqa: E402


METHOD_CONFIGS = {
    "f1": PROJECT_ROOT / "configs" / "capture_radius_pursuit_time_aligned_belief_dev.yaml",
    "f2": PROJECT_ROOT / "configs" / "capture_radius_pursuit_time_aligned_uncertainty_dev.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_CONFIGS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-cbf", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--keep-every-frame", action="store_true")
    return parser.parse_args()


def make_config(method: str, condition_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(METHOD_CONFIGS[method].read_text(encoding="utf-8"))
    # F1 predates the explicit uncertainty flag; make the replay contract
    # self-describing while preserving the checkpoint's original input shape.
    config["task"]["pursuit"].setdefault("include_uncertainty_features", method == "f2")
    condition = copy.deepcopy(CONDITIONS[condition_name])
    config["task"]["pursuit"].update(copy.deepcopy(condition["pursuit"]))
    config["experiments"] = [
        {
            "name": condition_name,
            "episodes": 1,
            "obstacle_count": int(condition["obstacle_count"]),
            "target_speed_scale": float(condition["target_speed_scale"]),
        }
    ]
    return config, condition


def find_ffmpeg() -> str | None:
    """Find FFmpeg on PATH or in the active Conda environment."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    for candidate in (Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe", Path(sys.prefix) / "bin" / "ffmpeg"):
        if candidate.is_file():
            return str(candidate)
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if bundled.is_file():
            return str(bundled)
    except (ImportError, RuntimeError, OSError):
        pass
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_trajectory_perspective_3d(
    trajectory_path: Path,
    output_path: Path,
    title: str,
    result: dict[str, Any],
) -> None:
    """Render a fixed three-dimensional perspective from the saved raw trajectory."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    data = np.load(trajectory_path)
    defenders = np.asarray(data["defender_positions"], dtype=np.float64)
    target = np.asarray(data["target_positions"], dtype=np.float64)
    centers = np.asarray(data["obstacle_centers_xy"], dtype=np.float64)
    radii = np.asarray(data["obstacle_radii"], dtype=np.float64)
    heights = np.asarray(data["obstacle_heights"], dtype=np.float64)
    shapes = (
        np.asarray(data["obstacle_shapes"]).astype(str)
        if "obstacle_shapes" in data.files
        else np.full(len(centers), "cylinder", dtype="U16")
    )
    half_extents = (
        np.asarray(data["obstacle_half_extents_xy"], dtype=np.float64)
        if "obstacle_half_extents_xy" in data.files
        else np.repeat(radii[:, None], 2, axis=1)
    )
    extent = float(data["world_half_extent"])
    world_height = float(data["world_height"])
    capture_radius = float(data["capture_radius"])
    colors = ("#20cdf5", "#ffae36", "#7be75e", "#c082ff")

    figure = plt.figure(figsize=(11.5, 8.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    for center, radius, obstacle_height, shape, half_extent in zip(
        centers, radii, heights, shapes, half_extents, strict=True
    ):
        if shape == "cylinder":
            theta = np.linspace(0.0, 2.0 * np.pi, 32)
            z = np.linspace(0.0, float(obstacle_height), 8)
            theta_grid, z_grid = np.meshgrid(theta, z)
            axis.plot_surface(
                float(center[0]) + float(radius) * np.cos(theta_grid),
                float(center[1]) + float(radius) * np.sin(theta_grid),
                z_grid,
                color="#65758a",
                alpha=0.62,
                linewidth=0,
            )
        else:
            half_x, half_y = float(half_extent[0]), float(half_extent[1])
            axis.bar3d(
                float(center[0]) - half_x,
                float(center[1]) - half_y,
                0.0,
                2.0 * half_x,
                2.0 * half_y,
                float(obstacle_height),
                color="#718198" if shape == "box" else "#596b82",
                alpha=0.66,
                shade=True,
            )

    for defender_index in range(defenders.shape[1]):
        points = defenders[:, defender_index]
        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=colors[defender_index % len(colors)],
            linewidth=2.2,
            label=f"Pursuer {defender_index + 1}",
        )
        axis.scatter(*points[-1], color=colors[defender_index % len(colors)], s=38)
    axis.plot(target[:, 0], target[:, 1], target[:, 2], color="#ff4a5e", linewidth=2.8, label="Target")
    axis.scatter(*target[-1], color="#ff4a5e", s=52)

    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 16)
    sphere_x = target[-1, 0] + capture_radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = target[-1, 1] + capture_radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = target[-1, 2] + capture_radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        color="#5bebbe" if result.get("safe_capture_success") else "#ffd45b",
        alpha=0.55,
        linewidth=0.6,
    )

    axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(0.0, world_height))
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(
        f"{title}\n"
        + (
            f"SAFE CAPTURE CONFIRMED (r = {capture_radius:.2f} m)"
            if result.get("safe_capture_success")
            else "FAILURE TRAJECTORY"
        ),
        pad=18,
    )
    axis.view_init(elev=25, azim=-58)
    axis.set_box_aspect((2.0, 2.0, 1.0))
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(True, alpha=0.25)
    figure.savefig(output_path, dpi=170, facecolor="white")
    plt.close(figure)


def render_animation(
    trajectory_path: Path,
    output_dir: Path,
    title: str,
    fps: int,
    frame_stride: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    data = np.load(trajectory_path)
    defenders = np.asarray(data["defender_positions"], dtype=np.float64)
    target = np.asarray(data["target_positions"], dtype=np.float64)
    centers = np.asarray(data["obstacle_centers_xy"], dtype=np.float64)
    radii = np.asarray(data["obstacle_radii"], dtype=np.float64)
    heights = np.asarray(data["obstacle_heights"], dtype=np.float64)
    shapes = (
        np.asarray(data["obstacle_shapes"]).astype(str)
        if "obstacle_shapes" in data.files
        else np.full(len(centers), "cylinder", dtype="U16")
    )
    half_extents = (
        np.asarray(data["obstacle_half_extents_xy"], dtype=np.float64)
        if "obstacle_half_extents_xy" in data.files
        else np.repeat(radii[:, None], 2, axis=1)
    )
    if frame_stride <= 0 or fps <= 0:
        raise ValueError("fps and frame-stride must be positive.")
    frame_indices = np.arange(0, len(target), frame_stride, dtype=np.int64)
    if frame_indices[-1] != len(target) - 1:
        frame_indices = np.append(frame_indices, len(target) - 1)

    colors = ((32, 205, 245), (255, 174, 54), (123, 231, 94), (192, 130, 255))
    target_color = (255, 74, 94)
    width, height_px = 1280, 760
    extent = float(data["world_half_extent"])
    world_height = float(data["world_height"])
    capture_radius = float(data["capture_radius"])
    nearest_distances = np.min(np.linalg.norm(defenders - target[:, None, :], axis=2), axis=1)
    min_clearance = result.get("min_clearance_m")
    map_box = (36, 102, 704, 696)
    panel_box = (730, 102, 1244, 696)
    map_width = map_box[2] - map_box[0]
    map_height = map_box[3] - map_box[1]
    map_scale = min(map_width, map_height) / max(1.0, 2.0 * extent)

    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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

    f_title, f_subtitle = font(25, True), font(13)
    f_section, f_metric, f_small = font(15, True), font(18, True), font(12)

    def world_to_map(point: np.ndarray) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        px = map_box[0] + (x + extent) / (2.0 * extent) * map_width
        py = map_box[3] - (y + extent) / (2.0 * extent) * map_height
        return int(px), int(py)

    def draw_frame(index: int) -> Image.Image:
        image = Image.new("RGBA", (width, height_px), (11, 17, 27, 255))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, width, 74), fill=(16, 25, 39, 255))
        draw.line((0, 74, width, 74), fill=(54, 73, 96, 210), width=1)
        draw.text((34, 17), "3D COOPERATIVE PURSUIT", font=f_title, fill=(239, 246, 255, 255))
        draw.text((36, 50), "PARTIALLY OBSERVABLE  /  LOCKED CHECKPOINT REPLAY", font=f_subtitle, fill=(139, 164, 193, 255))
        draw.text((width - 350, 25), title.upper(), font=f_section, fill=(113, 222, 255, 255))

        draw.rounded_rectangle(map_box, radius=8, fill=(20, 30, 44, 255), outline=(68, 92, 119, 255), width=1)
        draw.rounded_rectangle(panel_box, radius=8, fill=(20, 30, 44, 255), outline=(68, 92, 119, 255), width=1)
        grid_color = (101, 125, 151, 90)
        for coordinate in np.linspace(-extent, extent, 9):
            a = world_to_map(np.array([coordinate, -extent, 0.0]))
            b = world_to_map(np.array([coordinate, extent, 0.0]))
            c = world_to_map(np.array([-extent, coordinate, 0.0]))
            d = world_to_map(np.array([extent, coordinate, 0.0]))
            draw.line((a[0], a[1], b[0], b[1]), fill=grid_color, width=1)
            draw.line((c[0], c[1], d[0], d[1]), fill=grid_color, width=1)
        origin = world_to_map(np.zeros(3))
        draw.line((map_box[0], origin[1], map_box[2], origin[1]), fill=(187, 210, 235, 145), width=1)
        draw.line((origin[0], map_box[1], origin[0], map_box[3]), fill=(187, 210, 235, 145), width=1)
        draw.text((map_box[2] - 20, origin[1] + 7), "x", font=f_small, fill=(185, 210, 237, 220))
        draw.text((origin[0] + 8, map_box[1] + 5), "y", font=f_small, fill=(185, 210, 237, 220))

        for center, radius, obstacle_height, shape, half_extent in zip(
            centers, radii, heights, shapes, half_extents, strict=True
        ):
            px, py = world_to_map(np.array([center[0], center[1], 0.0]))
            if shape == "cylinder":
                half_x, half_y = float(radius), float(radius)
            else:
                half_x, half_y = float(half_extent[0]), float(half_extent[1])
            half_px_x = max(8, int(half_x * map_scale))
            half_px_y = max(6, int(half_y * map_scale))
            shadow = (px - half_px_x + 7, py - half_px_y + 8, px + half_px_x + 7, py + half_px_y + 8)
            draw.rectangle(shadow, fill=(35, 47, 62, 255))
            if shape == "cylinder":
                draw.ellipse(
                    (px - half_px_x, py - half_px_y, px + half_px_x, py + half_px_y),
                    fill=(75, 92, 114, 255),
                    outline=(166, 183, 205, 255),
                    width=2,
                )
                draw.ellipse(
                    (px - half_px_x + 4, py - half_px_y + 4, px + half_px_x - 4, py + half_px_y - 4),
                    outline=(132, 155, 181, 220),
                    width=1,
                )
            else:
                draw.rectangle(
                    (px - half_px_x, py - half_px_y, px + half_px_x, py + half_px_y),
                    fill=(91, 108, 130, 255),
                    outline=(190, 205, 222, 255),
                    width=2,
                )
                draw.line(
                    (px - half_px_x, py - half_px_y, px + half_px_x, py - half_px_y),
                    fill=(222, 231, 242, 230),
                    width=2,
                )
            draw.text(
                (px + half_px_x + 5, py - 12),
                f"{shape} h={float(obstacle_height):.1f}",
                font=f_small,
                fill=(170, 190, 215, 220),
            )

        for defender_index in range(defenders.shape[1]):
            points = [world_to_map(point) for point in defenders[: index + 1, defender_index]]
            if len(points) > 1:
                draw.line(points, fill=colors[defender_index % len(colors)] + (150,), width=3)
        target_points = [world_to_map(point) for point in target[: index + 1]]
        if len(target_points) > 1:
            draw.line(target_points, fill=target_color + (205,), width=4)

        target_xy = world_to_map(target[index])
        capture_px = max(10, int(capture_radius * map_scale))
        ring_color = (91, 235, 190) if result.get("safe_capture_success") and index == len(target) - 1 else (255, 212, 91)
        draw.ellipse((target_xy[0] - capture_px, target_xy[1] - capture_px, target_xy[0] + capture_px, target_xy[1] + capture_px), outline=ring_color + (240,), width=3)
        # This is a display-only pulse. The solid ring above remains the fixed
        # physical capture radius used by the environment success predicate.
        pulse_radius = capture_px + 6 + int(4.0 * (0.5 + 0.5 * np.sin(index * 0.55)))
        draw.ellipse(
            (target_xy[0] - pulse_radius, target_xy[1] - pulse_radius, target_xy[0] + pulse_radius, target_xy[1] + pulse_radius),
            outline=ring_color + (95,),
            width=1,
        )
        draw.ellipse((target_xy[0] - 16, target_xy[1] - 16, target_xy[0] + 16, target_xy[1] + 16), fill=(255, 255, 255, 28))
        draw.ellipse((target_xy[0] - 8, target_xy[1] - 8, target_xy[0] + 8, target_xy[1] + 8), fill=target_color + (255,), outline=(255, 235, 240, 255), width=2)
        draw.line((target_xy[0] - 13, target_xy[1], target_xy[0] + 13, target_xy[1]), fill=(255, 255, 255, 230), width=1)
        draw.line((target_xy[0], target_xy[1] - 13, target_xy[0], target_xy[1] + 13), fill=(255, 255, 255, 230), width=1)
        draw.text((target_xy[0] + 13, target_xy[1] - 24), "TARGET", font=f_small, fill=(255, 157, 169, 255))

        for defender_index, point in enumerate(defenders[index]):
            px, py = world_to_map(point)
            color = colors[defender_index % len(colors)]
            draw.ellipse((px - 13, py - 13, px + 13, py + 13), fill=color + (45,))
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=color + (255,), outline=(232, 248, 255, 255), width=1)
            if index > 0:
                velocity = defenders[index, defender_index, :2] - defenders[index - 1, defender_index, :2]
                velocity_norm = float(np.linalg.norm(velocity))
                if velocity_norm > 1e-6:
                    direction = velocity / velocity_norm
                    tip = (int(px + direction[0] * 18), int(py - direction[1] * 18))
                    left = (int(px - direction[1] * 6), int(py - direction[0] * 6))
                    right = (int(px + direction[1] * 6), int(py + direction[0] * 6))
                    draw.polygon((tip, left, right), fill=color + (235,))
            draw.text((px + 12, py + 8), f"D{defender_index + 1}", font=f_small, fill=color + (255,))

        nearest = float(nearest_distances[index])
        time_seconds = index * 0.1
        completed = index == len(target) - 1
        if completed and bool(result.get("safe_capture_success")):
            status, status_color = "SAFE CAPTURE CONFIRMED", (91, 235, 190)
        elif completed and bool(result.get("collision")):
            status, status_color = "SAFETY TERMINATED", (255, 108, 120)
        else:
            status, status_color = "PURSUIT ACTIVE", (255, 212, 91)

        draw.text((758, 126), "RUN STATUS", font=f_section, fill=(157, 181, 208, 255))
        draw.rounded_rectangle((756, 153, 1218, 205), radius=7, fill=(30, 51, 61, 255), outline=status_color + (220,), width=1)
        draw.text((778, 168), status, font=f_metric, fill=status_color + (255,))
        draw.text((758, 226), f"t = {time_seconds:05.1f} s", font=f_metric, fill=(240, 247, 255, 255))
        draw.text((758, 257), f"nearest distance     {nearest:5.2f} m", font=f_small, fill=(191, 210, 233, 255))
        draw.text((758, 280), f"capture radius       {capture_radius:5.2f} m", font=f_small, fill=(191, 210, 233, 255))
        draw.text((758, 303), f"minimum clearance    {float(min_clearance):5.2f} m" if min_clearance is not None else "minimum clearance    n/a", font=f_small, fill=(191, 210, 233, 255))
        draw.line((758, 332, 1218, 332), fill=(70, 94, 120, 180), width=1)
        draw.text((758, 350), "SCENE", font=f_section, fill=(157, 181, 208, 255))
        draw.text((758, 378), f"defenders             {defenders.shape[1]}", font=f_small, fill=(191, 210, 233, 255))
        draw.text((758, 401), f"obstacles              {len(centers)}", font=f_small, fill=(191, 210, 233, 255))
        draw.text((758, 424), f"world                  {2 * extent:.0f} x {2 * extent:.0f} x {world_height:.0f} m", font=f_small, fill=(191, 210, 233, 255))

        chart = (758, 470, 1218, 635)
        draw.text((758, 448), "ALTITUDE PROFILE", font=f_section, fill=(157, 181, 208, 255))
        draw.rectangle(chart, fill=(15, 23, 35, 230), outline=(70, 94, 120, 180), width=1)
        chart_w, chart_h = chart[2] - chart[0], chart[3] - chart[1]
        for z_value in np.linspace(0.0, world_height, 5):
            y = chart[3] - int(z_value / max(world_height, 1e-6) * chart_h)
            draw.line((chart[0], y, chart[2], y), fill=(82, 105, 130, 75), width=1)
            draw.text((chart[0] + 5, y - 14), f"{z_value:.0f}", font=f_small, fill=(131, 155, 180, 190))
        for defender_index in range(defenders.shape[1]):
            altitude_points = []
            for sample_index, z in enumerate(defenders[: index + 1, defender_index, 2]):
                x = chart[0] + int(sample_index / max(1, len(target) - 1) * chart_w)
                y = chart[3] - int(float(z) / max(world_height, 1e-6) * chart_h)
                altitude_points.append((x, y))
            if len(altitude_points) > 1:
                draw.line(altitude_points, fill=colors[defender_index % len(colors)] + (210,), width=2)
        target_altitudes = []
        for sample_index, z in enumerate(target[: index + 1, 2]):
            x = chart[0] + int(sample_index / max(1, len(target) - 1) * chart_w)
            y = chart[3] - int(float(z) / max(world_height, 1e-6) * chart_h)
            target_altitudes.append((x, y))
        if len(target_altitudes) > 1:
            draw.line(target_altitudes, fill=target_color + (230,), width=2)
        draw.text((758, 652), "top-down x-y map  |  altitude encoded at right", font=f_small, fill=(118, 145, 173, 220))
        return image.convert("RGB")
    base_name = output_dir / ("capture_cbf" if result.get("use_cbf") else "capture_raw")
    gif_path = base_name.with_suffix(".gif")
    frames = [draw_frame(int(index)) for index in frame_indices]
    capture_freeze_frames = 0
    if bool(result.get("safe_capture_success")):
        # Keep the capture confirmation readable after the final motion frame.
        capture_freeze_frames = max(1, int(round(1.75 * fps)))
        frames.extend([frames[-1].copy() for _ in range(capture_freeze_frames)])
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=max(1, int(1000 / fps)), loop=0, optimize=False)
    final_png_path = base_name.with_suffix(".png")
    frames[-1].save(final_png_path)
    perspective_path = output_dir / f"{base_name.name}_3d.png"
    render_trajectory_perspective_3d(trajectory_path, perspective_path, title, result)
    final_nearest_distance = float(nearest_distances[-1])
    media: dict[str, Any] = {
        "gif": str(gif_path),
        "png": str(final_png_path),
        "perspective_3d_png": str(perspective_path),
        "simulation_frames": int(len(frame_indices)),
        "capture_freeze_frames": int(capture_freeze_frames),
        "frames": int(len(frames)),
        "fps": int(fps),
        "trajectory_sha256": _file_sha256(trajectory_path),
        "trajectory_samples": int(len(target)),
        "capture_radius_m": capture_radius,
        "final_nearest_distance_m": final_nearest_distance,
        "final_frame_inside_capture_radius": final_nearest_distance <= capture_radius,
    }
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        mp4_path = base_name.with_suffix(".mp4")
        try:
            png_stream = bytearray()
            for frame in frames:
                buffer = io.BytesIO()
                frame.save(buffer, format="PNG")
                png_stream.extend(buffer.getvalue())
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(mp4_path),
                ],
                input=bytes(png_stream),
                check=True,
            )
            media["mp4"] = str(mp4_path)
        except Exception as error:  # pragma: no cover - depends on host codecs
            media["mp4_error"] = str(error)
    else:
        media["mp4_error"] = "ffmpeg executable not found; GIF was generated."
    return media


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.frame_stride <= 0:
        raise ValueError("fps and frame-stride must be positive.")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config, condition = make_config(args.method, args.condition)
    device = select_device(args.device)
    experiment = config["experiments"][0]
    prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(experiment["obstacle_count"]),
        target_speed_scale=float(experiment["target_speed_scale"]),
    )
    policy, action_scale, checkpoint_metadata = load_policy(
        checkpoint,
        prototype,
        prototype.reset(seed=args.seed),
        device,
    )
    row, env = rollout_episode(
        policy,
        config,
        obstacle_count=int(condition["obstacle_count"]),
        target_speed_scale=float(condition["target_speed_scale"]),
        seed=args.seed,
        device=device,
        action_scale=action_scale,
        use_cbf=args.use_cbf,
        record_history=True,
    )
    row.update(
        {
            "method": args.method,
            "condition": args.condition,
            "use_cbf": bool(args.use_cbf),
            "checkpoint": str(checkpoint),
            "device": str(device),
        }
    )
    trajectory_path = output_dir / "trajectory.npz"
    save_trajectory(env, trajectory_path)
    (output_dir / "episode.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    media = render_animation(
        trajectory_path,
        output_dir,
        title=f"{args.method} / {args.condition} / {'CBF' if args.use_cbf else 'raw'}",
        fps=args.fps,
        frame_stride=args.frame_stride,
        result=row,
    )
    row["media"] = media
    (output_dir / "episode.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({"episode": row, "media": media}, indent=2), flush=True)


if __name__ == "__main__":
    main()
