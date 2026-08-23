"""Render a dependency-light 3-D perspective animation from a saved trajectory.

The renderer deliberately uses Pillow rather than Matplotlib's 3-D backend.
This keeps PNG/GIF/FFmpeg MP4 generation stable in the Windows conda runtime
used by the project while preserving obstacle volumes and altitude cues.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_ffmpeg() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    for candidate in (Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe", Path(sys.prefix) / "bin" / "ffmpeg"):
        if candidate.is_file():
            return str(candidate)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--tail-length", type=int, default=0, help="0 keeps the complete trajectory tail.")
    parser.add_argument("--freeze-seconds", type=float, default=1.75)
    return parser.parse_args()


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


def _project(points: np.ndarray, extent: float, world_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Project world coordinates through a fixed elevated perspective camera."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    azimuth = np.deg2rad(-52.0)
    elevation = np.deg2rad(25.0)
    x_rot = np.cos(azimuth) * values[:, 0] - np.sin(azimuth) * values[:, 1]
    depth_axis = np.sin(azimuth) * values[:, 0] + np.cos(azimuth) * values[:, 1]
    y_rot = np.cos(elevation) * depth_axis - np.sin(elevation) * values[:, 2]
    depth = np.sin(elevation) * depth_axis + np.cos(elevation) * values[:, 2]
    camera_distance = 31.0
    scale = 26.0 * min(1.0, 10.0 / max(extent, 1e-6))
    perspective = camera_distance / np.maximum(6.0, camera_distance + depth)
    screen_x = 455.0 + x_rot * scale * perspective
    screen_y = 512.0 - y_rot * scale * perspective
    return np.column_stack((screen_x, screen_y)), depth


def _line(draw: ImageDraw.ImageDraw, points: np.ndarray, fill: tuple[int, int, int, int], width: int = 1) -> None:
    if len(points) >= 2:
        draw.line([tuple(map(float, point)) for point in points], fill=fill, width=width, joint="curve")


def _draw_grid(draw: ImageDraw.ImageDraw, extent: float, world_height: float) -> None:
    grid = (112, 136, 161, 75)
    for coordinate in np.linspace(-extent, extent, 9):
        projected, _ = _project(np.array([[coordinate, -extent, 0.0], [coordinate, extent, 0.0]]), extent, world_height)
        _line(draw, projected, grid)
        projected, _ = _project(np.array([[-extent, coordinate, 0.0], [extent, coordinate, 0.0]]), extent, world_height)
        _line(draw, projected, grid)
    outline = np.array(
        [
            [-extent, -extent, 0.0],
            [extent, -extent, 0.0],
            [extent, extent, 0.0],
            [-extent, extent, 0.0],
            [-extent, -extent, 0.0],
        ]
    )
    projected, _ = _project(outline, extent, world_height)
    _line(draw, projected, (176, 198, 218, 150), width=2)
    for corner in outline[:-1]:
        projected, _ = _project(np.vstack((corner, corner + np.array([0.0, 0.0, world_height]))), extent, world_height)
        _line(draw, projected, (122, 149, 176, 85), width=1)


def _draw_prism(
    draw: ImageDraw.ImageDraw,
    center: np.ndarray,
    half_extent: np.ndarray,
    height: float,
    extent: float,
    world_height: float,
    color: tuple[int, int, int],
) -> None:
    half_x, half_y = map(float, half_extent)
    x, y = map(float, center)
    ground = np.array([[x - half_x, y - half_y, 0.0], [x + half_x, y - half_y, 0.0], [x + half_x, y + half_y, 0.0], [x - half_x, y + half_y, 0.0]])
    top = ground + np.array([0.0, 0.0, height])
    vertices = np.vstack((ground, top))
    projected, depth = _project(vertices, extent, world_height)
    faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    for face in sorted(faces, key=lambda indices: float(np.mean(depth[list(indices)]))):
        polygon = [tuple(map(float, projected[index])) for index in face]
        shade = 0.72 + 0.20 * (face == (4, 5, 6, 7))
        fill = tuple(int(channel * shade) for channel in color) + (185,)
        draw.polygon(polygon, fill=fill, outline=(214, 228, 239, 195))


def _draw_cylinder(
    draw: ImageDraw.ImageDraw,
    center: np.ndarray,
    radius: float,
    height: float,
    extent: float,
    world_height: float,
    color: tuple[int, int, int],
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
    ring_xy = np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)))
    bottom = np.column_stack((ring_xy, np.zeros(len(theta))))
    top = bottom + np.array([0.0, 0.0, height])
    top_projected, _ = _project(top, extent, world_height)
    bottom_projected, _ = _project(bottom, extent, world_height)
    draw.polygon([tuple(map(float, point)) for point in top_projected], fill=color + (175,), outline=(218, 231, 241, 200))
    for index in range(0, len(theta), 4):
        _line(draw, np.vstack((bottom_projected[index], top_projected[index])), color + (165,), width=2)
    _line(draw, np.vstack((top_projected, top_projected[0])), (219, 232, 243, 220), width=2)


def _draw_obstacles(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    centers = data["centers"]
    radii = data["radii"]
    heights = data["heights"]
    shapes = data["shapes"]
    half_extents = data["half_extents"]
    extent = data["extent"]
    world_height = data["world_height"]
    colors = {"cylinder": (83, 111, 140), "box": (81, 105, 134), "wall": (106, 88, 125)}
    ordering: list[tuple[float, int]] = []
    for index, center in enumerate(centers):
        _, depth = _project(np.array([[center[0], center[1], 0.0]]), extent, world_height)
        ordering.append((float(depth[0]), index))
    for _, index in sorted(ordering):
        shape = str(shapes[index])
        color = colors.get(shape, (92, 114, 139))
        if shape == "cylinder":
            _draw_cylinder(draw, centers[index], float(radii[index]), float(heights[index]), extent, world_height, color)
        else:
            _draw_prism(draw, centers[index], half_extents[index], float(heights[index]), extent, world_height, color)


def _draw_frame(data: dict[str, Any], frame_index: int, tail_length: int, safe_capture: bool, final_frame: bool) -> Image.Image:
    image = Image.new("RGBA", (1280, 760), (12, 20, 32, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font, body_font, small_font = _font(24, True), _font(15), _font(12)
    draw.rectangle((0, 0, 1280, 78), fill=(18, 30, 47, 255))
    draw.line((0, 78, 1280, 78), fill=(84, 113, 143, 180), width=1)
    draw.text((34, 19), "3-D COOPERATIVE PURSUIT", font=title_font, fill=(237, 246, 255, 255))
    draw.text((36, 51), "PERSPECTIVE OBSTACLE-VOLUME REPLAY", font=small_font, fill=(151, 177, 205, 255))
    draw.rounded_rectangle((896, 19, 1246, 58), radius=6, fill=(27, 54, 64, 255), outline=(91, 235, 190, 220) if safe_capture and final_frame else (255, 213, 91, 210), width=1)
    status = "CAPTURE CONFIRMED" if safe_capture and final_frame else "PURSUIT ACTIVE"
    status_color = (91, 235, 190, 255) if safe_capture and final_frame else (255, 213, 91, 255)
    draw.text((918, 31), status, font=body_font, fill=status_color)

    _draw_grid(draw, data["extent"], data["world_height"])
    _draw_obstacles(draw, data)

    defenders = data["defenders"]
    target = data["target"]
    start = 0 if tail_length == 0 else max(0, frame_index - tail_length)
    colors = ((32, 205, 245), (255, 174, 54), (123, 231, 94), (192, 130, 255))
    for defender_index in range(defenders.shape[1]):
        projected, _ = _project(defenders[start : frame_index + 1, defender_index], data["extent"], data["world_height"])
        _line(draw, projected, colors[defender_index % len(colors)] + (215,), width=3)
    target_projected, _ = _project(target[start : frame_index + 1], data["extent"], data["world_height"])
    _line(draw, target_projected, (255, 74, 94, 235), width=4)

    target_position = target[frame_index]
    target_screen, _ = _project(target_position[None, :], data["extent"], data["world_height"])
    radius_points, _ = _project(
        np.array([target_position + [data["capture_radius"], 0.0, 0.0], target_position + [0.0, data["capture_radius"], 0.0]]),
        data["extent"],
        data["world_height"],
    )
    capture_px = max(12, int(np.mean(np.linalg.norm(radius_points - target_screen[0], axis=1))))
    capture_color = (91, 235, 190, 215) if safe_capture and final_frame else (255, 213, 91, 220)
    point = target_screen[0]
    draw.ellipse((point[0] - capture_px, point[1] - capture_px, point[0] + capture_px, point[1] + capture_px), outline=capture_color, width=3)
    draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=(255, 74, 94, 255), outline=(255, 236, 240, 255), width=2)

    for defender_index, position in enumerate(defenders[frame_index]):
        projected, _ = _project(position[None, :], data["extent"], data["world_height"])
        point = projected[0]
        color = colors[defender_index % len(colors)]
        draw.ellipse((point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9), fill=color + (255,), outline=(239, 249, 255, 255), width=2)
        draw.text((point[0] + 11, point[1] - 7), f"D{defender_index + 1}", font=small_font, fill=color + (255,))

    distances = np.linalg.norm(defenders[frame_index] - target_position[None, :], axis=1)
    draw.rounded_rectangle((34, 632, 715, 724), radius=7, fill=(21, 35, 53, 235), outline=(84, 113, 143, 180), width=1)
    draw.text((54, 650), f"t = {frame_index * 0.1:05.1f} s", font=body_font, fill=(237, 246, 255, 255))
    draw.text((54, 681), f"nearest defender = {float(np.min(distances)):.2f} m     capture radius = {data['capture_radius']:.2f} m", font=body_font, fill=(185, 209, 235, 255))
    draw.text((749, 690), "solid volumes = obstacles  |  colored paths = defenders  |  red path = target", font=small_font, fill=(164, 187, 212, 235))
    return image.convert("RGB")


def _load_scene(trajectory_path: Path) -> dict[str, Any]:
    raw = np.load(trajectory_path)
    centers = np.asarray(raw["obstacle_centers_xy"], dtype=np.float64)
    radii = np.asarray(raw["obstacle_radii"], dtype=np.float64)
    return {
        "defenders": np.asarray(raw["defender_positions"], dtype=np.float64),
        "target": np.asarray(raw["target_positions"], dtype=np.float64),
        "centers": centers,
        "radii": radii,
        "heights": np.asarray(raw["obstacle_heights"], dtype=np.float64),
        "shapes": np.asarray(raw["obstacle_shapes"]).astype(str) if "obstacle_shapes" in raw.files else np.full(len(centers), "cylinder", dtype="U16"),
        "half_extents": np.asarray(raw["obstacle_half_extents_xy"], dtype=np.float64) if "obstacle_half_extents_xy" in raw.files else np.repeat(radii[:, None], 2, axis=1),
        "extent": float(raw["world_half_extent"]),
        "world_height": float(raw["world_height"]),
        "capture_radius": float(raw["capture_radius"]),
    }


def render_static_perspective(trajectory_path: Path, output_path: Path, result: dict[str, Any]) -> None:
    """Write the final-frame perspective used by legacy replay output."""
    scene = _load_scene(trajectory_path)
    image = _draw_frame(scene, len(scene["target"]) - 1, 0, bool(result.get("safe_capture_success")), True)
    image.save(output_path)


def _write_mp4(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found.")
    png_stream = bytearray()
    for frame in frames:
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        png_stream.extend(buffer.getvalue())
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
        input=bytes(png_stream),
        check=True,
    )


def render_animation(trajectory_path: Path, result_path: Path, output_dir: Path, fps: int, frame_stride: int, tail_length: int, freeze_seconds: float) -> dict[str, Any]:
    if fps <= 0 or frame_stride <= 0 or tail_length < 0 or freeze_seconds < 0:
        raise ValueError("fps, frame-stride, tail-length and freeze-seconds must be non-negative as appropriate")
    scene = _load_scene(trajectory_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    safe_capture = bool(result.get("safe_capture_success", False))
    indices = np.arange(0, len(scene["target"]), frame_stride, dtype=np.int64)
    if indices[-1] != len(scene["target"]) - 1:
        indices = np.append(indices, len(scene["target"]) - 1)
    frames = [
        _draw_frame(scene, int(index), tail_length, safe_capture, int(index) == len(scene["target"]) - 1)
        for index in indices
    ]
    if safe_capture and freeze_seconds > 0:
        frames.extend([frames[-1].copy() for _ in range(max(1, int(round(freeze_seconds * fps))))])
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "capture_3d.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=max(1, int(1000 / fps)), loop=0, optimize=False)
    mp4_path = output_dir / "capture_3d.mp4"
    _write_mp4(frames, mp4_path, fps)
    final_png = output_dir / "capture_3d_final.png"
    frames[-1].save(final_png)
    distances = np.linalg.norm(scene["defenders"][-1] - scene["target"][-1][None, :], axis=1)
    return {
        "gif": str(gif_path),
        "mp4": str(mp4_path),
        "final_png": str(final_png),
        "frames": len(frames),
        "fps": fps,
        "capture_radius_m": scene["capture_radius"],
        "final_nearest_distance_m": float(np.min(distances)),
        "safe_capture_success": safe_capture,
    }


def main() -> None:
    args = parse_args()
    trajectory = args.trajectory.resolve()
    result = args.result.resolve()
    output_dir = args.output_dir.resolve()
    if not trajectory.is_file() or not result.is_file():
        raise FileNotFoundError("trajectory or result JSON does not exist")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    media = render_animation(trajectory, result, output_dir, args.fps, args.frame_stride, args.tail_length, args.freeze_seconds)
    (output_dir / "media.json").write_text(json.dumps(media, indent=2), encoding="utf-8")
    print(json.dumps(media, indent=2), flush=True)


if __name__ == "__main__":
    main()
