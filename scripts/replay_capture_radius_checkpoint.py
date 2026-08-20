"""Replay one frozen capture checkpoint and render its full 3D episode.

The command evaluates one episode with the same local-information rollout used
by the formal experiments, stores every simulator frame as a compressed NPZ,
and renders a GIF. If FFmpeg is installed, an MP4 is written as well.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

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
    if frame_stride <= 0 or fps <= 0:
        raise ValueError("fps and frame-stride must be positive.")
    frame_indices = np.arange(0, len(target), frame_stride, dtype=np.int64)
    if frame_indices[-1] != len(target) - 1:
        frame_indices = np.append(frame_indices, len(target) - 1)

    colors = ((40, 120, 220), (240, 150, 35), (35, 170, 95), (220, 65, 65))
    width, height_px = 1000, 760
    extent = float(data["world_half_extent"])
    world_height = float(data["world_height"])
    scale = min(30.0, 0.44 * width / max(1.0, extent * 2.0))
    def project(point: np.ndarray) -> tuple[int, int]:
        x, y, z = (float(point[0]), float(point[1]), float(point[2]))
        return (int(width * 0.50 + scale * 0.82 * (x - y)), int(height_px * 0.60 + scale * (0.30 * (x + y) - 0.95 * z)))
    def draw_frame(index: int) -> Image.Image:
        image = Image.new("RGB", (width, height_px), (247, 249, 252))
        draw = ImageDraw.Draw(image, "RGBA")
        origin = project(np.array([0.0, 0.0, 0.0]))
        draw.line((origin[0] - 260, origin[1], origin[0] + 260, origin[1]), fill=(180, 190, 200, 150), width=1)
        draw.line((origin[0], origin[1] - 220, origin[0], origin[1] + 40), fill=(180, 190, 200, 150), width=1)
        for center, radius, obstacle_height in zip(centers, radii, heights, strict=True):
            base = project(np.array([center[0], center[1], 0.0]))
            top = project(np.array([center[0], center[1], obstacle_height]))
            rx, ry = max(6, int(scale * radius * 0.82)), max(3, int(scale * radius * 0.30))
            draw.polygon([(base[0]-rx, base[1]), (base[0]+rx, base[1]), (top[0]+rx, top[1]), (top[0]-rx, top[1])], fill=(100, 110, 125, 55), outline=(90, 100, 115, 170))
            draw.ellipse((top[0]-rx, top[1]-ry, top[0]+rx, top[1]+ry), fill=(110, 120, 135, 90), outline=(80, 90, 105, 180))
        for defender_index in range(defenders.shape[1]):
            pts = [project(p) for p in defenders[: index + 1, defender_index]]
            if len(pts) > 1: draw.line(pts, fill=colors[defender_index % len(colors)] + (190,), width=3)
        target_pts = [project(p) for p in target[: index + 1]]
        if len(target_pts) > 1: draw.line(target_pts, fill=(20, 20, 20, 210), width=4)
        target_xy = project(target[index])
        capture_px = max(8, int(scale * float(data["capture_radius"])))
        draw.ellipse((target_xy[0]-capture_px, target_xy[1]-max(4, capture_px//3), target_xy[0]+capture_px, target_xy[1]+max(4, capture_px//3)), outline=(235, 180, 20, 210), width=2)
        for defender_index, point in enumerate(defenders[index]):
            px, py = project(point); r = 7
            draw.ellipse((px-r, py-r, px+r, py+r), fill=colors[defender_index % len(colors)] + (255,), outline=(255,255,255,255), width=1)
        tx, ty = target_xy
        draw.line((tx-8, ty-8, tx+8, ty+8), fill=(15,15,15,255), width=3); draw.line((tx-8, ty+8, tx+8, ty-8), fill=(15,15,15,255), width=3)
        nearest = float(np.min(np.linalg.norm(defenders[index] - target[index], axis=1)))
        draw.text((24, 22), title, fill=(20, 30, 45, 255))
        draw.text((24, 52), f"t = {index * 0.1:.1f} s    nearest distance = {nearest:.2f} m    capture radius = {float(data['capture_radius']):.2f} m", fill=(30, 40, 55, 255))
        draw.text((width - 220, 24), "blue/orange/green/red: defenders\nblack X: target\ngold ellipse: capture radius", fill=(45, 55, 70, 255))
        return image
    base_name = output_dir / ("capture_cbf" if result.get("use_cbf") else "capture_raw")
    gif_path = base_name.with_suffix(".gif")
    frames = [draw_frame(int(index)) for index in frame_indices]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=max(1, int(1000 / fps)), loop=0, optimize=False)
    final_png_path = base_name.with_suffix(".png")
    frames[-1].save(final_png_path)
    media: dict[str, Any] = {
        "gif": str(gif_path),
        "png": str(final_png_path),
        "frames": int(len(frame_indices)),
        "fps": int(fps),
    }
    ffmpeg = shutil.which("ffmpeg")
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
