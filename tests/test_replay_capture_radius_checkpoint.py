from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.replay_capture_radius_checkpoint import METHOD_CONFIGS, find_ffmpeg, make_config, render_animation


def test_replay_renderer_has_f1_and_f2_configs() -> None:
    assert set(METHOD_CONFIGS) == {"f1", "f2"}
    f1, _ = make_config("f1", "delayed_measurements")
    f2, _ = make_config("f2", "delayed_measurements")
    assert f1["task"]["pursuit"]["include_uncertainty_features"] is False
    assert f2["task"]["pursuit"]["include_uncertainty_features"] is True


def test_replay_renderer_keeps_locked_condition_parameters() -> None:
    config, condition = make_config("f2", "burst_occlusion")
    assert condition["obstacle_count"] == 5
    assert config["experiments"][0]["obstacle_count"] == 5
    assert config["task"]["pursuit"]["detection_loss_burst_duration_steps"] == 5


def test_replay_renderer_exports_gif_and_final_png(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.npz"
    np.savez_compressed(
        trajectory,
        defender_positions=np.array(
            [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], [[0.2, 0.0, 1.0], [0.8, 0.0, 1.0]]], dtype=np.float64
        ),
        target_positions=np.array([[0.5, 0.0, 1.0], [0.5, 0.0, 1.0]], dtype=np.float64),
        obstacle_centers_xy=np.array([[2.0, 2.0]], dtype=np.float64),
        obstacle_radii=np.array([0.5], dtype=np.float64),
        obstacle_heights=np.array([3.0], dtype=np.float64),
        capture_radius=0.8,
        world_half_extent=5.0,
        world_height=5.0,
    )
    media = render_animation(
        trajectory,
        tmp_path,
        "test",
        fps=2,
        frame_stride=1,
        result={"use_cbf": True, "safe_capture_success": True},
    )
    assert media["simulation_frames"] == 2
    assert media["frames"] == 6
    assert media["capture_freeze_frames"] == 4
    assert media["capture_radius_m"] == 0.8
    assert media["final_frame_inside_capture_radius"] is True
    assert len(media["trajectory_sha256"]) == 64
    assert (tmp_path / "capture_cbf.gif").is_file()
    assert (tmp_path / "capture_cbf.png").is_file()
    assert (tmp_path / "capture_cbf_3d.png").is_file()


def test_replay_renderer_can_find_conda_ffmpeg() -> None:
    resolved = find_ffmpeg()
    assert resolved is None or Path(resolved).is_file()
