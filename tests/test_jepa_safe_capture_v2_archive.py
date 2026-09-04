from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.audit_jepa_safe_capture_v2_archive import candidate_group_audit
from scripts.generate_jepa_safe_capture_v2_archive import (
    _validate_contract,
    _roll_counterfactual_v2,
    episode_seed,
    pairwise_time_to_collision,
)
from encirclement3d.pursuit_controllers import DynamicEncirclementController, PursuitCBFSafetyFilter
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


def test_pairwise_ttc_only_marks_approaching_pairs_that_reach_the_barrier() -> None:
    positions = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 5.0, 1.0]])
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ttc = pairwise_time_to_collision(positions, velocities, radius=0.25, margin=0.35, clip_seconds=10.0)
    np.testing.assert_allclose(ttc[:2], [0.575, 0.575])
    assert ttc[2] == pytest.approx(10.0)


def test_episode_seed_offset_creates_disjoint_reproducible_block() -> None:
    collection = {"seed_blocks": {"train": 100, "validation": 200, "calibration": 300}}
    assert episode_seed(collection, "calibration", 2, 7) == 20307
    assert episode_seed(collection, "calibration", 2, 7, seed_offset=1_000_000) == 1_020_307


def test_candidate_group_audit_groups_by_explicit_keys_not_npz_row_order() -> None:
    arrays = {
        "episode_seed": np.array([1, 1, 1, 2, 2, 2]),
        "time_index": np.array([3, 3, 3, 4, 4, 4]),
        "agent_id": np.array([0, 0, 0, 1, 1, 1]),
        "candidate_index": np.array([2, 0, 1, 1, 2, 0]),
        "candidate_is_nominal": np.array([0, 1, 0, 0, 0, 1], dtype=np.float32),
    }
    report = candidate_group_audit(arrays, candidate_count=3)
    assert report["all_groups_have_expected_candidate_count"] is True
    assert report["invalid_group_count"] == 0


def test_candidate_group_audit_rejects_missing_candidate() -> None:
    arrays = {
        "episode_seed": np.array([1, 1, 1]),
        "time_index": np.array([3, 3, 3]),
        "agent_id": np.array([0, 0, 0]),
        "candidate_index": np.array([0, 1, 1]),
        "candidate_is_nominal": np.array([1, 0, 0], dtype=np.float32),
    }
    report = candidate_group_audit(arrays, candidate_count=3)
    assert report["all_groups_have_expected_candidate_count"] is False
    assert report["invalid_group_count"] == 1


def test_counterfactual_target_label_uses_post_action_defender_position() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").read_text(encoding="utf-8")
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.45)
    observation = env.reset(seed=20260911)
    controller = DynamicEncirclementController(env)
    chunk = np.zeros((3, env.n_defenders, 3), dtype=np.float32)
    labels = _roll_counterfactual_v2(
        env,
        controller,
        observation,
        chunk,
        horizon_steps=[1],
        chunk_length_steps=3,
        clip_m=10.0,
        extent=10.0,
        ttc_clip_seconds=10.0,
        cbf_max_correction_norm_mps=5.0,
    )

    clone = copy.deepcopy(env)
    clone_observation = copy.deepcopy(observation)
    executed, _diagnostics = PursuitCBFSafetyFilter(clone).filter(chunk[0], clone_observation)
    clone.step(executed)
    expected = (clone.target_position[None, :] - clone.defender_positions) / 10.0
    np.testing.assert_allclose(labels["target_relative"][:, 0], expected, atol=1e-7)


def test_v11_collection_and_protocol_form_a_corrected_frame_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    collection = yaml.safe_load(
        (root / "configs" / "jepa_safe_capture_v2_corrected_frame_v11_collection.yaml").read_text(encoding="utf-8")
    )
    protocol = yaml.safe_load(
        (root / "configs" / "jepa_safe_capture_v2_corrected_frame_v11_protocol.yaml").read_text(encoding="utf-8")
    )
    _validate_contract(protocol, collection, "calibration")
    assert collection["archive_contract"]["dataset_version"] == "jepa_safe_capture_v2_p1_corrected_frame"
    assert protocol["archive_contract"]["target_relative_frame"] == "post_action_defender_position"
    assert protocol["world"]["half_extent_xy_m"] == pytest.approx(10.0)
