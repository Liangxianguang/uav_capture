from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_jepa_safe_capture_v2_archive import candidate_group_audit
from scripts.generate_jepa_safe_capture_v2_archive import pairwise_time_to_collision


def test_pairwise_ttc_only_marks_approaching_pairs_that_reach_the_barrier() -> None:
    positions = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 5.0, 1.0]])
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ttc = pairwise_time_to_collision(positions, velocities, radius=0.25, margin=0.35, clip_seconds=10.0)
    np.testing.assert_allclose(ttc[:2], [0.575, 0.575])
    assert ttc[2] == pytest.approx(10.0)


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
