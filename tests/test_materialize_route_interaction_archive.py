from __future__ import annotations

import json

import numpy as np

from scripts.materialize_route_interaction_archive import materialize


def test_materialize_derives_other_defender_relative_action_and_preserves_source(tmp_path):
    source_dataset = tmp_path / "source.npz"
    source_metadata = tmp_path / "source_metadata.json"
    output_dir = tmp_path / "derived"
    tensorboard_dir = tmp_path / "tb"
    actions = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        source_dataset,
        route_action_chunk=actions,
        inputs=np.zeros((4, 8, 63), dtype=np.float32),
        scenario_index=np.zeros(4, dtype=np.int64),
        time_index=np.zeros(4, dtype=np.int64),
        route_candidate_index=np.zeros(4, dtype=np.int64),
        sample_type=np.zeros(4, dtype=np.float32),
        route_geometry_valid=np.ones(4, dtype=np.float32),
        labels_cbf_feasible=np.ones((4, 5), dtype=np.float32),
        earliest_failure_step=np.full(4, 5, dtype=np.int64),
        labels_boundary_ttc=np.full((4, 5), 10.0, dtype=np.float32),
        labels_pairwise_ttc=np.full((4, 5), 10.0, dtype=np.float32),
        labels_acceleration_slack=np.zeros((4, 5), dtype=np.float32),
        labels_boundary_clearance=np.ones((4, 5), dtype=np.float32),
    )
    source_metadata.write_text(
        json.dumps(
            {
                "development_only": True,
                "locked_test_opened": False,
                "route_labels": [],
                "route_counts": {},
            }
        ),
        encoding="utf-8",
    )

    result = materialize(source_dataset, source_metadata, output_dir, tensorboard_dir, "interaction_v1")

    assert result["sample_count"] == 4
    with np.load(output_dir / "route_identity_counterfactual.npz") as archive:
        relative = archive["route_relative_action_chunk"][:, 0, 0]
    np.testing.assert_allclose(relative, [-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0], atol=1e-6)
    output_metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert output_metadata["interaction_action_conditioned_route_chunk"] is True
    assert source_dataset.exists()


def test_materialize_pairwise_derives_nine_action_features(tmp_path):
    source_dataset = tmp_path / "source.npz"
    source_metadata = tmp_path / "source_metadata.json"
    output_dir = tmp_path / "derived_pairwise"
    tensorboard_dir = tmp_path / "tb_pairwise"
    actions = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        source_dataset,
        route_action_chunk=actions,
        scenario_index=np.zeros(4, dtype=np.int64),
        time_index=np.zeros(4, dtype=np.int64),
        route_candidate_index=np.zeros(4, dtype=np.int64),
        sample_type=np.zeros(4, dtype=np.float32),
        route_geometry_valid=np.ones(4, dtype=np.float32),
        labels_cbf_feasible=np.ones((4, 5), dtype=np.float32),
        earliest_failure_step=np.full(4, 5, dtype=np.int64),
        labels_boundary_ttc=np.full((4, 5), 10.0, dtype=np.float32),
        labels_pairwise_ttc=np.full((4, 5), 10.0, dtype=np.float32),
        labels_acceleration_slack=np.zeros((4, 5), dtype=np.float32),
        labels_boundary_clearance=np.ones((4, 5), dtype=np.float32),
    )
    source_metadata.write_text(
        json.dumps({"development_only": True, "locked_test_opened": False}), encoding="utf-8"
    )

    materialize(
        source_dataset,
        source_metadata,
        output_dir,
        tensorboard_dir,
        "interaction_pairwise_v1",
        pairwise=True,
    )

    with np.load(output_dir / "route_identity_counterfactual.npz") as archive:
        pairwise = archive["route_pairwise_relative_action_chunk"]
        relative = archive["route_relative_action_chunk"]
    assert pairwise.shape == (4, 1, 9)
    assert relative.shape == (4, 1, 3)
    np.testing.assert_allclose(pairwise[0, 0, [0, 3, 6]], [-1.0, -2.0, -3.0], atol=1e-6)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pairwise_action_conditioned_route_chunk"] is True
