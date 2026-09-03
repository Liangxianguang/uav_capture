from __future__ import annotations

import numpy as np

from scripts.generate_jepa_v3_counterfactual_dataset import _append_candidate_samples, _empty_samples


def test_counterfactual_action_history_uses_runtime_action_normalization() -> None:
    samples = _empty_samples()
    observation_history = [np.full((4, 63), step, dtype=np.float32) for step in range(8)]
    executed_actions = [np.full((4, 3), float(step + 1), dtype=np.float32) for step in range(7)]
    candidate = np.full((4, 3), 5.0, dtype=np.float32)
    labels = {
        "target_relative": np.zeros((4, 4, 3), dtype=np.float32),
        "obstacle_clearance": np.zeros((4, 4), dtype=np.float32),
        "inter_agent_clearance": np.zeros((4, 4), dtype=np.float32),
        "target_visible": np.zeros((4, 4), dtype=np.float32),
        "cbf_correction": np.zeros((4, 4), dtype=np.float32),
        "cbf_intervention": np.zeros((4, 4), dtype=np.float32),
        "collision": np.zeros((4, 4), dtype=np.float32),
        "boundary": np.zeros((4, 4), dtype=np.float32),
    }

    _append_candidate_samples(
        samples,
        observation_history,
        executed_actions,
        candidate,
        labels,
        episode_seed=1,
        scenario_index=2,
        time_index=7,
        candidate_index=0,
        chunk_length_steps=1,
        action_scale=5.0,
    )

    np.testing.assert_allclose(samples["action_history"][0][0], np.full(3, 0.2, dtype=np.float32))
    np.testing.assert_allclose(samples["action_history"][0][-1], np.ones(3, dtype=np.float32))
    assert np.isclose(samples["candidate_action_norm_mps"][0], np.sqrt(75.0))


def test_counterfactual_candidate_groups_are_complete_without_contiguous_row_assumption() -> None:
    # Collection loops over candidates and then defenders, so one group need
    # not occupy five consecutive rows. Group by its explicit state-agent key.
    episode_seed = np.array([7, 7, 7, 8, 8, 8])
    time_index = np.array([3, 3, 3, 4, 4, 4])
    agent_id = np.array([0, 0, 0, 1, 1, 1])
    candidate_index = np.array([2, 0, 1, 1, 2, 0])
    keys = np.stack([episode_seed, time_index, agent_id], axis=1)
    _groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    for group in range(int(np.max(inverse)) + 1):
        np.testing.assert_array_equal(np.sort(candidate_index[inverse == group]), np.arange(3))
