from __future__ import annotations

import torch

from scripts.train_route_identity_jepa import _RouteGroupBatchSampler, _route_progress_ranking_metrics


def test_route_group_sampler_never_splits_a_belief_group() -> None:
    tensors = {
        "scenario_index": torch.tensor([0, 0, 1, 1, 2, 2]),
        "time_index": torch.tensor([0, 0, 0, 0, 1, 1]),
    }
    sampler = _RouteGroupBatchSampler(tensors, batch_size=4, shuffle=False, seed=7)
    batches = list(iter(sampler))

    assert sorted(index for batch in batches for index in batch) == list(range(6))
    index_locations = {
        index: batch_index
        for batch_index, batch in enumerate(batches)
        for index in batch
    }
    keys = {
        (int(tensors["scenario_index"][index]), int(tensors["time_index"][index]))
        for index in range(6)
    }
    for key in keys:
        indices = [
            index
            for index in range(6)
            if (int(tensors["scenario_index"][index]), int(tensors["time_index"][index])) == key
        ]
        assert len({index_locations[index] for index in indices}) == 1


def test_route_progress_ranking_loss_penalizes_reversed_route_order() -> None:
    batch = {
        "sample_type": torch.zeros(4),
        "route_candidate_index": torch.tensor([0, 0, 1, 1]),
        "scenario_index": torch.zeros(4, dtype=torch.long),
        "time_index": torch.zeros(4, dtype=torch.long),
        "route_geometry_valid": torch.ones(4),
        "labels_cbf_feasible": torch.ones(4, 5),
        "labels_route_progress": torch.tensor(
            [[0.8, 0.0, 0.8, 0.0, 0.0], [0.8, 0.0, 0.8, 0.0, 0.0],
             [0.2, 0.0, 0.2, 0.0, 0.0], [0.2, 0.0, 0.2, 0.0, 0.0]]
        ),
    }
    auxiliary = {"route_progress": torch.tensor(
        [[0.1, 0.0, 0.1, 0.0, 0.0], [0.1, 0.0, 0.1, 0.0, 0.0],
         [0.9, 0.0, 0.9, 0.0, 0.0], [0.9, 0.0, 0.9, 0.0, 0.0]], requires_grad=True
    )}

    loss, accuracy, groups = _route_progress_ranking_metrics(
        auxiliary, batch, horizon_index=2, margin=0.05
    )

    assert float(loss) > 0.0
    assert float(accuracy) == 0.0
    assert float(groups) == 1.0
    loss.backward()
    assert auxiliary["route_progress"].grad is not None


def test_listwise_route_progress_loss_accepts_near_ties() -> None:
    batch = {
        "sample_type": torch.zeros(4),
        "route_candidate_index": torch.tensor([0, 0, 1, 1]),
        "scenario_index": torch.zeros(4, dtype=torch.long),
        "time_index": torch.zeros(4, dtype=torch.long),
        "route_geometry_valid": torch.ones(4),
        "labels_cbf_feasible": torch.ones(4, 5),
        "labels_route_progress": torch.tensor(
            [[0.50, 0.0, 0.50, 0.0, 0.0], [0.50, 0.0, 0.50, 0.0, 0.0],
             [0.49, 0.0, 0.49, 0.0, 0.0], [0.49, 0.0, 0.49, 0.0, 0.0]]
        ),
    }
    auxiliary = {"route_progress": torch.tensor(
        [[0.20, 0.0, 0.20, 0.0, 0.0], [0.20, 0.0, 0.20, 0.0, 0.0],
         [0.10, 0.0, 0.10, 0.0, 0.0], [0.10, 0.0, 0.10, 0.0, 0.0]], requires_grad=True
    )}
    loss, accuracy, groups = _route_progress_ranking_metrics(
        auxiliary, batch, horizon_index=2, margin=0.005, mode="listwise", temperature=0.02
    )
    assert float(loss) > 0.0
    assert float(accuracy) == 1.0
    assert float(groups) == 1.0
    loss.backward()
    assert auxiliary["route_progress"].grad is not None
