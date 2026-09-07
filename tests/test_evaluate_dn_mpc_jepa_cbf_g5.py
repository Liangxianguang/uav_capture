from types import SimpleNamespace

import numpy as np

from scripts.evaluate_dn_mpc_jepa_cbf_g5 import _shortlist_batch


def _runtime():
    candidates = tuple(
        SimpleNamespace(label=label, action_chunk=np.full((3, 2, 3), index, dtype=np.float64))
        for index, label in enumerate(("nominal", "left_detour", "right_detour", "braking"))
    )
    batch = SimpleNamespace(
        valid_mask=np.asarray([True, True, False, True]),
        rejection_reasons=((), (), ("cbf_infeasible",), ()),
    )
    route_batch = SimpleNamespace(candidates=candidates, chunks=np.stack([item.action_chunk for item in candidates]), labels=tuple(item.label for item in candidates))
    return SimpleNamespace(route_batch=route_batch, candidate_batch=batch)


def test_shortlist_keeps_nominal_anchor_and_orders_analytic_cost() -> None:
    runtime = _runtime()
    decision = SimpleNamespace(scores=(2.0, 0.5, 0.1, 1.0))
    batch, indices = _shortlist_batch(runtime, decision, 3)
    assert indices == [0, 1, 3]
    assert batch.labels == ("nominal", "left_detour", "braking")
    assert batch.valid_mask.tolist() == [True, True, True]
    assert batch.chunks.shape == (3, 3, 2, 3)


def test_shortlist_does_not_make_invalid_candidate_eligible() -> None:
    runtime = _runtime()
    decision = SimpleNamespace(scores=(2.0, 0.5, 0.1, 1.0))
    batch, indices = _shortlist_batch(runtime, decision, 4)
    assert indices == [0, 1, 3]
    assert all(bool(value) for value in batch.valid_mask)


def test_shortlist_keeps_previous_route_for_ranker_hysteresis() -> None:
    runtime = _runtime()
    decision = SimpleNamespace(scores=(2.0, 0.5, 0.1, 1.0))
    _batch, indices = _shortlist_batch(runtime, decision, 2, previous_route_index=3)
    assert indices[0] == 0
    assert 3 in indices
