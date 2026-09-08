from __future__ import annotations

from scripts.audit_dn_mpc_p28_hard_negative_replay import ROUTE_LABELS, SAMPLE_TYPES


def test_p28_sample_types_are_reserved_outside_runtime_candidates() -> None:
    assert ROUTE_LABELS == ("braking", "left_detour", "right_detour", "boundary_rescue")
    assert SAMPLE_TYPES == {"braking": 5, "left_detour": 6, "right_detour": 7, "boundary_rescue": 8}
    assert min(SAMPLE_TYPES.values()) > 0
