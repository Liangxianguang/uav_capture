from __future__ import annotations

import pytest

from scripts.audit_dn_mpc_p36_route_utility import (
    _at_grid_boundary,
    _parse_nonnegative_grid,
    _parse_switch_grid,
)


def test_switch_grid_parser_sorts_and_deduplicates_values() -> None:
    assert _parse_switch_grid("0.5, 0, 1, 0.5") == (0.0, 0.5, 1.0)


def test_switch_grid_parser_rejects_negative_or_empty_values() -> None:
    with pytest.raises(ValueError):
        _parse_switch_grid("0.0,-0.1")
    with pytest.raises(ValueError):
        _parse_switch_grid("")


def test_generic_weight_grid_parser_has_named_errors() -> None:
    assert _parse_nonnegative_grid("2, 0, 1, 1", name="--length-grid") == (0.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="--length-grid"):
        _parse_nonnegative_grid("-1", name="--length-grid")
    with pytest.raises(ValueError, match="--cbf-grid"):
        _parse_nonnegative_grid("nan", name="--cbf-grid")


def test_grid_boundary_is_computed_against_the_active_grid() -> None:
    grid = {"length": (0.0, 1.0), "switch": (0.0, 0.5, 1.0)}
    assert _at_grid_boundary({"length": 1.0, "switch": 0.5}, grid) == {
        "length": True,
        "switch": False,
    }
