from pathlib import Path
from argparse import Namespace

import numpy as np

from scripts.evaluate_dn_mpc_cbf_g5 import _buffer_observables, _planner_config


def test_s1_buffer_observables_are_separate_from_physical_clearance() -> None:
    class _Env:
        defender_positions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64)
        n_defenders = 2
        lower = np.array([-2.0, -2.0, 0.0], dtype=np.float64)
        upper = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        obstacles = ()
        agents = {"drone_radius": 0.25}

    class _Filter:
        obstacle_margin_m = 0.35
        inter_agent_margin_m = 0.35
        boundary_margin_m = 0.35

    values = _buffer_observables(_Env(), _Filter())
    assert np.isclose(values["minimum_pairwise_buffer_clearance_m"], 0.15)
    assert values["minimum_boundary_buffer_clearance_m"] > 0.0


def test_s1_result_contract_is_development_only() -> None:
    config = Path(__file__).parents[1] / "configs" / "dn_mpc_cbf_g5_development.yaml"
    text = config.read_text(encoding="utf-8")
    assert "development_only: true" in text
    assert "locked_test_opened: false" in text
    assert "jepa_enabled: false" in text
    assert "ledger_enabled: false" in text


def test_route_hysteresis_parameters_are_reflected_in_planner_contract() -> None:
    config = _planner_config(
        Namespace(minimum_hold_steps=3, switch_improvement_m=0.35, tangent_route_hold_steps=6),
        0.1,
    )
    assert config.minimum_hold_steps == 3
    assert np.isclose(config.switch_improvement_m, 0.35)
    assert config.tangent_route_hold_steps == 6
