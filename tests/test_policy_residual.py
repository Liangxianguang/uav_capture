from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml

from encirclement3d.environment import CylinderObstacle
from encirclement3d.pybullet_env import PyBulletEncirclement3DEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"
EVALUATOR_PATH = PROJECT_ROOT / "scripts" / "evaluate_pybullet.py"


def _load_apply_policy_residual():
    spec = importlib.util.spec_from_file_location("evaluate_pybullet_test_module", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load evaluate_pybullet.py for unit testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_policy_residual


def test_clearance_gate_scales_only_the_nearby_defender_residual() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    env = PyBulletEncirclement3DEnv(config, obstacle_count=0, target_speed_scale=0.3)
    env.defender_positions = np.array(
        [[1.30, 0.0, 1.0], [-5.0, 0.0, 1.0], [-6.0, 0.0, 1.0], [-7.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    env.obstacles = [CylinderObstacle(center_xy=np.array([0.0, 0.0]), radius=1.0, height=4.0)]
    observation = {
        "defender_positions": env.defender_positions.copy(),
        "slot_positions": env.defender_positions + np.array([[1.0, 0.0, 0.0]] * 4),
        "target_velocity": np.zeros(3),
    }
    apply_policy_residual = _load_apply_policy_residual()

    action, _norm, minimum_clearance, mean_scale = apply_policy_residual(
        np.zeros((4, 3), dtype=np.float64),
        observation,
        env,
        {
            "enabled": True,
            "slot_gain": 1.0,
            "activation_error": 2.0,
            "hold_only": True,
            "clearance_gate_threshold": 1.0,
        },
    )

    # The first defender is 0.05 m outside the obstacle plus vehicle radius,
    # so its unit slot residual is reduced to 5% of its original magnitude.
    np.testing.assert_allclose(action[0], np.array([0.05, 0.0, 0.0]), atol=1e-8)
    np.testing.assert_allclose(action[1:], np.array([[1.0, 0.0, 0.0]] * 3), atol=1e-8)
    np.testing.assert_allclose(minimum_clearance, 0.05, atol=1e-8)
    assert 0.75 < mean_scale < 0.77
