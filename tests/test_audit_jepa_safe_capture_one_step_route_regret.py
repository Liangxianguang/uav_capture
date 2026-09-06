from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_jepa_safe_capture_one_step_route_regret import _as_action, _finite


def test_route_regret_helpers_reject_nonfinite_actions() -> None:
    action = _as_action([[1.0, 0.0, 0.0]], "action")
    np.testing.assert_allclose(action, [[1.0, 0.0, 0.0]])
    assert _finite("0.25") == pytest.approx(0.25)
    assert _finite("nan") is None
    with pytest.raises(ValueError):
        _as_action([[float("nan"), 0.0, 0.0]], "bad")
