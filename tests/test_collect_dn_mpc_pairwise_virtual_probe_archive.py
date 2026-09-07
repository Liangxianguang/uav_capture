import numpy as np
import pytest

from scripts.collect_dn_mpc_pairwise_virtual_probe_archive import _probe_chunk


def test_probe_chunk_converges_selected_pair_without_execution() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [2.0, 3.0, 1.0]],
        dtype=np.float64,
    )
    chunk = _probe_chunk(positions, pair=(0, 1), max_speed=5.0)
    assert chunk.shape == (5, 4, 3)
    assert np.allclose(chunk[:, 0], np.asarray([5.0, 0.0, 0.0]))
    assert np.allclose(chunk[:, 1], np.asarray([-5.0, 0.0, 0.0]))
    assert np.allclose(chunk[:, 2:], 0.0)


def test_probe_chunk_hold_is_zero_and_validates_shape() -> None:
    positions = np.zeros((4, 3), dtype=np.float64)
    hold = _probe_chunk(positions, pair=None, max_speed=5.0)
    assert hold.shape == (5, 4, 3)
    assert np.allclose(hold, 0.0)
    with pytest.raises(ValueError):
        _probe_chunk(np.zeros((3, 3)), pair=None, max_speed=5.0)
