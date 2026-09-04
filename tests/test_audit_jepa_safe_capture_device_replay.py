from __future__ import annotations

from scripts.audit_jepa_safe_capture_device_replay import _numeric_equal


def test_numeric_trace_comparison_allows_small_device_roundoff() -> None:
    assert _numeric_equal([[1.0, 2.0]], [[1.0 + 1e-6, 2.0 - 1e-6]], atol=1e-5)
    assert not _numeric_equal([[1.0]], [[1.1]], atol=1e-5)


def test_numeric_trace_comparison_requires_matching_mapping_keys() -> None:
    assert _numeric_equal({"a": [1.0], "b": True}, {"a": [1.0], "b": True})
    assert not _numeric_equal({"a": [1.0]}, {"c": [1.0]})
