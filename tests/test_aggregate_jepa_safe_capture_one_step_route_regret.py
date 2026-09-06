from __future__ import annotations

import json
from pathlib import Path

from scripts.aggregate_jepa_safe_capture_one_step_route_regret import _read


def test_aggregate_reader_accepts_development_audit_json(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"audit_type": "test"}), encoding="utf-8")
    assert _read(path)["audit_type"] == "test"
