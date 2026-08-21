from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_fixed_wall_coverage.py"
SPEC = importlib.util.spec_from_file_location("analyze_fixed_wall_coverage", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def test_collect_reports_accepted_wall_strata_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "fixed"
    run_dir.mkdir()
    (run_dir / "expert_sequence_dataset.npz").write_bytes(b"archive")
    (run_dir / "expert_dataset_manifest.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "accepted": True,
                        "layout": "wall",
                        "defender_side": "left",
                        "initial_side_distance": 5.0,
                        "target_speed_scale": 0.55,
                        "target_motion_mode": "flee_persistence",
                        "target_crossing_required": False,
                        "required_defender_zone_entries": 2,
                    },
                    {
                        "accepted": True,
                        "layout": "cylinder",
                        "defender_side": "left",
                        "initial_side_distance": 5.5,
                        "target_speed_scale": 0.45,
                        "target_motion_mode": "flee_persistence",
                        "target_crossing_required": False,
                        "required_defender_zone_entries": 2,
                    },
                    {
                        "accepted": False,
                        "layout": "wall",
                        "defender_side": "right",
                        "initial_side_distance": 6.0,
                        "target_speed_scale": 0.55,
                        "target_motion_mode": "s_curve",
                        "target_crossing_required": False,
                        "required_defender_zone_entries": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    analysis = ANALYZER.collect(run_dir)

    assert analysis["accepted_expert_episodes"] == 2
    assert analysis["accepted_layout_episodes"] == 1
    assert analysis["strata"][0]["target_speed_scale"] == 0.55
    assert "Wall Strata" in ANALYZER.render_markdown(analysis)
