from pathlib import Path

from scripts.smoke_dn_mpc_route_selection import run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dn_mpc_route_smoke_writes_provenance_and_tensorboard(tmp_path: Path) -> None:
    output = tmp_path / "output"
    tensorboard = tmp_path / "tensorboard"
    report = run(
        PROJECT_ROOT / "configs" / "dn_mpc_jepa_safe_capture_development.yaml",
        output,
        tensorboard,
        20260907,
    )

    assert report["summary"]["scenario_count"] == 4
    assert report["summary"]["all_scenarios_selected_valid_route"] is True
    assert (output / "summary.json").is_file()
    assert (output / "provenance.json").is_file()
    assert (output / "candidate_rows.jsonl").is_file()
    assert list(tensorboard.glob("events.out.tfevents.*"))
