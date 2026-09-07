from pathlib import Path

from scripts.replay_dn_mpc_multicycle import run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_multicycle_replay_has_valid_routes_and_tensorboard(tmp_path: Path) -> None:
    output = tmp_path / "output"
    tensorboard = tmp_path / "tensorboard"
    report = run(
        PROJECT_ROOT / "configs" / "dn_mpc_jepa_safe_capture_development.yaml",
        output,
        tensorboard,
        20260907,
        cycles=4,
    )
    assert report["summary"]["scenario_count"] == 4
    assert report["summary"]["all_cycles_selected_valid_route"] is True
    assert (output / "cycle_traces.jsonl").is_file()
    assert (output / "provenance.json").is_file()
    assert list(tensorboard.glob("events.out.tfevents.*"))


def test_multicycle_replay_records_route_phase_and_progress(tmp_path: Path) -> None:
    output = tmp_path / "output"
    tensorboard = tmp_path / "tensorboard"
    run(
        PROJECT_ROOT / "configs" / "dn_mpc_jepa_safe_capture_development.yaml",
        output,
        tensorboard,
        20260908,
        cycles=3,
    )
    first = (output / "cycle_traces.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert '"route_phase"' in first
    assert '"selected_terminal_progress_cost"' in first
    assert '"selected_stopping_distance_cost"' in first
