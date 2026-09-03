from __future__ import annotations

from pathlib import Path

from scripts.audit_jepa_safe_capture_v2_candidate_ranking import summarize, write_tensorboard


def _row(step: int, *, trusted: float = 1.0, fallback: float = 0.0, hold: float = 0.0) -> dict[str, float | int]:
    return {
        "step": step,
        "valid_candidate_fraction": 1.0,
        "eligible_candidate_fraction": trusted,
        "non_nominal_selection": 1.0 if step % 2 == 0 else 0.0,
        "trusted_fraction": trusted,
        "fallback_nominal": fallback,
        "safe_hold": hold,
        "action_following_mean_abs_delta": 0.02,
        "score_margin_to_nominal": 0.1,
        "rank_latency_ms": 2.0,
    }


def test_p4_audit_summary_aggregates_synthetic_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    ledger = tmp_path / "ledger.json"
    ledger.write_bytes(b"ledger")
    report = summarize(
        [_row(0), _row(1, trusted=0.0, fallback=1.0)],
        checkpoint=checkpoint,
        ledger=ledger,
        device="cpu",  # type: ignore[arg-type]
    )
    assert report["steps"] == 2
    assert report["valid_candidate_fraction"] == 1.0
    assert report["trusted_fraction"] == 0.5
    assert report["fallback_nominal_fraction"] == 0.5
    assert report["non_nominal_selection_fraction"] == 0.5


def test_p4_audit_tensorboard_has_provenance_and_ranking_scalars(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = summarize([_row(0)], checkpoint=checkpoint, ledger=None, device="cpu")  # type: ignore[arg-type]
    metadata = write_tensorboard(report, tmp_path / "tb")
    assert metadata["required_text_complete"] is True
    assert metadata["scalar_tag_count"] >= 10
    assert metadata["text_tag_count"] == 3
