"""Audit v2 candidate generation/ranking on a deterministic synthetic replay.

This is a P4 development audit, not a closed-loop or locked-test evaluator.
The synthetic observations contain only policy-safe features and the geometric
fields exposed by the online observation contract.  The script verifies that
the real v2 checkpoint follows candidate actions, that ledger abstention is
observable, and that every result is recorded in TensorBoard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.jepa_safe_capture_candidates import (  # noqa: E402
    SafeCaptureCandidateConfig,
    SafeCaptureCandidateHistory,
    make_safe_capture_candidate_chunks,
)
from encirclement3d.jepa_safe_capture_ranker import (  # noqa: E402
    SafeCaptureJEPARanker,
    SafeCaptureRankerConfig,
)
from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    build_action_conditioned_predictor,
)
from encirclement3d.reliability import SafeCaptureReliabilityLedger  # noqa: E402


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_safe_capture_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--layout-signature", default="scenario_0")
    parser.add_argument("--target-motion-mode", default="flee_persistence")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def load_predictor(path: Path, device: torch.device) -> tuple[InteractionAwareActionConditionedSafeCaptureJEPAPredictor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(f"P4 requires {MODEL_TYPE}, got {checkpoint.get('model_type')!r}.")
    model_config = checkpoint.get("model")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must contain model and model_state_dict mappings.")
    predictor = build_action_conditioned_predictor(MODEL_TYPE, model_config)
    if not isinstance(predictor, InteractionAwareActionConditionedSafeCaptureJEPAPredictor):
        raise TypeError("Checkpoint factory returned an unexpected v2 predictor.")
    predictor.load_state_dict(state_dict, strict=True)
    predictor.to(device).eval()
    return predictor, checkpoint


def load_ledger(path: Path | None, checkpoint_path: Path) -> SafeCaptureReliabilityLedger | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Ledger JSON must be an object.")
    expected = sha256(checkpoint_path)
    actual = str(payload.get("source", {}).get("checkpoint_sha256", ""))
    if actual != expected:
        raise ValueError("Ledger checkpoint hash does not match the P4 checkpoint.")
    return SafeCaptureReliabilityLedger(payload)


def synthetic_observation(rng: np.random.Generator, defender_count: int = 4) -> tuple[dict[str, Any], np.ndarray]:
    positions = rng.normal(0.0, 1.0, size=(defender_count, 3)).astype(np.float64)
    positions[:, 2] = np.clip(positions[:, 2] + 4.0, 0.7, 8.0)
    beliefs = positions + rng.normal(0.0, 1.5, size=(defender_count, 3))
    velocities = rng.normal(0.0, 0.5, size=(defender_count, 3))
    base = rng.normal(0.0, 0.15, size=(defender_count, 63)).astype(np.float32)
    observation: dict[str, Any] = {
        "defender_positions": positions,
        "target_belief_positions": beliefs,
        "target_belief_velocities": velocities,
        "target_visible": np.asarray(rng.random(defender_count) > 0.15, dtype=bool),
        "target_observation_age_steps": rng.integers(0, 8, size=defender_count).astype(np.float64),
        "message_age_steps": rng.integers(0, 8, size=defender_count).astype(np.float64),
        "obstacles": [{"shape": "cylinder"}, {"shape": "box"}, {"shape": "wall"}],
    }
    return observation, base


def summarize(rows: list[dict[str, Any]], *, checkpoint: Path, ledger: Path | None, device: torch.device) -> dict[str, Any]:
    if not rows:
        raise ValueError("P4 audit produced no rows.")
    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in rows]))

    return {
        "audit_type": "jepa_safe_capture_v2_p4_synthetic_candidate_ranking",
        "model_type": MODEL_TYPE,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "ledger": str(ledger.resolve()) if ledger is not None else None,
        "ledger_sha256": sha256(ledger) if ledger is not None else None,
        "device": str(device),
        "steps": len(rows),
        "candidate_count": 5,
        "chunk_length_steps": 3,
        "execute_only_first_step": True,
        "valid_candidate_fraction": mean("valid_candidate_fraction"),
        "eligible_candidate_fraction": mean("eligible_candidate_fraction"),
        "non_nominal_selection_fraction": mean("non_nominal_selection"),
        "trusted_fraction": mean("trusted_fraction"),
        "fallback_nominal_fraction": mean("fallback_nominal"),
        "safe_hold_fraction": mean("safe_hold"),
        "action_following_mean_abs_delta": mean("action_following_mean_abs_delta"),
        "mean_score_margin_to_nominal": mean("score_margin_to_nominal"),
        "mean_rank_latency_ms": mean("rank_latency_ms"),
        "rows": rows,
    }


def write_tensorboard(report: Mapping[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/audit", json.dumps({key: report[key] for key in ("audit_type", "model_type", "candidate_count", "chunk_length_steps", "execute_only_first_step")}, indent=2), 0)
        writer.add_text("Provenance/sources", json.dumps({key: report[key] for key in ("checkpoint", "checkpoint_sha256", "ledger", "ledger_sha256", "not_a_locked_test", "locked_test_opened")}, indent=2), 0)
        writer.add_text("Provenance/environment", json.dumps({"git_revision": git_revision(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "tensorboard": version("tensorboard")}, indent=2), 0)
        for row in report["rows"]:
            step = int(row["step"])
            writer.add_scalar("Ranking/valid_candidate_fraction", float(row["valid_candidate_fraction"]), step)
            writer.add_scalar("Ranking/eligible_candidate_fraction", float(row["eligible_candidate_fraction"]), step)
            writer.add_scalar("Ranking/non_nominal_selection_fraction", float(row["non_nominal_selection"]), step)
            writer.add_scalar("Ranking/action_following_mean_abs_delta", float(row["action_following_mean_abs_delta"]), step)
            writer.add_scalar("Ranking/score_margin_to_nominal", float(row["score_margin_to_nominal"]), step)
            writer.add_scalar("Reliability/trusted_fraction", float(row["trusted_fraction"]), step)
            writer.add_scalar("Fallback/nominal_fraction", float(row["fallback_nominal"]), step)
            writer.add_scalar("Fallback/safe_hold_fraction", float(row["safe_hold"]), step)
            writer.add_scalar("Latency/rank_ms", float(row["rank_latency_ms"]), step)
        writer.add_scalar("Aggregate/valid_candidate_fraction", float(report["valid_candidate_fraction"]), 0)
        writer.add_scalar("Aggregate/eligible_candidate_fraction", float(report["eligible_candidate_fraction"]), 0)
        writer.add_scalar("Aggregate/action_following_mean_abs_delta", float(report["action_following_mean_abs_delta"]), 0)
        writer.add_scalar("Aggregate/non_nominal_selection_fraction", float(report["non_nominal_selection_fraction"]), 0)
        writer.add_scalar("Aggregate/rank_latency_ms", float(report["mean_rank_latency_ms"]), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {"Config/audit/text_summary", "Provenance/sources/text_summary", "Provenance/environment/text_summary"}
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    if missing:
        raise ValueError(f"P4 TensorBoard provenance is incomplete: {missing}")
    return {
        "logdir": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_text_complete": not missing,
    }


def run_audit(
    predictor: InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    *,
    ledger: SafeCaptureReliabilityLedger | None,
    device: torch.device,
    steps: int,
    seed: int,
    layout_signature: str,
    target_motion_mode: str,
) -> list[dict[str, Any]]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    rng = np.random.default_rng(seed)
    history = SafeCaptureCandidateHistory(
        predictor,
        defender_count=4,
        device=device,
        history_length=8,
        action_scale=5.0,
    )
    observation, base = synthetic_observation(rng)
    history.reset(base)
    ranker = SafeCaptureJEPARanker(
        history,
        config=SafeCaptureRankerConfig(horizon_index=2, horizon_seconds=0.30, position_extent_m=10.0),
        reliability_ledger=ledger,
        context_defaults={"layout_signature": layout_signature, "target_motion_mode": target_motion_mode},
    )
    previous_action = np.zeros((4, 3), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for step in range(steps):
        if step == 0:
            nominal = np.clip(rng.normal(0.0, 0.25, size=(4, 3)), -0.45, 0.45).astype(np.float64)
        else:
            # Keep the synthetic actor trajectory within the declared slew
            # contract so the audit exercises ranking rather than rejecting
            # every candidate for an intentionally discontinuous actor.
            nominal = np.clip(previous_action + rng.normal(0.0, 0.03, size=(4, 3)), -1.5, 1.5).astype(np.float64)
        # The first synthetic frame represents an actor action already held for
        # one control period, so its acceleration reference is the nominal
        # action itself. Subsequent frames use the previously selected action.
        action_reference = nominal if step == 0 else previous_action
        batch = make_safe_capture_candidate_chunks(
            nominal,
            observation,
            config=SafeCaptureCandidateConfig(),
            previous_action=action_reference,
        )
        valid_indices = np.flatnonzero(batch.valid_mask)
        action_following = 0.0
        if valid_indices.size > 1:
            before = time.perf_counter()
            means, _std, _aux = history.predict_candidates_multitask(
                batch.chunks[valid_indices, 0], horizon_index=2
            )
            del before
            action_following = float(np.mean(np.abs(means[1:] - means[:1])))
        started = time.perf_counter()
        result = ranker.rank(observation, batch, previous_action=previous_action)
        latency_ms = (time.perf_counter() - started) * 1000.0
        states = result.trace.ledger_states
        rows.append(
            {
                "step": step,
                "selected_index": result.selected_index,
                "execution_mode": result.execution_mode,
                "valid_candidate_fraction": float(np.mean(batch.valid_mask)),
                "eligible_candidate_fraction": float(np.mean(result.trace.eligible_mask)),
                "non_nominal_selection": float(result.selected_index != 0),
                "trusted_fraction": float(np.mean(np.asarray(states) == "trusted")),
                "fallback_nominal": float(result.execution_mode == "fallback_nominal"),
                "safe_hold": float(result.execution_mode == "safe_hold"),
                "action_following_mean_abs_delta": action_following,
                "score_margin_to_nominal": float(result.trace.scores[0] - result.trace.scores[result.selected_index]) if np.isfinite(result.trace.scores[0]) and np.isfinite(result.trace.scores[result.selected_index]) else 0.0,
                "rank_latency_ms": latency_ms,
            }
        )
        previous_action = result.selected_action.astype(np.float64, copy=True)
        next_observation, next_base = synthetic_observation(rng)
        history.observe_after_action(next_base, result.selected_action)
        observation = next_observation
    return rows


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output}")
    device = choose_device(args.device)
    predictor, _checkpoint_payload = load_predictor(checkpoint, device)
    ledger_path = args.ledger.resolve() if args.ledger is not None else None
    ledger = load_ledger(ledger_path, checkpoint)
    rows = run_audit(
        predictor,
        ledger=ledger,
        device=device,
        steps=int(args.steps),
        seed=int(args.seed),
        layout_signature=str(args.layout_signature),
        target_motion_mode=str(args.target_motion_mode),
    )
    report = summarize(rows, checkpoint=checkpoint, ledger=ledger_path, device=device)
    tensorboard = write_tensorboard(report, args.tensorboard_logdir)
    report["tensorboard"] = tensorboard
    report["provenance"] = {
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorboard": version("tensorboard"),
        "source_hashes": {
            "scripts/audit_jepa_safe_capture_v2_candidate_ranking.py": sha256(Path(__file__).resolve()),
            "src/encirclement3d/jepa_safe_capture_candidates.py": sha256(PROJECT_ROOT / "src/encirclement3d/jepa_safe_capture_candidates.py"),
            "src/encirclement3d/jepa_safe_capture_ranker.py": sha256(PROJECT_ROOT / "src/encirclement3d/jepa_safe_capture_ranker.py"),
            "configs/jepa_safe_capture_v2_protocol.yaml": sha256(PROJECT_ROOT / "configs/jepa_safe_capture_v2_protocol.yaml"),
        },
        "command": " ".join(sys.argv),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
