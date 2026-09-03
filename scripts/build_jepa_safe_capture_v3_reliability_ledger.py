"""Build a v3 checkpoint-bound Reliability Ledger from calibration data.

The v3 trainer keeps the v2 runtime tensor shape for compatibility, but the
ledger is explicitly versioned and bound to the v3 protocol, checkpoint, and
calibration archive.  Calibration remains offline-only; CBF remains the final
execution boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from build_jepa_safe_capture_v2_reliability_ledger import (
    _load_arrays,
    _load_metadata,
    _predict,
    build_payload,
    choose_device,
    forecast,
    render_report,
    sha256,
    write_tensorboard,
)

from encirclement3d.reliability import SafeCaptureReliabilityLedger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPE = "interaction_aware_action_conditioned_jepa_safe_capture_v2"
TRAINING_VARIANT = "hard_context_weighted_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/jepa_safe_capture_v3_next_phase.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--minimum-sample-count", type=int, default=128)
    parser.add_argument("--minimum-credit", type=float, default=0.65)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def _load_protocol(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("v3 protocol must be a YAML mapping.")
    if value.get("phase") != "development_only" or value.get("locked_test_opened") is not False:
        raise ValueError("v3 ledger calibration requires a closed development protocol.")
    if value.get("reliability_ledger", {}).get("version") != 3:
        raise ValueError("v3 protocol must declare reliability_ledger.version=3.")
    if value.get("reliability_ledger", {}).get("source_split") != "calibration_only":
        raise ValueError("v3 ledger requires calibration-only source split.")
    return value


def _fallback_audit(payload: dict[str, Any]) -> dict[str, Any]:
    """Exercise explicit OOD, stale, and non-finite runtime fallbacks."""

    ledger = SafeCaptureReliabilityLedger(payload)
    base: dict[str, Any] = {
        "visibility_condition": 1.0,
        "observation_age_steps": 0.0,
        "obstacle_count": 3,
        "layout_signature": "scenario_0",
        "target_motion_mode": "flee_persistence",
        "minimum_clearance_m": 1.0,
        "pairwise_ttc_s": 2.0,
        "uncertainty": 0.05,
        "cbf_risk": 0.1,
        "candidate_separation_m": 0.3,
    }
    cases: dict[str, dict[str, Any]] = {
        "ood": {**base, "ood": True},
        "stale": {**base, "observation_age_steps": 46.0},
        "non_finite": {**base, "uncertainty": float("nan")},
    }
    expected = {"ood": "ood", "stale": "stale_observation", "non_finite": "non_finite_context"}
    results: dict[str, Any] = {}
    for name, context in cases.items():
        try:
            decision = ledger.decision(0, context)
            passed = decision.state == "safe_hold" and decision.fallback_reason == expected[name]
            results[name] = {
                "state": decision.state,
                "fallback_reason": decision.fallback_reason,
                "passed": bool(passed),
            }
        except Exception as error:  # pragma: no cover - defensive audit record
            results[name] = {"state": "exception", "fallback_reason": None, "error": repr(error), "passed": False}
    return {
        "cases": results,
        "all_required_fallbacks_pass": bool(all(item.get("passed") for item in results.values())),
    }


def _write_extra_tensorboard(
    logdir: Path,
    protocol: dict[str, Any],
    checkpoint: dict[str, Any],
    fallback_audit: dict[str, Any],
) -> dict[str, Any]:
    with SummaryWriter(log_dir=str(logdir.resolve()), flush_secs=1) as writer:
        writer.add_text("Config/v3_protocol", yaml.safe_dump(protocol, sort_keys=False), 0)
        writer.add_text(
            "Config/checkpoint",
            json.dumps(
                {
                    "model_type": checkpoint.get("model_type"),
                    "training_variant": checkpoint.get("training_variant"),
                    "checkpoint_sha256": checkpoint.get("_checkpoint_sha256"),
                },
                indent=2,
            ),
            0,
        )
        writer.add_text("Reliability/fallback_audit", json.dumps(fallback_audit, indent=2), 0)
        for index, (name, result) in enumerate(fallback_audit["cases"].items(), start=1):
            writer.add_scalar(f"Reliability/fallback_gate/{name}", float(bool(result.get("passed"))), index)
    return {
        "path": str(logdir.resolve()),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "v3_text_logged": True,
        "fallback_gate_logged": True,
    }


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.dataset, args.metadata, args.protocol):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    if args.batch_size <= 0 or args.minimum_sample_count <= 0 or not 0.0 <= args.minimum_credit <= 1.0:
        raise ValueError("Invalid v3 ledger policy or batch size.")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite an existing v3 ledger or report.")
    if args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir()):
        raise FileExistsError("Refusing to overwrite a non-empty v3 TensorBoard directory.")
    protocol = _load_protocol(args.protocol)
    metadata = _load_metadata(args.metadata.resolve(), args.dataset.resolve())
    metadata["protocol"] = str(args.protocol.resolve())
    arrays = _load_arrays(args.dataset.resolve())
    checkpoint_path = args.checkpoint.resolve()
    device = choose_device(args.device)
    predictions, checkpoint = _predict(checkpoint_path, arrays, args.batch_size, device)
    if checkpoint.get("training_variant") != TRAINING_VARIANT:
        raise ValueError("v3 ledger requires the hard-context weighted checkpoint variant.")
    checkpoint["_checkpoint_sha256"] = sha256(checkpoint_path)
    payload, diagnostics = build_payload(
        arrays,
        predictions,
        metadata,
        checkpoint_path,
        args.dataset.resolve(),
        args.metadata.resolve(),
        args.minimum_sample_count,
        args.minimum_credit,
    )
    payload["ledger_type"] = SafeCaptureReliabilityLedger.LEDGER_TYPE_V3
    payload["ledger_version"] = 3
    payload["training_variant"] = TRAINING_VARIANT
    payload["source"]["protocol"] = str(args.protocol.resolve())
    payload["source"]["protocol_sha256"] = sha256(args.protocol.resolve())
    payload["source"]["model_type"] = checkpoint.get("model_type")
    payload["source"]["training_variant"] = TRAINING_VARIANT
    payload["source"]["builder"] = str(Path(__file__).resolve())
    payload["source"]["builder_sha256"] = sha256(Path(__file__).resolve())
    fallback_audit = _fallback_audit(payload)
    if not fallback_audit["all_required_fallbacks_pass"]:
        raise RuntimeError("v3 fallback audit failed; refusing to publish ledger.")
    collection = yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))
    extent = float(collection["world"]["half_extent_xy"])
    maximum_observation_age = float(collection["task"]["pursuit"]["maximum_message_age_steps"])
    forecast_diagnostics = forecast(payload, arrays, predictions, metadata, extent, maximum_observation_age)
    payload["diagnostics"] = diagnostics
    payload["forecast"] = forecast_diagnostics
    payload["fallback_audit"] = fallback_audit
    tensorboard = write_tensorboard(payload, diagnostics, forecast_diagnostics, args.tensorboard_logdir)
    tensorboard.update(_write_extra_tensorboard(args.tensorboard_logdir, protocol, checkpoint, fallback_audit))
    payload["tensorboard"] = tensorboard
    report_text = render_report(payload, diagnostics, forecast_diagnostics)
    report_text = report_text.replace("# JEPA Safe-Capture v2 P3 Reliability Ledger", "# JEPA Safe-Capture v3 WP3 Reliability Ledger")
    report_text += (
        "\n## v3 Fallback Audit\n\n"
        f"`{json.dumps(fallback_audit, sort_keys=True)}`\n\n"
        "OOD, stale, and non-finite contexts all require explicit safe-hold; this audit is separate from closed-loop performance.\n"
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.report.resolve().write_text(report_text, encoding="utf-8")
    print(json.dumps({"ledger": str(args.output.resolve()), "fallback_audit": fallback_audit, "forecast": forecast_diagnostics, "tensorboard": tensorboard}, indent=2))


if __name__ == "__main__":
    main()
