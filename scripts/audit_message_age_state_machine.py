"""Audit the explicit communication/observation age state machine.

This is a development-only contract audit.  It runs synthetic transitions on
the partial-observation environment and never produces episode performance
claims or modifies an existing result directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, _BeliefPacket


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_dev.yaml"
STATE_CODES = {"never_received": 0, "fresh": 1, "delayed": 2, "saturated": 3}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _fresh(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {label}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _config(base: dict[str, Any], **pursuit: Any) -> dict[str, Any]:
    result = copy.deepcopy(base)
    result["task"]["pursuit"].update(pursuit)
    return result


def _snapshot(env: CaptureRadiusPursuit3DEnv, label: str) -> dict[str, Any]:
    observation = env.observe()
    return {
        "label": label,
        "step": int(env.step_count),
        "message_age_steps": observation["message_age_steps"],
        "message_received": observation["message_received"],
        "message_age_state": observation["message_age_state"],
        "target_observation_age_steps": observation["target_observation_age_steps"],
        "target_observation_received": observation["target_observation_received"],
        "target_observation_age_state": observation["target_observation_age_state"],
        "target_visible": observation["target_visible"],
    }


def _empty_pending_stream(env: CaptureRadiusPursuit3DEnv) -> None:
    """Construct a deterministic never-received stream without target truth."""

    env._message_queue = []
    env.message_received[:] = False
    env.message_age_steps[:] = int(env.pursuit["maximum_message_age_steps"])
    env.target_observation_timestamps[:] = -1
    env.detection_loss_burst_remaining[:] = int(env.pursuit["maximum_message_age_steps"]) + 10


def _synthetic_packet(env: CaptureRadiusPursuit3DEnv, receiver: int = 0) -> _BeliefPacket:
    zeros = np.zeros(3, dtype=np.float64)
    return _BeliefPacket(
        delivery_step=int(env.step_count),
        receiver=receiver,
        source=receiver,
        timestamp_step=int(env.step_count),
        position=zeros.copy(),
        velocity=zeros.copy(),
        confidence=1.0,
        covariance=np.eye(3, dtype=np.float64),
        via_message=False,
    )


def _run_cases(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    env = CaptureRadiusPursuit3DEnv(
        _config(base_config, maximum_message_age_steps=5, belief_update_mode="time_aligned"),
        obstacle_count=0,
        target_speed_scale=0.1,
    )
    env.reset(seed=520123)
    _empty_pending_stream(env)
    snapshots = [_snapshot(env, "reset_never_received")]
    for index in range(3):
        env.step(np.zeros((env.n_defenders, 3)))
        snapshots.append(_snapshot(env, f"never_received_step_{index + 1}"))
    cases.append({"name": "never_received", "snapshots": snapshots})

    delayed_config = _config(
        base_config,
        maximum_message_age_steps=10,
        observation_delay_steps=4,
        belief_update_mode="time_aligned",
        detection_dropout_probability=0.0,
        message_dropout_probability=1.0 - 1e-9,
        communication_link_dropout_probability=1.0 - 1e-9,
    )
    env = CaptureRadiusPursuit3DEnv(delayed_config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=520124)
    snapshots = [_snapshot(env, "delayed_pending_reset")]
    for index in range(3):
        env.step(np.zeros((env.n_defenders, 3)))
        snapshots.append(_snapshot(env, f"visible_pending_step_{index + 1}"))
    env.step(np.zeros((env.n_defenders, 3)))
    snapshots.append(_snapshot(env, "delivered_step_4"))
    env._message_queue = []
    env.step(np.zeros((env.n_defenders, 3)))
    snapshots.append(_snapshot(env, "visible_without_delivery_step_5"))
    cases.append({"name": "delayed_delivery_and_visible_gap", "snapshots": snapshots})

    env = CaptureRadiusPursuit3DEnv(
        _config(base_config, maximum_message_age_steps=2, belief_update_mode="time_aligned"),
        obstacle_count=0,
        target_speed_scale=0.1,
    )
    env.reset(seed=520125)
    env._message_queue = []
    env.detection_loss_burst_remaining[:] = 20
    assert env._deliver_belief_packet(_synthetic_packet(env))
    snapshots = [_snapshot(env, "recovery_fresh")]
    for index in range(2):
        env._message_queue = []
        env.detection_loss_burst_remaining[:] = 20
        env.step(np.zeros((env.n_defenders, 3)))
        snapshots.append(_snapshot(env, f"saturation_step_{index + 1}"))
    assert env._deliver_belief_packet(_synthetic_packet(env))
    snapshots.append(_snapshot(env, "recovery_after_saturation"))
    cases.append({"name": "saturation_and_recovery", "snapshots": snapshots})
    return cases


def _validate(cases: list[dict[str, Any]]) -> None:
    by_name = {str(case["name"]): case for case in cases}
    never = by_name["never_received"]["snapshots"]
    assert all(set(snapshot["message_age_state"]) == {"never_received"} for snapshot in never)
    assert all(
        all(int(age) == 5 for age in snapshot["message_age_steps"])
        for snapshot in never
    )

    delayed = by_name["delayed_delivery_and_visible_gap"]["snapshots"]
    assert set(delayed[0]["message_age_state"]) == {"never_received"}
    assert delayed[4]["message_received"][0]
    assert delayed[4]["message_age_steps"][0] == 4
    assert delayed[4]["message_age_state"][0] == "delayed"
    assert delayed[5]["target_visible"][0]
    assert delayed[5]["message_age_steps"][0] == 5
    assert delayed[5]["message_age_state"][0] == "delayed"

    recovery = by_name["saturation_and_recovery"]["snapshots"]
    assert recovery[0]["message_age_state"][0] == "fresh"
    assert recovery[1]["message_age_state"][0] == "delayed"
    assert recovery[2]["message_age_state"][0] == "saturated"
    assert recovery[3]["message_age_state"][0] == "fresh"


def _write_report(path: Path, provenance: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    lines = [
        "# Message Age State-Machine Audit",
        "",
        "Status: development-only synthetic contract audit; no episode performance result.",
        "",
        f"- Config SHA-256: `{provenance['config_sha256']}`",
        f"- Git revision at audit start: `{provenance['git_revision']}`",
        "- `locked_test_opened`: `false`",
        "",
        "| Case | Transition evidence | Result |",
        "|---|---|---|",
        "| `never_received` | Numeric age stays at the compatibility ceiling while state remains `never_received` | pass |",
        "| `delayed_delivery_and_visible_gap` | Age advances only on missing delivery, including while target is visible | pass |",
        "| `saturation_and_recovery` | Received stream moves fresh -> delayed -> saturated -> fresh after recovery | pass |",
        "",
        "The numeric `message_age_steps` field is retained for frozen actor compatibility.  "
        "The explicit `message_received` and `message_age_state` fields are authoritative for "
        "diagnostics and reliability routing, so a saturated compatibility value is not labeled "
        "as stale when no packet has ever been accepted.",
        "",
        "All cases passed and were written to `audit.json`; the TensorBoard event file contains "
        "state codes, received masks, and transition snapshots.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tensorboard(path: Path, provenance: dict[str, Any], cases: list[dict[str, Any]]) -> list[str]:
    path = _fresh(path, "TensorBoard directory")
    with SummaryWriter(log_dir=str(path), flush_secs=1) as writer:
        writer.add_text("Config/Provenance", json.dumps(_jsonable(provenance), indent=2), 0)
        for case_index, case in enumerate(cases):
            writer.add_text(f"Case/{case['name']}", json.dumps(_jsonable(case), indent=2), 0)
            for step, snapshot in enumerate(case["snapshots"]):
                states = [STATE_CODES[str(value)] for value in snapshot["message_age_state"]]
                writer.add_scalar(f"{case['name']}/message_age_state_code_max", float(max(states)), step)
                writer.add_scalar(
                    f"{case['name']}/message_received_fraction",
                    float(np.mean(np.asarray(snapshot["message_received"], dtype=np.float64))),
                    step,
                )
                writer.add_scalar(
                    f"{case['name']}/message_age_max_steps",
                    float(max(snapshot["message_age_steps"])),
                    step,
                )
            writer.add_scalar("Audit/case_pass", 1.0, case_index)
        writer.add_scalar("Audit/all_cases_passed", 1.0, 0)
        writer.flush()
    files = sorted(item.name for item in path.glob("events.out.tfevents.*"))
    if not files:
        raise RuntimeError("TensorBoard writer did not create an event file.")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("The age-state audit requires --development-only.")
    config_path = args.environment_config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    output_dir = _fresh(args.output_dir, "audit output directory")
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError("Environment config must be a mapping.")
    provenance = {
        "audit": "message_age_state_machine",
        "development_only": True,
        "locked_test_opened": False,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "git_revision": _git_revision(),
    }
    cases = _run_cases(base_config)
    _validate(cases)
    events = _write_tensorboard(args.tensorboard_dir.resolve(), provenance, cases)
    provenance["tensorboard_event_files"] = events
    (output_dir / "provenance.json").write_text(
        json.dumps(_jsonable(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "audit.json").write_text(
        json.dumps(_jsonable({"all_cases_passed": True, "cases": cases}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", provenance, cases)
    print(json.dumps(_jsonable({"all_cases_passed": True, "output_dir": str(output_dir), "tensorboard": events})))


if __name__ == "__main__":
    main()
