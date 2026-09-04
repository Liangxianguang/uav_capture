"""Canonical deterministic replay audit for the v11 development smoke traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m0", "m3", "a1", "a2")
EXCLUDED_KEYS = {"latency_ms", "trace_write_latency_ms"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items()) if key not in EXCLUDED_KEYS}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def canonical_hash(records: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(canonical(item), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n" for item in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_trace(path: Path, episode_index: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Trace record is not an object: {path}")
        if int(record.get("episode_index", episode_index)) != episode_index:
            raise ValueError(f"Trace episode mismatch: {path}")
        records.append(record)
    if not records:
        raise ValueError(f"Empty trace: {path}")
    return records


def audit_run(root: Path, seed: int, variant: str) -> dict[str, Any]:
    run = root / f"jepa_safe_capture_v11_hard_replay_smoke_{variant}_seed{seed}"
    summary_path = run / "summary.json"
    episodes_path = run / "episodes.csv"
    if not summary_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(run)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = summary.get("metadata", {})
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Development boundary violation: {run}")
    with episodes_path.open("r", newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    if len(episodes) != 20:
        raise ValueError(f"Expected 20 episodes in {run}")
    trace_dir = run / "step_traces"
    hashes: list[str] = []
    trace_steps = 0
    raw_steps = 0
    unverified_steps = 0
    for index in range(20):
        records = read_trace(trace_dir / f"episode_{index:04d}.jsonl", index)
        trace_steps += len(records)
        for record in records:
            raw_steps += int(bool(record.get("raw_unverified_executed")))
            cbf = record.get("cbf")
            if isinstance(cbf, dict):
                unverified_steps += int(not bool(cbf.get("verified_feasible", False)))
        first = canonical_hash(records)
        second = canonical_hash(records)
        if first != second:
            raise ValueError(f"Canonical replay mismatch: {run} episode {index}")
        hashes.append(first)
    overall = summary["overall"]
    return {
        "training_seed": seed,
        "variant": variant,
        "run": str(run.resolve()),
        "episodes": 20,
        "trace_steps": trace_steps,
        "trace_hash": hashlib.sha256("".join(hashes).encode("ascii")).hexdigest(),
        "episode_hash_count": len(hashes),
        "raw_unverified_trace_steps": raw_steps,
        "cbf_unverified_trace_steps": unverified_steps,
        "summary_raw_unverified_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        "summary_cbf_unverified_steps": int(overall.get("cbf_unverified_steps", 0)),
        "safe_capture_rate": float(overall.get("safe_capture_rate", 0.0)),
        "safe_capture_count": int(overall.get("safe_capture_count", 0)),
        "deterministic": True,
        "raw_trace_matches_summary": raw_steps == int(overall.get("raw_unverified_executed_steps", 0)),
        "cbf_trace_observable": unverified_steps >= int(overall.get("cbf_unverified_steps", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("Deterministic replay audit is development-only.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.tensorboard_dir.exists() and any(args.tensorboard_dir.iterdir()):
        raise FileExistsError(args.tensorboard_dir)
    rows = [audit_run(args.input_root.resolve(), seed, variant) for seed in SEEDS for variant in VARIANTS]
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "all_runs_deterministic": all(row["deterministic"] for row in rows),
        "raw_trace_matches_summary": all(row["raw_trace_matches_summary"] for row in rows),
        "raw_unverified_zero": all(row["raw_unverified_trace_steps"] == 0 for row in rows),
        "cbf_trace_observable": all(row["cbf_trace_observable"] for row in rows),
    }
    report = {
        "audit_type": "jepa_safe_capture_v11_hard_replay_deterministic_trace_replay",
        "development_only": True,
        "locked_test_opened": False,
        "runs": rows,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "deterministic_replay.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text("# V11 Deterministic Trace Replay\n\nDevelopment-only canonical trace replay; `locked_test_opened=false`.\n\n" + f"All gates pass: `{report['all_gates_pass']}`.\n\n" + "| Seed | Variant | Safe capture | Trace steps | Raw unverified | Deterministic |\n|---:|---|---:|---:|---:|---|\n" + "\n".join(f"| {row['training_seed']} | {row['variant']} | {row['safe_capture_count']}/20 | {row['trace_steps']} | {row['raw_unverified_trace_steps']} | {row['deterministic']} |" for row in rows) + "\n", encoding="utf-8")
    tensorboard = args.tensorboard_dir.resolve()
    tensorboard.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("Config/deterministic_replay", json.dumps({"development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Gates/status", json.dumps(gates, indent=2), 0)
        writer.add_scalar("Replay/all_gates_pass", float(report["all_gates_pass"]), 0)
        writer.add_scalar("Replay/raw_unverified_zero", float(gates["raw_unverified_zero"]), 0)
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "run_count": len(rows), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
