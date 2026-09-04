"""Replay every M3 episode in the frozen WP-7 tie3 paired block.

This is a read-only trace audit.  It does not resimulate the environment or
load target future ground truth.  The M3 traces are paired with the aggregate
M0 outcome at the same seed and episode index, then written twice using
canonical JSON.  The two hashes must agree for every episode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aggregate_jepa_safe_capture_v2_paired import canonical_scene_manifest_sha256, sha256  # noqa: E402
from index_jepa_safe_capture_failures import read_trace  # noqa: E402
from replay_jepa_safe_capture_failures import (  # noqa: E402
    _as_bool,
    _as_int,
    _hash_manifest,
    _read_json,
    _reduce_trace,
    _validate_episode_source,
    _validate_run,
    _write_canonical_jsonl,
)


REPLAY_TYPE = "jepa_safe_capture_wp8_tie3_paired_failure_replay"
PAIR_LABELS = ("degraded", "improved", "tied")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-index", type=Path, required=True)
    parser.add_argument("--aggregate-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _canonical_pair_key(seed: Any, episode_index: Any) -> tuple[int, int]:
    return _as_int(seed, "training_seed"), _as_int(episode_index, "episode_index")


def _load_failure_index(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = path.resolve()
    report = _read_json(path)
    if report.get("index_type") != "jepa_safe_capture_v3_wp1_failure_index":
        raise ValueError(f"Unexpected failure-index type: {report.get('index_type')!r}")
    if report.get("input_format") not in {"v2", "v3"}:
        raise ValueError(f"Unsupported failure-index format: {report.get('input_format')!r}")
    if report.get("development_only") is not True or report.get("locked_test_opened") is not False:
        raise ValueError("Failure index crossed the locked-test boundary")
    if not isinstance(report.get("runs"), list) or not isinstance(report.get("rows"), list):
        raise ValueError("Failure index must contain runs and rows lists")
    return report, {"failure_index_json_sha256": sha256(path)}


def _load_runs(report: Mapping[str, Any]) -> dict[tuple[int, str], Any]:
    runs: dict[tuple[int, str], Any] = {}
    for raw in report["runs"]:
        if not isinstance(raw, Mapping):
            raise ValueError("Failure-index run record is not an object")
        run = _validate_run(raw)
        if run.key in runs:
            raise ValueError(f"Duplicate source run identity: {run.key}")
        runs[run.key] = run
    return runs


def _load_pairs(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    summary = _read_json(path.resolve())
    if summary.get("stage") != "full" or summary.get("locked_test_opened") is not False:
        raise ValueError("Aggregate summary is not a development-only full result")
    if summary.get("not_a_locked_test") is not True:
        raise ValueError("Aggregate summary crossed the locked-test boundary")
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for comparison in summary.get("paired_comparisons", []):
        if not isinstance(comparison, Mapping):
            raise ValueError("Invalid paired comparison record")
        if comparison.get("base_variant") != "m0" or comparison.get("candidate_variant") != "m3":
            continue
        seed = _as_int(comparison.get("training_seed"), "paired comparison seed")
        raw_pairs = comparison.get("pairs")
        if not isinstance(raw_pairs, list) or len(raw_pairs) != 40:
            raise ValueError(f"Expected 40 M3 pairs for seed {seed}")
        for raw in raw_pairs:
            if not isinstance(raw, Mapping):
                raise ValueError("Invalid episode pair")
            key = _canonical_pair_key(seed, raw.get("episode_index"))
            if key in pairs:
                raise ValueError(f"Duplicate M3 pair: {key}")
            delta = _as_int(raw.get("delta"), "paired delta")
            if delta not in {-1, 0, 1}:
                raise ValueError(f"Unexpected paired delta for {key}: {delta}")
            pairs[key] = {
                "training_seed": key[0],
                "episode_index": key[1],
                "episode_seed": _as_int(raw.get("episode_seed"), "episode_seed"),
                "base_safe_capture": _as_bool(raw.get("base_safe_capture")),
                "candidate_safe_capture": _as_bool(raw.get("candidate_safe_capture")),
                "delta": delta,
                "pair_label": "improved" if delta > 0 else "degraded" if delta < 0 else "tied",
            }
    if len(pairs) != 120:
        raise ValueError(f"Expected 120 M3 paired episodes, found {len(pairs)}")
    counts = Counter(item["pair_label"] for item in pairs.values())
    if counts != Counter({"tied": 80, "degraded": 30, "improved": 10}):
        raise ValueError(f"Unexpected pair label counts: {dict(counts)}")
    return pairs


def _validate_pair_rows(
    report: Mapping[str, Any], pairs: Mapping[tuple[int, int], Mapping[str, Any]]
) -> dict[tuple[int, int], dict[str, Any]]:
    rows = report["rows"]
    m3_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or str(raw.get("variant")) != "m3":
            continue
        key = _canonical_pair_key(raw.get("training_seed"), raw.get("episode_index"))
        if key in m3_rows:
            raise ValueError(f"Duplicate M3 failure-index row: {key}")
        pair = pairs.get(key)
        if pair is None:
            raise ValueError(f"M3 failure-index row has no aggregate pair: {key}")
        if _as_int(raw.get("episode_seed"), "failure-index episode_seed") != pair["episode_seed"]:
            raise ValueError(f"Episode seed mismatch between index and aggregate: {key}")
        if _as_bool(raw.get("safe_capture")) != pair["candidate_safe_capture"]:
            raise ValueError(f"M3 safe-capture mismatch between index and aggregate: {key}")
        if _as_bool(raw.get("baseline_safe_capture")) != pair["base_safe_capture"]:
            raise ValueError(f"M0 baseline mismatch between index and aggregate: {key}")
        m3_rows[key] = dict(raw)
    if set(m3_rows) != set(pairs):
        raise ValueError(f"M3 failure-index/aggregate key mismatch: {set(pairs) ^ set(m3_rows)}")
    return m3_rows


def _replay_one(
    *,
    pair: Mapping[str, Any],
    row: Mapping[str, Any],
    run: Any,
    output_dir: Path,
    repeats: int,
) -> dict[str, Any]:
    episode_index = _as_int(pair["episode_index"], "episode_index")
    trace_path, scene_hash = _validate_episode_source(row, run)
    trace = read_trace(trace_path, episode_index)
    identifier = f"{pair['training_seed']}:m3:{episode_index:04d}"
    payload = _reduce_trace(
        trace,
        row,
        identifier=identifier,
        categories=[f"paired_{pair['pair_label']}"],
        scene_hash=scene_hash,
    )
    pair_meta = {
        "base_safe_capture": bool(pair["base_safe_capture"]),
        "candidate_safe_capture": bool(pair["candidate_safe_capture"]),
        "delta": int(pair["delta"]),
        "pair_label": str(pair["pair_label"]),
    }
    for record in payload:
        record["paired_outcome"] = pair_meta
    replay_dir = output_dir / "replays" / identifier.replace(":", "_")
    replay_dir.mkdir(parents=True, exist_ok=False)
    hashes = [
        _write_canonical_jsonl(replay_dir / f"replay_{repeat}.jsonl", payload)
        for repeat in range(1, repeats + 1)
    ]
    if len(set(hashes)) != 1:
        raise ValueError(f"Non-deterministic replay hash for {identifier}: {hashes}")
    return {
        "identifier": identifier,
        "training_seed": int(pair["training_seed"]),
        "variant": "m3",
        "episode_index": episode_index,
        "episode_seed": int(pair["episode_seed"]),
        **pair_meta,
        "trace_steps": len(payload),
        "repeat_sha256": hashes,
        "repeat_deterministic": True,
        "cbf_unverified_steps": sum(int(record["cbf"]["unverified"]) for record in payload),
        "scene_hash": scene_hash,
    }


def _write_tensorboard(result: Mapping[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text(
            "Config/wp8_tie3_replay",
            json.dumps(
                {"replay_type": REPLAY_TYPE, "development_only": True, "locked_test_opened": False, "repeats": result["repeats"]},
                indent=2,
            ),
            0,
        )
        writer.add_text("Provenance/inputs", json.dumps(result["inputs"], indent=2), 0)
        writer.add_text("Provenance/source_runs", json.dumps(result["source_runs"], indent=2), 0)
        writer.add_text("Selection/pair_contract", json.dumps(result["pair_counts"], indent=2), 0)
        for label in PAIR_LABELS:
            writer.add_scalar(f"Pairs/{label}", float(result["pair_counts"].get(label, 0)), 0)
        for index, episode in enumerate(result["episodes"]):
            writer.add_scalar(f"Replay/episode_{index:03d}/trace_steps", float(episode["trace_steps"]), 0)
            writer.add_scalar(f"Replay/episode_{index:03d}/delta", float(episode["delta"]), 0)
            writer.add_scalar(f"Replay/episode_{index:03d}/repeat_deterministic", 1.0, 0)
            writer.add_scalar(f"Replay/episode_{index:03d}/cbf_unverified_steps", float(episode["cbf_unverified_steps"]), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/wp8_tie3_replay/text_summary",
        "Provenance/inputs/text_summary",
        "Provenance/source_runs/text_summary",
        "Selection/pair_contract/text_summary",
    }
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"WP-8 TensorBoard audit failed: missing_text={missing}, event_files={events}")
    return {
        "logdir": str(logdir),
        "event_files": events,
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_provenance": True,
    }


def replay_tie3(
    failure_index_path: Path,
    aggregate_summary_path: Path,
    output_dir: Path,
    tensorboard_logdir: Path,
    *,
    repeats: int = 2,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("WP-8 requires at least two deterministic replay copies")
    report, index_hashes = _load_failure_index(failure_index_path)
    pairs = _load_pairs(aggregate_summary_path)
    runs = _load_runs(report)
    rows = _validate_pair_rows(report, pairs)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    used_runs: set[tuple[int, str]] = set()
    for key in sorted(pairs):
        pair = pairs[key]
        run_key = (key[0], "m3")
        if run_key not in runs:
            raise ValueError(f"Missing validated M3 source run: {run_key}")
        episodes.append(_replay_one(pair=pair, row=rows[key], run=runs[run_key], output_dir=output_dir, repeats=repeats))
        used_runs.add(run_key)
    source_runs = []
    for run_key in sorted(used_runs):
        run = runs[run_key]
        source_runs.append({"training_seed": run_key[0], "variant": run_key[1], "path": str(run.path), **run.source_hashes, "canonical_manifest_sha256": run.canonical_manifest_sha256})
    pair_counts = dict(Counter(item["pair_label"] for item in episodes))
    result: dict[str, Any] = {
        "replay_type": REPLAY_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "repeats": repeats,
        "episode_count": len(episodes),
        "pair_counts": pair_counts,
        "inputs": {
            "failure_index": str(failure_index_path.resolve()),
            "failure_index_sha256": index_hashes["failure_index_json_sha256"],
            "aggregate_summary": str(aggregate_summary_path.resolve()),
            "aggregate_summary_sha256": sha256(aggregate_summary_path.resolve()),
            "canonical_scene_manifest_sha256": canonical_scene_manifest_sha256(next(iter(runs.values())).manifest_path),
        },
        "source_runs": source_runs,
        "episodes": episodes,
        "provenance": {
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source_hashes": {"scripts/replay_jepa_safe_capture_wp8_tie3.py": sha256(Path(__file__).resolve())},
        },
    }
    (output_dir / "replay_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "replay_manifest.json").write_text(json.dumps({key: result[key] for key in ("replay_type", "development_only", "locked_test_opened", "inputs", "source_runs", "provenance")}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output_dir / "replay_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["identifier", "training_seed", "variant", "episode_index", "episode_seed", "pair_label", "delta", "base_safe_capture", "candidate_safe_capture", "trace_steps", "repeat_deterministic", "cbf_unverified_steps", "scene_hash", "repeat_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            writer.writerow({**episode, "repeat_sha256": json.dumps(episode["repeat_sha256"], separators=(",", ":"))})
    lines = [
        "# WP-8 tie3 Paired Deterministic Failure Replay",
        "",
        "**Status:** development-only; `locked_test_opened=false`",
        "**Method:** read-only canonical derivation from all frozen M3 source traces; no environment rollout.",
        f"**Episodes:** {len(episodes)}; **repeats:** {repeats}",
        "",
        "## Pair Contract",
        "",
        "| Pair label | Episodes | Meaning |",
        "|---|---:|---|",
        "| `degraded` | %d | M0 safe, M3 unsafe |" % pair_counts.get("degraded", 0),
        "| `improved` | %d | M0 unsafe, M3 safe |" % pair_counts.get("improved", 0),
        "| `tied` | %d | Same settled safe-capture outcome |" % pair_counts.get("tied", 0),
        "",
        "## Determinism",
        "",
        f"All {len(episodes)} episodes have identical repeat hashes and finite action fields.",
        "Source summaries, provenance, scene manifests, episode seeds, candidate ranking, ledger state, CBF status, fallback and termination fields were revalidated.",
        "No target future ground truth or unrecorded causal variable is inferred.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["tensorboard"] = _write_tensorboard(result, tensorboard_logdir)
    (output_dir / "replay_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "hash_manifest.json").write_text(json.dumps(_hash_manifest(output_dir), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("WP-8 requires --development-only")
    result = replay_tie3(
        args.failure_index,
        args.aggregate_summary,
        args.output_dir,
        args.tensorboard_logdir,
        repeats=args.repeats,
    )
    print(json.dumps({"episodes": result["episode_count"], "pair_counts": result["pair_counts"], "tensorboard": result["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
