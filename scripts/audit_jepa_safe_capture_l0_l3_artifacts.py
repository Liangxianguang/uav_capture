"""Audit the persisted L0-L3 paired development artifact matrix.

The evaluator and aggregate already validate the experiment contract.  This
read-only audit adds an explicit inventory check for the files that make the
run reproducible: all 21 runs, episode/trace counts, paired manifest hashes,
per-run TensorBoard events, and the aggregate event file.  It never opens a
locked-test split and never modifies an evaluator run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m0", "m1", "m2", "m3", "a1", "a2", "a3")
RUN_RE = re.compile(
    r"^jepa_safe_capture_l0_l3_paired_full_seed(?P<seed>\d+)_(?P<variant>m0|m1|m2|m3|a1|a2|a3)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def canonical_manifest_sha256(path: Path) -> str:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Manifest record is not an object: {path}")
        record = dict(value)
        record.pop("training_seed", None)
        records.append(record)
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for record in records
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_run_names() -> tuple[str, ...]:
    return tuple(
        f"jepa_safe_capture_l0_l3_paired_full_seed{seed}_{variant}"
        for seed in SEEDS
        for variant in VARIANTS
    )


def _require_files(run: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run}: missing {missing}")


def audit_run(run: Path, tensorboard_root: Path, expected_episodes: int) -> dict[str, Any]:
    match = RUN_RE.fullmatch(run.name)
    if match is None:
        raise ValueError(f"Unexpected run name: {run.name}")
    seed = int(match.group("seed"))
    variant = match.group("variant")
    if seed not in SEEDS:
        raise ValueError(f"Unexpected training seed: {seed}")
    _require_files(run, ("summary.json", "provenance.json", "episodes.csv", "scene_manifest.jsonl", "scenes.jsonl"))
    summary = read_json(run / "summary.json")
    provenance = read_json(run / "provenance.json")
    metadata = summary.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing metadata: {run / 'summary.json'}")
    for label, payload in (("summary", metadata), ("provenance", provenance)):
        if payload.get("development_only") is not True or payload.get("locked_test_opened") is not False:
            raise ValueError(f"{label} crossed locked-test boundary: {run}")
    if int(metadata.get("episodes", -1)) != expected_episodes:
        raise ValueError(f"Unexpected episode count in {run}: {metadata.get('episodes')}")
    declared = metadata.get("variant")
    if not isinstance(declared, dict) or declared.get("variant") != variant:
        raise ValueError(f"Variant metadata mismatch: {run}")
    if int(metadata.get("training_seed", -1)) != seed:
        raise ValueError(f"Training seed metadata mismatch: {run}")
    with (run / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_episodes:
        raise ValueError(f"Episode table count mismatch in {run}: {len(rows)}")
    indices = sorted(int(row["episode_index"]) for row in rows)
    if indices != list(range(expected_episodes)):
        raise ValueError(f"Episode indices are not contiguous: {run}")
    manifest = run / "scene_manifest.jsonl"
    declared_manifest = str(metadata.get("inputs", {}).get("scene_manifest_sha256", ""))
    if declared_manifest != sha256(manifest):
        raise ValueError(f"Manifest hash mismatch: {run}")
    trace_dir = run / "step_traces"
    traces = sorted(trace_dir.glob("episode_*.jsonl")) if trace_dir.is_dir() else []
    if len(traces) != expected_episodes or any(not path.stat().st_size for path in traces):
        raise ValueError(f"Step-trace inventory mismatch: {run}")
    tb_dir = (tensorboard_root / f"full_seed{seed}_{variant}").resolve()
    events = sorted(tb_dir.glob("events.out.tfevents.*")) if tb_dir.is_dir() else []
    if not events or any(not event.stat().st_size for event in events):
        raise ValueError(f"TensorBoard event missing/empty: {tb_dir}")
    declared_tb = metadata.get("tensorboard", {}).get("logdir")
    if declared_tb is None:
        declared_tb = metadata.get("tensorboard_dir")
    if Path(str(declared_tb)).resolve() != tb_dir:
        raise ValueError(f"TensorBoard provenance mismatch: {run}")
    overall = summary.get("overall", {})
    return {
        "run": run.name,
        "training_seed": seed,
        "variant": variant,
        "episodes": len(rows),
        "trace_count": len(traces),
        "scene_manifest_sha256": declared_manifest,
        "canonical_scene_manifest_sha256": canonical_manifest_sha256(manifest),
        "tensorboard_event_count": len(events),
        "tensorboard_logdir": str(tb_dir),
        "safe_capture_count": int(overall.get("safe_capture_count", 0)),
        "safe_capture_rate": float(overall.get("safe_capture_rate", 0.0)),
        "collision_count": int(overall.get("collision_count", 0)),
        "boundary_count": int(overall.get("boundary_violation_count", 0)),
        "pairwise_count": int(overall.get("pairwise_violation_count", 0)),
        "raw_unverified_steps": int(overall.get("raw_unverified_executed_steps", 0)),
    }


def audit_matrix(
    runs_root: Path,
    tensorboard_root: Path,
    aggregate_dir: Path,
    expected_episodes: int,
) -> dict[str, Any]:
    expected = set(expected_run_names())
    actual = {path.name for path in runs_root.iterdir() if path.is_dir() and RUN_RE.fullmatch(path.name)}
    if actual != expected:
        raise ValueError(f"Run matrix mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    runs = [audit_run(runs_root / name, tensorboard_root, expected_episodes) for name in sorted(expected)]
    canonical = {item["canonical_scene_manifest_sha256"] for item in runs}
    if len(canonical) != 1:
        raise ValueError(f"Paired canonical manifest mismatch: {sorted(canonical)}")
    aggregate_summary_path = aggregate_dir / "summary.json"
    aggregate_report_path = aggregate_dir / "report.md"
    if not aggregate_summary_path.is_file() or not aggregate_report_path.is_file():
        raise FileNotFoundError(f"Aggregate report missing under {aggregate_dir}")
    aggregate = read_json(aggregate_summary_path)
    if aggregate.get("stage") != "full" or aggregate.get("locked_test_opened") is not False:
        raise ValueError(f"Aggregate is not a development-only full result: {aggregate_dir}")
    if int(aggregate.get("episodes_per_run", -1)) != expected_episodes:
        raise ValueError("Aggregate episode count mismatch")
    if aggregate.get("canonical_scene_manifest_sha256") != next(iter(canonical)):
        raise ValueError("Aggregate canonical manifest mismatch")
    aggregate_tb = aggregate_dir / "tensorboard"
    aggregate_events = sorted(aggregate_tb.glob("events.out.tfevents.*")) if aggregate_tb.is_dir() else []
    if not aggregate_events or any(not event.stat().st_size for event in aggregate_events):
        raise ValueError(f"Aggregate TensorBoard event missing/empty: {aggregate_tb}")
    return {
        "audit_type": "jepa_safe_capture_l0_l3_artifact_matrix",
        "development_only": True,
        "locked_test_opened": False,
        "expected_run_count": len(expected),
        "run_count": len(runs),
        "expected_episodes_per_run": expected_episodes,
        "trace_count": sum(item["trace_count"] for item in runs),
        "tensorboard_run_count": sum(item["tensorboard_event_count"] > 0 for item in runs),
        "aggregate_tensorboard_event_count": len(aggregate_events),
        "canonical_scene_manifest_sha256": next(iter(canonical)),
        "safety_hard_gate": all(
            item[field] == 0
            for item in runs
            for field in ("collision_count", "boundary_count", "pairwise_count", "raw_unverified_steps")
            if item["variant"] != "a3"
        ),
        "runs": runs,
        "aggregate_summary_sha256": sha256(aggregate_summary_path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# L0-L3 R2 Artifact Audit",
        "",
        "Read-only audit of the development-only paired artifact matrix; `locked_test_opened=false`.",
        "",
        f"- Runs: `{report['run_count']}/{report['expected_run_count']}`",
        f"- Episodes per run: `{report['expected_episodes_per_run']}`",
        f"- Step traces: `{report['trace_count']}`",
        f"- Per-run TensorBoard directories with events: `{report['tensorboard_run_count']}`",
        f"- Aggregate TensorBoard events: `{report['aggregate_tensorboard_event_count']}`",
        f"- Canonical scene manifest SHA-256: `{report['canonical_scene_manifest_sha256']}`",
        f"- Safety hard gate (A3 excluded): **{'PASS' if report['safety_hard_gate'] else 'FAIL'}**",
        "",
        "| Seed | Variant | Episodes | Safe capture | Collision | Boundary | Pairwise | Raw-unverified |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["runs"]:
        lines.append(
            f"| {item['training_seed']} | {item['variant'].upper()} | {item['episodes']} | "
            f"{item['safe_capture_count']}/{item['episodes']} ({item['safe_capture_rate']:.1%}) | "
            f"{item['collision_count']} | {item['boundary_count']} | {item['pairwise_count']} | "
            f"{item['raw_unverified_steps']} |"
        )
    lines += [
        "",
        "This audit does not reinterpret development performance and does not open or modify a locked-test split.",
    ]
    return "\n".join(lines) + "\n"


def write_tensorboard(report: dict[str, Any], logdir: Path) -> dict[str, Any]:
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=5) as writer:
        writer.add_text("Audit/provenance", json.dumps({k: report[k] for k in report if k != "runs"}, indent=2), 0)
        writer.add_scalar("Audit/run_count", report["run_count"], 0)
        writer.add_scalar("Audit/trace_count", report["trace_count"], 0)
        writer.add_scalar("Audit/tensorboard_run_count", report["tensorboard_run_count"], 0)
        writer.add_scalar("Audit/safety_hard_gate", float(report["safety_hard_gate"]), 0)
        for index, item in enumerate(report["runs"]):
            writer.add_scalar(f"SafeCapture/{item['variant']}/rate", item["safe_capture_rate"], index)
            writer.add_scalar(f"Safety/{item['variant']}/raw_unverified_steps", item["raw_unverified_steps"], index)
        writer.flush()
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if not events:
        raise RuntimeError(f"No TensorBoard event created: {logdir}")
    return {"logdir": str(logdir.resolve()), "event_files": events, "required_provenance": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--tensorboard-root", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-run", type=int, default=64)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("Artifact audit requires --development-only.")
    if args.episodes_per_run <= 0:
        raise ValueError("episodes-per-run must be positive.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    report = audit_matrix(
        args.runs_root.resolve(), args.tensorboard_root.resolve(), args.aggregate_dir.resolve(), args.episodes_per_run
    )
    output.mkdir(parents=True, exist_ok=True)
    report["tensorboard"] = write_tensorboard(report, output / "tensorboard")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("run_count", "trace_count", "tensorboard_run_count", "safety_hard_gate", "tensorboard")}, indent=2))


if __name__ == "__main__":
    main()
