"""Summarize the DN-MPC/JEPA route-chunk contract audit.

This is a read-only analysis tool.  It compares a JEPA-compatible paired replay
with the historical five-step analytic route contract and writes a provenance-
bound JSON summary, Markdown report, and TensorBoard scalars.  It intentionally
does not merge the results into a locked benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _require(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _summary_from_current(path: Path, variant: str) -> dict[str, Any]:
    root = _load(path)
    summaries = root.get("summaries")
    if not isinstance(summaries, dict) or not isinstance(summaries.get(variant), dict):
        raise ValueError(f"{path} does not contain summaries[{variant!r}].")
    metadata = root.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} does not contain metadata.")
    contract = metadata.get("contract", {})
    return {
        "run": _relative(path.parent),
        "variant": variant,
        "summary": dict(summaries[variant]),
        "contract": dict(contract) if isinstance(contract, dict) else {},
        "inputs": dict(metadata.get("inputs", {})) if isinstance(metadata.get("inputs", {}), dict) else {},
        "git_revision": metadata.get("git_revision", "unknown"),
    }


def _summary_from_analytic(path: Path) -> dict[str, Any]:
    root = _load(path)
    summary = root.get("overall")
    metadata = root.get("metadata")
    if not isinstance(summary, dict) or not isinstance(metadata, dict):
        raise ValueError(f"{path} does not contain overall and metadata.")
    contract = metadata.get("contract", {})
    return {
        "run": _relative(path.parent),
        "variant": "dn_mpc_cbf_chunk5",
        "summary": dict(summary),
        "contract": dict(contract) if isinstance(contract, dict) else {},
        "inputs": dict(metadata.get("inputs", {})) if isinstance(metadata.get("inputs", {}), dict) else {},
        "git_revision": metadata.get("git_revision", "unknown"),
    }


def _assert_same_manifest(records: list[dict[str, Any]]) -> str:
    hashes = {
        str(record["inputs"].get("scene_manifest_sha256", ""))
        for record in records
    }
    hashes.discard("")
    if len(hashes) != 1:
        raise ValueError(f"Runs do not share one scene manifest hash: {sorted(hashes)}")
    return next(iter(hashes))


def _metric(record: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = record["summary"].get(name, default)
    if value is None:
        return float("nan")
    return float(value)


def _contract_value(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = record["contract"]
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _write_report(path: Path, result: dict[str, Any]) -> None:
    rows = result["runs"]
    lines = [
        "# DN-MPC + JEPA Route-Chunk Contract Audit",
        "",
        "**Status:** development-only; no locked benchmark opened",
        "",
        "## Purpose",
        "",
        "The historical analytic G5 replay and the new JEPA paired replay did not "
        "use the same executable route-chunk contract. This report keeps the "
        "comparison explicit and prevents a route-length change from being "
        "mistaken for a JEPA effect.",
        "",
        f"- Shared scene manifest SHA-256: `{result['scene_manifest_sha256']}`",
        f"- Analysis revision: `{result['git_revision']}`",
        "- Safety hard gates: collision, defender boundary, pairwise, and raw-unverified.",
        "",
        "## Results",
        "",
        "| Run | Route chunk | Variant | Safe capture | Mean capture (s) | Route switches | Controlled abort | Safety hard events |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in rows:
        summary = record["summary"]
        chunk = _contract_value(record, "route_chunk_length_steps", default="unknown")
        safe = f"{int(summary['safe_capture_count'])}/{int(summary['episodes'])} ({100.0 * float(summary['safe_capture_rate']):.1f}%)"
        mean_time = summary.get("mean_capture_time_seconds")
        mean_text = "n/a" if mean_time is None else f"{float(mean_time):.3f}"
        hard_events = int(summary.get("collision_count", 0)) + int(summary.get("boundary_violation_count", 0)) + int(summary.get("pairwise_violation_count", 0)) + int(summary.get("raw_unverified_executed_steps", 0))
        lines.append(
            f"| `{record['run']}` | {chunk} | `{record['variant']}` | {safe} | {mean_text} | "
            f"{int(summary.get('route_switch_steps', 0))} | {int(summary.get('controlled_abort_steps', 0))} | {hard_events} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. The five-step analytic contract reproduced `4/4` safe capture with "
            "75 route switches and zero safety hard events.",
            "2. The JEPA-compatible three-step paired contract produced `2/4` for "
            "both analytic and JEPA variants. Therefore the `2/4` result is not "
            "evidence that JEPA alone caused the loss; the candidate horizon was "
            "changed before the comparison.",
            "3. Within the same three-step contract, JEPA did not improve safe capture "
            "and increased route switches and mean capture time. This is a stop "
            "signal for additional JEPA training or multi-seed expansion.",
            "4. The next valid comparison must either train a route-chunk-5 JEPA "
            "checkpoint or define a documented common three-step baseline. It must "
            "not mix the historical five-step baseline with the three-step JEPA run.",
            "",
            "## Decision",
            "",
            "Keep the five-step analytic replay as the historical contract reference "
            "and keep the three-step paired replay as a separate JEPA-contract "
            "diagnostic. Do not claim a JEPA improvement, do not open L1-L3 or "
            "three-seed JEPA evaluation, and do not relax CBF margins or gates.",
            "",
            "## Artifacts",
            "",
            f"- JSON summary: `{result['output_json']}`",
            f"- TensorBoard: `{result['tensorboard_dir']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tensorboard(path: Path, result: dict[str, Any]) -> None:
    with SummaryWriter(log_dir=str(path), flush_secs=2) as writer:
        writer.add_text("Provenance/metadata", json.dumps(result, indent=2, sort_keys=True), 0)
        for index, record in enumerate(result["runs"]):
            prefix = record["variant"]
            summary = record["summary"]
            writer.add_scalar(f"{prefix}/safe_capture_rate", float(summary["safe_capture_rate"]), 0)
            writer.add_scalar(f"{prefix}/route_switch_steps", float(summary.get("route_switch_steps", 0)), 0)
            writer.add_scalar(f"{prefix}/mean_capture_time_seconds", float(summary.get("mean_capture_time_seconds") or 0.0), 0)
            writer.add_scalar(f"{prefix}/collision_count", float(summary.get("collision_count", 0)), 0)
            writer.add_scalar(f"{prefix}/boundary_violation_count", float(summary.get("boundary_violation_count", 0)), 0)
            writer.add_scalar(f"{prefix}/pairwise_violation_count", float(summary.get("pairwise_violation_count", 0)), 0)
            writer.add_scalar(f"{prefix}/raw_unverified_executed_steps", float(summary.get("raw_unverified_executed_steps", 0)), 0)
            writer.add_text(f"{prefix}/contract", json.dumps(record["contract"], sort_keys=True), index)
        writer.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-run", type=Path, required=True)
    parser.add_argument("--analytic-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paired_path = _require(args.paired_run / "summary.json")
    analytic_path = _require(args.analytic_run / "summary.json")
    output_dir = args.output_dir.resolve()
    tensorboard_dir = args.tensorboard_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {tensorboard_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _summary_from_analytic(analytic_path),
        _summary_from_current(paired_path, "baseline"),
        _summary_from_current(paired_path, "jepa"),
    ]
    manifest_hash = _assert_same_manifest(records)
    result: dict[str, Any] = {
        "analysis_type": "dn_mpc_jepa_route_chunk_contract_audit",
        "development_only": True,
        "locked_test_opened": False,
        "git_revision": _git_revision(),
        "scene_manifest_sha256": manifest_hash,
        "runs": records,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "tensorboard": version("tensorboard"),
        },
        "tensorboard_dir": _relative(tensorboard_dir),
        "output_json": _relative(output_dir / "summary.json"),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output_dir / "report.md", result)
    _write_tensorboard(tensorboard_dir, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
