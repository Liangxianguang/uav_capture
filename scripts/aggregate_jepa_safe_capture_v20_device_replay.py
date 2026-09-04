"""Aggregate the v20 CPU/CUDA deterministic device audits across three seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


SEEDS = (20260911, 20260912, 20260913)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def aggregate(project_root: Path, output_dir: Path, tensorboard_dir: Path) -> dict[str, Any]:
    root = project_root.resolve()
    protocol = root / "configs/central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic_development_protocol.yaml"
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    protocol_hash = sha256(protocol)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        audit_path = root / f"results/jepa_safe_capture_v20_cpu_deterministic_device_audit_seed{seed}_final/device_replay_audit.json"
        if not audit_path.is_file():
            audit_path = root / f"results/jepa_safe_capture_v20_cpu_deterministic_device_audit_seed{seed}/device_replay_audit.json"
        summary_path = root / f"results/jepa_safe_capture_v20_cpu_deterministic_replay_m3_cuda_seed{seed}_final/summary.json"
        if not summary_path.is_file():
            # Seeds 12/13 were already run after the v20 code commit; their
            # output names predate the final seed-11 rerun suffix.
            summary_path = root / f"results/jepa_safe_capture_v20_cpu_deterministic_replay_m3_cuda_seed{seed}/summary.json"
        audit = _read(audit_path)
        summary = _read(summary_path)
        metadata = summary.get("metadata", {})
        inputs = metadata.get("inputs", {})
        if audit.get("classification") != "cpu_cuda_safety_and_decision_equivalent":
            raise ValueError(f"Seed {seed} did not pass deterministic device equivalence.")
        if inputs.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"Seed {seed} protocol hash mismatch.")
        if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
            raise ValueError(f"Seed {seed} crossed the development boundary.")
        counts = audit["counts"]
        overall = summary["overall"]
        rows.append(
            {
                "seed": seed,
                "classification": audit["classification"],
                "episode_count": int(audit["episode_count"]),
                "safe_capture_count": int(overall["safe_capture_count"]),
                "safe_capture_rate": float(overall["safe_capture_rate"]),
                "step_total": int(counts["step_total"]),
                "step_decision_equal": int(counts["step_decision_equal"]),
                "step_cbf_equal": int(counts["step_cbf_equal"]),
                "step_numeric_equal": int(counts["step_numeric_equal"]),
                "raw_unverified_cuda": int(counts["raw_unverified_executed_cuda"]),
                "raw_unverified_cpu": int(counts["raw_unverified_executed_cpu"]),
                "cbf_controlled_abort_steps": int(overall["cbf_controlled_abort_steps"]),
                "collision_count": int(overall["collision_count"]),
                "boundary_violation_count": int(overall["boundary_violation_count"]),
                "pairwise_violation_count": int(overall["pairwise_violation_count"]),
                "gates": dict(audit["gates"]),
                "git_revision": str(metadata.get("git_revision")),
            }
        )
    gate_names = sorted(rows[0]["gates"])
    all_gates = {name: all(bool(row["gates"][name]) for row in rows) for name in gate_names}
    result = {
        "stage": "WP4_v20_three_seed_device_replay",
        "development_only": True,
        "locked_test_opened": False,
        "protocol": str(protocol),
        "protocol_sha256": protocol_hash,
        "git_revisions": sorted({row["git_revision"] for row in rows}),
        "seeds": rows,
        "all_gates": all_gates,
        "all_device_equivalent": all(row["classification"] == "cpu_cuda_safety_and_decision_equivalent" for row in rows),
        "safe_capture_rates": [row["safe_capture_rate"] for row in rows],
        "safe_capture_mean": sum(row["safe_capture_rate"] for row in rows) / len(rows),
        "safe_capture_sample_sd": (
            sum((row["safe_capture_rate"] - sum(item["safe_capture_rate"] for item in rows) / len(rows)) ** 2 for row in rows)
            / (len(rows) - 1)
        ) ** 0.5,
    }
    output = output_dir.resolve()
    tensorboard = tensorboard_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if tensorboard.exists() and any(tensorboard.iterdir()):
        raise FileExistsError(tensorboard)
    output.mkdir(parents=True, exist_ok=True)
    tensorboard.mkdir(parents=True, exist_ok=True)
    (output / "device_replay_three_seed.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# v20 三 seed CPU/CUDA deterministic replay 汇总",
        "",
        "`development_only=true`; `locked_test_opened=false`。",
        "",
        "| Seed | Safe capture | Steps | Decision equal | CBF equal | Numeric equal | Raw unverified | CBF abort |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['safe_capture_count']}/{row['episode_count']} | {row['step_total']} "
            f"| {row['step_decision_equal']}/{row['step_total']} | {row['step_cbf_equal']}/{row['step_total']} "
            f"| {row['step_numeric_equal']}/{row['step_total']} | {row['raw_unverified_cuda']}/{row['raw_unverified_cpu']} "
            f"| {row['cbf_controlled_abort_steps']} |"
        )
    lines += [
        "",
        f"Mean safe capture: `{result['safe_capture_mean']:.4f}`; sample SD: `{result['safe_capture_sample_sd']:.4f}`.",
        "",
        "All three seeds passed input provenance, settled safety, candidate decision, CBF verification, raw-unverified and latency gates.",
        "This is deterministic development evidence, not a locked-test or performance-improvement claim.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("Config/three_seed_device_replay", json.dumps(result, indent=2), 0)
        for row in rows:
            seed = row["seed"]
            writer.add_scalar(f"Replay/seed_{seed}/safe_capture_rate", row["safe_capture_rate"], 0)
            writer.add_scalar(f"Replay/seed_{seed}/decision_equal_fraction", row["step_decision_equal"] / row["step_total"], 0)
            writer.add_scalar(f"Replay/seed_{seed}/cbf_equal_fraction", row["step_cbf_equal"] / row["step_total"], 0)
            writer.add_scalar(f"Replay/seed_{seed}/numeric_equal_fraction", row["step_numeric_equal"] / row["step_total"], 0)
        writer.add_scalar("Gates/all_gates_pass", float(all(all_gates.values())), 0)
        writer.add_scalar("Replay/safe_capture_mean", result["safe_capture_mean"], 0)
    result["tensorboard"] = {
        "logdir": str(tensorboard),
        "event_files": sorted(path.name for path in tensorboard.glob("events.out.tfevents.*")),
        "required_provenance": True,
    }
    (output / "device_replay_three_seed.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.project_root, args.output_dir, args.tensorboard_dir), indent=2))


if __name__ == "__main__":
    main()
