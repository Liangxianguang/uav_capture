"""Validate the safe-capture-first JEPA system protocol and frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    _mapping(protocol, "protocol")
    if protocol.get("protocol_name") != "jepa_safe_capture_system_v2":
        raise ValueError("Unexpected safe-capture protocol name.")
    if int(protocol.get("protocol_version", -1)) != 2:
        raise ValueError("Unexpected safe-capture protocol version.")
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("Safe-capture protocol must remain closed development-only.")

    objective = _mapping(protocol.get("objective"), "objective")
    if objective.get("world_model_role") != "candidate_trajectory_evaluator_only":
        raise ValueError("World model must be an evaluator only.")
    if objective.get("primary_endpoint") != "safe_capture":
        raise ValueError("safe_capture must be the primary endpoint.")
    if objective.get("capture_time_is_rejection_gate") is not False:
        raise ValueError("Capture time cannot be an automatic rejection gate.")

    candidate = _mapping(protocol.get("candidate_contract"), "candidate_contract")
    if int(candidate.get("count", -1)) != 5 or candidate.get("jepa_can_generate_final_action") is not False:
        raise ValueError("Candidate contract must contain five candidates and forbid direct JEPA actions.")
    if candidate.get("candidate_must_be_dynamics_feasible") is not True:
        raise ValueError("Candidates must pass dynamics feasibility before JEPA.")
    if candidate.get("execute_only_first_step_then_reobserve") is not True:
        raise ValueError("Rolling-horizon first-step execution is required.")

    jepa = _mapping(protocol.get("jepa_contract"), "jepa_contract")
    if jepa.get("uses_online_target_truth") is not False or jepa.get("updates_actor_online") is not False:
        raise ValueError("JEPA cannot use online target truth or update the actor.")

    ledger = _mapping(protocol.get("reliability_ledger"), "reliability_ledger")
    if ledger.get("source_split") != "calibration_only" or ledger.get("checkpoint_hash_bound") is not True:
        raise ValueError("Ledger must be calibration-only and checkpoint-bound.")
    if ledger.get("immutable_after_calibration") is not True:
        raise ValueError("Ledger must be immutable after calibration.")
    if set(ledger.get("states", [])) != {"trusted", "fallback_nominal", "safe_hold"}:
        raise ValueError("Ledger must expose trusted, fallback_nominal, and safe_hold states.")

    cbf = _mapping(protocol.get("cbf_contract"), "cbf_contract")
    if cbf.get("enabled_for_baseline_and_candidate") is not True or cbf.get("final_safety_filter") is not True:
        raise ValueError("CBF must be enabled for both paths and remain the final filter.")
    required_constraints = {
        "obstacle_separation",
        "inter_agent_separation",
        "world_boundary",
        "altitude",
        "speed",
        "acceleration",
        "target_capture_approach",
    }
    if not required_constraints.issubset(set(cbf.get("constraints", []))):
        raise ValueError("CBF contract is missing a required safety constraint.")
    if cbf.get("qp_infeasible_action") != "safe_hold_then_nominal_cbf":
        raise ValueError("QP infeasibility must use the deterministic safe-hold fallback.")
    if cbf.get("stale_observation_action") != "fallback_nominal_then_cbf":
        raise ValueError("Stale observations must fall back through CBF.")

    evaluation = _mapping(protocol.get("evaluation"), "evaluation")
    if evaluation.get("training_seeds") != [20260911, 20260912, 20260913]:
        raise ValueError("The three declared training seeds are part of the protocol contract.")
    gates = _mapping(protocol.get("decision_gates"), "decision_gates")
    safety = _mapping(gates.get("safety_hard_gate"), "decision_gates.safety_hard_gate")
    if safety.get("candidate_collision_count_max") != 0 or safety.get("candidate_boundary_violation_count_max") != 0:
        raise ValueError("Safety hard gate must forbid collision and boundary events.")

    training = _mapping(protocol.get("training"), "training")
    tensorboard = _mapping(training.get("tensorboard"), "training.tensorboard")
    if tensorboard.get("required") is not True or tensorboard.get("log_config_text") is not True:
        raise ValueError("TensorBoard configuration and provenance logging are required.")
    if protocol.get("provenance", {}).get("require_git_commit_per_phase") is not True:
        raise ValueError("Each phase must have a git commit.")
    return protocol


def verify_frozen_inputs(protocol: dict[str, Any], project_root: Path) -> dict[str, Any]:
    inputs = _mapping(protocol.get("frozen_inputs"), "frozen_inputs")
    artifacts: dict[str, Any] = {}
    for name, relative in inputs.items():
        path = (project_root / str(relative)).resolve()
        exists = path.is_file()
        artifacts[name] = {
            "path": str(path),
            "exists": exists,
            "sha256": sha256(path) if exists else None,
            "bytes": path.stat().st_size if exists else None,
        }
    return artifacts


def verify(protocol_path: Path, project_root: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    artifacts = verify_frozen_inputs(protocol, project_root)
    return {
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "protocol_name": protocol["protocol_name"],
        "protocol_version": protocol["protocol_version"],
        "locked_test_opened": protocol["locked_test_opened"],
        "primary_endpoint": protocol["objective"]["primary_endpoint"],
        "world_model_role": protocol["objective"]["world_model_role"],
        "cbf_final_safety_filter": protocol["cbf_contract"]["final_safety_filter"],
        "all_frozen_inputs_exist": all(item["exists"] for item in artifacts.values()),
        "frozen_inputs": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/jepa_safe_capture_v2_protocol.yaml")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.protocol.resolve(), args.project_root.resolve())
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_frozen_inputs_exist"]:
        raise SystemExit("Safe-capture frozen input verification failed.")


if __name__ == "__main__":
    main()
