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
    protocol_name = str(protocol.get("protocol_name", ""))
    if (
        protocol_name.startswith("central_random_mixed_obstacle_s3_v5_v12_calibrated_clearance")
        or protocol_name.startswith("central_random_mixed_obstacle_s3_v5_v13_fixedpoint")
        or protocol_name.startswith("central_random_mixed_obstacle_s3_v5_v14_fixedpoint")
        or protocol_name.startswith("central_random_mixed_obstacle_s3_v5_v15_fixedpoint")
        or protocol_name.startswith("central_random_mixed_obstacle_s3_v5_v16_fixedpoint")
        or protocol_name.startswith("central_random_mixed_obstacle_s3_v5_v17_fixedpoint")
        or protocol_name.startswith("central_random_mixed_obstacle_s3_v5_v18_fixedpoint")
        or protocol_name.startswith("central_random_mixed_obstacle_s3_v5_v19_cpu_ranker")
        or protocol_name.startswith("central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic")
    ):
        _validate_v12_protocol(protocol)
        return protocol
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


def _validate_v12_protocol(protocol: dict[str, Any]) -> None:
    """Validate the compact v12 development protocol contract.

    v12 is an evaluation protocol, so its schema deliberately differs from
    the original system-wide v2 protocol.  Keep the safety invariants strict
    while accepting the versioned v12 layout.
    """

    protocol_version = int(protocol.get("protocol_version", -1))
    if protocol_version not in {12, 13, 14, 15, 16, 17, 18, 19, 20}:
        raise ValueError("Unexpected calibrated safe-capture protocol version.")
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("Calibrated protocol must remain closed development-only.")

    model = _mapping(protocol.get("model_contract"), "model_contract")
    if model.get("world_model_role") != "candidate_trajectory_evaluator_only":
        raise ValueError("v12 world model must be an evaluator only.")
    if model.get("candidate_ranking_only") is not True or model.get("cbf_is_final_safety_filter") is not True:
        raise ValueError("v12 must rank candidates and use CBF as the final filter.")

    candidate = _mapping(protocol.get("candidate_contract"), "candidate_contract")
    if int(candidate.get("candidate_count", -1)) != 5:
        raise ValueError("v12 candidate contract must contain five candidates.")
    if int(candidate.get("action_chunk_length_steps", -1)) != 3:
        raise ValueError("v12 action chunks must contain three steps.")
    if candidate.get("execute_first_step_then_replan") is not True:
        raise ValueError("v12 requires first-step execution followed by replanning.")
    if candidate.get("require_finite_reachability_before_jepa") is not True:
        raise ValueError("v12 candidates must pass finite reachability checks.")
    if candidate.get("nominal_anchor_required") is not True:
        raise ValueError("v12 must retain the nominal candidate anchor.")
    if protocol_version >= 13:
        action_quantum = float(candidate.get("action_comparison_quantum_mps", -1.0))
        if action_quantum <= 0.0 or action_quantum != action_quantum or action_quantum in (float("inf"), float("-inf")):
            raise ValueError("v13 must declare a positive finite action comparison quantum.")

    ranking = _mapping(protocol.get("candidate_ranking"), "candidate_ranking")
    if float(ranking.get("score_comparison_quantum_m", 0.0)) <= 0.0:
        raise ValueError("v12 must declare a positive comparison quantum.")
    safety_band = float(ranking.get("score_comparison_safety_band_m", 0.0))
    if safety_band < 0.0 or safety_band != safety_band or safety_band in (float("inf"), float("-inf")):
        raise ValueError("v12 abstention safety band must be finite and non-negative.")
    if float(ranking.get("top_two_abstention_margin_m", -1.0)) < 0.0:
        raise ValueError("v12 abstention margin must be non-negative.")
    if ranking.get("cbf_margin_changed") is not False:
        raise ValueError("v12 cannot change the CBF margin.")
    if protocol_version >= 13:
        expected_profile = {
            13: "p13_fixedpoint_v1",
            14: "p14_fixedpoint_robust_v1",
            15: "p15_fixedpoint_robust_v1",
            16: "p16_fixedpoint_robust_v1",
            17: "p17_fixedpoint_robust_v1",
            18: "p18_fixedpoint_robust_v1",
            19: "p19_cpu_ranker_v1",
            20: "p20_cpu_deterministic_v1",
        }[protocol_version]
        if ranking.get("profile") != expected_profile:
            raise ValueError(f"v{protocol_version} must declare the fixed-point ranking profile.")
        if ranking.get("fixed_point_score_comparison") is not True:
            raise ValueError("v13 must enable fixed-point score comparison.")
        if protocol_version >= 19 and ranking.get("ranking_device") != "cpu":
            raise ValueError("v19 must freeze the candidate ranking backend to CPU.")
        if protocol_version >= 19 and ranking.get("actor_device") != "cpu":
            raise ValueError("v19 must freeze the actor backend to CPU for deterministic replay.")

    ledger = _mapping(protocol.get("reliability_ledger"), "reliability_ledger")
    if ledger.get("source_split") != "calibration_only" or ledger.get("immutable_after_calibration") is not True:
        raise ValueError("v12 ledger must be immutable and calibration-only.")
    if (
        ledger.get("checkpoint_hash_bound") is not True
        or ledger.get("protocol_hash_bound") is not True
        or ledger.get("clearance_calibration_hash_bound") is not True
    ):
        raise ValueError("v12 ledger must bind checkpoint, protocol, and calibration hashes.")
    if set(ledger.get("states", [])) != {"trusted", "fallback_nominal", "safe_hold"}:
        raise ValueError("v12 ledger states are incomplete.")
    tolerance_payload = ledger.get("bucket_boundary_tolerances")
    if tolerance_payload is None:
        # Preserve validation of the original v12 protocol; the deterministic
        # revision below must declare all tolerances explicitly.
        if "deterministic" in str(protocol.get("protocol_name", "")):
            raise ValueError("Deterministic v12 protocol must declare bucket tolerances.")
        tolerance_payload = {
            "visibility_fraction": 0.0,
            "observation_age_steps": 0.0,
            "clearance_m": 0.0,
            "ttc_s": 0.0,
            "uncertainty": 0.0,
            "cbf_risk": 0.0,
            "candidate_separation_m": 0.0,
        }
    tolerances = _mapping(tolerance_payload, "reliability_ledger.bucket_boundary_tolerances")
    required_tolerances = {
        "visibility_fraction",
        "observation_age_steps",
        "clearance_m",
        "ttc_s",
        "uncertainty",
        "cbf_risk",
        "candidate_separation_m",
    }
    if set(tolerances) != required_tolerances:
        raise ValueError("v12 ledger bucket tolerances are incomplete.")
    if any(
        float(value) < 0.0
        or float(value) != float(value)
        or float(value) in (float("inf"), float("-inf"))
        for value in tolerances.values()
    ):
        raise ValueError("v12 ledger bucket tolerances must be finite and non-negative.")

    tensorboard = _mapping(protocol.get("tensorboard"), "tensorboard")
    if tensorboard.get("required") is not True:
        raise ValueError("v12 requires TensorBoard records.")
    provenance = _mapping(protocol.get("provenance"), "provenance")
    if provenance.get("target_truth_used_only_for_offline_labels") is not True:
        raise ValueError("v12 target truth must remain offline-only.")
    if provenance.get("results_are_not_locked_evidence") is not True:
        raise ValueError("v12 results must remain development-only.")


def verify_frozen_inputs(protocol: dict[str, Any], project_root: Path) -> dict[str, Any]:
    raw_inputs = protocol.get("frozen_inputs")
    if raw_inputs is None:
        # Compact evaluation protocols bind their checkpoints and calibration
        # archive at the run level rather than in one static input map.
        return {}
    inputs = _mapping(raw_inputs, "frozen_inputs")
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
    is_calibrated = (
        str(protocol.get("protocol_name", "")).startswith(
            "central_random_mixed_obstacle_s3_v5_v12_calibrated_clearance"
        )
        or str(protocol.get("protocol_name", "")).startswith(
            "central_random_mixed_obstacle_s3_v5_v13_fixedpoint"
        )
        or str(protocol.get("protocol_name", "")).startswith(
            "central_random_mixed_obstacle_s3_v5_v14_fixedpoint"
        )
        or str(protocol.get("protocol_name", "")).startswith(
            "central_random_mixed_obstacle_s3_v5_v15_fixedpoint"
        )
        or str(protocol.get("protocol_name", "")).startswith(
            "central_random_mixed_obstacle_s3_v5_v16_fixedpoint"
        )
        or str(protocol.get("protocol_name", "")).startswith(
            "central_random_mixed_obstacle_s3_v5_v17_fixedpoint"
        )
        or str(protocol.get("protocol_name", "")).startswith(
            "central_random_mixed_obstacle_s3_v5_v18_fixedpoint"
        )
        or str(protocol.get("protocol_name", "")).startswith(
            "central_random_mixed_obstacle_s3_v5_v19_cpu_ranker"
        )
        or str(protocol.get("protocol_name", "")).startswith(
            "central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic"
        )
    )
    if is_calibrated:
        primary_endpoint = "safe_capture"
        world_model_role = protocol["model_contract"]["world_model_role"]
        cbf_final_safety_filter = protocol["model_contract"]["cbf_is_final_safety_filter"]
    else:
        primary_endpoint = protocol["objective"]["primary_endpoint"]
        world_model_role = protocol["objective"]["world_model_role"]
        cbf_final_safety_filter = protocol["cbf_contract"]["final_safety_filter"]
    return {
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "protocol_name": protocol["protocol_name"],
        "protocol_version": protocol["protocol_version"],
        "locked_test_opened": protocol["locked_test_opened"],
        "primary_endpoint": primary_endpoint,
        "world_model_role": world_model_role,
        "cbf_final_safety_filter": cbf_final_safety_filter,
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
