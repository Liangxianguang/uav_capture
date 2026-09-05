from pathlib import Path

import pytest
import yaml

from scripts.verify_jepa_safe_capture_protocol import load_protocol, verify_frozen_inputs


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "jepa_safe_capture_v2_protocol.yaml"
V12_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v12_calibrated_clearance_development_protocol.yaml"
V13_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v13_fixedpoint_development_protocol.yaml"
V14_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v14_fixedpoint_robust_development_protocol.yaml"
V15_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v15_fixedpoint_robust_development_protocol.yaml"
V16_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v16_fixedpoint_robust_development_protocol.yaml"
V17_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v17_fixedpoint_robust_development_protocol.yaml"
V18_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v18_fixedpoint_robust_development_protocol.yaml"
V19_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v19_cpu_ranker_development_protocol.yaml"
V20_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic_development_protocol.yaml"
V21_PROTOCOL_PATH = ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml"


def test_safe_capture_protocol_freezes_evaluator_ledger_and_cbf_contract() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol["objective"]["primary_endpoint"] == "safe_capture"
    assert protocol["candidate_contract"]["jepa_can_generate_final_action"] is False
    assert protocol["reliability_ledger"]["checkpoint_hash_bound"] is True
    assert protocol["cbf_contract"]["final_safety_filter"] is True
    assert protocol["cbf_contract"]["qp_infeasible_action"] == "safe_hold_then_nominal_cbf"


def test_safe_capture_protocol_frozen_inputs_exist() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert all(item["exists"] for item in artifacts.values())
    assert all(item["sha256"] for item in artifacts.values())


def test_v12_calibrated_clearance_protocol_is_closed_and_hash_bound() -> None:
    protocol = load_protocol(V12_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 12
    assert protocol["locked_test_opened"] is False
    assert protocol["model_contract"]["world_model_role"] == "candidate_trajectory_evaluator_only"
    assert protocol["candidate_contract"]["execute_first_step_then_replan"] is True
    assert protocol["reliability_ledger"]["clearance_calibration_hash_bound"] is True
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


def test_v13_fixedpoint_protocol_enables_discrete_ranking_and_stays_closed() -> None:
    protocol = load_protocol(V13_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 13
    assert protocol["locked_test_opened"] is False
    assert protocol["candidate_ranking"]["profile"] == "p13_fixedpoint_v1"
    assert protocol["candidate_ranking"]["fixed_point_score_comparison"] is True
    assert protocol["candidate_ranking"]["cbf_margin_changed"] is False
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


def test_v14_robust_fixedpoint_protocol_widens_device_comparison_quantization() -> None:
    protocol = load_protocol(V14_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 14
    assert protocol["locked_test_opened"] is False
    assert protocol["candidate_ranking"]["profile"] == "p14_fixedpoint_robust_v1"
    assert protocol["candidate_ranking"]["score_comparison_quantum_m"] == pytest.approx(0.0015)
    assert protocol["candidate_contract"]["action_comparison_quantum_mps"] == pytest.approx(0.001)
    assert protocol["candidate_ranking"]["cbf_margin_changed"] is False
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


def test_v15_robust_fixedpoint_protocol_widens_score_and_risk_boundary_policy() -> None:
    protocol = load_protocol(V15_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 15
    assert protocol["locked_test_opened"] is False
    assert protocol["candidate_ranking"]["profile"] == "p15_fixedpoint_robust_v1"
    assert protocol["candidate_ranking"]["score_comparison_quantum_m"] == pytest.approx(0.002)
    assert protocol["reliability_ledger"]["bucket_boundary_tolerances"]["cbf_risk"] == pytest.approx(0.005)
    assert protocol["candidate_ranking"]["cbf_margin_changed"] is False
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


def test_v16_robust_fixedpoint_protocol_freezes_state_drift_quantization() -> None:
    protocol = load_protocol(V16_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 16
    assert protocol["locked_test_opened"] is False
    assert protocol["candidate_ranking"]["profile"] == "p16_fixedpoint_robust_v1"
    assert protocol["candidate_ranking"]["score_comparison_quantum_m"] == pytest.approx(0.004)
    assert protocol["candidate_contract"]["action_comparison_quantum_mps"] == pytest.approx(0.005)
    assert protocol["reliability_ledger"]["bucket_boundary_tolerances"]["cbf_risk"] == pytest.approx(0.005)
    assert protocol["candidate_ranking"]["cbf_margin_changed"] is False
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


def test_v17_robust_fixedpoint_protocol_freezes_conservative_abstention_band() -> None:
    protocol = load_protocol(V17_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 17
    assert protocol["locked_test_opened"] is False
    assert protocol["candidate_ranking"]["profile"] == "p17_fixedpoint_robust_v1"
    assert protocol["candidate_ranking"]["score_comparison_safety_band_m"] == pytest.approx(0.004)
    assert protocol["candidate_contract"]["action_comparison_quantum_mps"] == pytest.approx(0.005)
    assert protocol["candidate_ranking"]["cbf_margin_changed"] is False
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


def test_v18_robust_fixedpoint_protocol_freezes_nominal_anchor_semantics() -> None:
    protocol = load_protocol(V18_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 18
    assert protocol["locked_test_opened"] is False
    assert protocol["candidate_ranking"]["profile"] == "p18_fixedpoint_robust_v1"
    assert protocol["candidate_ranking"]["score_comparison_safety_band_m"] == pytest.approx(0.004)
    assert protocol["candidate_ranking"]["cbf_margin_changed"] is False
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


def test_v19_protocol_freezes_cpu_candidate_ranking_backend() -> None:
    protocol = load_protocol(V19_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 19
    assert protocol["locked_test_opened"] is False
    assert protocol["candidate_ranking"]["profile"] == "p19_cpu_ranker_v1"
    assert protocol["candidate_ranking"]["ranking_device"] == "cpu"
    assert protocol["candidate_ranking"]["cbf_margin_changed"] is False
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


def test_v20_protocol_freezes_cpu_actor_and_ranking_backend() -> None:
    protocol = load_protocol(V20_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 20
    assert protocol["locked_test_opened"] is False
    assert protocol["candidate_ranking"]["profile"] == "p20_cpu_deterministic_v1"
    assert protocol["candidate_ranking"]["ranking_device"] == "cpu"
    assert protocol["candidate_ranking"]["actor_device"] == "cpu"
    assert protocol["candidate_ranking"]["cbf_margin_changed"] is False
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


def test_v21_protocol_freezes_positive_candidate_separation_gate() -> None:
    protocol = load_protocol(V21_PROTOCOL_PATH)
    assert protocol["protocol_version"] == 21
    assert protocol["locked_test_opened"] is False
    assert protocol["candidate_ranking"]["profile"] == "p21_cpu_separation_gate_v1"
    assert protocol["candidate_ranking"]["minimum_candidate_separation_m"] == pytest.approx(0.002)
    assert protocol["candidate_ranking"]["cbf_margin_changed"] is False
    artifacts = verify_frozen_inputs(protocol, ROOT)
    assert artifacts
    assert all(item["exists"] and item["sha256"] for item in artifacts.values())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("objective", "primary_endpoint", "capture_time_seconds"),
        ("candidate_contract", "jepa_can_generate_final_action", True),
        ("cbf_contract", "final_safety_filter", False),
        ("reliability_ledger", "source_split", "development"),
    ],
)
def test_safe_capture_protocol_rejects_unsafe_contract_mutations(section: str, key: str, value) -> None:
    raw = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw[section][key] = value
    temporary = PROTOCOL_PATH.with_name(".jepa_safe_capture_protocol_test.yaml")
    temporary.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            load_protocol(temporary)
    finally:
        temporary.unlink(missing_ok=True)
