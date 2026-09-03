from pathlib import Path

import pytest
import yaml

from scripts.verify_jepa_safe_capture_protocol import load_protocol, verify_frozen_inputs


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "jepa_safe_capture_v2_protocol.yaml"


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
