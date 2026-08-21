from __future__ import annotations

from scripts.aggregate_e1_rule_expert_feasibility import evaluate_profile, report


def rows(*, cooperative: int, collision: int, boundary: int, total: int = 60) -> list[dict[str, str]]:
    values = []
    for index in range(total):
        values.append(
            {
                "profile": "E0",
                "execution_mode": "execution_aware_cbf",
                "method": "dynamic_encirclement",
                "cooperative_safe_capture": str(index < cooperative),
                "collision": str(index < collision),
                "world_violation_steps": "1" if index < boundary else "0",
                "termination_reason": "safe_capture" if index < cooperative else "timeout",
                "case_sha256": f"case-{index}",
            }
        )
    return values


def test_feasibility_gate_requires_all_three_preregistered_thresholds() -> None:
    passed = evaluate_profile(rows(cooperative=57, collision=3, boundary=3), expected_episodes=60, profile="E0")
    assert passed["gate"]["passed"] is True
    capture_failure = evaluate_profile(rows(cooperative=56, collision=3, boundary=3), expected_episodes=60, profile="E0")
    collision_failure = evaluate_profile(rows(cooperative=57, collision=4, boundary=3), expected_episodes=60, profile="E0")
    boundary_failure = evaluate_profile(rows(cooperative=57, collision=3, boundary=4), expected_episodes=60, profile="E0")
    assert capture_failure["gate"]["passed"] is False
    assert collision_failure["gate"]["passed"] is False
    assert boundary_failure["gate"]["passed"] is False


def test_report_marks_a_failed_profile_without_suppressing_its_metrics() -> None:
    value = evaluate_profile(rows(cooperative=30, collision=20, boundary=10), expected_episodes=60, profile="E0")
    text = report({"profiles": {"E0": value}, "decision": "stop"})
    assert "50.0%" in text
    assert "FAIL" in text
