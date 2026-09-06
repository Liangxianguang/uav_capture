from scripts.audit_wp2_route_abort import classify_abort


def _step(*, step: int, geometry_valid: int, accepted: int, selected: str, abort: bool, active: list[str]):
    return {
        "step": step,
        "selected_route": {"label": selected, "route_id": f"{selected}:obstacle-4"},
        "route_geometry": {"valid_count": geometry_valid},
        "candidate_cbf": {
            "checks": 10,
            "accepted": accepted,
            "probes": [
                {"label": "left_detour", "accepted": accepted > 0},
                {"label": "right_detour", "accepted": accepted > 1},
            ],
        },
        "cbf": {
            "fallback_mode": "controlled_abort" if abort else "none",
            "infeasible": abort,
            "timed_out": False,
            "active_constraints": active,
        },
        "independent_cbf": [],
    }


def test_abort_classifies_late_anticipation_and_recovery_infeasibility() -> None:
    previous = _step(
        step=14,
        geometry_valid=10,
        accepted=10,
        selected="nominal",
        abort=False,
        active=["obstacle_0_defender_1"],
    )
    abort = _step(
        step=15,
        geometry_valid=10,
        accepted=0,
        selected="nominal",
        abort=True,
        active=["obstacle_0_defender_1", "acceleration_defender_1"],
    )

    finding = classify_abort(abort, previous)

    assert finding["primary"] == "joint_cbf_recovery_infeasibility"
    assert "late_obstacle_anticipation" in finding["labels"]
    assert "obstacle_barrier_recovery" in finding["labels"]


def test_abort_distinguishes_geometric_exhaustion() -> None:
    abort = _step(
        step=8,
        geometry_valid=0,
        accepted=0,
        selected="verified_safe_hold",
        abort=True,
        active=[],
    )

    finding = classify_abort(abort, None)

    assert finding["primary"] == "all_route_geometric_rejection"
