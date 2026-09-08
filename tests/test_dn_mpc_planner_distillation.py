from __future__ import annotations

import torch

from scripts.train_dn_mpc_planner_distillation import _distillation_from_aux


def _batch(*, abstention: float = 0.0, include_ineligible: bool = True) -> dict[str, torch.Tensor]:
    # Two candidates in one state group, with two replicated defender rows per
    # candidate. Candidate 1 is the analytic planner teacher target.
    candidate = torch.tensor([0, 0, 1, 1] + ([2, 2] if include_ineligible else []))
    n = int(candidate.numel())
    return {
        "sample_type": torch.zeros(n, dtype=torch.long),
        "route_candidate_index": candidate,
        "route_geometry_valid": torch.ones(n),
        "labels_cbf_feasible": torch.ones(n, 5),
        "scenario_index": torch.zeros(n, dtype=torch.long),
        "time_index": torch.zeros(n, dtype=torch.long),
        "planner_teacher_label": torch.tensor([0, 0, 1, 1] + ([-1, -1] if include_ineligible else []), dtype=torch.float32),
        "planner_eligible": torch.tensor([1, 1, 1, 1] + ([0, 0] if include_ineligible else []), dtype=torch.float32),
        "planner_abstention": torch.full((n,), abstention),
        "planner_teacher_selected": torch.full((n,), 1, dtype=torch.long),
    }


def test_group_distillation_masks_ineligible_rows_and_selects_teacher() -> None:
    batch = _batch()
    auxiliary = {"route_progress": torch.tensor([[0.1, 0.1, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.1, 0.1, 0.8, 0.1, 0.1], [0.1, 0.1, 0.8, 0.1, 0.1], [99.0, 99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 99.0, 99.0, 99.0]])}
    loss, agreement, groups, used_rows = _distillation_from_aux(auxiliary, batch, horizon_index=2, temperature=0.05)
    assert loss.item() > 0.0
    assert agreement.item() == 1.0
    assert groups.item() == 1.0
    assert used_rows.item() == 4.0


def test_group_distillation_ignores_abstention_group() -> None:
    batch = _batch(abstention=1.0, include_ineligible=False)
    auxiliary = {"route_progress": torch.zeros(4, 5)}
    loss, agreement, groups, used_rows = _distillation_from_aux(auxiliary, batch, horizon_index=2, temperature=0.05)
    assert loss.item() == 0.0
    assert agreement.item() == 0.0
    assert groups.item() == 0.0
    assert used_rows.item() == 0.0
