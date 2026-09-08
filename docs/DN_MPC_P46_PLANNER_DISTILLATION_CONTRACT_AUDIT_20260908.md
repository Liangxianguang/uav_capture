# DN-MPC P46 Planner-Distillation Contract Audit

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** teacher-label contract passes; distillation training remains gated

## Objective

P46 converts the analytic DN-MPC selection into an explicit offline teacher
label. A candidate is positive only when it equals
`planner_selected_candidate_index` in a non-abstention group and is both
geometry-valid and first-step Joint-CBF-feasible. Other eligible alternatives
are negative; ineligible candidates, offline branches and zero-eligible
planner-abstention groups remain unknown (`-1`). No action is executed.

## Archive audit

| Split | Runtime groups | Eligible groups | Abstention groups | Invalid identity groups | Teacher-labeled rows | Positive fraction | Trace groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train8 | 268 | 267 | 1 | 0 | 9,896 | 10.79% | 267 |
| Calibration8 | 420 | 419 | 1 | 0 | 14,836 | 11.30% | 419 |
| Validation8 | 339 | 337 | 2 | 0 | 11,624 | 11.60% | 337 |

Selected route counts and candidate-family coverage are retained in the
machine-readable audit. The `lower_detour` family remains geometry-invalid in
the solid-ground scene contract and is not silently turned into a normal
negative.

## Contract checks

- Planner selected/previous/switch identity is state-consistent in every group.
- Selected routes are always eligible or explicit planner abstentions.
- Unknown labels are preserved for ineligible candidates and abstention groups.
- Independent selected/nominal/safe-hold CBF traces are present for all
  eligible runtime groups except the explicit abstentions.
- Train, calibration and validation episode seeds remain disjoint.
- Target truth is used only for offline labels; no raw-unverified action is
  executed.

## Artifacts

- Builder/audit: `scripts/audit_dn_mpc_p46_planner_distillation.py`
- Audit JSON: `results/dn_mpc_jepa_safe_capture_dev/p46_planner_distillation_contract_audit.json`
- Audit Markdown: `results/dn_mpc_jepa_safe_capture_dev/p46_planner_distillation_contract_audit.md`
- Train labels: `results/dn_mpc_jepa_safe_capture_dev/p46_planner_distillation_train8/`
- Calibration labels: `results/dn_mpc_jepa_safe_capture_dev/p46_planner_distillation_calibration8/`
- Validation labels: `results/dn_mpc_jepa_safe_capture_dev/p46_planner_distillation_validation8/`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p46_planner_distillation_contract_audit/`

## Decision and next gate

The label contract is suitable for a bounded planner-distillation experiment,
but it does not prove JEPA route-selection quality. Before training, the
distillation task must use group-level masked loss with explicit abstention
handling and retain the same independent safety traces. The analytic DN-MPC +
CBF planner remains the only execution authority; Ledger-Lite, online JEPA
override, paired replay and locked testing remain closed.
