# DN-MPC P51 Group-Level Planner Distillation Audit

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** implementation complete; planner-selection promotion gate failed

## Objective

P51 implements the group-level planner-distillation loss requested after P46.
The analytic DN-MPC planner remains the execution authority. For each
`(scenario_index, time_index)` group, the loss averages the learned
`route_progress` score over each candidate's defender rows and applies a
temperature-scaled cross-entropy target to
`planner_selected_candidate_index`. Only rows with
`planner_teacher_label >= 0`, `planner_eligible == 1` and
`planner_abstention == 0` enter the loss. Ineligible candidates, offline rows
and explicit abstentions remain unknown and are never made negative examples.

No action was executed. CBF margins, stale/OOD/non-finite gates and
`controlled_abort` were unchanged.

## Baseline and runs

The frozen P42 evaluator (`seed=393702`) was evaluated with the same masked
group metric before training:

- validation planner agreement: **39.47%** (`133/337` groups);
- finite predictions: `True`;
- masked groups/rows: `337 / 11,624`;
- unsafe ranked candidates: `0`.

| Run | Trainable scope | Best epoch | Validation agreement | Validation distillation loss | Calibration agreement | Stop |
|---|---|---:|---:|---:|---:|---|
| P51 full, seed 513001 | all evaluator parameters, base-loss weight 0.1 | 1 | **42.73%** (`144/337`) | `1.582387` | `43.68%` (`183/419`) | 5 stale epochs |
| P51 head-only, seed 513001 | `route_progress_decoder` only | 1 | **46.88%** (`158/337`) | `1.635579` | `40.57%` (`170/419`) | 3 stale epochs |

The head-only run is the strongest bounded result, improving over the frozen
baseline by **+7.41 percentage points**, but it remains below the P18 top-1
promotion threshold of `50%`. Training agreement rose while held-out
agreement and calibration degraded after the first epoch, showing
overfitting on the small P39 archive.

The P18 route audit for the head-only checkpoint reports validation
route-progress truth top-1 `37.09%` and pairwise agreement `80.88%`. Its
selected-candidate, OOD and full three-way counterfactual promotion gates
remain closed; the two explicit P46 abstention groups explain the missing
selected trace entries rather than a new safety violation.

## Safety audit

| Check | Validation result |
|---|---:|
| Runtime groups | `339` |
| Independent selected/nominal/safe-hold trace coverage | `99.41% / 99.41% / 99.41%` |
| Unknown rows preserved | `4,648` |
| Unsafe ranked candidates | `0` |
| Finite predictions | `True` |
| Raw-unverified actions executed | `False` |
| CBF contract changed | `False` |

The `99.41%` trace coverage is the expected `337/339` non-abstention group
coverage. No safety event was introduced by the offline trainer.

## Artifacts

- Trainer: `scripts/train_dn_mpc_planner_distillation.py`
- Unit tests: `tests/test_dn_mpc_planner_distillation.py` (`2 passed`)
- Full checkpoint: `results/dn_mpc_jepa_safe_capture_checkpoints/p51_full_seed513001_v2/checkpoint.pt`
- Head-only checkpoint: `results/dn_mpc_jepa_safe_capture_checkpoints/p51_headonly_seed513002_v2/checkpoint.pt`
- Full TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p51_full_seed513001_v2/`
- Head-only TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p51_headonly_seed513002_v2/`
- P18 audit: `results/dn_mpc_jepa_safe_capture_dev/p51_headonly_p18_ranking/`

All training configuration, losses, agreement, masked-row counts, parameter
histograms and provenance are recorded in the corresponding TensorBoard runs.

## Gate decision and next action

- [x] Group-level masked planner-distillation loss implemented.
- [x] Unknown/ineligible/abstention rows excluded from loss.
- [x] TensorBoard and checkpoint provenance recorded.
- [x] Validation safety audit has zero unsafe-ranked candidates and no CBF
  contract change.
- [ ] Validation planner-agreement promotion gate (`>=50%`) passed.
- [ ] Online JEPA route override, Ledger-Lite or locked benchmark authorized.

P51 must therefore remain offline-only. Do not add more P51 epochs or seeds.
The evidence points to archive-scale overfitting and a route-progress score
that is not sufficiently identifiable from the current candidate features. The
next bounded task is to audit/collect a larger, independently seeded teacher
archive with explicit route-level counterfactual outcome labels, then rerun a
single pre-registered distillation comparison. Analytic DN-MPC + CBF remains
the only execution path.
