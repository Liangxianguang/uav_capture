# DN-MPC P38 Route-Agreement Diagnosis

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** route-contract repair is required; JEPA remains an offline evaluator

## Scope

P38 stratifies the P37 pairwise action-conditioned JEPA evaluator by route
family, candidate eligibility, utility ties, explicit CBF abstention, and
route identity. It executes no candidate action, does not alter CBF margins,
and does not open a locked benchmark.

The source is the P37 full train/validation/calibration archive and its
independent utility audit. The generated artifacts are:

- JSON: `results/dn_mpc_jepa_safe_capture_dev/p38_route_agreement_diagnosis_v2/diagnosis.json`
- Markdown: `results/dn_mpc_jepa_safe_capture_dev/p38_route_agreement_diagnosis_v2/report.md`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p38_route_agreement_diagnosis_v2/`
- Details SHA-256: `770a86ba5af3b06f292b259d94fb92edc4067904832738cb08ff5405c4ea3a1e`
- Utility audit SHA-256: `443a0d8fec73f25ae92609f06db65eafcb7af161e7792f479618e53b487815c5`
- Metadata SHA-256: `b913925af3fd0b148fc24029c10ad9fab5c892d52394bdb0c9dd8d84f5e1cf61`

## Main findings

| Split | Groups | Valid groups (>=2 eligible) | Mean eligible | CBF abstention | Model vs truth | Model vs selected | Near tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 268 | 267 | 9.23 | 0.37% | 98.13% | 34.08% | 4.87% |
| Validation | 339 | 337 | 8.57 | 0.59% | 99.41% | 34.42% | 3.56% |
| Calibration | 420 | 419 | 8.83 | 0.24% | 97.85% | 35.56% | 5.25% |

The evaluator reproduces the offline utility labels well, including `100.00%`
informative exact agreement and `99.07%` pairwise agreement on the P37
validation audit. However, it agrees with the analytic DN-MPC route actually
selected from the same public state only `34.42%` of the time. Therefore the
P37 failure is not primarily a lack of JEPA capacity or an excessive CBF
abstention rate.

Route-family counts make the mismatch concrete. On validation, the analytic
planner selects `formation_split` 110 times, `safe_intercept` 59 times and
`formation_contract` 15 times, while the learned evaluator's best route is
mostly `nominal` (275 groups), with `braking` (41) and `visibility_hold` (20)
as the remaining major families. The evaluator's model-vs-truth agreement is
perfect for the nominal and visibility-hold groups, but that does not imply it
implements the planner's tie-break or formation-role contract.

The `lower_detour` family has zero eligible candidates in all three splits
(`0/339`, `0/268`, `0/420` eligible rows). It must not be treated as a useful
negative training signal until its candidate geometry and reachable-dynamics
projection are audited; otherwise the archive encodes a permanently
unavailable route as if it were a meaningful alternative.

## Route-switch interpretation

P38 reports two different comparisons:

1. `planner_selected_vs_previous_executed_route_fraction` compares the
   analytic planner's current selection with the frozen actor's previous
   executed route. Validation is `76.60%`.
2. `model_vs_previous_executed_route_fraction` compares the JEPA model-best
   route with that same previous executed route. Validation is `15.20%`.

Neither is the planner's own route-switch rate. The previous route is produced
by the frozen actor, while `selected` is produced by the analytic DN-MPC
planner. Calling the first number a planner route-switch rate would mix two
decision chains and would make the switch penalty label invalid. This is the
central execution-contract mismatch to repair in P39.

The explicit CBF abstentions are small and traceable: validation has two groups
with fewer than two eligible candidates, and all unknown route identities are
associated with first-step CBF infeasibility. This is a safety-positive result,
but it does not authorize relaxing the safety gate.

## Decision

P38 is complete as a diagnosis gate, but P37 is not promoted:

- keep analytic DN-MPC + strict Joint CBF as the only runtime-eligible route;
- keep JEPA offline-only;
- do not create or update Ledger-Lite;
- do not run a three-seed paired replay or a locked benchmark;
- do not lower CBF margins or disable stale/OOD/non-finite gates;
- do not interpret `34.42%` as a safe-capture rate.

## P39 entry criteria

Before another evaluator training run, the archive and utility code must record
separate identities for:

- `previous_executed_route_index` from the frozen actor after CBF filtering;
- `previous_selected_candidate_index` from the analytic planner;
- the planner's own route-switch outcome;
- candidate eligibility after reachable-dynamics projection and first-step CBF;
- a valid outcome label for routes that are unavailable by construction.

P39 must then regenerate disjoint train/validation/calibration archives and
recalibrate utility weights. Only a fresh evaluator that agrees with the
planner's corrected selected-candidate contract on the promotion threshold may
proceed to Ledger-Lite or paired closed-loop replay.
