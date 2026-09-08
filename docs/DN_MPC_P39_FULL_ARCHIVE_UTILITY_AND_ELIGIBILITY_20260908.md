# DN-MPC P39 Full Archive, Utility and Eligibility Audit

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** contract repair passes; utility evidence is positive; online promotion remains gated

## P39 contract repair

The archive now keeps the two decision chains separate:

| Chain | Previous identity | Current identity | Switch outcome |
|---|---|---|---|
| Frozen actor after final CBF | `previous_executed_route_index` | `executed_route_index` | `route_switch_outcome` |
| Analytic DN-MPC planner | `previous_selected_candidate_index` | `planner_selected_candidate_index` | `planner_route_switch_outcome` |

The route utility audit uses the planner identity for the switch penalty when
all three P39 fields are present. It refuses mixed split contracts; legacy P36
archives remain readable through an explicit executed-route fallback.

## Full archives

All split episode seeds are disjoint and all runs use chunk length 5, strict
CBF horizon 5, reachable-dynamics projection, pairwise hard-negative branches,
independent selected/nominal/safe-hold CBF probes and TensorBoard provenance.

| Split | Rows | Runtime rows | Geometry valid | First-step CBF feasible | Executed route match | Planner current known | Planner previous known |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train8 | 17,152 | 12,864 | 77.24% | 99.63% | 99.63% | 12,816/12,864 | 12,480/12,864 |
| Validation8 | 21,696 | 16,272 | 71.78% | 99.41% | 99.41% | 16,176/16,272 | 15,888/16,272 |
| Calibration8 | 26,880 | 20,160 | 73.77% | 99.76% | 99.76% | 20,112/20,160 | 19,776/20,160 |

Archive hashes:

- Train NPZ `a90090332b1a4fa7eb97771c96b10fc72a45677a87972503d73accbfc6bb2a4b`; metadata `aa5ffda2a9818214f9f6b1e7567ebf2c8619a55af0e903410744fe0f0c5681db`.
- Validation NPZ `72898cccc407a107820201b54cdb0152a62220df16d13d7fd89c0737e5cadcd4`; metadata `b783324390f380302184361233dea27df1227ca3bf2606c5d0a1e951599fa04a`.
- Calibration NPZ `de37a54d1800b4bc8352cc20712e9dcf0676dbf60a4d44300f96441326a64ceb`; metadata `c9b51617310a2368fdc0f87ba1a45f9f128cfc9e4b207b0b4c8c650f076d1a65`.

## Utility result

The existing P37 pairwise JEPA checkpoint (`seed=373701`) was evaluated without
retraining. Utility weights were selected only on the independent P39
calibration split:

`(length=0.5, escape=0.0, cbf=0.5, switch=0.5)`.

| Split | Exact top-1 | Informative top-1 | Pairwise | Model vs planner selected |
|---|---:|---:|---:|---:|
| Train | 97.00% | 99.60% | 98.45% | 75.28% |
| Validation | 99.11% | 100.00% | 97.42% | 80.71% |
| Calibration | 96.90% | 100.00% | 97.65% | 76.85% |

Compared with P38's incorrectly mixed identity contract (`34.42%` validation
model-vs-selected), corrected P39 validation agreement is `80.71%` (`+46.29
percentage points`). This is offline route-selection evidence, not a
safe-capture rate or an online closed-loop result. The selected switch weight
is at the current grid boundary, so the calibration is not yet a final global
weight choice.

## Eligibility result

The eligibility audit separates geometry, first-step CBF and complete
eligibility. On validation, complete eligible row fractions are:

| Route family | Eligible fraction | Interpretation |
|---|---:|---|
| `braking` | 99.41% | CBF-feasible hold branch |
| `formation_split` | 82.01% | Geometry rejects a minority of states |
| `formation_contract` | 82.30% | Geometry rejects a minority of states |
| `left_detour` | 22.12% | Side corridor is often blocked |
| `right_detour` | 69.91% | Side corridor is less often blocked |
| `upper_detour` | 63.72% | Upper corridor has boundary/obstacle rejects |
| `lower_detour` | 0.00% | Geometry contract rejects every row |
| `safe_intercept` | 82.30% | Same public corridor as nominal |
| `visibility_hold` | 87.91% | Mostly geometry-valid |

The first-step CBF-feasible fraction is `99.41%` for every candidate family on
validation. Therefore the observed family coverage loss is overwhelmingly a
route-geometry issue, not a strict-CBF issue. Source inspection shows the lower
route explicitly marks `lower_face_blocked` when solid obstacles extend to the
ground plane (`z=0`); it is not a meaningful training negative until a
ground-clearance route is designed or the obstacle model is changed.

## Artifacts and verification

- P39 utility audit: `results/dn_mpc_jepa_safe_capture_dev/p39_route_utility_audit_v1/`.
- P39 route diagnosis: `results/dn_mpc_jepa_safe_capture_dev/p39_route_agreement_diagnosis_v2/`.
- P39 eligibility audit: `results/dn_mpc_jepa_safe_capture_dev/p39_candidate_eligibility_v1/`.
- TensorBoard runs are in the matching `results/dn_mpc_jepa_safe_capture_tensorboard/p39_*` directories.
- 27 focused regression tests pass after the contract changes.

No action was executed by the utility/eligibility audits. CBF margins,
stale/OOD/non-finite gates and controlled-abort behavior were unchanged. No
Ledger-Lite, online JEPA route override, three-seed replay or locked benchmark
was opened.

## Decision and next gate

P39 contract repair is complete and materially improves offline agreement, but
promotion is still closed because the switch weight is grid-boundary selected
and `lower_detour` has zero coverage. Next work is bounded:

1. Expand the switch-weight calibration grid and confirm the `80.71%` result is
   not a boundary artifact.
2. Decide whether to remove `lower_detour` from the learned label vocabulary or
   implement a separately verified above-ground/ground-clearance route.
3. Re-run fresh calibration and an independent evaluator seed. Only if the
   corrected planner agreement remains above the promotion gate may Ledger-Lite
   or online paired replay be considered.
