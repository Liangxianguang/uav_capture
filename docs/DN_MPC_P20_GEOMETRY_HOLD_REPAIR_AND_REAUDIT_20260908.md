# DN-MPC P20 Geometry-Hold Repair and P18 Re-audit

**Date:** 2026-09-08
**Status:** development-only; online promotion remains stopped

## Root cause repaired

P19 found 10/172 validation route-state groups with no eligible candidate. The
CBF first-step labels were feasible in those groups, but the route geometry
pre-check rejected every candidate because the current centroid was already
inside the obstacle-clearance margin. This incorrectly applied a traversing
corridor test to `braking` and `verified_safe_hold`, which do not traverse the
obstacle field; their safety must be decided by the downstream independent CBF.

The route generator now keeps the geometric diagnostic but does not let the
current-centroid clearance alone invalidate these two zero/low-motion fallback
routes. No CBF margin, acceleration bound, stale/OOD gate, or controlled-abort
rule was changed.

## Re-audit evidence

The same eight P17 validation episodes, frozen actor, seed block, five-step
chunk, and interaction archive contract were recollected into:

`results/dn_mpc_jepa_safe_capture_dev/p20_route_validation8_geometry_hold_repair`

| Metric | P19 original | P20 repaired |
|---|---:|---:|
| runtime groups with zero eligible candidates | 10/172 | **0/172** |
| minimum eligible candidates per group | 0 | **2** |
| first-step CBF feasible fraction | 100.00% | **100.00%** |
| validation top-1 route-progress agreement | 14.81% | **19.19%** |
| informative pairwise score direction | 84.31% | **84.32%** |
| route identity accuracy | 100.00% | **100.00%** |

The P20 re-audit JSON and ranking details are stored locally at:

- `results/dn_mpc_jepa_safe_capture_dev/p20_p18_candidate_ranking_reaudit/audit.json`
- `results/dn_mpc_jepa_safe_capture_dev/p20_p18_candidate_ranking_reaudit/ranking_details.csv`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p20_p18_candidate_ranking_reaudit`

## Decision

The geometry-eligibility gate is repaired, but the P18 checkpoint still fails
the route selector gate: top-1 progress agreement is below the 50% promotion
threshold. Pairwise direction remains useful diagnostically, but it is not
enough to authorize online selection. Selected/nominal/safe-hold independent
CBF traces and a fresh OOD/disagreement calibration are also still missing.

Therefore P20 **does not enter online replay or three-seed evaluation**. The
next task is to audit route-progress labels and the scalar ranking score,
including whether invalid/hold candidates are being compared with task routes;
then recalibrate or retrain the route-progress head. Safety constraints remain
unchanged.

## Verification

- targeted geometry regression: `2 passed`;
- full repository regression after the P19 stage: `586 passed`;
- P20 archive generation completed with `11,008` rows and TensorBoard output;
- no action was executed and no raw-unverified branch was enabled.
