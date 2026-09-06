# JEPA Route Runtime Alignment Audit

**Date:** 2026-09-06
**Phase:** development-only; `locked_test_opened=false`
**Primary metric:** `safe_capture`
**Safety policy:** no CBF margin relaxation, no stale/OOD bypass, no raw unverified action.

## Why This Audit Was Run

The ranked route checkpoint improved held-out archive route ranking but produced
`safe_capture=0/1` in the online route smoke. The audit replays the frozen scene
manifest and rolls out each valid three-step route in an isolated environment
clone. Simulator target truth is used only for these offline branch metrics; it
is never passed to the online planner, JEPA, Ledger, or CBF path.

## Evidence

| Run | Route archive/state source | Steps | Predicted vs actual top-1 | Pairwise sign agreement | Selected predicted progress | Selected actual progress | Safe capture |
|---|---|---:|---:|---:|---:|---:|---:|
| `wp3_route_runtime_smoke_ranked_v3_m3_seed20260911_sidefix` | rule-controller archive | 50 | 34.0% | 73.6% | 0.309 m | 0.065 m | 0/1 |
| `wp3_route_runtime_smoke_actor_m3_seed20260911` | frozen runtime actor archive | 50 | 16.0% | 78.8% | 0.603 m | 0.240 m | 0/1 |

Both runs replayed the recorded target-clearance trace exactly (`max error =
0.0 m`). This validates the audit replay alignment, not a task success claim.

The actor-matched archive reduced route-execution fallback from `136/250` to
`41/250` steps and retained zero collision, boundary, pairwise, timeout, and
raw-unverified actions. The task still timed out, so the state-source correction
is necessary but not sufficient.

## Findings

1. The former archive was generated while the main state trajectory followed
   `DynamicEncirclementController`, whereas runtime follows
   `v5_development_exact_reactive_seed661606.pt`. This is a real train/runtime
   distribution mismatch.
2. `verified_safe_hold` had a route-side metadata mismatch (`hold=8` at
   runtime versus archive `boundary_shadow=11`). The runtime mapping and a
   regression test now agree with the archive vocabulary.
3. Runtime route ranking remains poorly calibrated under branch replay. The
   actor-matched model often predicts a route with higher progress than its
   actual CBF-filtered branch, so enabling a larger score weight or opening a
   multi-seed block before recalibration would be premature.
4. CBF is not the current bottleneck in this smoke: all candidate probes and
   independent selected/nominal/safe-hold probes were accepted, and no safety
   hard gate was violated.

## Next Gate

The next experiment is a full actor-matched archive block using the frozen
runtime actor for train, validation, and independent calibration scenes. The
checkpoint and Ledger must be retrained/rebuilt from those new archives. The
new checkpoint first has to pass this same 50-step route alignment audit with
improved top-1/pairwise agreement and nonzero safe-capture smoke before any
three-seed experiment is authorized.

No result in this report is a locked-test result or a formal improvement claim.
