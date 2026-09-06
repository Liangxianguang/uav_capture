# Distinct Model-Seed L0 Paired Gate Stop Report

**Date:** 2026-09-07
**Scope:** development-only independent model-seed replay
**Locked test:** not opened
**Hardware:** NVIDIA GeForce RTX 5050

## Decision

The independent model-seed gate failed. The run is stopped before expanding to
additional seeds, L1-L3, larger episode blocks, or another training run.

| Variant | Safe capture | Timeout | Collision | Defender boundary | Pairwise | Raw-unverified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0: actor + CBF | 2/3 (66.7%) | 1 | 0 | 0 | 0 | 0 |
| M3: JEPA + Ledger + route candidates + CBF | 1/3 (33.3%) | 2 | 0 | 0 | 0 | 0 |

The paired M3 minus M0 delta is **-33.3 percentage points**. The safety hard
gate passed, but the capability gate did not. This is a negative independent
model-seed result, not evidence for a method improvement.

## Contract

Both variants used the same three-episode manifest from validation seed block
`650101`, the same actor checkpoint, strict-buffer CBF, anticipatory horizon 5,
reachable-dynamics projection, and first-step-only rolling execution. M3 used
the distinct active-search JEPA checkpoint and its hash-bound Ledger.

- Protocol SHA-256: `5fc57476361b016f474600900de1887849a853b30ae776f3b51476be7041e119`
- Canonical scene manifest SHA-256: `130f9602a87bcc392b10b2fa8bddaeaa88dab21bae4d932f95ddfeb7647da644`
- Actor checkpoint SHA-256: `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`
- M3 JEPA checkpoint SHA-256: `1f0b169d0b845d0c1591abba5e17b7fb520818c8bdbc0c2dc3bd0f7d48891f1a`
- M3 Ledger SHA-256: `919aeb5c1ec59715868e7478295a5a2fab6860b285474f17399dd63b7d20f959`
- CBF: `strict_buffer`, margin `0.35 m`, horizon `5`

## Safety and execution evidence

M3 did not fail because its safety filter rejected every route or because the
solver was infeasible:

- candidate CBF prefilter: `5939/5939` accepted;
- independent selected/nominal/safe-hold probes: `1755/1755` accepted;
- CBF infeasible steps: `0`;
- CBF timeout steps: `0`;
- raw-unverified execution: `0`;
- cautious active-search attempts: `0`.

The M3 target-boundary diagnostic was `1/3`. It is reported separately from the
UAV defender safety hard gate and does not turn into a UAV boundary violation.

## Failure attribution

Episode `650102` is the decisive regression:

- M0 captured safely at step 62;
- M3 ran 250 steps and timed out;
- M3 recorded a target-boundary diagnostic;
- M3 had 2,750 geometry-valid and independently CBF-accepted route probes;
- M3 selected mostly `left_detour` and `nominal` routes while the target
  clearance increased to approximately `10.8 m`.

Episode `650103` timed out for both M0 and M3, so it is a tied failure rather
than an M3 regression. M3 episode `650101` was a safe capture. Across the two
M3 failures, the route set and CBF were available, but the distinct checkpoint
did not produce a reliable capture trajectory. The current evidence points to
JEPA route-ranking / target-motion generalization and target-escape handling,
not to a CBF safety violation or a missing verified action.

## TensorBoard and artifacts

Each run wrote its own TensorBoard event and provenance record:

- M0: `results/wp1_active_search_v4_ranker_fix_distinct_m0_seed20260912_tensorboard/`
- M3: `results/wp1_active_search_v4_ranker_fix_distinct_m3_seed20260912_tensorboard/`
- M0 event SHA-256: `ed7a556a2d51176164f76caaf28fac4c464d31b5247c56f18fca4884b99a72ac`
- M3 event SHA-256: `c482aecdb70fde79425712adf99d3aac4cdea2d159099560f6da4ceefd92c58b`

Run summaries:

- M0: `results/wp1_active_search_v4_ranker_fix_distinct_m0_seed20260912/summary.json`
- M3: `results/wp1_active_search_v4_ranker_fix_distinct_m3_seed20260912/summary.json`

## Stop rule and next permitted work

No additional training or scene expansion is authorized by this result. The
The completed bounded route-ranking audit is recorded in
[JEPA_SAFE_CAPTURE_DISTINCT_SEED_ROUTE_RANKING_AUDIT_20260907.md](JEPA_SAFE_CAPTURE_DISTINCT_SEED_ROUTE_RANKING_AUDIT_20260907.md).
The next permitted task is an offline route-regret audit for episode `650102`:

1. compare selected route, score argmin, and settled-best route;
2. separate JEPA target-progress error from visibility and Ledger routing;
3. test whether target-boundary-aware progress/escape supervision would reject
   the failed route without weakening CBF;
4. only after that audit, make a small checkpoint-bound recalibration or
   auxiliary-head change and rerun a new paired L0 gate.

CBF margins, stale/OOD gates, controlled-abort semantics, and raw-action guards
must remain unchanged.
