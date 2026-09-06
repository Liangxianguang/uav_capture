# V5 Nominal-Prefilter Alternative-Route Fix

**Date:** 2026-09-07
**Scope:** development-only paired replay; `locked_test_opened=false`
**Change:** when nominal geometry/CBF prefilter fails, rank independently verified eligible alternatives instead of forcing `safe_hold`

## Root cause

The unique degraded episode in the prior-observation active-search block was
not caused by a CBF collision, boundary failure, or solver timeout. At the
first step after the nominal route became geometrically invalid, the trace
contained valid and CBF-accepted alternatives (`braking` and
`visibility_hold`). The ranker nevertheless entered `safe_hold` because it
treated `valid[0] == false` as a global failure of the whole candidate batch.

The same episode accumulated 250 safe-hold steps and ended in `timeout`. This
was an execution-contract error: a nominal anchor is a preference and an
audit reference, not a prerequisite for using a separately verified route.

## Minimal fix

`SafeCaptureJEPARanker` now:

1. marks the nominal anchor ineligible when its geometry/CBF prefilter fails;
2. checks whether any non-nominal candidate is both valid and Ledger-trusted;
3. ranks those candidates when available, with trace reason
   `nominal_infeasible_alternative_route`;
4. retains the previous `safe_hold` behavior when no trusted alternative
   exists; and
5. leaves OOD, stale, never-received, non-finite, CBF margin, and raw-action
   guards unchanged.

The regression test is
`test_ranker_uses_verified_alternative_when_nominal_prefilter_fails` in
`tests/test_jepa_safe_capture_candidates.py`.

## Paired replay result

The old M0 runs and the fixed M3 runs use the same three manifests, actor,
JEPA checkpoint, Ledger, protocol, `horizon=5`, strict buffer, and RTX 5050
CUDA environment.

| Seed | M0 | Fixed M3 | Improved | Degraded | Tied | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 1/3 | 3/3 | 2 | 0 | 1 | +66.7 pp |
| 20260912 | 1/3 | 3/3 | 2 | 0 | 1 | +66.7 pp |
| 20260913 | 1/3 | 3/3 | 2 | 0 | 1 | +66.7 pp |
| **Pooled** | **3/9** | **9/9** | **6** | **0** | **3** | **+66.7 pp** |

Fixed M3 safety results:

- collision: `0`
- defender boundary: `0`
- pairwise: `0`
- raw-unverified execution: `0`
- CBF controlled abort: `0`
- CBF timeout: `0`
- candidate CBF prefilter: `3687/3687` accepted for each seed
- cautious reacquisition: `0` steps for each seed
- transit success: `100%` for each seed

One independent selected/nominal/safe-hold probe was rejected in each fixed
M3 run and remained a verified fallback; it did not become raw execution or a
safety violation.

## Interpretation and gate

This is strong evidence that the nominal-anchor routing bug caused the prior
timeout. It is still a development replay result, not an independent
three-checkpoint seed result: all three fixed M3 runs use one JEPA checkpoint
and one Ledger on one canonical scene manifest. The active-search branch also
did not trigger, so this result isolates route fallback behavior rather than
proving active-search efficacy.

The expansion gate remains closed until one of the following is completed:

- a distinct model-seed checkpoint is evaluated on a non-repeated scene block;
- the fixed route contract passes the L0 paired gate on that independent block;
- the same safety hard gates and provenance checks remain intact.

No CBF margin, stale/OOD policy, `controlled_abort`, or raw-action guard was
relaxed.

## Artifacts

- Ranker fix: `src/encirclement3d/jepa_safe_capture_ranker.py`
- Regression test: `tests/test_jepa_safe_capture_candidates.py`
- Aggregate script:
  `scripts/aggregate_jepa_safe_capture_v5_prior_observation_active_search.py`
- Aggregate output:
  `results/jepa_safe_capture_v5_prior_observation_ranker_fix_three_seed_aggregate_v2/`
- TensorBoard:
  `results/jepa_safe_capture_v5_prior_observation_ranker_fix_three_seed_aggregate_v2_tensorboard/`

The aggregate TensorBoard contains 61 scalar tags and 3 required text
summaries. All three fixed M3 runs contain their own event file and provenance
record.

## Verification

- Candidate/ranker and paired artifact tests: `55 passed`.
- TensorBoard aggregate audit: passed.
- Locked test opened: no.
- Development-only boundary: preserved.
