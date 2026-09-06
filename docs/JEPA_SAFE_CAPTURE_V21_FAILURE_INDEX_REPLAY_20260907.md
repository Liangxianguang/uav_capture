# V21 Failure Index and Deterministic Replay

**Date:** 2026-09-07
**Scope:** development-only read-only S2 diagnostic
**Locked test:** `false`
**Online contract:** unchanged

## Result

The current V21 paired smoke traces were indexed again using a new output
prefix. The input contains 12 runs (M0/M3/A1/A2 across three training seeds),
240 episodes, and the same settled local rows used only for offline diagnosis.

| Metric | Result |
|---|---:|
| Safe capture | `117/240 = 48.75%` |
| Failed episodes | `123` |
| Primary `cbf_controlled_abort` | `116` |
| Primary `timeout` | `7` |
| Collision / boundary / pairwise | `0 / 0 / 0` |
| Raw-unverified executed steps | `0` |
| M3 safe capture | `28/60 = 46.67%` |
| M3 settled-rank-mismatch episodes | `24` |

The safety hard gate passes, but the capability gate does not. M3 remains below
the M0 reference (`30/60 = 50.0%`) and is not eligible for a larger block or
new training run.

## Failure attribution

The dominant termination path is a CBF controlled abort, not a physical safety
violation. The diagnostic index also records candidate settled-regret,
high-credit failure, fallback, visibility, and clearance-gap labels. The
communication-age field reaches its configured saturation value in all failed
episodes, so it is currently non-discriminating and must not be treated as
proof of target-observation staleness or target drift.

The evidence supports three follow-up checks before any new checkpoint:

1. inspect the earliest CBF infeasibility cycle and stopping-distance/pairwise
   constraints;
2. separate candidate eligibility and nominal-anchor fallback from JEPA route
   score error;
3. repair the communication-age state-machine semantics and rerun only the
   offline contract audit.

## Deterministic replay

The selection policy requested three examples per diagnostic category. Fifteen
episodes were available and selected: three candidate-regression, three
high-credit, three nominal-fallback, three stale/noisy, and three timeout
episodes. No candidate-oscillation example existed, so the shortage is
reported rather than fabricated.

Each selected trace was canonically derived twice. All `15/15` repeat hashes
matched. The replay terminations were `12 cbf_controlled_abort` and `3
timeout`; replay itself executed no environment rollout and inferred no future
target ground truth.

## Gate decision

This S2 result is diagnostic progress, not a method improvement. Keep CBF
margins, stale/OOD/non-finite gates, controlled-abort semantics, independent
counterfactual probes, and raw-action guards unchanged. Do not train a new
JEPA, enlarge L1-L3, or open a locked test until S3 contract diagnosis passes.
If the next offline repair still leaves settled ranking below the promotion
threshold, retain the label `ranking_unresolved` or
`prediction_signal_no_control_gain`.

## Provenance

- Failure index script: [index_jepa_safe_capture_v21_failures.py](../scripts/index_jepa_safe_capture_v21_failures.py)
- Replay script: [replay_jepa_safe_capture_failures.py](../scripts/replay_jepa_safe_capture_failures.py)
- Failure-index JSON: `results/jepa_safe_capture_v21_failure_index_20260907/failure_index.json`
- Failure-index JSON SHA-256: `48e45c5a944d11180a34bac24864b99ef46d3fce1f1eaab09ff3f6b4a071f960`
- Failure-index CSV SHA-256: `63e5e3f538cbb3ad5af8ed88e25b0980cd5c3c8dcc5ddcdf8e524637630bf0e1`
- Replay summary: `results/jepa_safe_capture_v21_failure_replay_20260907/replay_summary.json`
- Replay summary SHA-256: `c72b731676a990ed341b2946ac5dff592749610c5705fbcedac2782c569e23cf`
- Replay hash manifest SHA-256: `468632cb87486baa9b61a7baabcdf7d18b3f64bfdfb37b9ce67827710ebda5df`
- Failure-index TensorBoard event SHA-256: `ec2a400ec3eefbbc611a25c6f58d9559a1e43be7536aadb3bdbf60f95f7fe29a`
- Replay TensorBoard event SHA-256: `4ea0073d6faf7ff5826d2274295f97624776d53e19fe8ce28d93d54099c820fc3`
