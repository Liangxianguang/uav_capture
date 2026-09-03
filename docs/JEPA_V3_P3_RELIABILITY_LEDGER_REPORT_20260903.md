# JEPA-v3 P3 Reliability Ledger Report

**Status:** implemented and admitted to further development smoke tests

**Scope:** development-only. This ledger does not open a V4/V5 locked block,
does not update online, and does not replace CBF.

## Design

For each JEPA training seed, P3 builds an immutable JSON ledger from the
`146,400`-sample counterfactual **validation** split. Each label was obtained
after a completed cloned-environment rollout. The ledger is never built from
the frozen 60-episode S3 development scenarios.

A bucket is indexed by prediction horizon, current visibility, normalized
message age, predicted minimum obstacle/inter-agent clearance, and physical
candidate-action magnitude. Its credit combines settled candidate-ranking
regret (weight `0.45`), target error (`0.20`), clearance error (`0.15`),
visibility error (`0.10`), and CBF-intervention error (`0.10`).

The pre-specified policy in
`configs/jepa_v3_development_protocol.yaml` is:

| Setting | Value |
| --- | ---: |
| Candidate count | 5 |
| Candidate perturbation | 0.10 m/s |
| Prediction horizon index | 3 (0.5 s) |
| Minimum local samples | 128 |
| Minimum credit | 0.65 |

When a selected candidate lacks local credit, its action is replaced with the
unchanged frozen V5 nominal action. The existing CBF filter is then applied in
all cases, including fallback. A ledger checksum must match the loaded JEPA
checkpoint before the evaluator will run.

## Offline Ledger Evidence

| JEPA seed | Global credit at 0.1 / 0.2 / 0.3 / 0.5 s | Candidate-ranking win rate at 0.5 s | Source-validation fallback at 0.5 s |
| ---: | --- | ---: | ---: |
| 20260911 | 0.8453 / 0.8459 / 0.8418 / 0.8298 | 0.8729 | 1.05% |
| 20260912 | 0.8318 / 0.8333 / 0.8297 / 0.8184 | 0.8685 | 1.72% |
| 20260913 | 0.8219 / 0.8322 / 0.8246 / 0.8182 | 0.8630 | 1.24% |

These quantities are offline agreement diagnostics, not estimated capture
rates. They are sufficient to audit a conservative fallback policy, not to
claim safety of the learned predictor.

## Zero-Perturbation Regression

The baseline and ledger-enabled runs replayed the same first 20 frozen S3
development scenes with the same V5 actor, CBF, per-step recurrent reset and
reference transit evidence. The ledger run used `K=5` candidates with
`perturbation=0`, so every candidate exactly equalled the nominal action.

| Check | Result |
| --- | --- |
| Episodes | 20 / 20 paired |
| Non-JEPA episode fields compared | 87 |
| Field differences | 0 |
| `scenes.jsonl` | byte-identical |
| Scene SHA-256 in both runs | `1402bf6429814f7638625025bc75a3b4ca04ac3c0bc107eef13ac0cdf2a18b99` |
| Baseline / zero-ledger safe capture | 18/20 / 18/20 |
| Baseline / zero-ledger collision and boundary | 0% / 0%; 0% / 0% |

This passes the deterministic zero-perturbation regression gate.

## Non-Zero Ledger Smoke

Only after the regression passed, seed `20260911` ran the same 20 frozen scenes
with the pre-specified `0.10 m/s` alternatives:

| Metric | V5 + CBF baseline | Multitask JEPA + ledger + CBF |
| --- | ---: | ---: |
| Safe capture | 18/20 (90.0%) | 19/20 (95.0%) |
| Collision / boundary | 0% / 0% | 0% / 0% |
| Paired improved / degraded / tied | \- | 1 / 0 / 19 |
| Mean total defender path | 85.37 m | 78.08 m |
| Mean capture time | 6.17 s | 5.99 s |
| Mean runtime nominal fallback | \- | 11.37% |
| Mean runtime global/OOD fallback | \- | 5.34% |

The runtime fallback rate is higher than its source-validation forecast because
some frozen S3 contexts are absent from local ledger buckets. That conservative
behavior is intended. The smoke is one seed and 20 episodes, so it is not
evidence of a general capture-rate improvement and cannot be used to tune the
ledger thresholds.

## Verification

The reliability core, multitask candidate decoder, low-credit nominal fallback,
S3 evaluator, and existing prediction paths passed:

```text
18 passed
```

P3 is therefore complete as a safety-preserving runtime mechanism. P4/P5 must
still be evaluated against paired controls before the final three-seed decision.
