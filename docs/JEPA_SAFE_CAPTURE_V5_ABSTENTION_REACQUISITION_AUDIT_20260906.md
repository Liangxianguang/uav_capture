# V5 Abstention and Reacquisition Audit

**Date:** 2026-09-06  
**Status:** development-only diagnostic  
**Locked test:** not opened  
**Safety policy:** CBF margins, stale/OOD gates, and `controlled_abort` are unchanged

## Purpose

This audit closes the next permitted work item after the route-12 settled
three-seed result. It separates two questions:

1. Does removing nominal-anchor/abstention interference recover control
   performance on already-settled candidate branches?
2. Is the bounded reacquisition route causally justified by the public
   observation, and does it actually produce a search action?

No online decision was changed and no new checkpoint was trained.

## Settled Abstention Counterfactual

The input is the frozen route-CBF settled replay for training seeds
`20260911`, `20260912`, and `20260913` (1550 decisions). `score_argmin` is an
offline diagnostic that chooses the minimum finite JEPA score among the
recorded eligible routes. `settled_best` is an oracle upper bound and is not a
deployable policy.

| Quantity | Recorded | Score argmin |
| --- | ---: | ---: |
| Multi-eligible decisions | 1549 | 1549 |
| Selected-not-settled-best | 78.37% | 79.92% |
| Settled safety | 99.87% | 99.87% |
| Settled safe-capture | 0.721% | 0.710% |
| Mean settled progress (m) | 0.1764 | 0.1765 |

Recorded and score-argmin selections agree on `93.48%` of multi-eligible
decisions, while score-argmin agrees with the settled-best oracle on only
`20.08%`. Removing the anchor/abstention effect therefore does not recover a
control gain; it is marginally worse in this replay.

**Decision:** do not disable abstention, do not continue JEPA training or data
expansion, and do not enlarge the L1-L3 matrix. The remaining problem is the
mapping from predicted route score to settled outcome, together with the
quality of the reacquisition action itself.

## Public Observation Contract Audit

The frozen M3 validation trace was inspected using only its pre-action public
observation snapshots. Target truth is not used by the controller and is only
available to the audit as an offline label.

| Episode | Steps | Never received | Zero-belief while never received | Cautious steps | Non-zero cautious actions | Safety hard gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 125 | 0 | n/a | 0 | 0 | 0 |
| 1 | 250 | 4 | 4/4 | 3 | 0 | 0 |
| 2 | 45 | 0 | n/a | 0 | 0 | 0 |

Episode 1 confirms that a never-initialized belief is represented as zero and
cannot justify an active target-directed search. The three cautious steps
used the configured `visibility_hold`, but its requested action was zero, so
the route did not create useful information. This is a route-design failure,
not evidence that stale/OOD gates should be relaxed.

## Contract Change

`SafeCaptureRankerConfig` now exposes
`cautious_reacquisition_requires_prior_observation` and defaults it to `true`.
When enabled, `cautious_reacquisition` requires at least one public
`target_observation_received` flag. A never-received target remains on the
existing `safe_hold` path (`observation_never_received`); it cannot be
promoted to a trusted route. A target that was previously received but is
temporarily invisible may still receive the existing short reacquisition
budget, subject to the independent selected/nominal/safe-hold CBF probes.

This preserves:

- `collision=0`, `boundary=0`, `pairwise=0`, and `raw_unverified=0` hard gates;
- stale, OOD, and non-finite prediction handling;
- reachable projection, residual verification, and `controlled_abort`;
- the requirement to execute only the first verified action step.

## Next Stop Condition

The next implementation block is a bounded **public-geometry active-search**
contract. It may use only a finite interior/lateral scan waypoint, a three-step
budget, and the same three independent CBF probes. It must record a non-zero
projected action and a follow-up observation before it can be evaluated for
capture benefit. If a fixed micro-scene replay does not increase useful
reacquisition steps or safe-capture while preserving all hard gates, stop and
return to route-score/auxiliary-head diagnosis. Do not train a larger model or
add more random data before this gate passes.

## Provenance

- Settled counterfactual report: `results/jepa_safe_capture_v5_route12_abstention_cf_three_seed/`
- Settled counterfactual TensorBoard: `results/jepa_safe_capture_v5_route12_abstention_cf_three_seed_tensorboard/`
- Observation audit episode 0: `results/jepa_safe_capture_v5_prior_observation_contract_audit/`
- Observation audit episode 1: `results/jepa_safe_capture_v5_prior_observation_contract_audit_ep1/`
- Observation audit episode 2: `results/jepa_safe_capture_v5_prior_observation_contract_audit_ep2/`
- TensorBoard is required for each audit; the listed directories contain the
  corresponding event files and provenance text summaries.
- Source trace hashes and checkpoint/protocol hashes remain in the original
  audit JSON/provenance files; no locked benchmark artifact was overwritten.

