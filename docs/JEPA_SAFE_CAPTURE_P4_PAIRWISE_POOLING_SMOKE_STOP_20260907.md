# JEPA Safe Capture P4 Pairwise-Pooling Smoke Stop

**Date:** 2026-09-07
**Phase:** development-only, offline prediction audit
**Decision:** `prediction_signal_improved_but_pairwise_gate_failed`

## Purpose

The preceding hazard-head smoke showed that the frozen route representation did
not generalize pairwise TTC risk from validation to calibration. The bounded
follow-up added an explicit pooling block for the latest public teammate
relative position/velocity features and trained only that block plus the five
risk heads. The shared JEPA backbone, legacy route heads, CBF contract, Ledger
contract, and runtime controller were unchanged.

This was a prediction-only experiment. No candidate was executed, no new
Ledger was built, and no locked split was opened.

## Contract

| item | value |
|---|---|
| model | `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2` |
| base checkpoint | `results/jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt` |
| base checkpoint SHA-256 | `4b68f8e61df978b79cb58113e6e960fc930912e992469c604abfe5f972a3bdf1` |
| output checkpoint | `results/jepa_route_identity_pairwise_pool_headonly_smoke_seed20260907/checkpoint.pt` |
| output checkpoint SHA-256 | `9777714342a16c3c8c0e6dcf9c2b54ec8abe5a974620b14d340bef2ba90ff0b9` |
| training | 8 epochs, batch 256, seed `20260907` |
| trainable scope | pairwise pooling + risk heads only |
| device | CUDA, NVIDIA GeForce RTX 5050 |
| training TensorBoard | `results/jepa_route_identity_pairwise_pool_headonly_smoke_seed20260907_tensorboard/` |
| prediction TensorBoard | `results/jepa_route_identity_pairwise_pool_prediction_audit_seed20260907_tensorboard/` |
| prediction audit | `results/jepa_route_identity_pairwise_pool_prediction_audit_seed20260907.json` |

The pooling block reads the public local interaction block `[15:33]`:
three relative teammate positions followed by three relative teammate
velocities. It does not use target ground truth or future simulator state.

## Held-out pairwise hazard result

Metrics below use the default probability cutoff `0.5` and are computed on
all five horizon labels. The prior hazard-head smoke is included as the
comparison, not as a new baseline run.

| split | prior `<=1s` recall | prior precision | pooling `<=1s` recall | pooling precision |
|---|---:|---:|---:|---:|
| validation | 51.4% | 35.9% | 61.5% | 48.4% |
| calibration | 37.9% | 37.8% | 42.9% | 40.1% |

On calibration, the threshold sweep still exposes the unsafe tradeoff:

- cutoff `0.2`: `84.6%` recall, `32.6%` precision;
- cutoff `0.5`: `42.9%` recall, `40.1%` precision;
- cutoff `0.7`: `31.9%` recall, `66.2%` precision.

Using the predeclared bounded gate `recall >= 80%` and `precision >= 50%`, no
cutoff passes both conditions. The pooling block therefore gives a modest
held-out improvement but does not provide a reliable pairwise risk gate.

## Interpretation

The interaction-tail audit found that the public observation contains useful
pairwise information: a direct observation TTC proxy had calibration AUC
`0.796` for the `<=1s` label. The pooling smoke improves validation and
calibration somewhat, so the issue is not simply “no teammate information.”
However, the calibration precision/recall frontier remains inadequate and is
not evidence of safe-capture improvement. The remaining limitation is the
action-conditioned future interaction representation and/or its calibration
under route and split shift, rather than a reason to relax CBF safety.

## Stop rule and next action

Stop the following until the pairwise gate is passed on a fresh held-out
calibration archive:

- full JEPA fine-tuning or three-seed expansion;
- new Reliability Ledger construction or threshold selection;
- L0-L3 closed-loop evaluation;
- changes to CBF margins, stale/OOD/non-finite gates, or `controlled_abort`.

The only permitted next action is another bounded offline design iteration:
make pairwise pooling explicitly action-conditioned (relative velocity projected
under the candidate chunk, formation topology, and per-pair risk) and evaluate
the same calibration protocol. If that does not pass the gate, stop adding
model capacity and collect targeted disjoint transition data instead.

## Safety and provenance

- `collision`, `boundary`, `pairwise`, and `raw_unverified` were not generated
  because no runtime episode was executed.
- `locked_test_opened=false`.
- `raw_unverified_execution_allowed=false`.
- CBF margin and stale/OOD/non-finite gates were unchanged.
- All training and prediction metrics/configuration are recorded in the two
  TensorBoard directories above.
