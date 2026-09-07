# DN-MPC P7 Interaction Transitions and Pairwise Pooling Stop

**Date:** 2026-09-08
**Phase:** development-only; locked test not opened

## Decision

The targeted interaction-transition archive and pairwise-pooling JEPA
checkpoint are **not promoted** to online route ranking. The analytic DN-MPC
plus strict joint CBF path remains the only runtime-eligible controller.

This is a calibration stop, not a safety relaxation. CBF margins, reachable
projection, stale/OOD/non-finite gates, and controlled abort are unchanged.

## What Changed

Three explicit, offline-only action branches were added at every sampled
state:

- `near_pass` (sample type `2`),
- `formation_crossing` (sample type `3`),
- `split_merge` (sample type `4`).

Each branch is projected through the environment's reachable acceleration and
speed envelope, evaluated by `_route_rollout`, and checked by the same strict
CBF verifier. None of these synthetic branches is executed by `env.step()` or
eligible for runtime route identity/ranking supervision.

## Archive Contract and Audit

The new development dataset version is
`dn_mpc_route_identity_chunk5_interaction_transitions_v1`.

| Split | Total rows | Runtime rows | Interaction rows | Episodes | Dataset SHA-256 |
|---|---:|---:|---:|---:|---|
| train | 66,496 | 49,872 | 12,468 | 40 | `ab601719...dc50812` |
| validation | 56,192 | 42,144 | 10,536 | 40 | `ebae3bac...eeb664` |
| calibration | 32,704 | 24,528 | 6,132 | 20 | `d5c20cc2...98582ca` |

The three splits are seed-disjoint (`645101-645140`, `646101-646140`, and
`648101-648120`). The archive audit passed finite-label checks, visibility and
feasibility class coverage, runtime boundary non-negativity, TensorBoard
provenance, and the explicit sample-type contract.

Pairwise tail coverage improved materially. On calibration, all 20 episodes
contain a pairwise TTC `<1 s` window; the best observable public feature,
observation pairwise TTC, has AUC `0.706` for that tail.

Archive audit:
`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_interaction_transitions_v1_archive_audit.json`

Pairwise-tail audit:
`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_interaction_transitions_v1_pairwise_tail_audit.json`

## Training

The fresh checkpoint used the new train/validation archives and an explicit
latest-frame pairwise pooling block. All training metrics and hyperparameters
were recorded in TensorBoard.

| Setting | Value |
|---|---|
| model | `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2` |
| route chunk | 5 steps; interaction feature dimension 3 |
| pairwise pooling | enabled |
| seed | `20260909` |
| batch size | 512 |
| requested/completed epochs | 20 / 19 |
| best epoch | 14 |
| best validation loss | `-1.2731487226` |
| device | RTX 5050, CUDA (`torch 2.7.1+cu128`) |
| checkpoint SHA-256 | `8f0a0cc46c71783f7cee1b941906e0443dc4a3dc5c7d888ff08681f587e74347` |

Checkpoint:
`results/dn_mpc_jepa_safe_capture_checkpoints/route_identity_chunk5_interaction_transitions_v1_pairwise_pooling_seed20260909/checkpoint.pt`

TensorBoard:
`results/dn_mpc_jepa_safe_capture_tensorboard/route_identity_chunk5_interaction_transitions_v1_pairwise_pooling_seed20260909`

## Independent Calibration Gate

Hazard results for TTC `<1 s` at the prespecified probability cutoff `0.5`:

| Head | Recall | Precision | Gate |
|---|---:|---:|---|
| obstacle TTC | 98.30% | 58.36% | pass |
| boundary TTC | 92.55% | 54.49% | pass |
| pairwise TTC | 79.85% | 37.11% | **fail** |

Pairwise recall improved from the preceding P6 value of `66.9%` to `79.85%`,
but precision remains well below `50%`, and recall is fractionally below the
`80%` requirement. The threshold sweep has no cutoff satisfying both
`recall >= 80%` and `precision >= 50%`; therefore changing the cutoff would
not constitute a valid promotion.

Prediction audit:
`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_interaction_transitions_v1_pairwise_pooling_seed20260909_prediction_audit.json`

Prediction TensorBoard:
`results/dn_mpc_jepa_safe_capture_tensorboard/route_identity_chunk5_interaction_transitions_v1_pairwise_pooling_seed20260909_prediction_audit`

## Consequences

- Do not connect this checkpoint to online JEPA route ranking.
- Do not create or update a reliability ledger from this checkpoint.
- Do not claim an L0-L3 JEPA closed-loop improvement from this run.
- Keep the strict analytic DN-MPC + joint CBF controller as the baseline.
- Keep the new archive and TensorBoard runs as development evidence; do not
  delete them.

## Next Bounded Experiment

The remaining bottleneck is pairwise hazard calibration, not the amount of
nominal trajectory data. The next experiment should add an explicit
per-agent/per-pair interaction pooling or pairwise relational head and
recalibrate it on a fresh disjoint block. It must preserve the same sample
types, reachable projection, CBF counterfactual, and safety gates. If the
pairwise `80%/50%` gate is not met again, stop expanding JEPA capacity and
continue with analytic DN-MPC route ranking plus CBF for the closed-loop
benchmark.
