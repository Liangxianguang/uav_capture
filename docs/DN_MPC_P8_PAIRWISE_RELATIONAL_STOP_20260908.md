# DN-MPC P8 Pairwise Relational JEPA Stop Report

**Date:** 2026-09-08
**Phase:** development-only offline prediction audit
**Decision:** do not promote this checkpoint to online route ranking

## Scope

P8 added a per-agent, per-teammate action-conditioned relational feature to the
DN-MPC route archive. For each defender and each of the other three defenders,
the feature contains the candidate action chunk minus the teammate action
chunk. The three pair features are encoded separately and pooled before the
pairwise risk heads. The existing route candidates, reachable projection,
strict CBF counterfactual, controlled abort, and stale/OOD/non-finite safety
gates were unchanged.

The feature contract is `route_pairwise_relative_action_chunk` with shape
`[N, 5, 9]`. The train, validation and calibration archives are seed-disjoint
and remain development-only; no locked test data was used.

## Archive and provenance

| Split | Rows | Runtime rows | Interaction rows | Dataset SHA-256 |
|---|---:|---:|---:|---|
| train | 66,496 | 49,872 | 12,468 | `552b04e40b34062062e2f222de6e3192b86df38b1979e6fb0a97f9d735342671` |
| validation | 56,192 | 42,144 | 10,536 | `0f52f67d536e4e428724c6c1c57302449c94632f9289ec004b12114a1e1e2a87` |
| calibration | 32,704 | 24,528 | 6,132 | `f8f0a337752a2382876f6c45c3aadf0e4c2d07bd46c718f353636d6e94e03d84` |

The unified archive audit passed seed disjointness, finite labels, visibility
and feasibility class coverage, runtime boundary non-negativity, strict CBF
provenance, and TensorBoard archive scalars:

`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_archive_audit.json`

## Training

| Setting | Value |
|---|---|
| model | `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2` |
| relational head | enabled; per-agent/per-pair action-conditioned encoding |
| seed | `20260910` |
| epochs | 20 requested / 20 completed |
| batch size | 512 |
| hazard positive weight | 8 |
| device | RTX 5050, CUDA (`torch 2.7.1+cu128`) |
| best epoch | 18 |
| best validation loss | `-1.2768174467` |
| checkpoint SHA-256 | `e7a393d49ce5c5b47933e5aa2f3bf095c4510bdfa959262db774b059e377dc2d` |

Checkpoint:

`results/dn_mpc_jepa_safe_capture_checkpoints/route_identity_chunk5_pairwise_relational_v1_seed20260910/checkpoint.pt`

Training TensorBoard:

`results/dn_mpc_jepa_safe_capture_tensorboard/route_identity_chunk5_pairwise_relational_v1_seed20260910`

## Calibration gate

The prespecified gate is evaluated on the held-out calibration split for
pairwise TTC `<1 s` at probability cutoff `0.5`:

| Head | Recall | Precision | Gate |
|---|---:|---:|---|
| obstacle TTC | 96.20% | 62.63% | pass |
| boundary TTC | 94.90% | 54.12% | pass |
| pairwise TTC | 81.04% | 40.38% | **fail** |

The threshold sweep has no valid operating point. At cutoff `0.7`, pairwise
precision reaches `52.53%`, but recall falls to `67.93%`; at cutoff `0.5`,
recall is above the target but precision remains below `50%`.

Compared with P7 pairwise pooling (`79.85% / 37.11%`), P8 improves recall by
`+1.19 pp` and precision by `+3.27 pp`, but the calibration contract is still
not satisfied.

Prediction audit and its TensorBoard record:

`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_seed20260910_prediction_audit.json`

`results/dn_mpc_jepa_safe_capture_tensorboard/route_identity_chunk5_pairwise_relational_v1_seed20260910_prediction_audit`

## Stop decision

- Do not connect this checkpoint to online JEPA route ranking.
- Do not create or update a reliability ledger from this checkpoint.
- Do not run a JEPA closed-loop benchmark or claim a `safe_capture` gain.
- Keep strict analytic DN-MPC plus joint CBF as the only runtime-eligible
  controller.
- Retain the archive, checkpoint, audit JSON and TensorBoard event files as
  development evidence.

The result indicates that explicit relational action conditioning helps the
pairwise signal modestly, but false-positive control remains the bottleneck.
The next experiment should be a bounded calibration/data-contract analysis
(for example, per-interaction-mode calibration and hard-negative weighting),
not a larger backbone or an online safety decision based on this model.
