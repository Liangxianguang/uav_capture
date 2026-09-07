# DN-MPC P6 Interaction-Conditioned Route Feature Stop

**Date:** 2026-09-08  
**Phase:** development-only; locked test not opened

## Objective

This bounded experiment tests one contract-level feature change after the
dense hard-negative checkpoint failed calibration: append the candidate action
relative to the mean candidate action of the other three defenders to the
route-JEPA input. Both the route chunk and the new feature are normalized by
the archive `action_scale` before entering the model.

For defender `i`, the derived feature is

```text
u_i - mean(u_j for j != i)
```

The feature is public and action-conditioned. It is not a control output and
does not replace the analytic DN-MPC or strict joint CBF filter.

## Archive Contract

The source dense archives were not modified. The new revision was materialized
from complete route-state groups, each containing exactly four defenders. The
source archive hash and the derived archive hash are recorded in each
`derived_feature_contract` and `provenance.json`.

| Split | Samples | Route-state groups | Episode seeds |
|---|---:|---:|---|
| train | 27,820 | 6,955 | 645101-645108 |
| validation | 36,244 | 9,061 | 646101-646108 |
| calibration | 39,312 | 9,828 | 648101-648108 |

The archive label audit passed: seed-disjoint splits, both validation CBF
classes, boundary-shadow negatives, visibility classes, finite labels, and the
required TensorBoard scalars. All archives remain development-only.

## Training

The model was trained with the same architecture and hazard-positive weighting
as the strongest previous dense ablation. No network-size or loss-weight sweep
was performed.

| Setting | Value |
|---|---|
| model | `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2` |
| route interaction dimension | 3 |
| seed | 20260908 |
| batch size | 512 |
| requested/completed epochs | 20 / 9 |
| best epoch | 4 |
| best validation loss | 0.9812448398 |
| hazard positive weight | 8 |
| stop reason | validation early stop after 5 stale epochs |
| checkpoint SHA-256 | `eef6397ec6e60f198c9b85f3fd47b3c0251b35256535f809294d1cf1b0bb8be8` |

TensorBoard logs:

`results/dn_mpc_jepa_safe_capture_tensorboard/p6_archive_chunk5_interaction_train8_seed20260908`

`results/dn_mpc_jepa_safe_capture_tensorboard/p6_archive_chunk5_interaction_validation8_seed20260908`

`results/dn_mpc_jepa_safe_capture_tensorboard/p6_archive_chunk5_interaction_calibration8_seed20260908`

`results/dn_mpc_jepa_safe_capture_tensorboard/p6b_train_chunk5_interaction_v1_scaled_hazard8_seed20260908`

`results/dn_mpc_jepa_safe_capture_tensorboard/p6b_prediction_chunk5_interaction_v1_scaled_hazard8_seed20260908`

## Calibration Result

The gate is evaluated on the independent calibration split for TTC `<1 s`.

| Model | Obstacle recall / precision | Boundary recall / precision | Pairwise recall / precision |
|---|---:|---:|---:|
| dense v2, hazard weight 8 | 88.7% / 76.6% | 85.6% / 54.5% | 61.6% / 47.3% |
| P6 interaction feature (scaled) | 92.3% / 61.3% | 90.3% / 51.0% | **66.9% / 38.3%** |

The pairwise recall gain is `+5.3 pp`, while precision falls by `9.0 pp`.
The required pairwise gate (`recall >= 80%` and `precision >= 50%`) is not
met. The model therefore cannot safely route candidate actions without
reproducing the over-conservative or false-positive behavior that motivated
this experiment.

A first smoke run used the unnormalized derived feature and was discarded
after the loader audit identified the scale mismatch. Its checkpoint and
TensorBoard run are retained only as debugging provenance and are not part of
the comparison above.

## Decision

This bounded feature experiment is **stopped at calibration**.

- Do not connect this checkpoint to online JEPA ranking.
- Do not create or update a Ledger-Lite from this checkpoint.
- Do not run an L0-L3 JEPA closed-loop claim from this checkpoint.
- Keep the strict analytic DN-MPC + joint CBF path as the runtime-eligible
  baseline.
- Keep CBF margins, stale/OOD/non-finite gates, reachable projection, and
  controlled abort unchanged.

The result does not prove that JEPA is unusable in general. It shows that this
single candidate-action feature, derived from the existing dense state
distribution, is insufficient to make the pairwise risk head reliable under
the current contract. The next change must add genuinely diverse interaction
transitions (formation crossing, near-pass, split/merge, and CBF-infeasible
branches) rather than increase model capacity or tune a threshold.

## Artifacts

- Archive audit:
  `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_interaction_v1_archive_audit.json`
- Prediction audit:
  `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_interaction_v1_scaled_hazard8_prediction_audit.json`
- Checkpoint:
  `results/dn_mpc_jepa_safe_capture_checkpoints/route_identity_chunk5_interaction_v1_scaled_hazard8_seed20260908/checkpoint.pt`
- Materialization utility:
  `scripts/materialize_route_interaction_archive.py`
