# DN-MPC P5 Dense Hard-Negative Replay

**Date:** 2026-09-07
**Protocol:** development-only; locked test not opened

## Purpose

The four-episode chunk-5 pilot had too few interaction transitions for a
reliable pairwise hazard head. This stage reduced archive sampling stride from
`8` to `2` and collected eight disjoint episodes per split. The CBF contract,
frozen actor, candidate generator and information boundary were unchanged.

Two fresh checkpoints were trained on the same dense train/validation archives:

- `dense_v2`: default hazard positive weight `4`;
- `dense_v2_hazard8`: hazard positive weight `8` for the controlled weighting
  ablation.

Both runs wrote complete TensorBoard event files, metadata and provenance.

## Archive Evidence

| Split | Episodes | Samples | Runtime rows | Route-state groups | First-step CBF feasible |
|---|---:|---:|---:|---:|---:|
| train | 8 | 27,820 | 25,680 | 535 | 99.63% |
| validation | 8 | 36,244 | 33,456 | 697 | 99.71% |
| calibration | 8 | 39,312 | 36,288 | 756 | 99.87% |

The archive label audit passed with seed-disjoint splits, both validation
classes, finite labels, non-negative runtime boundary clearance and all required
TensorBoard scalars. The calibration block now exceeds 500 route-state groups;
it is still development evidence, not a locked test.

## Prediction Audit

Recall/precision/Brier below are for the predicted hazard band `TTC < 1 s` on
the independent calibration split.

| Checkpoint | Obstacle recall / precision | Boundary recall / precision | Pairwise recall / precision |
|---|---:|---:|---:|
| pilot v1 | 91.0% / 58.9% | 87.2% / 37.8% | 40.5% / 46.3% |
| dense v2 | 91.6% / 74.3% | 76.7% / 56.1% | 54.8% / 53.7% |
| dense v2, hazard weight 8 | 88.7% / 76.6% | 85.6% / 54.5% | **61.6% / 47.3%** |

The dense archive and positive weighting improve pairwise detection, but the
main safety failure remains interaction under-detection. A threshold of `0.2`
raises hazard-8 pairwise recall to `90.3%` but lowers precision to `27.1%`;
using that threshold online would likely reject too many otherwise useful
routes and reproduce the over-conservative safe-capture bottleneck.

## Gate Decision

**Calibration gate: not passed.** The checkpoint is not connected to online
JEPA ranking, no Ledger-Lite is generated, and no L0-L3 closed-loop result is
reported from it. The strict analytic DN-MPC + CBF path remains the only
runtime-eligible path.

This is the second bounded calibration attempt. Per the stop rule, further
network-size or loss-weight sweeps are stopped. The next improvement must be a
contract-level data/feature change targeted at interaction transitions, followed
by a fresh archive and calibration protocol revision.

## Artifacts

- Dense archive audit:
  `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_dense_v2_archive_audit.json`
- Default prediction audit:
  `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_dense_v2_prediction_audit.json`
- Hazard-weight prediction audit:
  `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_dense_v2_hazard8_prediction_audit.json`
- Dense training TensorBoard:
  `results/dn_mpc_jepa_safe_capture_tensorboard/p5_train_chunk5_dense_v2_seed20260907`
- Hazard-weight TensorBoard:
  `results/dn_mpc_jepa_safe_capture_tensorboard/p5_train_chunk5_dense_v2_hazard8_seed20260907`

## Next Allowed Work

1. Freeze the analytic DN-MPC + CBF G5 baseline as the comparison anchor.
2. Design a new archive revision with explicit pairwise interaction transitions
   (formation crossing, near-pass, route split/merge and CBF-infeasible branches)
   rather than more nominal dense replay.
3. Add a public interaction/topology feature or balanced sampling rule only in
   that new protocol; keep the current checkpoints immutable.
4. Re-run calibration before any online JEPA ranking or Ledger-Lite work.
