# DN-MPC P5 Chunk-5 Archive and Training Smoke

**Date:** 2026-09-07
**Protocol:** development-only; no locked test opened
**Direction:** DN-MPC-Hybrid

## Scope

This stage validates the new five-step route-identity archive contract and a
small JEPA evaluator training smoke. It does not claim a benchmark improvement
and does not connect the checkpoint to the online evaluator or Ledger-Lite.

The archive collector uses the frozen runtime actor
`models/v5_development_exact_reactive_seed661606.pt`. Target truth is used only
for offline labels. Every runtime route branch is first-step CBF verified; the
offline boundary shadow is never executed.

## Contract Repair

The first chunk-5 pilot exposed a real archive bug: normal route candidates had
five-step chunks while the offline `boundary_shadow` path was hard-coded to
three steps. The collector now passes the requested chunk length through the
shadow rollout and validates positive horizon lengths. The archive collector
also reports field-level shape inconsistencies instead of silently coercing or
padding them.

Focused route/archive regression: **37 passed**.

## Archive Results

| Split | Episodes | Samples | Runtime rows | `route_action_chunk` | Geometry valid | First-step CBF feasible |
|---|---:|---:|---:|---|---:|---:|
| train pilot | 4 | 3536 | 3264 | `[3536, 5, 3]` | 56.4% | 100% |
| validation pilot | 4 | 2808 | 2592 | `[2808, 5, 3]` | 74.5% | 100% |
| calibration pilot | 4 | 5096 | 4704 | `[5096, 5, 3]` | 62.7% | 100% |

All three archives contain explicit boundary-shadow negatives and branch
failure labels. No raw-unverified action was executed. TensorBoard event files
were written for all collection runs under
`results/dn_mpc_jepa_safe_capture_tensorboard/`.

## Training Smoke

The evaluator was trained from scratch for at most 10 epochs on the train and
validation pilots, with a fresh chunk-5 model and a separate TensorBoard run.
The old three-step checkpoint and old Reliability Ledger were not reused.

| Metric | Result |
|---|---:|
| Best epoch | 6 |
| Best validation loss | 1.6204 |
| Epoch-1 validation loss | 3.6232 |
| Epoch-10 validation loss | 2.8237 |
| Validation route identity accuracy | 100% |
| Stop reason | no validation improvement for 4 epochs |

The training loss kept falling while validation loss worsened after epoch 6.
This is an overfitting/data-coverage warning, not evidence that JEPA improves
safe capture. Per the master plan, expansion is stopped until labels, route
diversity and calibration are audited. The checkpoint is development-only and
has not been used for online control.

## Artifacts

- Train archive: `results/dn_mpc_route_identity_archive_chunk5_pilot_train4_seed20260907/`
- Validation archive: `results/dn_mpc_route_identity_archive_chunk5_pilot_validation4_seed20260907/`
- Calibration archive: `results/dn_mpc_route_identity_archive_chunk5_pilot_calibration4_seed20260907/`
- Training checkpoint: `results/dn_mpc_jepa_safe_capture_checkpoints/route_identity_chunk5_pilot_seed20260907/`
- Training TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p5_train_chunk5_pilot_seed20260907/`

These generated artifacts remain local and are not added to Git. Their metadata
and provenance files contain the input hashes, code revision and environment
information needed for replay.

## Next Gate

Before any online JEPA ranking or Ledger-Lite implementation:

1. Audit calibration loss and route-ranking calibration on the independent
   calibration archive.
2. Add more scene families only if the calibration audit shows a coverage gap;
   do not simply increase network size.
3. Compare analytic DN-MPC + CBF against JEPA-disabled and JEPA-ranking replay
   on the same fixed G5 manifest, with safety events and controlled abort as
   hard gates.
