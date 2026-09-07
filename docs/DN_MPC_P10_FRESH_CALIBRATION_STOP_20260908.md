# DN-MPC P10 Fresh Calibration Stop

**Date:** 2026-09-08
**Phase:** development-only, fresh disjoint calibration
**Decision:** keep P8 relational JEPA out of runtime

## Purpose

P9 identified runtime and interaction-mode false positives as the source of
the pairwise calibration failure. P10 collected a fresh calibration block
with a new seed range to test whether that result was an artifact of the
original calibration scenes.

The new block uses the same frozen runtime actor, candidate route contract,
reachable projection, strict CBF horizon 5, offline-only interaction branches,
and unchanged safety gates. Its seed range is `650101-650120`, disjoint from
the train (`645101-645140`), validation (`646101-646140`) and original
calibration (`648101-648120`) blocks.

## Fresh archive

The archive contract audit passed:

`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_p9_fresh_calibration_archive_audit.json`

| Quantity | Value |
|---|---:|
| total rows | 32,896 |
| runtime rows | 24,672 |
| boundary-shadow rows | 2,056 |
| rows per interaction mode | 2,056 |
| CBF first-step feasible fraction | 100% |
| runtime negative boundary-clearance rows | 0 |
| archive SHA-256 | `b27b9ca6064dc34600eda64c3671a11e41274a9c07339f477426d097b8a84b11` |

The source archive and collection TensorBoard are retained at:

`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_p9_fresh_calibration_source`

`results/dn_mpc_jepa_safe_capture_tensorboard/route_identity_chunk5_pairwise_relational_v1_p9_fresh_calibration_source`

## Fresh prediction gate

The frozen P8 checkpoint was evaluated without retraining:

`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_seed20260910_fresh_calibration_prediction_audit.json`

For pairwise TTC `<1 s`:

| Probability cutoff | Recall | Precision | Gate |
|---:|---:|---:|---|
| 0.5 | 84.50% | 40.40% | fail |
| 0.7 | 72.42% | 53.58% | fail |
| 0.9 | 49.47% | 84.89% | fail |

The fresh block therefore has no operating point satisfying recall `>=80%`
and precision `>=50%`. The failure is not caused by reusing the original
calibration scenes.

At cutoff 0.5, the mode-stratified results are:

| Sample type | Count | Positive rate | Recall | Precision |
|---|---:|---:|---:|---:|
| runtime | 123,360 | 11.2% | 71.8% | 28.0% |
| boundary shadow | 10,280 | 100.0% | 100.0% | 100.0% |
| near pass | 10,280 | 15.8% | 79.8% | 28.5% |
| formation crossing | 10,280 | 28.4% | 89.7% | 36.2% |
| split/merge | 10,280 | 13.1% | 90.9% | 23.4% |

Detailed stratification and TensorBoard:

`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_seed20260910_fresh_calibration_by_mode.json`

`results/dn_mpc_jepa_safe_capture_tensorboard/route_identity_chunk5_pairwise_relational_v1_seed20260910_fresh_calibration_by_mode`

## Stop decision

- Do not retrain a larger JEPA or tune a global hazard cutoff from these data.
- Do not connect P8 to online route ranking, a reliability ledger, or a
  safe-capture closed-loop benchmark.
- Keep strict analytic DN-MPC plus joint CBF as the only runtime controller.
- Treat pairwise hazard as an offline diagnostic until a new model passes the
  same gate on a fresh calibration block.

The repeated failure narrows the next research task to representation/label
calibration for runtime and split/merge interactions. It is not evidence that
CBF margins should be relaxed and it is not evidence of an end-to-end JEPA
improvement.
