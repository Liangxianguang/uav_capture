# DN-MPC P9 Pairwise Calibration Stratification

**Date:** 2026-09-08
**Phase:** development-only, read-only calibration analysis

## Purpose

P8 improved pairwise hazard recall and precision slightly but still failed the
joint calibration gate. P9 stratifies the held-out calibration predictions by
archive sample type to identify the source of the remaining false positives.
It does not retrain the model, change any threshold used by runtime, or alter
CBF/controlled-abort behavior.

Analysis script:

`scripts/analyze_jepa_pairwise_calibration.py`

The script records every hazard-band and cutoff in TensorBoard and writes the
following artifact:

`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_seed20260910_calibration_by_mode.json`

TensorBoard:

`results/dn_mpc_jepa_safe_capture_tensorboard/route_identity_chunk5_pairwise_relational_v1_seed20260910_calibration_by_mode`

## Calibration evidence

Pairwise TTC `<1 s`, probability cutoff `0.5`:

| Sample type | Count | Positive rate | Recall | Precision | False-positive rate |
|---|---:|---:|---:|---:|---:|
| all | 163,520 | 21.4% | 81.0% | 40.4% | 32.6% |
| runtime | 122,640 | 14.9% | 69.7% | 30.6% | 27.8% |
| boundary shadow | 10,220 | 100.0% | 100.0% | 100.0% | n/a |
| near pass | 10,220 | 17.9% | 77.2% | 27.2% | 45.1% |
| formation crossing | 10,220 | 31.2% | 86.3% | 37.1% | 66.4% |
| split/merge | 10,220 | 13.8% | 84.2% | 21.2% | 50.1% |

The boundary-shadow rows are all pairwise positives under this label contract,
so they do not contribute false positives. The precision loss is concentrated
in runtime rows and the three interaction modes, with split/merge the weakest
mode.

No interaction mode has a cutoff satisfying both recall `>=80%` and precision
`>=50%`. Among cutoffs retaining recall `>=80%`, the best precision is only
`25.6%` for near pass, `37.1%` for formation crossing and `21.2%` for
split/merge. This is a calibration/data-distribution failure, not evidence
that a more permissive safety threshold is justified.

## Decision and next bounded step

- Keep P8 out of online route ranking, reliability ledgers and closed-loop
  evaluation.
- Keep strict analytic DN-MPC plus joint CBF as the runtime baseline.
- Do not tune a global probability cutoff on this calibration block.
- Design the next data experiment around runtime and split/merge hard negatives,
  with a fresh disjoint calibration block and explicit per-mode coverage.

The next candidate is a bounded, mode-balanced hard-negative/calibration
experiment. It must pass the same joint `80% recall / 50% precision` gate on a
fresh calibration split before any runtime integration is considered.
