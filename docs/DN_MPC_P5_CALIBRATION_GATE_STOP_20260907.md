# DN-MPC P5 Calibration Gate Stop

**Date:** 2026-09-07
**Protocol:** development-only; locked test not opened

## Decision

The chunk-5 JEPA checkpoint is **not promoted** to online candidate ranking and
no Ledger-Lite is created. The calibration gate is stopped for two independent
reasons:

1. The current calibration archive is a four-episode pilot, not the planned
   500-window independent calibration block.
2. Safety-critical pairwise and boundary hazard predictions are not reliable
   enough for routing decisions.

This is a planned stop, not a CBF relaxation or an execution failure. The
analytic DN-MPC + strict CBF path remains the only eligible runtime path.

## Contract Audit

The train, validation and calibration archives were audited with the same
chunk-5 contract:

- dataset version: `dn_mpc_route_identity_chunk5_v1`;
- route action shape: `[N, 5, 3]`;
- seed-disjoint train/validation/calibration splits;
- finite risk labels and both validation feasibility/visibility classes;
- runtime boundary clearance never negative;
- all required archive TensorBoard scalars present.

The archive audit output is
`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pilot_archive_audit.json`.

## Calibration Prediction Audit

The checkpoint was evaluated on the independent calibration archive using
`audit_jepa_route_hard_negative_predictions.py`. The following values are
recall/precision/Brier for the predicted hazard band `TTC < 1 s`:

| Head | Recall | Precision | Brier | Interpretation |
|---|---:|---:|---:|---|
| obstacle TTC | 91.0% | 58.9% | 0.123 | useful as a conservative low-weight signal |
| boundary TTC | 87.2% | 37.8% | 0.208 | too many false alarms and misses for routing authority |
| pairwise TTC | 40.5% | 46.3% | 0.188 | safety-critical under-detection; reject promotion |

The checkpoint has useful stopping-distance correlation (`0.869`) but that does
not compensate for weak interaction hazard recall. The prediction audit is
stored at
`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pilot_prediction_audit.json`.

All prediction metrics and provenance text were written to
`results/dn_mpc_jepa_safe_capture_tensorboard/p5_prediction_audit_chunk5_pilot_seed20260907`.

## Consequence

- Do not build a new Ledger-Lite from this pilot.
- Do not let JEPA override analytic DN-MPC route selection.
- Do not use pairwise hazard as a hard safety certificate; Joint CBF remains
  the only safety filter.
- Do not expand network capacity or open L0-L3 online experiments yet.

## Next Corrective Work

1. Expand only the data needed to cover pairwise interaction transitions,
   boundary approach and route-switch hard negatives; keep scene seeds disjoint.
2. Add per-agent interaction context and balanced near-collision/near-boundary
   windows rather than simply repeating nominal episodes.
3. Refit a fresh chunk-5 checkpoint and rerun calibration. Promotion requires
   an independent calibration block of at least 500 windows plus a prespecified
   pairwise hazard recall gate.
4. After the gate passes, compare JEPA-disabled and JEPA-ranking replay on the
   same G5 manifest before implementing Ledger-Lite.
