# DN-MPC P30 Route-Utility Diagnostic

**Date:** 2026-09-08
**Status:** development-only; offline-only; no action executed

## 1. Question

P29 showed that the P28 augmented route-JEPA was sensitive to the random seed.
Before collecting more data or creating a Ledger-Lite, this diagnostic tests a
specific contract hypothesis: route progress alone omits a public and useful
cost, namely the geometric length of each candidate route. The audit therefore
adds a public route-length prior to both the offline label and the model score.

This is a read-only ranking diagnostic. It does not retrain a checkpoint,
change CBF margins, change candidate eligibility, execute a route, or open the
locked test.

## 2. Protocol

- Source details: `results/dn_mpc_jepa_safe_capture_dev/p28_augmented_ranking_audit/details.csv`
- Train archive: `results/dn_mpc_jepa_safe_capture_dev/p28_augmented_train8_v2/route_identity_counterfactual.npz`
- Validation archive: `results/dn_mpc_jepa_safe_capture_dev/p25_trace_validation8/route_identity_counterfactual.npz`
- Calibration archive: `results/dn_mpc_jepa_safe_capture_dev/p28_augmented_calibration8_v2/route_identity_counterfactual.npz`
- Sweep: `lambda` in `{0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2}`
- Utility transformation: `progress - lambda * route_length_m / 10`
- Selection rule: choose `lambda` using calibration informative exact top-1,
  then calibration pairwise agreement and smaller weight as tie breakers.

The details CSV hash is
`e3ef5f046a2201ca8da11af08501f1c30f773261b0c794fdcd287660ac8d803e`.
The selected value is `lambda=0.3` based only on calibration.

## 3. Results

| Validation metric | lambda 0.0 | calibration-selected lambda 0.3 | change |
|---|---:|---:|---:|
| exact top-1 | `47.09%` | **`66.28%`** | `+19.19 pp` |
| tie-aware top-1 | `70.93%` | **`80.81%`** | `+9.88 pp` |
| informative top-1 | `80.00%` | **`89.13%`** | `+9.13 pp` |
| pairwise agreement | `87.74%` | **`97.79%`** | `+10.05 pp` |

The calibration-selected weight itself has calibration informative top-1
`89.13%` and pairwise agreement `98.01%`. The validation gain is therefore
not caused by selecting the weight on validation.

The improvement is consistent with route-length being a missing public utility
term. It does not prove that a trained JEPA has learned the correct causal
trade-off, and it does not establish online safety or capture performance.

## 4. Interpretation

The P28 ranking failure is not only a representation problem. A model can have
reasonable pairwise progress predictions while still choosing a geometrically
long detour because route length is absent or underweighted in the ranking
contract. The P30 result gives a concrete next training target:

1. define a route utility label containing progress, route length, feasibility,
   target escape cost, and route-switch penalty;
2. fit the weight on an independent calibration split;
3. re-evaluate both P28 and P29 checkpoints with the same utility contract;
4. require selected/nominal/safe-hold CBF traces and fresh OOD/disagreement
   calibration before any online integration.

The current result is still offline-only. It must not be described as a
`66.28%` capture rate or as evidence that JEPA is promotion-ready.

## 5. Artifacts

- audit JSON: `results/dn_mpc_jepa_safe_capture_dev/p29_route_utility_audit/audit.json`
- audit Markdown: `results/dn_mpc_jepa_safe_capture_dev/p29_route_utility_audit/AUDIT.md`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p29_route_utility_audit`

