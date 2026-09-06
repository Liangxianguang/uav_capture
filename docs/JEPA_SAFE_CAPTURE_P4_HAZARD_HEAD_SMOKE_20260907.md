# JEPA Safe Capture P4 Hazard-Head Smoke

**Date:** 2026-09-07  
**Phase:** development-only  
**Decision:** `prediction_signal_improved_but_ledger_blocked`

## Purpose

The previous head-only smoke optimized only continuous TTC regression and missed the safety-critical tail. This smoke adds three hazard bands (`<=0.5 s`, `<=1 s`, `<=2 s`) for obstacle, boundary and pairwise TTC, trained with positive-class-weighted BCE, while retaining continuous regression and a lower quantile pinball objective. The v1 route-JEPA encoder remains frozen.

## Contract

| item | value |
|---|---|
| model | `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2` |
| base checkpoint | `results/jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt` |
| base checkpoint SHA-256 | `4b68f8e61df978b79cb58113e6e960fc930912e992469c604abfe5f972a3bdf1` |
| output checkpoint | `results/jepa_route_identity_hazard_headonly_basev1_smoke_seed20260907/checkpoint.pt` |
| output SHA-256 | `7df680d724170e311b10d22e7c55a956467427a6437650594acfc2f43cfe5a09` |
| epochs / batch | `8 / 256` |
| hazard positive weight | `6.0` |
| quantile | `0.10` |
| device | CUDA (RTX 5050 environment) |
| TensorBoard training | `results/jepa_route_identity_hazard_headonly_basev1_smoke_seed20260907_tensorboard/` |
| TensorBoard audit | `results/jepa_route_identity_hazard_prediction_audit_seed20260907_v3_tensorboard/` |

No CBF margin, stale/OOD/non-finite gate, Ledger state, runtime candidate or locked split was changed.

## Training result

On validation, hazard loss decreased from `1.3857` at epoch 1 to `0.9888` at epoch 8; quantile loss decreased from `0.1195` to `0.0519`. Pairwise recall at the default probability cutoff of `0.5` ended at `56.6% / 53.4% / 68.6%` for `<=0.5 / <=1 / <=2 s`.

## Held-out result

| head / band | validation recall | validation precision | calibration recall | calibration precision |
|---|---:|---:|---:|---:|
| obstacle `<=1s` | `99.94%` | `45.40%` | `98.87%` | `55.89%` |
| boundary `<=1s` | `94.81%` | `45.98%` | `98.73%` | `30.42%` |
| pairwise `<=1s` | `51.44%` | `35.90%` | `37.92%` | `37.83%` |
| pairwise `<=2s` | `68.80%` | `34.98%` | `49.51%` | `34.96%` |

The checkpoint improves obstacle and boundary hazard recall, but pairwise hazard recall remains unstable across validation and calibration. Threshold sweep shows the tradeoff clearly: on calibration pairwise `<=1s`, cutoff `0.2` gives `84.84%` recall at only `31.84%` precision, while cutoff `0.5` gives `37.92%` recall at `37.83%` precision. The model therefore cannot provide a reliable pairwise risk gate without rejecting a large fraction of safe routes.

## Failure attribution and stop rule

This is a prediction-level improvement, not an end-to-end safe-capture result. The remaining failure is interaction-tail generalization: the frozen route representation does not separate imminent pairwise conflicts sufficiently, and class weighting trades recall for broad false positives. Connecting this head directly to the Ledger would recreate the earlier `safe_hold` over-abstention failure.

Accordingly, full JEPA fine-tuning, three-seed training, new Ledger calibration, and closed-loop L0-L3 evaluation are stopped. The next permitted work is a bounded pairwise-specific representation/data audit (relative velocity, formation topology and hard-tail coverage), followed by another held-out audit. No threshold may be chosen from development safe-capture results.

