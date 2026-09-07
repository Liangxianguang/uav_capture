# DN-MPC P11 Pairwise Hazard-Weight Sweep Stop

**Date:** 2026-09-08
**Phase:** development-only, offline calibration sweep
**Decision:** stop pairwise positive-weight tuning; no checkpoint is runtime eligible

## Motivation

P8/P10 showed high pairwise hazard recall but poor precision. P11 adds one
explicit training parameter, `--pairwise-hazard-positive-weight`, so the
pairwise hazard BCE can be calibrated independently from the obstacle and
boundary heads. The model architecture, route archive, reachable projection,
strict CBF contract, and runtime code are unchanged.

All three variants use 20 epochs, batch size 512, CUDA on the RTX 5050, the
same train/validation archives and the same relational JEPA architecture. The
held-out original calibration block and the fresh disjoint block are both
audited. TensorBoard contains the complete arguments and training curves for
each run.

## Weight sweep

The gate is pairwise TTC `<1 s`, evaluated at cutoff `0.5`.

| Variant | Pairwise positive weight | Original calibration recall / precision | Fresh calibration recall / precision | Gate |
|---|---:|---:|---:|---|
| P8 baseline | 8 | 81.04% / 40.38% | 84.50% / 40.40% | fail |
| P11-A | 2 | 52.31% / 66.53% | 56.67% / 70.34% | fail |
| P11-B | 4 | 70.13% / 50.74% | 73.54% / 53.43% | fail |

No tested weight satisfies recall `>=80%` and precision `>=50%` on either
calibration block. The fresh block confirms that this is not an artifact of
the original scenes. Raising the weight improves recall but produces false
positives; lowering it improves precision but loses too many true pairwise
hazards.

## Fresh-block stratification

At weight `4` and cutoff `0.5` on the fresh block:

| Sample type | Recall | Precision | False-positive rate |
|---|---:|---:|---:|
| runtime | 56.3% | 37.7% | 11.7% |
| boundary shadow | 100.0% | 100.0% | n/a |
| near pass | 59.2% | 38.9% | 17.4% |
| formation crossing | 69.9% | 41.5% | 39.1% |
| split/merge | 72.8% | 32.9% | 22.4% |

The interaction modes remain below the precision target, while runtime rows
also lose recall. This is a representation/label calibration limitation, not
a reason to relax CBF margins or disable controlled abort.

## Artifacts

P11-A checkpoint:

`results/dn_mpc_jepa_safe_capture_checkpoints/route_identity_chunk5_pairwise_relational_v1_pairwise_weight2_seed20260911/checkpoint.pt`

SHA-256: `d82df5253279ff9f0fbf1b2001647253d765a3fe1b1e397692cd23fc2acce03a`

P11-B checkpoint:

`results/dn_mpc_jepa_safe_capture_checkpoints/route_identity_chunk5_pairwise_relational_v1_pairwise_weight4_seed20260912/checkpoint.pt`

SHA-256: `512726b5c61cd05e98587aec4ad01829224b4ac7e06722fd72db0fb8f3113749`

Original calibration audits:

- `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_pairwise_weight2_seed20260911_prediction_audit.json`
- `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_pairwise_weight4_seed20260912_prediction_audit.json`

Fresh calibration audits:

- `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_pairwise_weight2_seed20260911_fresh_calibration_prediction_audit.json`
- `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_pairwise_weight4_seed20260912_fresh_calibration_prediction_audit.json`

Fresh weight-4 stratification:

`results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_pairwise_weight4_seed20260912_fresh_calibration_by_mode.json`

## Stop decision

- Do not promote P11-A or P11-B to online route ranking, reliability ledger,
  or safe-capture closed-loop evaluation.
- Do not continue a blind scalar weight sweep; it is tracing a recall/precision
  frontier rather than solving the joint gate.
- Keep strict analytic DN-MPC plus joint CBF as the runtime controller.
- The next model change must address pairwise label semantics or relational
  representation (for example, time-to-collision event definition and
  action-conditioned relative velocity), followed by a new disjoint archive.
