# DN-MPC P19 P18 Candidate-Ranking Audit

**Date:** 2026-09-08
**Status:** development-only; **stopped before online integration**

## Scope

This audit runs the P18 action-conditioned interaction-aware route-JEPA
checkpoint on disjoint train, validation, and calibration route archives. It
checks the five-step route contract, finite model outputs, candidate eligibility,
route-progress score direction, and the traceability needed for selected,
nominal, and verified-safe-hold CBF counterfactuals. It is offline-only: no
control action was executed and no CBF gate was relaxed.

## Contract and provenance

| Item | Value |
|---|---|
| checkpoint | `results/dn_mpc_jepa_safe_capture_checkpoints/p18_route_jepa_train_seed181801/checkpoint.pt` |
| checkpoint SHA-256 | `474e4178a41d3765e2b978c0eb1411f759c52ce36b5c0d31f1c682785f938848` |
| model | `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2` |
| route action chunk | `[N, 5, 3]` |
| pairwise route chunk | `[N, 5, 9]` |
| candidates | 12 |
| device | RTX 5050 / CUDA (`torch 2.7.1+cu128`) |
| TensorBoard | `results/dn_mpc_jepa_safe_capture_tensorboard/p19_p18_candidate_ranking_audit` |

The train and validation archives are P17 route archives. Calibration is an
independently generated P15 balanced archive with a different dataset version,
but the structural action/history/chunk contract is identical. Full hashes and
the dirty-file provenance snapshot are in the JSON result:
`results/dn_mpc_jepa_safe_capture_dev/p19_p18_candidate_ranking_audit/audit.json`.

## Results

| Split | runtime groups | route identity | route side | top-1 progress agreement | informative pairwise agreement | geometry accuracy |
|---|---:|---:|---:|---:|---:|---:|
| train | 137 | 100.00% | 100.00% | 29.13% | 88.82% | 89.69% |
| validation | 172 | 100.00% | 100.00% | **14.81%** | **84.31%** | 83.26% |
| calibration | 214 | 100.00% | 100.00% | 20.50% | 84.25% | 83.53% |

The pairwise direction is above the development diagnostic threshold of 70%,
but top-1 selection is far below the 50% promotion threshold. Therefore the
checkpoint is not yet a reliable route selector even though its route identity
head is accurate.

Candidate eligibility is also incomplete. On validation, 10 of 172 runtime
groups have zero candidates that are simultaneously geometrically valid and
first-step CBF-feasible for all four defenders. In these groups the CBF first
step remains feasible, but every route is rejected by the geometry contract;
the issue must be fixed in route construction/geometry labels before JEPA is
allowed to choose a route.

## Counterfactual traceability

The original P19 archives contain the required per-candidate CBF labels and include runtime
rows for both candidate 0 (`nominal`) and candidate 11
(`verified_safe_hold`). They do **not** contain an independent per-step
`selected_candidate_index` together with a selected-action CBF replay. Thus:

- nominal counterfactual: present;
- verified safe-hold counterfactual: present;
- selected-candidate independent CBF counterfactual: missing;
- three-way promotion gate: failed.

The audit also finds no bound OOD-distance field or fresh disagreement ledger
for the P18 checkpoint. Finite JEPA outputs alone are not an OOD guarantee.

## Decision and next work

P19 **stops before online integration and before three-seed replay**. The
failure is diagnostic rather than a reason to relax safety:

1. repair or explicitly represent route-geometry invalid states so every
   runtime group has at least one verified fallback route;
2. collect selected/nominal/safe-hold independent CBF counterfactual traces;
3. bind a new calibration archive to P18 uncertainty and rollout-disagreement
   fields; and
4. retrain or recalibrate route progress and repeat P19. Only a passing audit
   may proceed to paired replay.

No raw-unverified action, online Ledger promotion, locked test, or CBF margin
change occurred in this stage.

## Reproduction

```text
python scripts/audit_dn_mpc_p18_candidate_ranking.py \
  --checkpoint results/dn_mpc_jepa_safe_capture_checkpoints/p18_route_jepa_train_seed181801/checkpoint.pt \
  --train-dataset results/dn_mpc_jepa_safe_capture_dev/p17_route_train8_paired/route_identity_counterfactual.npz \
  --train-metadata results/dn_mpc_jepa_safe_capture_dev/p17_route_train8_paired/metadata.json \
  --validation-dataset results/dn_mpc_jepa_safe_capture_dev/p17_route_validation8/route_identity_counterfactual.npz \
  --validation-metadata results/dn_mpc_jepa_safe_capture_dev/p17_route_validation8/metadata.json \
  --calibration-dataset results/dn_mpc_jepa_safe_capture_dev/p15_balanced_pairwise_archive_calibration8_seed20260908/route_identity_counterfactual.npz \
  --calibration-metadata results/dn_mpc_jepa_safe_capture_dev/p15_balanced_pairwise_archive_calibration8_seed20260908/metadata.json \
  --output results/dn_mpc_jepa_safe_capture_dev/p19_p18_candidate_ranking_audit/audit.json \
  --markdown-output docs/DN_MPC_P19_P18_CANDIDATE_RANKING_AUDIT_20260908.md \
  --details-output results/dn_mpc_jepa_safe_capture_dev/p19_p18_candidate_ranking_audit/ranking_details.csv \
  --tensorboard-logdir results/dn_mpc_jepa_safe_capture_tensorboard/p19_p18_candidate_ranking_audit \
  --device cuda
```
