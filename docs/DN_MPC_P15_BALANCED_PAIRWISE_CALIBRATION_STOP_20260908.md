# DN-MPC P15 Balanced Pairwise Calibration Stop

**Date:** 2026-09-08
**Status:** development-only collection complete; promotion gate not passed
**Locked test:** not opened

## Objective

P15 recollected an independent calibration block after P14. It combines the
normal route candidates with three explicitly offline interaction probes
(`near_pass`, `formation_crossing`, and `split_merge`) while preserving the
same reachable projection, horizon-5 strict CBF, controlled-abort behavior,
and raw-unverified execution prohibition.

## Contract and provenance

| Item | Value |
|---|---|
| protocol | `central_random_mixed_obstacle_s3_route_v1_p15_balanced_calibration_protocol` |
| archive contract | `dn_mpc_route_identity_chunk5_p15_balanced_calibration_v1` |
| split | calibration, development-only |
| episodes | 8 |
| rows | `13,696` |
| runtime route rows | `10,272` |
| boundary-shadow rows | `856` |
| rows per interaction mode | `856` |
| CBF horizon | 5 |
| actor | `v5_development_exact_reactive_seed661606.pt` |
| GPU | NVIDIA GeForce RTX 5050 |
| raw-unverified execution | false |

Source archive SHA-256:
`e5881864042d0dbaa87d105c413d612c0bf93bc0a262d62bba9032659b614364`.
The materialized four-label archive SHA-256 is
`c65512cdc786f0b26250458520539e769cbb5f1f718b2ab4c4f6364ccaf6840c`.
All collection and materialization parameters were recorded in TensorBoard.

## Results

| Metric | Result |
|---|---:|
| geometry-valid runtime fraction | `72.35%` |
| first-step CBF-feasible runtime fraction | `99.53%` |
| runtime branch-failure rows | `352 / 10,272 = 3.43%` |
| runtime strict-margin positive rows | `0` |
| interaction branch failures | `near_pass=1`, `formation_crossing=1`, `split_merge=0` |
| boundary-shadow strict-margin positive rows | `856` (offline synthetic only) |
| all-archive strict-margin cell rate | `6.25%` |
| all-archive branch-failure row rate | `3.24%` |

After materialization, the per-sample-type outcome rates were:

| Sample type | Rows | Strict-margin positive cells | CBF-infeasible cells | Branch-failure rows |
|---|---:|---:|---:|---:|
| runtime | 10,272 | `0%` | `1.68%` | `3.43%` |
| boundary shadow | 856 | `100%` | `100%` | `0%` |
| near pass | 856 | `0%` | `1.59%` | `3.27%` |
| formation crossing | 856 | `0%` | `1.59%` | `3.27%` |
| split merge | 856 | `0%` | `2.06%` | `4.21%` |

## Gate decision

P15 passes the collection and provenance checks, but it does **not** pass the
JEPA/Ledger promotion gate. The strict-margin positives are still confined to
offline boundary-shadow rows. Runtime and interaction branches contain real
CBF infeasibility and branch-failure outcomes, but no strict-margin crossing;
this is the expected consequence of never executing an unverified action.

Therefore:

1. Do not train a new pairwise strict-margin head from P15 alone.
2. Do not build Ledger-Lite or connect it to online route ranking.
3. Keep analytic DN-MPC plus strict joint CBF as the only runtime controller.
4. Treat P14 virtual positives and P15 runtime branch outcomes as separate,
   non-interchangeable sources until an explicit multi-source calibration
   contract and independent validation archive are implemented.

This is a useful negative result: lowering the CBF margin or allowing unsafe
execution would manufacture labels by violating the safety contract, so that
option is rejected. The next eligible data task is a source-bound, multi-source
calibration/validation archive that preserves the distinction between physical
margin probes and verified execution failures.

## Cleanup audit

Nine empty generated directories were identified under `results/`; no files
were removed because all non-empty result, TensorBoard, checkpoint, report,
and `tmp` archive directories are still part of the reproducibility evidence.
The empty-directory list is recorded in the execution log rather than being
silently deleted.

## Artifacts

- Protocol: `configs/central_random_mixed_obstacle_s3_route_v1_p15_balanced_calibration_protocol.yaml`
- Archive contract: `configs/jepa_safe_capture_route_identity_archive_dn_mpc_chunk5_p15_balanced_calibration.yaml`
- Collector: `scripts/collect_jepa_route_identity_archive.py`
- Label materializer: `scripts/materialize_dn_mpc_pairwise_label_contract.py`
- Source archive: `results/dn_mpc_jepa_safe_capture_dev/p15_balanced_pairwise_archive_calibration8_seed20260908/route_identity_counterfactual.npz`
- Label archive: `results/dn_mpc_jepa_safe_capture_dev/p15_balanced_pairwise_labels/route_identity_pairwise_outcome_labels.npz`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p15_balanced_pairwise_archive_calibration8_seed20260908`
