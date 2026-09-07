# DN-MPC P17 Independent Multi-Source Validation

**Date:** 2026-09-08
**Status:** development validation complete; online promotion remains closed
**Locked test:** not opened

## Objective

P17 repeats both sides of the P16 source contract on a new validation seed
block. It checks whether the complementary evidence is reproducible outside
the P14/P15 calibration scenes:

- virtual probes provide physical strict-margin positives without executing an
  unverified action;
- route branches provide verified CBF and branch-outcome labels.

No P17 archive is used for training or online route ranking.

## Contract and provenance

| Item | Value |
|---|---|
| protocol | `central_random_mixed_obstacle_s3_route_v1_p17_validation_protocol` |
| route contract | `dn_mpc_route_identity_chunk5_p17_validation_v1` |
| split | validation, development-only |
| episodes per source | 8 |
| CBF horizon | 5 |
| source actor | `v5_development_exact_reactive_seed661606.pt` |
| GPU | NVIDIA GeForce RTX 5050 |
| raw-unverified execution | false |
| locked test | not opened |

The virtual source archive SHA-256 is
`97efdcaca6d02feb4b13b5fcd4718f0d39f24975da3db38ebbd4046b65ce0147`.
The route source archive SHA-256 is
`927ad5cc37f531d15dcc003e0874ee07b6b320023a3b17faaebc0b9b3d36760b`.
The source-bound P17 validation bundle SHA-256 is
`1fde949a6399fad8cf5a7c9983f28db01f339313b1210755a64e0ed687b26589`.
Every collector and materializer recorded its parameters and hashes in
TensorBoard under the `P17/...` namespace.

## Results

The bundle contains `15,824` rows:

| Source | Rows | Strict-margin cell rate | Strict-margin row rate | Branch-failure row rate | CBF-infeasible cell rate |
|---|---:|---:|---:|---:|---:|
| `p17_virtual_probe` | 4,816 | `17.49%` | `43.83%` | `0%` | `75.27%` |
| `p17_route_outcomes` | 11,008 | `6.25%`* | `6.25%`* | `2.98%` | `7.42%` |

`*` The route aggregate includes offline boundary-shadow rows. Runtime and
interaction route rows remain strict-margin negative; the branch-failure
label is still positive for verified route outcomes.

The virtual probe contains `688` safe-hold rows and never executes the probe
action. The route archive has `256` branch failures within the five-step
horizon, while first-step CBF feasibility is `100%` for the sampled runtime
routes. This confirms that first-step feasibility and multi-step branch
survival are distinct outcomes.

## Gate decision

P17 passes the independent source and provenance checks:

- both sources use a disjoint validation seed block;
- source IDs, source names, hashes, and split are explicit;
- P17 TensorBoard tags are correctly namespaced;
- the two label semantics are preserved rather than merged;
- no raw-unverified action was executed.

P17 still does **not** authorize JEPA training or Ledger-Lite deployment. The
validation confirms reproducibility of the complementary evidence, but it does
not demonstrate that a model trained on virtual physical-margin probes will
calibrate on verified branch outcomes. A dedicated cross-source calibration
metric and a model-independent acceptance threshold are still required before
online promotion.

The runtime controller remains analytic DN-MPC plus strict joint CBF.

## Artifacts

- Protocols:
  - `configs/central_random_mixed_obstacle_s3_route_v1_p17_validation_protocol.yaml`
  - `configs/jepa_safe_capture_route_identity_archive_dn_mpc_chunk5_p17_validation.yaml`
- Collector: `scripts/collect_dn_mpc_pairwise_virtual_probe_archive.py`
- Route collector: `scripts/collect_jepa_route_identity_archive.py`
- Label materializer: `scripts/materialize_dn_mpc_pairwise_label_contract.py`
- Multisource builder: `scripts/build_dn_mpc_pairwise_multisource_calibration.py`
- Bundle: `results/dn_mpc_jepa_safe_capture_dev/p17_multisource_validation/pairwise_multisource_validation.npz`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p17_multisource_validation`

Generated validation archives and event files remain local and are not
committed.
