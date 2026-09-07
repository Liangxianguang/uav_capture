# DN-MPC P16 Multi-Source Calibration Bundle

**Date:** 2026-09-08
**Status:** source-bound calibration audit complete; JEPA/Ledger promotion still closed
**Locked test:** not opened

## Purpose

P14 and P15 exposed complementary evidence under the same strict safety
contract. P14 contains offline-only physical strict-margin positives but no
branch failures. P15 contains verified route/interaction branch failures but no
runtime strict-margin crossings. P16 packages these sources without merging
their meanings.

## Bundle contract

| Source | Role | Rows | Unsafe action executed |
|---|---|---:|---|
| P14 virtual probe | physical strict-margin counterfactual | `11,144` | no |
| P15 route outcomes | verified CBF and branch outcomes | `13,696` | no |
| **Total** | source-labelled calibration bundle | **`24,840`** | **no** |

Every row carries `calibration_source_id` and
`calibration_source_name`. P14-only fields (`virtual_probe_mode` and
`virtual_probe_pair_index`) are filled with `-1` for P15 rows, which is an
explicit missing-value marker rather than an inferred label.

## Source-specific evidence

| Metric | P14 virtual probe | P15 route outcomes |
|---|---:|---:|
| strict-margin positive cell rate | `15.57%` | `6.25%` overall, all from offline boundary shadow |
| strict-margin positive row rate | `40.33%` | `6.25%` overall, all from offline boundary shadow |
| branch-failure positive rows | `0` | `352` runtime rows plus interaction failures |
| CBF-infeasible cell rate | `76.34%` | source-specific values preserved |
| raw-unverified execution | `false` | `false` |

The P15 aggregate strict-margin rate includes its explicitly offline
boundary-shadow rows. Runtime and interaction rows remain strict-margin
negative, as established in the P15 stop report. The bundle does not relabel
those rows as physical violations.

## Gate decision

P16 passes the **source-binding and provenance gate**:

- both archives are calibration-only and closed;
- source hashes and row counts are stored in metadata;
- labels remain semantically separate;
- TensorBoard uses the explicit `P16/...` namespace;
- no raw-unverified action was executed.

P16 does **not** pass the **online promotion gate**. It is not sufficient to
train a new JEPA outcome head or create Ledger-Lite because P14 positives are
virtual and P15 branch outcomes are verified execution events. A separate
independent validation archive must demonstrate calibration transfer across
these source types before model training or online route ranking is authorized.

The runtime controller therefore remains analytic DN-MPC plus strict joint CBF.
This is a deliberate safety result, not a failed collection: manufacturing a
runtime strict-margin positive by executing an unverified action would violate
the experiment contract and invalidate the comparison.

## Artifacts

- Protocol: `configs/dn_mpc_pairwise_multisource_calibration_v1.yaml`
- Builder: `scripts/build_dn_mpc_pairwise_multisource_calibration.py`
- Test: `tests/test_build_dn_mpc_pairwise_multisource_calibration.py`
- Bundle: `results/dn_mpc_jepa_safe_capture_dev/p16_multisource_calibration/pairwise_multisource_calibration.npz`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p16_multisource_calibration`

Generated NPZ and event files remain local and are not committed.
