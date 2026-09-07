# DN-MPC P18 Source Stability and Route-JEPA Training

**Date:** 2026-09-08
**Status:** development-only; online promotion and locked testing remain closed

## Scope

P18 closes the model-independent source-stability gate between the P16
calibration bundle and the disjoint P17 validation bundle, then trains one
route-aware action-conditioned JEPA checkpoint using the paired route archive.
This is a contract and training milestone, not evidence of an online capture
improvement.

The experiment preserves the existing safety contract:

- CBF margin and stale/OOD/non-finite gates were not changed;
- `controlled_abort` remains enabled;
- no raw-unverified action was executed;
- the JEPA checkpoint was not connected to online route override or the locked
  evaluator.

## Source-stability gate

The audit compares the P16 calibration bundle with the independent P17
validation bundle separately for `virtual_probe` and `route_outcomes`. Each
source has at least 1,000 rows and is checked on strict-margin cell rate,
strict-margin row rate, verified branch-failure row rate, and CBF-infeasible
cell rate.

| Source | Calibration rows | Validation rows | Maximum absolute delta | Gate |
|---|---:|---:|---:|---|
| `virtual_probe` | 11,144 | 4,816 | 3.51 pp | pass |
| `route_outcomes` | 13,696 | 11,008 | 0.42 pp | pass |

The exact report is stored locally at
`results/dn_mpc_jepa_safe_capture_dev/p18_source_stability/source_stability_report.json`.
TensorBoard provenance is under
`results/dn_mpc_jepa_safe_capture_tensorboard/p18_source_stability` with the
`P18/...` namespace. The gate is explicitly model-independent and does not
authorize online training or deployment by itself.

## JEPA training

The trainer used the paired P17 training archive, not the earlier unpaired
directory. The paired-contract correction is recorded in
`configs/dn_mpc_jepa_safe_capture_p18_training.yaml`.

| Item | Value |
|---|---|
| model | `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2` |
| train archive | `p17_route_train8_paired`, 8,768 rows |
| validation archive | `p17_route_validation8`, 11,008 rows |
| seed | `181801` |
| requested / completed epochs | 20 / 12 |
| best epoch | 7 |
| best validation loss | `0.8683711339` |
| route identity accuracy | `100%` |
| route progress pairwise accuracy | `84.38%` |
| CBF-feasibility Brier score | `0.01294` |
| device | NVIDIA RTX 5050 (`cuda`) |
| TensorBoard | `results/dn_mpc_jepa_safe_capture_tensorboard/p18_route_jepa_train_seed181801` |
| checkpoint | `results/dn_mpc_jepa_safe_capture_checkpoints/p18_route_jepa_train_seed181801/checkpoint.pt` |
| checkpoint SHA-256 | `474e4178a41d3765e2b978c0eb1411f759c52ce36b5c0d31f1c682785f938848` |

Training stopped after five validation epochs without improvement. The
validation metrics are useful for selecting the checkpoint, but they do not
establish candidate-ranking superiority or safe-capture improvement.

## Interpretation and next gate

P18 demonstrates that the complementary calibration sources are stable across
disjoint seed blocks and that a paired route-JEPA checkpoint can be trained
under the current contract. It does **not** demonstrate that virtual-probe
labels calibrate JEPA predictions, that JEPA improves DN-MPC route selection,
or that the full closed loop is safe-capture superior.

The next authorized step is an offline candidate-ranking audit using the P18
checkpoint. It must verify candidate eligibility, selected/nominal/safe-hold
CBF counterfactuals, route-ranking direction, and disagreement/OOD behavior.
Only after that audit passes should the same checkpoint be evaluated in a
development-only paired replay. Three-seed replay and a new locked block remain
future gates.

## Reproduction references

- Configuration: `configs/dn_mpc_jepa_safe_capture_p18_training.yaml`
- Source-stability protocol: `configs/dn_mpc_pairwise_source_stability_v1.yaml`
- Audit: `scripts/audit_dn_mpc_multisource_source_stability.py`
- P17 source report: `docs/DN_MPC_P17_INDEPENDENT_MULTISOURCE_VALIDATION_20260908.md`
