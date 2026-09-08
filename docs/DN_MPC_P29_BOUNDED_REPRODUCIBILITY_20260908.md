# DN-MPC P29 Bounded Reproducibility Audit

**Date:** 2026-09-08
**Status:** development-only; offline-only; promotion rejected
**Scope:** independent-seed reproduction of the P28 augmented route-JEPA

## 1. Purpose

P28 seed `282801` improved offline route agreement after adding anticipatory
braking, left/right tangent, and boundary-rescue hard negatives. P29 repeats
the same training contract with an independent seed (`282802`) while keeping
the train, validation, and calibration archives byte-for-byte fixed. This is a
bounded reproducibility check, not an online integration or locked benchmark.

No CBF margin, stale/OOD/non-finite gate, controlled-abort rule, candidate
contract, or raw-unverified execution rule was changed. No hard-negative branch
was executed by the environment.

## 2. Reproduction contract

| Item | P29 value |
|---|---|
| training archive | `results/dn_mpc_jepa_safe_capture_dev/p28_augmented_train8_v2/route_identity_counterfactual.npz` |
| validation archive | `results/dn_mpc_jepa_safe_capture_dev/p25_trace_validation8/route_identity_counterfactual.npz` |
| calibration archive | `results/dn_mpc_jepa_safe_capture_dev/p28_augmented_calibration8_v2/route_identity_counterfactual.npz` |
| model | `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2` |
| seed | `282802` |
| architecture | input 63, hidden 128, latent 64, one layer, five-step chunk |
| route objective | listwise, temperature `0.005`, horizon index `2` |
| pairwise interaction | pooling enabled, relational encoding disabled |
| optimizer | AdamW, learning rate `1e-3`, weight decay `1e-5`, batch `256` |
| device | RTX 5050, CUDA, PyTorch `2.7.1+cu128` |
| protocol | development-only; locked test unopened |

Archive hashes are recorded in the generated audit and are identical to P28:

- train dataset: `a06393c50d0cc71d03f6fa8047c8b6ed6bd78c79cdf219fb55ac0cab60e4dbf6`
- validation dataset: `cd39846e184c8fa2f6e7f7fbf5ee2be30d20def4d715399e19a1566901fd321b`
- calibration dataset: `c944fd991ca3c9be7d92ebc52fdd8a9bfbd277c549cdf288d51f8e9b7687dd2e`

## 3. Training result

The run completed `23` epochs and stopped after eight stale validation epochs.
The best checkpoint was at epoch `15` with validation loss `2.0440067108287368`.

- checkpoint: `results/dn_mpc_jepa_safe_capture_checkpoints/p28_route_jepa_augmented_seed282802/checkpoint.pt`
- checkpoint SHA-256: `d1c32b986a832205542e7103c4186c1af1c6c8317fe524557e2c9a0cf0ff15e6`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p28_route_jepa_augmented_seed282802`
- training metadata: `results/dn_mpc_jepa_safe_capture_checkpoints/p28_route_jepa_augmented_seed282802/run_metadata.json`

The training event file contains epoch losses, task-specific metrics,
parameter/gradient histograms, hyperparameters, provenance, and the early-stop
reason. The prediction and calibration audits have separate TensorBoard runs.

## 4. Prediction and calibration results

| Split | model-vs-truth | model-vs-selected | OOD / coverage notes |
|---|---:|---:|---|
| train | `54.41%` | `47.79%` | OOD `0.71%`, rollout `99.68%` |
| validation | **`41.28%`** | **`26.74%`** | OOD `1.93%`, rollout `91.39%`, route-error `96.20%` |
| calibration | `33.33%` | `24.88%` | fresh thresholds finite |

All outputs were finite. Archive contracts, abstention bounds, OOD coverage,
rollout coverage, and route-error coverage passed. The candidate-agreement gate
failed because validation model-vs-truth and model-vs-selected were both below
the required `50%` threshold.

## 5. Comparison with P28 seed 282801

| Checkpoint | validation model-vs-truth | validation model-vs-selected |
|---|---:|---:|
| P28 `282801` | `47.09%` | `39.53%` |
| P29 `282802` | `41.28%` | `26.74%` |
| change | `-5.81 pp` | `-12.79 pp` |

The independent seed does not reproduce the P28 improvement at the same
reliability level. This is evidence of estimator/label sensitivity, not
evidence that JEPA is safe to deploy. It also means that a three-seed paired
replay would currently be premature: the single independent reproduction is
already directionally worse and no promotion gate is satisfied.

## 6. Decision and next action

P29 is a valid, fully logged bounded reproduction, but it is **not promoted**.
The following remain prohibited:

- online JEPA route selection;
- Ledger-Lite creation or reuse of an old ledger;
- three-seed paired replay;
- lowering CBF margins or disabling stale/OOD/non-finite gates;
- executing raw-unverified actions.

The next permitted task is a read-only route utility/label diagnosis. It must
test whether the P28 gain is concentrated in hard-negative rows, whether
listwise targets are unstable across near ties, and whether route utility,
candidate eligibility, and calibration source distributions are aligned. Only
after a revised label contract passes a fresh calibration and an independent
seed reproduces the candidate-agreement gate may training be expanded.

## 7. Artifact index

- prediction audit JSON: `results/dn_mpc_jepa_safe_capture_dev/p28_augmented_seed282802_prediction_audit/audit.json`
- prediction audit TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p28_augmented_seed282802_prediction_audit`
- P27 fresh calibration JSON: `results/dn_mpc_jepa_safe_capture_dev/p28_augmented_seed282802_p27/audit.json`
- P27 fresh calibration Markdown: `results/dn_mpc_jepa_safe_capture_dev/p28_augmented_seed282802_p27/AUDIT.md`
- P27 details: `results/dn_mpc_jepa_safe_capture_dev/p28_augmented_seed282802_p27/details.json`
- P27 TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p28_augmented_seed282802_p27`
