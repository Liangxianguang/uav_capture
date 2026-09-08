# DN-MPC P37 Pairwise Action-Conditioned JEPA Audit

**Date:** 2026-09-08
**Phase:** development-only; offline-only
**Decision:** promotion rejected; online route override remains disabled

## Objective

P37 adds explicit pairwise action-conditioned route features to the P36
route-switch archive. For each candidate chunk, the model receives the
candidate action relative to each teammate's action (`[H, 9]`) and an explicit
pairwise pooling block. Interaction hard-negative branches (near-pass,
formation-crossing, split-merge) are retained as offline sample types. The
JEPA remains a trajectory evaluator; it does not generate or execute control
actions.

## Data and contract

| Split | Total rows | Runtime rows | Pairwise hard-negative rows | Runtime route match | Unknown route rows |
|---|---:|---:|---:|---:|---:|
| Train8 | 17,152 | 12,864 | 4,288 | 99.63% | 48 |
| Validation8 | 21,696 | 16,272 | 5,424 | 99.41% | 96 |
| Calibration8 | 26,880 | 20,160 | 6,720 | 99.76% | 48 |

Train, validation, and calibration episode seeds are disjoint. All unknown
route rows are explicit first-step CBF-infeasible abstentions. Independent
selected/nominal/safe-hold CBF traces are present for every route-identified
runtime row. Collision, boundary, pairwise, and raw-unverified execution are
not opened by this archive collection.

## Training

| Item | Value |
|---|---|
| Seed | `373701` |
| Hidden/latent dimensions | `128 / 64` |
| Pairwise pooling/relational | enabled / enabled |
| Ranking loss | listwise, temperature `0.005` |
| Batch size | `512` |
| Requested/completed epochs | `40 / 20` |
| Best epoch | `12` |
| Best validation loss | `2.2782918` |
| Stop reason | early stop after 8 stale epochs |
| Checkpoint | `results/dn_mpc_jepa_safe_capture_checkpoints/p37_route_jepa_pairwise_seed373701/checkpoint.pt` |
| TensorBoard | `results/dn_mpc_jepa_safe_capture_tensorboard/p37_route_jepa_pairwise_seed373701` |

The complete effective model, task weights, source hashes, dataset hashes,
Python/PyTorch/CUDA and GPU information are in `run_metadata.json` and the
TensorBoard text/config records.

## Offline utility results

Route utility weights were selected only on the independent P37 calibration
split and then applied to validation.

| Model | Selected `(length, escape, cbf, switch)` | Validation exact | Informative exact | Pairwise | Model vs DN-MPC selected |
|---|---|---:|---:|---:|---:|
| P37 pairwise seed `373701` | `(1.0, 0.5, 0.0, 0.2)` | 99.41% | 100.00% | 99.07% | 34.42% |

The first three ranking metrics are agreement with offline route-utility labels;
the final metric measures agreement with the analytic DN-MPC route actually
selected from the same public state. The large gap is decisive: the learned
evaluator has not learned the planner's execution contract, even though it
orders the offline utility labels well. The `length=1.0` and `escape=0.5`
weights are grid-boundary selections and should not be interpreted as globally
optimal physical weights.

## Decision and next step

P37 passes data, pairwise feature, traceability, seed-disjoint, finite-value,
and TensorBoard provenance gates. It fails the candidate-agreement promotion
gate (`34.42%` versus the required contract-level agreement), so:

- do not create or update Ledger-Lite;
- do not run a three-seed paired replay;
- do not connect JEPA to online route selection;
- do not alter CBF margins or stale/OOD/non-finite gates;
- do not open a locked benchmark.

The next bounded task is stratified route-family and route-switch diagnosis to
locate whether disagreement comes from route IDs, utility labels, candidate
eligibility, or the DN-MPC tie-break/selection contract. If that diagnosis does
not improve agreement, retain the analytic DN-MPC + CBF planner and use JEPA
only as an offline evaluator.

Artifacts:

- Archive configs: `configs/jepa_safe_capture_route_identity_archive_dn_mpc_chunk5_p37_pairwise_*.yaml`
- Training config: `configs/dn_mpc_p37_pairwise_jepa_train.yaml`
- Archive contract audit: `results/dn_mpc_jepa_safe_capture_dev/p37_route_switch_pairwise_contract_audit8.json`
- Utility audit: `results/dn_mpc_jepa_safe_capture_dev/p37_route_utility_audit8_v2/audit.json`
- Utility details: `results/dn_mpc_jepa_safe_capture_dev/p37_route_utility_audit8_v2/details.csv`
