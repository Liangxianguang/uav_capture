# DN-MPC P13 Pairwise Outcome-Label Contract

**Date:** 2026-09-08
**Status:** development-only; label contract accepted, training stopped
**Locked test:** not opened

## Objective

P13 separates the pairwise TTC warning from actual safety and execution
outcomes. The four labels are stored independently and are not interchangeable:

| Label | Meaning | Runtime use |
| --- | --- | --- |
| `labels_predicted_ttc_hazard` | Constant-relative-velocity TTC is at most `1.0 s` | Warning feature only |
| `labels_strict_margin_violation` | Physical inter-agent clearance is below the unchanged `0.35 m` operational margin | Safety outcome target |
| `labels_cbf_infeasible` | The primary CBF feasibility label is false/non-finite | Execution-contract outcome |
| `labels_branch_failure` | Counterfactual branch fails at or before a horizon step | Rollout outcome |

`boundary_shadow` rows are offline synthetic negatives. They are allowed to
carry margin and CBF-negative labels, but are explicitly excluded from
`labels_branch_failure` because no unsafe action was executed.

## Disjoint data and artifacts

The source archive was recollected with calibration seed block `651101` and
the same frozen actor, route candidates, horizon-5 strict CBF, reachable
projection, and interaction hard-negative contract. It is disjoint from the
P9 fresh block (`650101`--`650120`).

| Artifact | Value |
| --- | --- |
| Source rows | 28,928 (`21,696` runtime; `1,808` per offline class) |
| Source archive SHA-256 | `e6588c0031ce498e599a4deb8f95ab83a1787b193fb08dcf582f986115e0b5ee` |
| Source metadata SHA-256 | `ae95959a5664b64d297f0ed20248e076b848237d7ef379e2e4a6e8d5b4e58c88` |
| Materialized archive SHA-256 | `53423931c48786fe8bdddfe52a02df6dd7e46589fc7f37d1aa133bf3a7d7db72` |
| Materialized metadata SHA-256 | `1136dbf4cac6ff2156e9f350d30d972fdf626004638aa3081768a3bb0f5c8a05` |

Artifacts:

- Source archive: `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_relational_v1_p13_source`
- Label archive: `results/dn_mpc_jepa_safe_capture_dev/route_identity_chunk5_pairwise_outcome_labels_v1_p13`
- TensorBoard source run: `results/dn_mpc_jepa_safe_capture_tensorboard/route_identity_chunk5_pairwise_relational_v1_p13_source`
- TensorBoard label run: `results/dn_mpc_jepa_safe_capture_tensorboard/route_identity_chunk5_pairwise_outcome_labels_v1_p13`
- Materializer: `scripts/materialize_dn_mpc_pairwise_label_contract.py`

## Label rates

Rates below are cell-level rates over the five horizon steps.

| Sample type | TTC warning | Strict margin violation | CBF infeasible | Branch failure |
| --- | ---: | ---: | ---: | ---: |
| runtime (`n=21,696`) | 12.49% | 0.00% | 1.64% | 1.64% |
| boundary_shadow (`n=1,808`) | 100.00% | 100.00% | 100.00% | 0.00% |
| near_pass (`n=1,808`) | 16.13% | 0.00% | 1.81% | 1.81% |
| formation_crossing (`n=1,808`) | 30.86% | 0.00% | 1.95% | 1.95% |
| split_merge (`n=1,808`) | 13.57% | 0.00% | 1.68% | 1.68% |

The source collector also reports runtime first-step CBF feasibility `99.56%`
(`21,600/21,696`) and zero raw-unverified execution. The materializer itself
does not rerun the environment or change any safety parameter.

## Decision

The P13 contract is accepted as an accounting contract, but no JEPA training
or runtime integration is opened. The strict-margin head has no positive
examples in valid runtime/interaction branches; using it directly would create
an unidentifiable target. This is expected from a collector that advances a
branch only after a primary CBF verification.

The next data task is therefore a P14 offline counterfactual probe archive:

1. Generate intentionally unsafe *virtual* next-state probes analytically or
   on an isolated simulator clone; never execute them in the runtime path.
2. Label predicted margin violation and actual physical margin violation
   separately from CBF feasibility.
3. Keep the probes disjoint from train/validation/calibration and mark them
   `offline_only=true` in metadata and TensorBoard.
4. Require positive-count and calibration coverage gates before training a new
   outcome head.

Until P14 has those positives, keep the analytic DN-MPC + strict joint CBF path
as the only runtime-eligible controller and do not create Ledger-Lite.
