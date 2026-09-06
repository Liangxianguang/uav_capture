# V5 Route-12 Settled Counterfactual Three-Seed Audit

`development_only=true`; `locked_test_opened=false`. This is an offline local-chunk attribution, not a new online benchmark.

## Result

The audit keeps geometry-valid, independently CBF-verified routes and compares them with the recorded Ledger eligibility. It does not relax the online Ledger or CBF contract.

| Seed | Decisions | Mean recorded eligible | Mean CBF-verified eligible | Ledger over-abstention | Selected-not-best | Selected settled safe | Best settled safe | Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 239 | 8.167 | 8.577 | 10 | 0.536 | 0.033 | 0.084 | -0.45540284072698944 |
| 20260912 | 561 | 9.032 | 9.207 | 10 | 0.692 | 0.005 | 0.014 | -0.1725951652312848 |
| 20260913 | 750 | 8.296 | 8.880 | 44 | 0.931 | 0.000 | 0.000 | 0.05443223013026524 |

Pooled decisions: **1550**; selected-not-best `0.783`; selected settled safe `0.007`; best settled safe `0.018`.

Pooled mean eligible candidates rise from `8.543` in the recorded Ledger view to `8.952` when only independently verified route CBF probes are retained. This is evidence of Ledger over-abstention, not permission to disable stale/OOD gates.

## Decision

The route-regret signal is not positive: selected routes are frequently not the settled-best route, and no seed shows a reliable capture improvement. Stop further JEPA training, data expansion, and larger scene matrices. The next work item is a bounded Ledger reacquisition contract plus route-ranking/auxiliary-head diagnosis, followed by a new calibration archive.

All source hard safety events remain zero in this audit block. `controlled_abort`, CBF margins, stale/OOD gates, and raw-unverified prohibitions remain unchanged.

TensorBoard: `D:\uav-capture\uav_capture\results\jepa_safe_capture_v5_route12_settled_three_seed_tensorboard_v2`.
