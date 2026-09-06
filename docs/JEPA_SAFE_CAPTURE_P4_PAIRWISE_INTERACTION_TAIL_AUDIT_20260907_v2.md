# JEPA Pairwise Interaction-Tail Audit

**Phase:** development-only offline audit
**Decision:** stop full JEPA training, new Ledger calibration, and closed-loop evaluation pending a pairwise gate
**Locked test:** not opened

## Scope

This audit reconstructs relative teammate position/velocity from the public 63-D observation contract and measures pairwise hard-tail coverage and simple observability. It does not execute actions, change CBF margins, disable stale/OOD gates, or certify safety.

## Coverage

| split | samples | episodes | TTC <=0.5s samples | TTC <=1s samples | TTC <=2s samples |
|---|---:|---:|---:|---:|---:|
| train | 1872 | 4 | 472 (25.2%) | 590 (31.5%) | 682 (36.4%) |
| validation | 1508 | 4 | 376 (24.9%) | 517 (34.3%) | 582 (38.6%) |
| calibration | 2652 | 4 | 1006 (37.9%) | 1150 (43.4%) | 1229 (46.3%) |

## Calibration observability

| feature score (higher means more risky) | AUC for pairwise TTC <=1s |
|---|---:|
| `observation_pairwise_ttc_s` | 0.796 |
| `min_pairwise_distance_m` | 0.662 |
| `max_relative_speed_mps` | 0.564 |
| `max_closing_speed_mps` | 0.336 |
| `history_min_pairwise_distance_m` | 0.604 |

## Diagnosis

- **Classification:** `pairwise_signal_observable_representation_or_calibration_limited`
- **Best calibration feature:** `observation_pairwise_ttc_s` with AUC `0.7964681873444104`
- **Pairwise-positive calibration episodes:** `4`
- **Next bounded action:** add explicit pairwise pooling/topology features and recalibrate the hazard head

The preceding hazard-head smoke did not pass the calibration pairwise recall/precision gate. This audit therefore stops further training and runtime integration; its AUC values are diagnostic evidence, not a safe-capture claim.

## Provenance

- JSON: `D:\uav-capture\uav_capture\results\jepa_safe_capture_pairwise_interaction_tail_audit_20260907_v2.json`
- TensorBoard: `D:\uav-capture\uav_capture\results\jepa_safe_capture_pairwise_interaction_tail_audit_20260907_v2_tensorboard`
- `locked_test_opened=false`
- `raw_unverified_execution_allowed=false`
- `cbf_margin_changed=false`
