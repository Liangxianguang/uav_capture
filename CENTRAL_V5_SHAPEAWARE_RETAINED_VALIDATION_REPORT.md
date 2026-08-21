# Central V5 Retained-BC Development Validation

This is a one-training-seed development-validation report. It is not a locked test and is not evidence of a multi-seed improvement.

## Training Integrity

- Checkpoint SHA-256: `849b5fd97664e7d11d0d96d9a95c2e4c0179dcdcaf3d720b51a516b50e313ae4`
- Expert archive SHA-256: `2a3896ff5da967c60358db1969f80ef3b2d842406acce3d24c6d0d4ae491b744`
- Data provenance: `reused_expert_archives`
- Archive source balance: `equal_sequences`; selected sequences balanced: `True`
- All source demonstrations safe/cooperative: `True`
- Source 0: `651/651` sequences selected; safe/cooperative `True/True`.
- Source 1: `651/457` sequences selected; safe/cooperative `True/True`.
- Training epochs with finite imitation loss: `64` / `True`

## Fixed S1/S2 Regression

| Scene | Execution | Cooperative Safe Capture | Collision | Boundary | Transit |
| --- | --- | ---: | ---: | ---: | ---: |
| s1_cylinder | RAW | 30.0% | 70.0% | 0.0% | 100.0% |
| s1_cylinder | CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| s1_box | RAW | 100.0% | 0.0% | 0.0% | 100.0% |
| s1_box | CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| s1_wall | RAW | 60.0% | 40.0% | 0.0% | 100.0% |
| s1_wall | CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| s2 | RAW | 0.0% | 100.0% | 0.0% | 100.0% |
| s2 | CBF | 100.0% | 0.0% | 0.0% | 100.0% |

## V5 Random S3 Validation

| Execution | Cooperative Safe Capture (95% Wilson CI) | Capture | Collision | Boundary | Transit | Time to Capture | Path / Defender | CBF Correction (mean / median / p95) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAW | 1.7% (0.3%, 8.9%) | 1.7% | 98.3% | 0.0% | 100.0% | 3.90 s | 6.28 m | 0.000 / 0.000 / 0.000 |
| CBF | 91.7% (81.9%, 96.4%) | 91.7% | 1.7% | 1.7% | 100.0% | 7.59 s | 22.82 m | 0.145 / 0.136 / 0.254 |

## S3 Raw/CBF Pairing

- Static maps, initial positions, target profile, and episode seeds exactly paired: `True`
- Static-scene SHA-256: `066b690d766a919a6c452eaa9fc9bf2b5b80093a8a32e920c1425274f0d4c1ab`

## CBF Failure Groups

### Observation condition

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| delayed_noisy | 31 | 3.2% | timeout: 1 |
| nominal | 29 | 13.8% | safety_failure: 1, timeout: 3 |

### Obstacle count

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| 3 | 21 | 4.8% | safety_failure: 1 |
| 4 | 18 | 11.1% | timeout: 2 |
| 5 | 21 | 9.5% | timeout: 2 |

### Planned clearance proxy

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| medium: planned clearance 0.65-0.80 m | 20 | 5.0% | timeout: 1 |
| narrow: planned clearance <0.65 m | 40 | 10.0% | safety_failure: 1, timeout: 3 |

### Target motion

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| flee_persistence | 31 | 12.9% | safety_failure: 1, timeout: 3 |
| s_curve | 29 | 3.4% | timeout: 1 |

## Gate Decision

- `s3_cooperative_safe_capture_at_least_85_percent`: `True`
- `s3_collision_at_most_2_percent`: `True`
- `s3_boundary_at_most_2_percent`: `True`
- `s3_transit_at_least_99_percent`: `True`
- `all_fixed_cbf_at_least_98_percent`: `True`
- Overall one-seed development candidate gate: `True`

Raw actor and CBF execution are separate artifacts. CBF safety improvement is not attributed entirely to the learned actor. A passing development gate only permits the pre-registered next step; it does not open or replace the V5 next-locked seed block.
