# Central V5 Retained-BC Development Validation

This is a one-training-seed development-validation report. It is not a locked test and is not evidence of a multi-seed improvement.

## Training Integrity

- Checkpoint SHA-256: `4fe54f86b033b1d5290ffdaa8d1fb097f7e8b8491071e64f7baf1f8dbbb36bf3`
- Expert archive SHA-256: `17d83cc902740323a651d021f15c22bfae62e5955568f5ec463c9d9d6bdc0285`
- Data provenance: `reused_expert_archives`
- Archive source balance: `equal_sequences`; selected sequences balanced: `True`
- All source demonstrations safe/cooperative: `True`
- Source 0: `654/654` sequences selected; safe/cooperative `True/True`.
- Source 1: `654/457` sequences selected; safe/cooperative `True/True`.
- Training epochs with finite imitation loss: `64` / `True`

## Fixed S1/S2 Regression

| Scene | Execution | Cooperative Safe Capture | Collision | Boundary | Transit |
| --- | --- | ---: | ---: | ---: | ---: |
| s1_cylinder | RAW | 0.0% | 100.0% | 0.0% | 100.0% |
| s1_cylinder | CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| s1_box | RAW | 95.0% | 5.0% | 0.0% | 100.0% |
| s1_box | CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| s1_wall | RAW | 70.0% | 30.0% | 0.0% | 100.0% |
| s1_wall | CBF | 95.0% | 5.0% | 5.0% | 100.0% |
| s2 | RAW | 90.0% | 10.0% | 0.0% | 100.0% |
| s2 | CBF | 100.0% | 0.0% | 0.0% | 100.0% |

## V5 Random S3 Validation

| Execution | Cooperative Safe Capture (95% Wilson CI) | Capture | Collision | Boundary | Transit | Time to Capture | Path / Defender | CBF Correction (mean / median / p95) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAW | 1.7% (0.3%, 8.9%) | 1.7% | 98.3% | 0.0% | 100.0% | 3.60 s | 5.58 m | 0.000 / 0.000 / 0.000 |
| CBF | 98.3% (91.1%, 99.7%) | 98.3% | 1.7% | 1.7% | 100.0% | 8.82 s | 24.51 m | 0.155 / 0.134 / 0.268 |

## S3 Raw/CBF Pairing

- Static maps, initial positions, target profile, and episode seeds exactly paired: `True`
- Static-scene SHA-256: `066b690d766a919a6c452eaa9fc9bf2b5b80093a8a32e920c1425274f0d4c1ab`

## CBF Failure Groups

### Observation condition

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| delayed_noisy | 31 | 3.2% | safety_failure: 1 |
| nominal | 29 | 0.0% | none |

### Obstacle count

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| 3 | 21 | 4.8% | safety_failure: 1 |
| 4 | 18 | 0.0% | none |
| 5 | 21 | 0.0% | none |

### Planned clearance proxy

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| medium: planned clearance 0.65-0.80 m | 20 | 0.0% | none |
| narrow: planned clearance <0.65 m | 40 | 2.5% | safety_failure: 1 |

### Target motion

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| flee_persistence | 31 | 3.2% | safety_failure: 1 |
| s_curve | 29 | 0.0% | none |

## Gate Decision

- `s3_cooperative_safe_capture_at_least_85_percent`: `True`
- `s3_collision_at_most_2_percent`: `True`
- `s3_boundary_at_most_2_percent`: `True`
- `s3_transit_at_least_99_percent`: `True`
- `all_fixed_cbf_at_least_98_percent`: `False`
- Overall one-seed development candidate gate: `False`

Raw actor and CBF execution are separate artifacts. CBF safety improvement is not attributed entirely to the learned actor. A passing development gate only permits the pre-registered next step; it does not open or replace the V5 next-locked seed block.
