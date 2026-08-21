# Central V5 Retained-BC Development Validation

This is a one-training-seed development-validation report. It is not a locked test and is not evidence of a multi-seed improvement.

## Training Integrity

- Checkpoint SHA-256: `44424cd4d0579ab6c5771e38c8ee799060c5ab304bea71d5af29b64ccb587558`
- Expert archive SHA-256: `57b6377e56ddc5e7ad1b26e16a2e0231e84df7aa7cbee5923dbe770a6f622787`
- Data provenance: `reused_expert_archives`
- Archive source balance: `equal_sequences`; selected sequences balanced: `True`
- All source demonstrations safe/cooperative: `True`
- Source 0: `647/647` sequences selected; safe/cooperative `True/True`.
- Source 1: `647/457` sequences selected; safe/cooperative `True/True`.
- Training epochs with finite imitation loss: `64` / `True`

## Fixed S1/S2 Regression

| Scene | Execution | Cooperative Safe Capture | Collision | Boundary | Transit |
| --- | --- | ---: | ---: | ---: | ---: |
| s1_cylinder | RAW | 25.0% | 75.0% | 0.0% | 100.0% |
| s1_cylinder | CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| s1_box | RAW | 100.0% | 0.0% | 0.0% | 100.0% |
| s1_box | CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| s1_wall | RAW | 100.0% | 0.0% | 0.0% | 100.0% |
| s1_wall | CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| s2 | RAW | 45.0% | 55.0% | 0.0% | 100.0% |
| s2 | CBF | 100.0% | 0.0% | 0.0% | 100.0% |

## V5 Random S3 Validation

| Execution | Cooperative Safe Capture (95% Wilson CI) | Capture | Collision | Boundary | Transit | Time to Capture | Path / Defender | CBF Correction (mean / median / p95) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAW | 3.3% (0.9%, 11.4%) | 3.3% | 96.7% | 0.0% | 100.0% | 4.10 s | 6.10 m | 0.000 / 0.000 / 0.000 |
| CBF | 96.7% (88.6%, 99.1%) | 96.7% | 1.7% | 1.7% | 100.0% | 6.45 s | 20.31 m | 0.141 / 0.137 / 0.230 |

## S3 Raw/CBF Pairing

- Static maps, initial positions, target profile, and episode seeds exactly paired: `True`
- Static-scene SHA-256: `066b690d766a919a6c452eaa9fc9bf2b5b80093a8a32e920c1425274f0d4c1ab`

## CBF Failure Groups

### Observation condition

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| delayed_noisy | 31 | 3.2% | timeout: 1 |
| nominal | 29 | 3.4% | safety_failure: 1 |

### Obstacle count

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| 3 | 21 | 9.5% | safety_failure: 1, timeout: 1 |
| 4 | 18 | 0.0% | none |
| 5 | 21 | 0.0% | none |

### Planned clearance proxy

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| medium: planned clearance 0.65-0.80 m | 20 | 10.0% | safety_failure: 1, timeout: 1 |
| narrow: planned clearance <0.65 m | 40 | 0.0% | none |

### Target motion

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| flee_persistence | 31 | 6.5% | safety_failure: 1, timeout: 1 |
| s_curve | 29 | 0.0% | none |

## Gate Decision

- `s3_cooperative_safe_capture_at_least_85_percent`: `True`
- `s3_collision_at_most_2_percent`: `True`
- `s3_boundary_at_most_2_percent`: `True`
- `s3_transit_at_least_99_percent`: `True`
- `all_fixed_cbf_at_least_98_percent`: `True`
- Overall one-seed development candidate gate: `True`

Raw actor and CBF execution are separate artifacts. CBF safety improvement is not attributed entirely to the learned actor. A passing development gate only permits the pre-registered next step; it does not open or replace the V5 next-locked seed block.
