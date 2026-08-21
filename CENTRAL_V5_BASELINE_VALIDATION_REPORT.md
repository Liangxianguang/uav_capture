# Central V5 Retained-BC Development Validation

This is a one-training-seed development-validation report. It is not a locked test and is not evidence of a multi-seed improvement.

## Training Integrity

- Checkpoint SHA-256: `4cfa2de38ce4cea8462f02a9e91b73ce9a2d4e0a4bf9b9aa356c5e4118cf51c5`
- Expert archive SHA-256: `de5386e512e3458b902c438cf1ada94cc6ab81acfa1a1f591d2ccbaf6bcbbaac`
- Accepted expert episodes: `320/320`
- Expert rejection rate: `8.57%` (limit `25.00%`)
- All accepted demonstrations safe/cooperative: `True/True`
- Training epochs with finite imitation loss: `64` / `True`

## Fixed S1/S2 Regression

| Scene | Execution | Cooperative Safe Capture | Collision | Boundary | Transit |
| --- | --- | ---: | ---: | ---: | ---: |
| s1_cylinder | RAW | 0.0% | 100.0% | 0.0% | 100.0% |
| s1_cylinder | CBF | 35.0% | 0.0% | 0.0% | 100.0% |
| s1_box | RAW | 0.0% | 100.0% | 5.0% | 100.0% |
| s1_box | CBF | 45.0% | 0.0% | 0.0% | 100.0% |
| s1_wall | RAW | 0.0% | 100.0% | 5.0% | 100.0% |
| s1_wall | CBF | 50.0% | 0.0% | 0.0% | 100.0% |
| s2 | RAW | 0.0% | 100.0% | 0.0% | 100.0% |
| s2 | CBF | 45.0% | 0.0% | 0.0% | 100.0% |

## V5 Random S3 Validation

| Execution | Cooperative Safe Capture (95% Wilson CI) | Capture | Collision | Boundary | Transit | Time to Capture | Path / Defender | CBF Correction (mean / median / p95) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAW | 0.0% (0.0%, 6.0%) | 0.0% | 100.0% | 0.0% | 100.0% | n/a | 5.43 m | 0.000 / 0.000 / 0.000 |
| CBF | 68.3% (55.8%, 78.7%) | 68.3% | 0.0% | 0.0% | 100.0% | 11.32 s | 45.90 m | 0.103 / 0.101 / 0.158 |

## S3 Raw/CBF Pairing

- Static maps, initial positions, target profile, and episode seeds exactly paired: `True`
- Static-scene SHA-256: `066b690d766a919a6c452eaa9fc9bf2b5b80093a8a32e920c1425274f0d4c1ab`

## CBF Failure Groups

### Observation condition

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| delayed_noisy | 31 | 29.0% | timeout: 9 |
| nominal | 29 | 34.5% | timeout: 10 |

### Obstacle count

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| 3 | 21 | 38.1% | timeout: 8 |
| 4 | 18 | 22.2% | timeout: 4 |
| 5 | 21 | 33.3% | timeout: 7 |

### Planned clearance proxy

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| medium: planned clearance 0.65-0.80 m | 20 | 30.0% | timeout: 6 |
| narrow: planned clearance <0.65 m | 40 | 32.5% | timeout: 13 |

### Target motion

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| flee_persistence | 31 | 25.8% | timeout: 8 |
| s_curve | 29 | 37.9% | timeout: 11 |

## Gate Decision

- `s3_cooperative_safe_capture_at_least_85_percent`: `False`
- `s3_collision_at_most_2_percent`: `True`
- `s3_boundary_at_most_2_percent`: `True`
- `s3_transit_at_least_99_percent`: `True`
- `all_fixed_cbf_at_least_98_percent`: `False`
- Overall one-seed development candidate gate: `False`

Raw actor and CBF execution are separate artifacts. CBF safety improvement is not attributed entirely to the learned actor. A passing development gate only permits the pre-registered next step; it does not open or replace the V5 next-locked seed block.
