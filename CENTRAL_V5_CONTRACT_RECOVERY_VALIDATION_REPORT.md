# Central V5 Retained-BC Development Validation

This is a one-training-seed development-validation report. It is not a locked test and is not evidence of a multi-seed improvement.

## Training Integrity

- Checkpoint SHA-256: `c3b87633ecfbd078451d39b777c3ceeeeb352aaf32f185c06d15c75ae90a9ee3`
- Expert archive SHA-256: `4196b43587743fec61cdd3902d6bacd05faeebc6625de41f101f44e4496bce27`
- Data provenance: `reused_expert_archives`
- Archive source balance: `equal_sequences`; selected sequences balanced: `True`
- All source demonstrations safe/cooperative: `True`
- Source 0: `457/322` sequences selected; safe/cooperative `True/True`.
- Source 1: `457/457` sequences selected; safe/cooperative `True/True`.
- Training epochs with finite imitation loss: `64` / `True`

## Fixed S1/S2 Regression

| Scene | Execution | Cooperative Safe Capture | Collision | Boundary | Transit |
| --- | --- | ---: | ---: | ---: | ---: |
| s1_cylinder | RAW | 10.0% | 90.0% | 0.0% | 100.0% |
| s1_cylinder | CBF | 20.0% | 0.0% | 0.0% | 100.0% |
| s1_box | RAW | 0.0% | 100.0% | 0.0% | 100.0% |
| s1_box | CBF | 95.0% | 0.0% | 0.0% | 100.0% |
| s1_wall | RAW | 0.0% | 100.0% | 0.0% | 100.0% |
| s1_wall | CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| s2 | RAW | 0.0% | 100.0% | 0.0% | 100.0% |
| s2 | CBF | 90.0% | 0.0% | 0.0% | 100.0% |

## V5 Random S3 Validation

| Execution | Cooperative Safe Capture (95% Wilson CI) | Capture | Collision | Boundary | Transit | Time to Capture | Path / Defender | CBF Correction (mean / median / p95) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAW | 1.7% (0.3%, 8.9%) | 1.7% | 98.3% | 0.0% | 100.0% | 3.50 s | 5.61 m | 0.000 / 0.000 / 0.000 |
| CBF | 50.0% (37.7%, 62.3%) | 50.0% | 0.0% | 0.0% | 100.0% | 14.68 s | 50.02 m | 0.108 / 0.108 / 0.158 |

## S3 Raw/CBF Pairing

- Static maps, initial positions, target profile, and episode seeds exactly paired: `True`
- Static-scene SHA-256: `066b690d766a919a6c452eaa9fc9bf2b5b80093a8a32e920c1425274f0d4c1ab`

## CBF Failure Groups

### Observation condition

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| delayed_noisy | 31 | 51.6% | timeout: 16 |
| nominal | 29 | 48.3% | timeout: 14 |

### Obstacle count

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| 3 | 21 | 57.1% | timeout: 12 |
| 4 | 18 | 38.9% | timeout: 7 |
| 5 | 21 | 52.4% | timeout: 11 |

### Planned clearance proxy

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| medium: planned clearance 0.65-0.80 m | 20 | 35.0% | timeout: 7 |
| narrow: planned clearance <0.65 m | 40 | 57.5% | timeout: 23 |

### Target motion

| Group | Episodes | Cooperative failure rate | Failure stages |
| --- | ---: | ---: | --- |
| flee_persistence | 31 | 51.6% | timeout: 16 |
| s_curve | 29 | 48.3% | timeout: 14 |

## Gate Decision

- `s3_cooperative_safe_capture_at_least_85_percent`: `False`
- `s3_collision_at_most_2_percent`: `True`
- `s3_boundary_at_most_2_percent`: `True`
- `s3_transit_at_least_99_percent`: `True`
- `all_fixed_cbf_at_least_98_percent`: `False`
- Overall one-seed development candidate gate: `False`

Raw actor and CBF execution are separate artifacts. CBF safety improvement is not attributed entirely to the learned actor. A passing development gate only permits the pre-registered next step; it does not open or replace the V5 next-locked seed block.
