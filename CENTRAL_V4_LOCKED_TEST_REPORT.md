# Central V4 Locked-Test Report (D2-D4)

## Contract and decision hygiene

The retained BC checkpoints were frozen before the locked test was opened. The metric replay used the same models, fixed seed `660501`, S3 seed block `647001`, and 100 episodes per artifact. Commit `6e53efa` adds path-length and CBF-correction instrumentation; commit `d8ddaf7` reuses the policy-independent Transit evidence and commit `5cd5360` restores the already-frozen S3 scenes without re-sampling. No model, hyperparameter, scene, or threshold was tuned after opening the locked block.

Replay outcomes identical to the original locked artifacts: **True**.

## Frozen checkpoints

| Training seed | SHA-256 |
| ---: | --- |
| 661201 | `ccab4a9fa899082c7a363c7f0b24a58bd13ecdfb59fc9a2e265dddfad320de8d` |
| 661202 | `8baeec9c8e21d5dd6bcb3a5848e1edd3ac3ba05b99d2930799a88d3a91e42494` |
| 661203 | `a0c809607954996f7367db8f7ce346cec62552de235aa21f3a91f95b78f8c22b` |

## Fixed S1/S2 CBF regression

Values are mean +/- sample standard deviation across the three independently trained checkpoints.

| Scene | Cooperative capture | Capture | Collision | Boundary | Transit | Mean path / defender | Mean CBF correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1_cylinder | 100.0 +/- 0.0% | 100.0 +/- 0.0% | 0.0 +/- 0.0% | 0.0 +/- 0.0% | 100.0 +/- 0.0% | 14.46 +/- 0.03 m | 0.016 +/- 0.001 |
| s1_box | 100.0 +/- 0.0% | 100.0 +/- 0.0% | 0.0 +/- 0.0% | 0.0 +/- 0.0% | 100.0 +/- 0.0% | 14.39 +/- 0.03 m | 0.027 +/- 0.003 |
| s1_wall | 98.7 +/- 0.6% | 98.7 +/- 0.6% | 1.3 +/- 0.6% | 1.3 +/- 0.6% | 100.0 +/- 0.0% | 14.31 +/- 0.04 m | 0.032 +/- 0.006 |
| s2 | 100.0 +/- 0.0% | 100.0 +/- 0.0% | 0.0 +/- 0.0% | 0.0 +/- 0.0% | 100.0 +/- 0.0% | 14.80 +/- 0.05 m | 0.043 +/- 0.008 |

## Random S3 locked test

| Execution | Cooperative capture | Capture | Target zone entry | Pursuer zone entries | Transit | Collision | Boundary | Time to capture | Min clearance | Path / defender | CBF correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAW | 2.3 +/- 1.2% | 2.3 +/- 1.2% | 0.0 +/- 0.0% | 3.14 +/- 0.03 | 99.0 +/- 0.0% | 97.7 +/- 1.2% | 0.0 +/- 0.0% | 4.33 +/- 0.28 s | -0.10 +/- 0.01 m | 5.81 +/- 0.08 m | 0.000 +/- 0.000 |
| CBF | 75.3 +/- 6.5% | 75.3 +/- 6.5% | 18.7 +/- 8.5% | 3.93 +/- 0.02 | 99.0 +/- 0.0% | 4.7 +/- 1.2% | 4.7 +/- 1.2% | 8.08 +/- 0.77 s | 0.35 +/- 0.01 m | 26.12 +/- 1.39 m | 0.175 +/- 0.008 |

The raw actor is not safely deployable: its S3 collision rate is the complement of its very low capture rate in nearly every episode. CBF is therefore part of the retained execution stack, not evidence attributable to the policy network alone.

## CBF failure groups on S3

Counts below pool the 300 checkpoint-scenario evaluations for descriptive failure analysis. Main uncertainty remains across the three training seeds. The planned-route clearance band is a reproducible proxy for channel width, not a direct geometric doorway-width measurement.

### Obstacle count

| Group | Evaluations | Cooperative failures | Collision | Boundary | Timeout | Transit failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 102 | 24 (23.5%) | 4 | 4 | 20 | 3 |
| 4 | 96 | 23 (24.0%) | 5 | 5 | 18 | 0 |
| 5 | 102 | 27 (26.5%) | 5 | 5 | 22 | 0 |

### Obstacle layout

| Group | Evaluations | Cooperative failures | Collision | Boundary | Timeout | Transit failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cylinder1+box1+wall1 | 102 | 24 (23.5%) | 4 | 4 | 20 | 3 |
| cylinder1+box1+wall2 | 36 | 9 (25.0%) | 4 | 4 | 5 | 0 |
| cylinder1+box1+wall3 | 15 | 7 (46.7%) | 2 | 2 | 5 | 0 |
| cylinder1+box2+wall1 | 30 | 6 (20.0%) | 0 | 0 | 6 | 0 |
| cylinder1+box2+wall2 | 30 | 9 (30.0%) | 1 | 1 | 8 | 0 |
| cylinder1+box3+wall1 | 9 | 2 (22.2%) | 0 | 0 | 2 | 0 |
| cylinder2+box1+wall1 | 30 | 8 (26.7%) | 1 | 1 | 7 | 0 |
| cylinder2+box1+wall2 | 15 | 4 (26.7%) | 0 | 0 | 4 | 0 |
| cylinder2+box2+wall1 | 27 | 3 (11.1%) | 1 | 1 | 2 | 0 |
| cylinder3+box1+wall1 | 6 | 2 (33.3%) | 1 | 1 | 1 | 0 |

### Channel-clearance proxy

| Group | Evaluations | Cooperative failures | Collision | Boundary | Timeout | Transit failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medium: planned clearance 0.65-0.80 m | 120 | 28 (23.3%) | 2 | 2 | 26 | 0 |
| narrow: planned clearance <0.65 m | 177 | 46 (26.0%) | 12 | 12 | 34 | 3 |
| wide: planned clearance >=0.80 m | 3 | 0 (0.0%) | 0 | 0 | 0 | 0 |

### Defender birth side

| Group | Evaluations | Cooperative failures | Collision | Boundary | Timeout | Transit failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| left | 147 | 35 (23.8%) | 6 | 6 | 29 | 3 |
| right | 153 | 39 (25.5%) | 8 | 8 | 31 | 0 |

### Observation condition

| Group | Evaluations | Cooperative failures | Collision | Boundary | Timeout | Transit failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| delayed_noisy | 141 | 44 (31.2%) | 8 | 8 | 36 | 0 |
| nominal | 159 | 30 (18.9%) | 6 | 6 | 24 | 3 |

### Target speed scale

| Group | Evaluations | Cooperative failures | Collision | Boundary | Timeout | Transit failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.45 | 144 | 40 (27.8%) | 7 | 7 | 33 | 3 |
| 0.55 | 156 | 34 (21.8%) | 7 | 7 | 27 | 0 |

### Target motion mode

| Group | Evaluations | Cooperative failures | Collision | Boundary | Timeout | Transit failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flee_persistence | 156 | 43 (27.6%) | 10 | 10 | 33 | 0 |
| s_curve | 144 | 31 (21.5%) | 4 | 4 | 27 | 3 |

## Shared Transit failure audit

There are 1 unique locked scenario(s) with Transit failure across 6 raw/CBF checkpoint evaluations.

| Episode | Episode seed | Layout seed | Capture | Collision | Boundary | Target Transit reason | Execution clearance |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 68 | 647069 | 1647069 | True | False | False | obstacle_clearance_violation | 0.599890 m |

## Locked conclusion

Retained BC+CBF achieves 75.3 +/- 6.5% Cooperative Safe Capture on S3. This is lower than validation and includes non-zero safety failures, so the claim is a reproducible partial success rather than solved robust capture. The locked block remains closed to tuning.
