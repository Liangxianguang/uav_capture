# Central V4 S3 Retention Validation

Date: 2026-08-20

## Scope

This is a validation result on the frozen S3 seed block, not a locked test and
not a three-training-seed result. The task remains partially observable 3D
capture-radius pursuit: target central-zone entry is not required, capture
ends an episode, and independent transit is reported separately.

The evaluation contains 40 reproducible random central layouts with 3-5
cylinder/box/wall obstacles, both defender birth sides, target speed scales
`0.45` and `0.55`, `flee_persistence` and `s_curve` target motion, and nominal
or delayed-noisy observations. Cooperative Safe Capture requires a safe
capture-radius event plus central-zone entry by at least two defenders.

## Candidate Construction

The first S3-only behavior-cloning run used 320 quality-gated expert episodes.
It passed expert data collection but lost the fixed S2 regression, so it is
rejected as a candidate. The retained candidate starts from the fixed-scene
shape-aware BC checkpoint and performs recurrent BC on two audited archives:

| Source archive | SHA-256 | Original sequences | Selected sequences |
| --- | --- | ---: | ---: |
| Fixed V4 expert archive | `7666a03c446ca04ecb802664bd92817547aac1736ed041ebd8a76d0050085f48` | 650 | 650 |
| Quality-gated progressive S3 archive | `69f20885b12596ee3b33f52e4bca61c38720ac393f360221902936808b6249c5` | 384 | 650 |

Selection uses deterministic equal-sequence balancing (`seed=661201`). The
candidate is initialized from checkpoint SHA-256
`f6b1026767781880c0a41b872a6e0307e744d2b5827f179c34ae8638642f5797`, trained
for 64 epochs at learning rate `5e-5`, and produces checkpoint SHA-256
`ccab4a9fa899082c7a363c7f0b24a58bd13ecdfb59fc9a2e265dddfad320de8d`.

## Fixed S2 Regression

Twenty frozen V4 S2 episodes were evaluated before S3 validation.

| Execution | Cooperative Safe Capture | Collision | Boundary violation | Transit | Mean capture time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw actor | 10 / 20 (50.0%) | 50.0% | 0.0% | 100% | 3.94 s |
| Actor + CBF | 20 / 20 (100%) | 0.0% | 0.0% | 100% | 3.82 s |

The CBF execution configuration recovers the fixed S2 gate. The raw actor
does not; raw and CBF behavior must therefore remain separate in all claims.

## S3 Validation Results

| Method / execution | Cooperative Safe Capture | Capture | Collision | Boundary violation | Transit | Mean capture time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dynamic Encirclement expert + CBF | 35 / 40 (87.5%) | 87.5% | 5.0% | 0.0% | 100% | 4.52 s |
| Earlier shape-aware BC + CBF | 1 / 40 (2.5%) | 2.5% | 2.5% | 0.0% | 100% | 4.00 s |
| Earlier MAPPO + CBF | 2 / 40 (5.0%) | 5.0% | 0.0% | 0.0% | 100% | 5.55 s |
| Retained S3 BC, raw actor | 3 / 40 (7.5%) | 10.0% | 92.5% | 0.0% | 100% | 4.03 s |
| Retained S3 BC + CBF | 34 / 40 (85.0%) | 85.0% | 2.5% | 2.5% | 100% | 8.07 s |

For the retained BC + CBF candidate, all 34 safe captures also met the
two-defender central-entry condition. Thus the central-entry condition removed
none of the ordinary safe captures in this validation. Target zone entry was
22.5%, but is a diagnostic only and is not required by the task.

The retained BC + CBF candidate achieved a worst observed clearance of
`0.301 m`. It captured 13/17 left-side and 21/23 right-side episodes. The six
failures are four left-side timeouts, one right-side timeout, and one
right-side safety failure; all random maps still pass independent transit.

## Decision

1. The quality-gated, retained BC curriculum is accepted as an S3 validation
   improvement for the **policy + CBF execution stack**.
2. The raw actor is rejected as a safe standalone deployment policy because it
   collided in 37/40 validation episodes. CBF safety is not attributed to the
   learned actor.
3. This is not a locked-test result. The locked S3 seeds remain untouched, and
   three independent training seeds are still required before any research-level
   robustness claim.
4. The next training experiment initializes recurrent MAPPO from the retained
   BC candidate while retaining the same balanced expert archive; it must first
   pass fixed S1/S2 raw/CBF regression and S3 validation before locked testing.

This report concerns only the 3D kinematic simulator. It does not demonstrate
physical contact capture, onboard sensing, vehicle-level control, or real
flight performance.
