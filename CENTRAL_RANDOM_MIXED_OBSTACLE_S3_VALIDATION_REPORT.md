# Central Random Mixed-Obstacle S3 Validation

## Scope

This document records the first randomized-central-layout S3 validation run.
It validates the map-generation and evaluation protocol and identifies the
current policy's failure boundary. It is **not** a locked-test result and is
not a multi-training-seed result.

## Protocol

- Protocol: `configs/central_random_mixed_obstacle_s3_protocol.yaml`.
- Split: validation, motion seeds `646001..646040`; layout seeds
  `1646001..1646040`.
- Each layout has 3-5 central obstacles and contains at least one cylinder,
  box, and wall. Walls are physically axis-aligned at 0 or 90 degrees because
  arbitrary yaw is not yet supported by the collision model.
- Before acceptance, every defender start has a conservative route to the
  target's initial side and all spawn/overlap/boundary checks pass.
- The 192-row full-factorial condition table is deterministically shuffled per
  split. The 40 validation rows therefore vary start side, obstacle count,
  initial separation, target speed, target motion, and observation condition
  without synchronizing direction with sensing quality.
- Detection range is 14 m. The observation conditions are `nominal` and
  `delayed_noisy`; CBF is enabled for both methods.

## Results

| Method | Safe Capture | Collision | Boundary Violation | Defender Crossing | All-Defender Crossing | Mean Capture Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dynamic Encirclement expert + CBF | 33 / 40 (82.5%) | 12.5% | 7.5% | 86.3% | 52.5% | 4.69 s |
| Curriculum MAPPO seed 640101 + CBF | 13 / 40 (32.5%) | 0% | 0% | 93.8% | 90.0% | 9.35 s |

The gap is primarily capture completion rather than collision: the current
MAPPO times out in 27/40 episodes despite completing its obstacle traversal.
It captures 13/17 left-side episodes but 0/23 right-side episodes. This agrees
with the separate fixed S2 probe and identifies reverse-side pursuit as the
first curriculum-training target. The expert establishes that the randomized
scenario is generally solvable, but its 12.5% safety-failure rate also means
that S3 must not yet be claimed as a robust expert baseline.

`All-Defender Crossing` is deliberately reported separately from Safe Capture.
The simulator terminates as soon as the first safe capture happens, so a
non-capturing defender may not have reached the far side before task success.
This is a coverage diagnostic, not an extra condition retroactively added to
the capture definition.

## Decision

The next model must train on both defender start sides and randomized central
layouts while preserving ordinary random episodes. Model selection must use a
separate S1/S2 validation block; the S3 locked-test block `647001..647100`
must remain untouched until three independent training seeds are available.
Raw-action and CBF-action evaluations must be reported separately.

## Reproduction

```powershell
python scripts/evaluate_random_central_mixed_obstacles.py `
  --baseline dynamic_encirclement --split validation `
  --output-dir results/showcase/s3_validation_dynamic_cbf_v2 `
  --use-cbf --device cuda

python scripts/evaluate_random_central_mixed_obstacles.py `
  --method f2 `
  --checkpoint results/showcase_curriculum/mappo_seed640101/checkpoint.pt `
  --split validation `
  --output-dir results/showcase/s3_validation_curriculum_mappo_cbf_v2 `
  --use-cbf --device cuda
```
