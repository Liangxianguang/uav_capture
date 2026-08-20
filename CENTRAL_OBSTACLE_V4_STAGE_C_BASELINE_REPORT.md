# V4 Central Obstacle Capture: Stage C Baseline and Retraining Decision

Date: 2026-08-20

## Scope

This is a controlled distribution-shift diagnostic, not a locked-test result.
It evaluates the old recurrent behavior-cloning (BC) checkpoint on the frozen
V4 task. The checkpoint was trained with a curriculum that eventually forced
the target to cross the central zone. V4 explicitly does not impose that
behavior: the target uses normal `flee_persistence` dynamics, capture ends the
episode, and target zone entry is only a diagnostic.

Checkpoint evaluated:

```text
results/showcase_s3_curriculum/bc_seed650202/checkpoint.pt
```

Protocol: `configs/central_bidirectional_v4.yaml`.

## Evaluation Contract

- S1: fixed central cylinder, box, and wall layouts; 20 episodes per layout.
- S2: frozen central mixed layout; 20 episodes.
- All starts use V4 opposite-side births and target speed scale `0.45`.
- Cooperative safe capture requires no collision or boundary violation, a
  capture-radius event, and at least two defenders that have entered the
  central zone before capture.
- The target is not required to enter or cross the central zone.
- Raw policy action and CBF-filtered action are reported separately.
- `Transit` is an independent map-feasibility check and is not credited as a
  learned-policy capture result.

## Results

| Scene | Raw cooperative safe capture | Raw collision | CBF cooperative safe capture | CBF collision | CBF mean capture time |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 cylinder | 0/20 (0%) | 20/20 (100%) | 17/20 (85%) | 0/20 (0%) | 11.86 s |
| S1 box | 0/20 (0%) | 20/20 (100%) | 18/20 (90%) | 0/20 (0%) | 10.26 s |
| S1 wall | 0/20 (0%) | 20/20 (100%) | 18/20 (90%) | 0/20 (0%) | 8.31 s |
| S2 mixed | 0/20 (0%) | 20/20 (100%) | 15/20 (75%) | 0/20 (0%) | 11.66 s |

All eight runs recorded independent Transit success of `20/20`. Raw S1-box
also had one boundary violation; the other raw runs had none. The residual CBF
failures are timeouts, not collisions.

Generated evidence, excluded from Git:

```text
results/central_v4/bc_seed650202_s1_cylinder_raw_20
results/central_v4/bc_seed650202_s1_cylinder_cbf_20
results/central_v4/bc_seed650202_s1_box_raw_20
results/central_v4/bc_seed650202_s1_box_cbf_20
results/central_v4/bc_seed650202_s1_wall_raw_20
results/central_v4/bc_seed650202_s1_wall_cbf_20
results/central_v4/bc_seed650202_s2_raw_20
results/central_v4/bc_seed650202_s2_cbf_20
```

## Decision

The old BC checkpoint is not an acceptable V4 policy baseline. CBF removes
collisions but cannot recover reliable capture, so its safety contribution must
not be attributed to the learned policy. The failure is expected because the
checkpoint learned from a task distribution with forced target transit and
different obstacle progression.

Start a new V4 behavior-cloning run with
`configs/capture_radius_recurrent_behavior_cloning_central_v4_flee.yaml`.
Its curriculum is deliberately narrow and matches V4 exactly:

1. Cylinder only.
2. Cylinder plus box.
3. Fixed cylinder, box, and wall layout.

Every stage uses left-side defenders, a right-side normally fleeing target,
initial side distance `5.5 m`, target speed scale `0.45`, and
`target_crossing_probability: 0.0`. Random S3 layouts, reverse-side births,
faster targets, and target crossing are deferred until a fixed-scene policy is
validated. Only a passing V4 BC checkpoint may initialize the matching MAPPO
pilot configuration.

## Training-Configuration Audit

An initial V4 BC collection run was stopped before optimization because its
expert manifest reported only `32.5%` safe capture despite zero collisions.
The run had inherited the source development configuration's `7.5 m` detection
range, while the frozen V4 expert feasibility result uses `14.0 m`. A controlled
20-episode cylinder check with the same expert and the frozen range restored
`20/20` safe captures and zero collisions.

The formal V4 BC and MAPPO configurations now both reference
`configs/capture_radius_pursuit_central_v4_flee.yaml`, which records the
frozen `14.0 m` range, mixed obstacle profile, V4 world and agent settings.
The aborted local directory `results/central_v4/bc_seed660701` is retained as
an excluded diagnostic artifact and must not be used as a checkpoint or an
expert dataset.

## Required Gate Before MAPPO

1. Generate a new expert dataset and verify its manifest reports high safe
   capture rate with no collision trend in every curriculum stage.
2. Evaluate the resulting BC checkpoint on V4 S1 and S2 in raw and CBF modes.
3. Use the new BC checkpoint for MAPPO initialization only if it demonstrates
   stable cooperative safe capture; otherwise fix expert demonstrations or the
   task design before changing the network or increasing training time.
