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

## First Corrected BC Run

The corrected run `results/central_v4/bc_seed660701_v4sensor` used the frozen
sensor contract and completed all 24 BC epochs. Its 320 expert episodes were
all safe captures with no collisions: cylinder `96/96`, cylinder-plus-box
`112/112`, and mixed `112/112`. The expert data therefore passes the C3 data
quality gate.

The learned actor does not yet pass the policy gate. On frozen S2 (20 episodes,
seeds 660301-660320), raw actions achieved `0/20` cooperative safe captures
with `20/20` collisions. CBF actions prevented all collisions but achieved
only `1/20` (5%) cooperative safe captures. This is not a MAPPO initialization
candidate. The training action MSE was still decreasing but finished at `0.824`
after 24 epochs, versus `0.699` for the older BC run.

The next experiment is a BC optimization pilot on this same frozen dataset
distribution: increase fitting time before changing task difficulty, confirm
S2 raw and CBF results first, then run the three S1 layouts only after S2 is
stable. MAPPO, S3 random maps, higher target speed, target crossing, and
execution-dynamics randomization remain blocked until this gate is passed.

## Shape-Aware BC Gate

The optimization audit found a second task-interface limitation: the original
actor received obstacle center, bounding radius, and height only. A box and a
wall with the same bounding radius could therefore be indistinguishable to the
policy. V4 now uses `shape_extents_and_type`, adding horizontal half-extents
and a cylinder/box/wall one-hot code for each observed obstacle. This is part
of the frozen V4 protocol and is checked in both training and showcase
evaluation.

The shape-aware BC run `results/central_v4/bc_shapeaware_seed660701` used 640
safe, collision-free expert episodes. Layout coverage was cylinder `301`, box
`106`, wall `94`, cylinder-plus-box `98`, and mixed `41`; the resulting actor
observation is 63-dimensional. Its final action MSE was `0.0274`.

| Scene | Raw cooperative safe capture | Raw collision | CBF cooperative safe capture | CBF collision | CBF mean capture time |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 cylinder | 19/20 (95%) | 1/20 (5%) | 19/20 (95%) | 0/20 (0%) | 3.49 s |
| S1 box | 15/20 (75%) | 5/20 (25%) | 18/20 (90%) | 0/20 (0%) | 3.05 s |
| S1 wall | 20/20 (100%) | 0/20 (0%) | 20/20 (100%) | 0/20 (0%) | 2.95 s |
| S2 mixed | 20/20 (100%) | 0/20 (0%) | 20/20 (100%) | 0/20 (0%) | 3.44 s |

Every one of these eight controlled evaluations recorded independent Transit
success `20/20`. The raw and CBF numbers remain separate: CBF removes the
remaining S1 collisions but is not attributed to the learned actor. This model
passes the C3 initialization gate for one MAPPO pilot because it is already
stable under CBF on every frozen S1/S2 layout and raw S2 is fully successful.

## Required Gate Before MAPPO

1. Generate a new expert dataset and verify its manifest reports high safe
   capture rate with no collision trend in every curriculum stage.
2. Evaluate the resulting BC checkpoint on V4 S1 and S2 in raw and CBF modes.
3. Use the new BC checkpoint for MAPPO initialization only if it demonstrates
   stable cooperative safe capture; otherwise fix expert demonstrations or the
   task design before changing the network or increasing training time.
