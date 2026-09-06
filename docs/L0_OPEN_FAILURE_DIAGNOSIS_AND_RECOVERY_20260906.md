# L0 Open Failure Diagnosis and Recovery

Date: 2026-09-06  
Scope: development-only L0 open recovery; no locked test was opened.

## Finding

The original L0 failure was caused by a short anticipatory CBF horizon combined
with late braking in the actor/expert action stream. It was not a timeout
failure and it was not caused by an obstacle. The scene has zero obstacles,
nominal observation, zero dropout, zero noise, and zero message delay.

With the historical default `anticipatory_horizon_steps=3`, the rollout reaches
the final approach while several defenders are still moving at roughly
4.0-4.8 m/s. Pairwise clearance is already near the 0.35 m operational
margin, while the upper boundary constraint is becoming active. The requested
deceleration cannot satisfy the joint CBF constraints and the acceleration
limits simultaneously. The correct safety behavior is `controlled_abort`;
the run does not execute an unverified action.

The diagnostic reproduces the transition on scene `l0_open_28400101`:

- step 31: CBF still feasible, minimum pairwise clearance about `0.369 m`;
- step 32: CBF infeasible, minimum constraint value about `-0.089`, with
  boundary, pairwise, and acceleration constraints active;
- `timed_out=false`, `raw_unverified_executed=false`, and the state remains
  collision-free.

## Recovery evidence

All results below are development evidence on the same frozen eight-scene L0
manifest. They are not historical V4/V5 locked-test results.

| Controller | CBF horizon | Safe capture | CBF abort | Safety violations |
|---|---:|---:|---:|---:|
| historical actor + CBF | 3 | 0/8 | 8/8 | 0 |
| recovery actor + CBF | 3 | 4/8 | 4/8 | 0 |
| recovery actor + CBF | 5 | 8/8 | 0 | 0 |
| recovery actor + JEPA/Ledger/CBF, extended candidates | 5 | 8/8 | 0 | 0 |

The standard paired M0 run with the recovery actor and `horizon=5` also gives
`8/8`, with `collision=0`, `boundary=0`, `pairwise=0`,
`raw_unverified=0`, and `cbf_timeout=0`. The M3 result is recorded by the
candidate recovery evaluator because the calibrated Ledger is bound to the
original L0-L3 collection protocol; its protocol binding is intentionally not
silently rewritten for the one-condition recovery protocol.

## Changes

- Added explicit `--cbf-horizon` plumbing to the paired evaluators. The
  historical default remains `3` so old runs remain reproducible.
- Added recurrent checkpoint metadata and evaluator reset handling for the
  `sequence_length=1` recovery actor.
- Added the L0 diagnosis and recovery evaluators with independent CBF probes.
- Added the development-only protocol
  `configs/jepa_safe_capture_l0_open_recovery_v1.yaml`.
- Trained a separate L0 recovery actor under
  `results/capture_radius_recurrent_behavior_cloning_l0_open_recovery_long_seed661606/`.

No CBF margin was reduced, no stale/OOD gate was disabled, and
`controlled_abort` was not removed.

## Interpretation and next gate

`horizon=5` is the current L0 recovery setting, not yet a global replacement
for every difficulty level. It increases CBF branch work and must be checked on
L1-L3 and at least three development seeds before being used in a formal
benchmark. The next gate is a multi-seed paired development run with the same
CBF contract, followed by a latency audit and a separate calibration-bound
Ledger for any new protocol.
