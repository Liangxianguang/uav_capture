# V5 Fixed-Wall Regression Diagnostic

## Scope

This is one deterministic replay of an already observed development fixed-regression
failure. It is diagnostic evidence only, not a new evaluation, a model-selection
run, or a locked test.

- Candidate checkpoint SHA-256:
  `4fe54f86b033b1d5290ffdaa8d1fb097f7e8b8491071e64f7baf1f8dbbb36bf3`
- Fixed scene: `s1_wall`, policy + frozen CBF-default.
- Reference evaluation seed: `660514` (the 14th seed in `660501`–`660520`).
- Reference `episodes.csv` SHA-256:
  `e9b18873729f5a2afa8dc972296962ac8eb9e1233e01015691808b9dcb701a1e`.
- Replay trace SHA-256:
  `8bc65e5345b9101783e446fcdd125a39198f4dc81b203d2907470d4e4fffbae1`.

## Reproduction check

The diagnostic replay exactly matched the original fixed-regression row for the
four task-level fields: `steps=35`, `collision=True`,
`safe_capture_success=False`, and `termination_reason=safety_failure`.

## Timestep evidence

- The sole safety event occurred at step 35: `target_x_upper`. The evader
  reached the world upper x-boundary (`x=10 m`), resulting in
  `world_violation_steps=1` and `safety_failure`.
- This was **not** a defender-wall or defender-defender collision:
  minimum obstacle clearance was `1.025074 m`; minimum inter-agent clearance
  was `0.390991 m`; minimum defender-boundary clearance was `0.469980 m`.
- The closest pursuer-target approach was `0.847717 m` at step 31 by pursuer 2,
  which is `0.047717 m` outside the fixed `r_capture=0.80 m` criterion.
- CBF was active without a violated defender barrier: maximum action correction
  was `0.553358`, and the minimum recorded CBF barrier was `0.040991`.
- The maximum message and observation ages were both 2 steps.

The top-down and oblique 3D diagnostic figures are retained locally alongside
the trace. They are not committed because media and NPZ artifacts are excluded
from the repository.

## Feasibility counterfactual

On the same fixed wall geometry and seed `660514`, the existing
`dynamic_encirclement` rule expert + identical CBF-default execution achieved
safe cooperative capture at `2.8 s`, with no collision or boundary violation.
The rule-expert episode CSV SHA-256 is
`3ded88ac109420374906f4e68c4b9978defb305d4c723365fba998e67416e843`.

Therefore this map is feasible under the frozen task contract. The failure is
not evidence for changing the capture rule, CBF margin, target boundary rule,
or test split.

## Coverage diagnosis

The paired fixed archive (`fixed_shapeaware_seed661605`) contains 94 accepted
wall demonstrations, but all are in exactly one stratum:
`defender_side=left`, `initial_side_distance=5.5 m`,
`target_speed_scale=0.45`, `target_motion_mode=flee_persistence`,
`target_crossing_required=False`, and `required_defender_zone_entries=2`.

The fixed wall regression uses the same ID wall layout with
`initial_side_distance=5.0 m`. Fixed-layout geometry is deliberately an ID
regression in V5, not an OOD geometry test, but the evaluated episode seed and
motion noise remain excluded from training.

## Decision

The only permitted follow-up is the pre-registered P3-A data-only recovery in
`V5_WALL_COVERAGE_P3A_PREREGISTRATION.md`: extend **only wall** fixed-stage
sampling to include `5.0 m` and `5.5 m` starts. It must not add the evaluation
seed `660514`, alter any task/CBF parameter, add a second intervention, or open
locked block `647201`.
