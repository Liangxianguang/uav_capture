# V4 Central Obstacle Capture: Stages A and B

Date: 2026-08-20

## Scope

This report freezes and validates the fixed V4 task before evaluating or
training a learned policy. It is not a locked-test result and it does not make
an entity-capture or flight-test claim.

The task is a 3D capture-radius pursuit simulation with four defenders and one
target. Cylinder, box, and wall obstacles are confined to the central zone.
The target starts on the right and follows normal evasion dynamics. It is not
required to enter the central zone in the main capture rollout. Capture is
valid only when it is safe and at least two defenders have entered the central
zone before capture. Target and defender crossing capability is verified in a
separate conservative Transit execution.

## Frozen V4 Contract

Source: `configs/central_bidirectional_v4.yaml`.

| Item | Frozen value |
| --- | --- |
| World | `x,y in [-10, 10] m`, `z in [0.5, 10] m`, 1.0 m boundary buffer |
| Central obstacle zone | `x in [-2.5, 3.0] m` |
| Fixed S2 births | defenders at `x=-5.5 m`, target at `x=5.5 m` |
| Minimum initial target-defender distance | 10.0 m (actual nearest distance is greater than 11 m) |
| Capture radius | 0.80 m |
| Safety margin | 0.35 m |
| Cooperative condition | at least 2 of 4 defenders enter the central zone before safe capture |
| Target central entry | diagnostic only; not a success requirement |
| Fixed target scale | 0.45 |
| S2 evaluation seeds | 660201 through 660220 |

The fixed S2 geometry contains one cylinder, one box, and one axis-aligned
wall. The validation path planner inflates obstacles by drone radius plus the
safety margin, rejects maps with out-of-zone geometry, invalid births, or
missing independent Transit routes.

## Implemented Evidence

- `configs/central_bidirectional_v4.yaml` records all task, seed, runtime,
  reporting, and evaluation values.
- `ShowcaseScenario` records the required number of defender zone entries and
  whether target zone entry is required.
- CSV and `episode.json` now include `defender_zone_entry_count`,
  `required_defender_zone_entries`, `target_zone_entry_required`, and
  `cooperative_safe_capture`.
- `v4_s2` loads the frozen protocol. The evaluator writes the protocol into
  `summary.json`; this prevents a rendered episode from losing its task
  definition.
- The showcase CLI can select `cylinder`, `box`, `wall`, `cylinder_box`, and
  `mixed` layouts. The V3 default behavior remains compatible.

## Automated Checks

Command:

```powershell
$pythonExe='C:\Users\liangxianguang\.conda\envs\uav-encirclement-gpu\python.exe'
& $pythonExe -m pytest `
  tests/test_mixed_obstacle_showcase.py `
  tests/test_mixed_obstacle_showcase_cli.py `
  tests/test_mixed_obstacle_showcase_evaluation.py `
  tests/test_random_central_mixed_obstacles.py -q -rA
```

Result: 23 passed.

The tests cover V4 protocol parsing, two-defender capture gating, optional
target zone entry, central-only obstacle extents, two-sided births, independent
Transit feasibility, three single-obstacle layouts, V4 CLI selection, and S3
random-layout regression behavior.

## Rule Expert + CBF Feasibility

All commands use the local `DynamicEncirclementController` with the CBF safety
filter. Result directories are ignored by Git because they contain generated
rollouts; the exact commands and summaries below are the reproducibility
record.

| Scene | Episode seeds | Episodes | Cooperative safe capture | Transit | Collision | Boundary | Worst clearance | Mean capture time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 cylinder | 660221-660240 | 20 | 20/20 (100%) | 20/20 | 0% | 0% | 0.479 m | 3.38 s |
| S1 box | 660241-660260 | 20 | 20/20 (100%) | 20/20 | 0% | 0% | 0.467 m | 2.90 s |
| S1 wall | 660261-660280 | 20 | 20/20 (100%) | 20/20 | 0% | 0% | 0.466 m | 2.91 s |
| S2 fixed mixed | 660201-660220 | 20 | 20/20 (100%) | 20/20 | 0% | 0% | 0.463 m | 3.56 s |

Every successful S1/S2 rollout recorded all four defenders entering the central
zone. The target did not enter the zone in the main rollout, as allowed by the
V4 contract. In the separately executed Transit check, the target safely
traversed right-to-left and all four defenders safely traversed left-to-right.

S2 reproduction command:

```powershell
$pythonExe='C:\Users\liangxianguang\.conda\envs\uav-encirclement-gpu\python.exe'
& $pythonExe scripts/evaluate_mixed_obstacle_showcase.py `
  --baseline dynamic_encirclement `
  --scenario v4_s2 `
  --protocol-config configs/central_bidirectional_v4.yaml `
  --use-cbf `
  --episodes 20 `
  --seed 660201 `
  --output-dir results/central_v4/expert_s2_v4_cbf_20_frozen
```

For S1, replace `--scenario v4_s2` with `--scenario s1 --layout cylinder`,
`--layout box`, or `--layout wall`, and use the listed seed blocks.

## Calibration Decision

The first V4 draft used a 6.5 m per-side birth distance. Under normal target
evasion, it produced boundary failures before capture and was rejected as the
frozen task. The accepted 5.5 m per-side configuration still has more than
11 m nearest initial separation, preserves opposite-side starts and central
obstacles, and gives the target enough legal airspace to evade without relying
on a boundary collision. Only the accepted configuration is reported above.

## Next Gate

Stage C starts with the existing BC checkpoint on frozen S1/S2, evaluating raw
actions and CBF-filtered actions separately. It must not use this expert result
as a learned-policy result. Retraining begins only if that checkpoint fails the
frozen task for a genuine distribution-shift reason.
