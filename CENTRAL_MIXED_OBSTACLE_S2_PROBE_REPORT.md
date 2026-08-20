# Central Mixed-Obstacle S2 Probe

## Scope

This report records the first reverse-side (S2) probe for the controlled
central mixed-obstacle scenario. It is a distribution-shift diagnostic, not a
locked-test result.

## S2 Contract

- Four defenders start at `x = +5.0 m`; the target starts at `x = -5.0 m`.
- The cylinder, box, and wall remain in the validated central layout.
- The target retains its normal escape behavior and moves away from the
  defenders; all defenders must complete the central-zone crossing before a
  showcase success can be recorded.
- `s2_cross`, in which the target is explicitly required to cross toward the
  opposite side before capture, remains a distinct future feasibility task. It
  is not mixed into ordinary S2 training or metrics.

## Controlled Probe

All runs use side distance `5.0 m`, detection range `14 m`, target speed scale
`0.55`, CBF enabled, and the fixed seed block `644001..644020`.

| Method | Safe Capture | Showcase Success | Defender crossing | Collision | Boundary violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dynamic Encirclement expert + CBF | 20 / 20 (100%) | 20 / 20 (100%) | 100% | 0% | 0% |
| Existing curriculum MAPPO + CBF | 0 / 20 (0%) | 0 / 20 (0%) | 100% | 0% | 0% |

The expert result verifies that the reversed scenario is geometrically and
dynamically solvable. The existing MAPPO checkpoint crosses the obstacle zone
safely but does not close the capture distance, which is a directional
distribution-shift failure rather than evidence of an impossible map.

## Decision

The next training run should retain original random episodes and late-stage
mixed-layout episodes, while sampling both `defender_sides: [left, right]`.
It should not include target-crossing episodes until the separate `s2_cross`
contract has passed an expert feasibility check. Checkpoints must be selected
on held-out S1/S2 validation seeds and then compared across at least three
independent training seeds.

## Reproduction

```powershell
python scripts/evaluate_mixed_obstacle_showcase.py `
  --baseline dynamic_encirclement --scenario s2 `
  --output-dir results/showcase/probe_dynamic_s2_cbf_v2 `
  --seed 644001 --episodes 20 `
  --initial-side-distance 5.0 --detection-range 14.0 `
  --target-speed-scale 0.55 --use-cbf --device cuda

python scripts/evaluate_mixed_obstacle_showcase.py `
  --method f2 `
  --checkpoint results/showcase_curriculum/mappo_seed640101/checkpoint.pt `
  --scenario s2 `
  --output-dir results/showcase/probe_curriculum_mappo_s2_cbf `
  --seed 644001 --episodes 20 `
  --initial-side-distance 5.0 --detection-range 14.0 `
  --target-speed-scale 0.55 --use-cbf --device cuda
```
