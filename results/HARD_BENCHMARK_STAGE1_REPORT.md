# Stage-1 hard pursuit benchmark report

Date: 2026-08-19

## Purpose

This report records the first difficult-target and unseen-map benchmark added
after the frozen capture-radius baseline. The benchmark changes target motion
and obstacle profiles but keeps the decentralized actor observation dimension
at 44 features and keeps the Safe Capture definition unchanged.

## Protocol

- 5 scenario families;
- 100 episodes per scenario;
- seed block starts at 620001;
- 500 episodes per controller;
- 4 defenders, 1 target, 20 m x 20 m x 10 m world;
- capture radius 0.80 m;
- no target truth exposed to the controller;
- results are kinematic simulation results only.

The scenario configuration is `configs/capture_radius_hard_benchmark.yaml`.
The complete protocol is in `HARD_PURSUIT_BENCHMARK_PROTOCOL.md`.

## Dynamic encirclement + CBF diagnostic

The existing audited dynamic-encirclement controller achieved 100% Safe
Capture and 0% collision episodes in all five scenario families. This result
shows that the current rule baseline remains too strong for these particular
initialization and speed ranges; it is not evidence that a learned policy has
already generalized.

## Prediction pursuit + CBF diagnostic

The prediction pursuit controller also achieved 100% Safe Capture and 0%
collision episodes in all five scenario families. It had larger CBF action
corrections and longer capture times than dynamic encirclement.

## Pure pursuit + CBF failure map

The pure pursuit controller exposes meaningful failure conditions:

| Scenario | Safe Capture | Boundary-violation episodes | Mean capture time (s) |
|---|---:|---:|---:|
| baseline_open | 100% | 0% | 1.513 |
| random_turn_clutter | 86% | 11% | 2.419 |
| s_curve_unseen_maps | 71% | 29% | 1.561 |
| burst_narrow_channels | 39% | 61% | 1.805 |
| boundary_escape_walls | 30% | 70% | 0.947 |

These are diagnostic results, not tuned final results. They identify
high-maneuver and boundary-biased scenarios where prediction, memory, and
better team coordination should provide measurable benefit.

## Important implementation correction

The environment now treats any prior world-boundary violation as a safety
failure. A later entry into the capture radius cannot retroactively become a
Safe Capture. This is covered by a regression test.

## Next experiment

The next scientifically meaningful step is to evaluate the frozen MAPPO
checkpoint on this locked hard benchmark. If its failure rate is too low to
separate methods, increase only the held-out target speed/maneuver block or
initial spawn distance, while preserving the 44-feature actor interface and
reserving the final test seed block.
