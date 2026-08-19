# Hard pursuit benchmark protocol

This protocol defines the first stage-1 difficulty evaluation for the
capture-radius task. It changes target maneuvers and obstacle layouts while
keeping the capture event, decentralized observation interface, and safety
metrics unchanged.

## Fixed task definition

Four defenders pursue one target in the 3D kinematic environment. A safe
capture is recorded when at least one defender satisfies

```text
min_i ||p_defender_i - p_target|| <= 0.80 m
```

before timeout and without a prior defender-obstacle collision,
defender-defender collision, or world-boundary violation.

The benchmark does not claim physical impact, camera perception, SITL, or
real-flight capture.

## Scenario design

`configs/capture_radius_hard_benchmark.yaml` contains five locked scenario
families:

| Scenario | Target | Obstacles | Speed scale |
|---|---|---:|---:|
| baseline_open | persistent flee | none | 0.55 |
| random_turn_clutter | random heading changes | mixed | 0.75 |
| s_curve_unseen_maps | lateral S-curve | boxes | 1.00 |
| burst_narrow_channels | burst acceleration | narrow walls | 1.00 |
| boundary_escape_walls | boundary-biased escape | walls | 1.25 |

Each family uses 100 episodes. Scenario overrides are applied to a deep copy
of the base configuration. The `map_seed_offset` ensures that the obstacle
layout stream differs from the base target/initialization stream.

## Evaluation command

Run the audited dynamic encirclement CBF baseline:

```powershell
.\scripts\run_capture_radius_pursuit.ps1 `
  -Config configs/capture_radius_hard_benchmark.yaml `
  -Controller encirclement_cbf `
  -Output results/hard_benchmark_encirclement_cbf
```

For a quick smoke run, override all scenarios to two episodes:

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python scripts/run_capture_radius_pursuit.py `
  --config configs/capture_radius_hard_benchmark.yaml `
  --controller encirclement_cbf `
  --episodes 2 `
  --output results/hard_benchmark_smoke
```

The output contains `episodes.csv`, `summary.json`, source hashes, effective
configuration, and one trajectory image per scenario. Do not tune parameters
on the 100-episode locked run.

## Required metrics

Report each scenario separately:

- Safe Capture and Capture rates;
- collision and boundary-violation episode rates;
- mean and worst minimum clearance;
- capture-time mean and dispersion;
- target visible fraction and teammate-message age;
- target motion mode and obstacle profile.

The hard benchmark is a diagnostic stage. Its purpose is to identify failure
conditions for the existing policy before adding learned prediction or
recurrent memory.
