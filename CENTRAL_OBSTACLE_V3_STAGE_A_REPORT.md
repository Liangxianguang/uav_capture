# Central Obstacle V3 — Stage A/B Evidence

Date: 2026-08-20

## Scope

This stage freezes the central-obstacle capture contract and verifies that the
fixed mixed layout is feasible before training a new policy. The main capture
rollout terminates at the first safe capture. Complete left-to-right/right-to-
left transit for every participant is therefore evaluated as an independent
transit experiment, not as a post-capture requirement.

## Implemented contract

- `target_zone_entered` and per-defender `defender_zone_entered` record entry into
  `obstacle_zone_x`.
- `safe_capture_in_pursuit` requires a safe capture and the required central-zone
  encounter. For the ordinary S3 protocol, defender entry is sufficient; the
  dedicated `s1_cross` task additionally requires target entry.
- `transit_route_feasible` uses a conservative, radius-inflated grid route with
  continuous segment clearance checks.
- `transit_success` independently executes the target and each defender under
  the configured speed/acceleration limits and checks obstacle clearance and
  boundary safety.
- Results are stored in CSV/JSON metadata, including first zone-entry steps,
  transit outcomes, termination reasons, and layout geometry.

## Verification

Command:

```powershell
$pythonExe='C:\Users\liangxianguang\.conda\envs\uav-encirclement-gpu\python.exe'
& $pythonExe -m pytest tests/test_mixed_obstacle_showcase.py tests/test_random_central_mixed_obstacles.py tests/test_mixed_obstacle_showcase_cli.py tests/test_mixed_obstacle_showcase_evaluation.py -q
```

Result: **17 passed**.

Fixed mixed central main task (`s1_cross`, Dynamic Encirclement + CBF, 20
episodes, seeds 651001–651020):

| Metric | Result |
|---|---:|
| Safe Capture | 20/20 (100%) |
| Safe Capture in Pursuit | 20/20 (100%) |
| Target entered central zone | 20/20 (100%) |
| At least one defender entered central zone | 20/20 (100%) |
| Independent Transit Success | 20/20 (100%) |
| Obstacle/inter-drone collision | 0% |
| Boundary violation | 0% |
| Mean capture time | 2.47 s |
| Worst rollout minimum clearance | 0.444 m |

Evidence files (generated locally; experiment outputs remain ignored by Git):

- `results/central_v3/expert_s1_cross_cbf_20_v2/summary.json`
- `results/central_v3/expert_s1_cross_cbf_20_v2/episodes.csv`

## Important limitation

The fixed expert gate is passed. The randomized S3 protocol has not yet passed
the strict target-crossing capture task; high-speed target crossing can cause
safety failures in dense five-obstacle maps. S3 is therefore retained as the
next training/generalization stage, not reported as completed.
