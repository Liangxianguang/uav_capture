# Central V4 Visualization Evidence (E1-E4)

The visualization artifacts are derived from frozen checkpoint `bc_s3_retained_seed661201/checkpoint.pt` and the existing locked-test protocol. They are evidence of individual trajectories only; the statistical conclusion remains in `CENTRAL_V4_LOCKED_TEST_REPORT.md`.

## Formal successful S2 episode

Artifact directory: `results/central_v4/showcase_s2_locked_success_seed660501_cbf/`

| Check | Value |
| --- | --- |
| Scenario / seed | locked S2 / `660501` |
| Cooperative Safe Capture | `true` |
| Independent Transit | `true` |
| Capture radius | `0.80 m` |
| Final nearest distance | `0.7651807 m` |
| Capture predicate | `true` |
| Simulation samples | `40` |
| Freeze | `21` frames at `12 fps` (`1.75 s`) |
| Trajectory SHA-256 | `948361ab9fb4bb9350853280e9532fef80c1cc6521b68c3361fe4b324d51b444` |

Files in the directory include the checkpoint reference, effective `config.yaml`, `scenario.json`, `episode.json`, `trajectory.npz`, fixed top-down `capture_cbf.png`, animated `capture_cbf.gif`, three-dimensional `capture_cbf_3d.png`, and H.264 `capture_cbf.mp4`.

The final PNG and 3D view show the physical `0.80 m` capture sphere, green `SAFE CAPTURE CONFIRMED` state, all four defender paths, the target path, obstacle geometry, and altitude profiles. The media renderer records the trajectory hash and final-distance predicate in `episode.json`.

## Failure trajectory

Artifact directory: `results/central_v4/showcase_s1_wall_locked_failure_seed660510_cbf/`

This frozen CBF episode ends with `safety_failure`, `cooperative_safe_capture=false`, and final nearest distance `0.9454000 m`; it is included to make the failure mode auditable rather than presenting only a success case. It has matching PNG, GIF, 3D PNG, H.264 MP4, `episode.json`, `trajectory.npz`, checkpoint reference, and effective configuration.

## Media verification

Both MP4 files were decoded successfully by the bundled FFmpeg runtime. SHA-256 values in each `episode.json` match the corresponding `trajectory.npz`; no media is generated from a separately edited trajectory.
