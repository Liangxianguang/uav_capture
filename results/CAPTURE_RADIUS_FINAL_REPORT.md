# Capture-radius pursuit final report

Date: 2026-08-19

## Objective and success event

The evaluated task is four defenders pursuing one kinematic evasive target in a
20 m x 20 m x 10 m 3D world with static cylindrical obstacles. The actor sees
only decentralized partial observations: local target detections, occlusion,
dropout, delayed/dropped teammate messages, belief state, teammate geometry,
and obstacle features. The centralized critic is used only during training.

The terminal success event is:

    min_i ||p_defender_i - p_target|| <= r_capture

with `r_capture = 0.80 m`. A Safe Capture additionally requires no earlier
obstacle/teammate collision or world-boundary violation. This report makes no
physical-contact, net-capture, SITL, hardware, or real-flight claim.

## Main frozen result

The main policy is parameter-sharing CTDE MAPPO initialized from a local
observation behavior-cloning checkpoint and evaluated with the explicitly
named local-information CBF execution filter. Each training seed used 131,072
environment steps. The locked test block is identical for all three seeds:
100 episodes each in `open`, `clutter` (3 obstacles), and `occluded` (5
obstacles), starting at test seed 610001.

| Training seed | Episodes | Safe Capture | Collision | Mean capture time (s) | Mean minimum clearance (m) |
|---:|---:|---:|---:|---:|---:|
| 521001 | 300 | 100.0% | 0.0% | 1.373 | 0.843 |
| 521002 | 300 | 100.0% | 0.0% | 1.391 | 0.812 |
| 521003 | 300 | 100.0% | 0.0% | 1.334 | 0.874 |
| Mean +/- population std | 900 | 100.0% +/- 0.0% | 0.0% +/- 0.0% | 1.366 +/- 0.023 | 0.843 +/- 0.025 |

The individual evidence directories are:

- `results/capture_radius_mappo_warmstart_locked100_cpu_seed610001_v1`
- `results/capture_radius_mappo_warmstart_locked100_cpu_seed610001_from521002_v1`
- `results/capture_radius_mappo_warmstart_locked100_cpu_seed610001_from521003_v1`

Each contains `episodes.csv`, `summary.json`, trajectory `.npz`/`.png`,
`evaluation_protocol.json`, source hashes, and the exact checkpoint hash.

## Ablations and controls

| Policy / execution | Episodes | Safe Capture | Collision | Mean capture time (s) |
|---|---:|---:|---:|---:|
| MAPPO warm-start, no prediction, raw actions | 300 | 98.33% | 1.67% | 1.328 |
| MAPPO warm-start, no prediction + CBF | 300 | 100.00% | 0.00% | 1.373 |
| MAPPO warm-start + constant-velocity prediction, raw actions | 300 | 96.67% | 3.33% | 1.340 |
| MAPPO warm-start + prediction + CBF | 300 | 100.00% | 0.00% | 1.397 |

The prediction variant adds four actor features: a constant-velocity future
belief position and an age-dependent uncertainty scalar. It uses no target
truth. Its interface can later be replaced by the user's learned trajectory
predictor without changing the capture event or evaluation protocol.

The raw-vs-CBF rows quantify the safety layer rather than hiding collisions in
the success metric. The rule expert development baseline also achieved 100%
Safe Capture with 0 collisions on 30 episodes per scenario, and is retained
under `results/capture_radius_dev_encirclement_cbf_v4`.

The from-scratch 131,072-step MAPPO control run is retained as a negative
control under `results/capture_radius_mappo_dev_gpu_seed521001_v1`; its locked
evaluation had 0% Safe Capture. This documents why the warm-start protocol is
explicit rather than silently selecting a successful checkpoint.

## Reproduction commands

From `F:\uav_capture\three_d_encirclement`:

    conda run --no-capture-output -n uav-encirclement-gpu python -m pytest -q

Train the local-observation expert initialization:

    .\scripts\run_capture_radius_behavior_cloning.ps1 -Config configs/capture_radius_behavior_cloning_dev.yaml -Output results/capture_radius_behavior_cloning_dev_gpu_seed521001_v2 -Seed 521001

Train MAPPO from that checkpoint:

    .\scripts\run_capture_radius_mappo.ps1 -Config configs/capture_radius_mappo_warmstart_dev.yaml -Output results/capture_radius_mappo_warmstart_dev_gpu_seed521001_reproduction -Seed 521001 -InitializeFrom results/capture_radius_behavior_cloning_dev_gpu_seed521001_v2/checkpoint.pt

Run a locked evaluation (the output directory must be new and empty):

    .\scripts\evaluate_capture_radius_mappo.ps1 -Config configs/capture_radius_pursuit_dev.yaml -Checkpoint results/capture_radius_mappo_warmstart_dev_gpu_seed521001_v1/checkpoint.pt -Output results/capture_radius_reproduction_eval_seed610001 -Seed 610001 -Episodes 100 -Device cpu -UseCbf

TensorBoard logs are stored in each training result's `tensorboard` directory.
For example:

    .\scripts\start_tensorboard.ps1 -LogDir results -Port 6006

