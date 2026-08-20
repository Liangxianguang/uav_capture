# Central V4 Recurrent MAPPO Pilot Report

Date: 2026-08-20

## Protocol

- Main task: partially observable 3D cooperative capture in a central mixed-obstacle scene.
- Capture: any defender enters the target capture radius (`0.80 m`); the rollout terminates immediately.
- Target central-zone entry is **not required**.
- Cooperative capture additionally requires at least two defenders to have entered the central zone before capture.
- Transit is a separate feasibility test. It does not continue after capture and is not part of the main capture success condition.
- World and obstacle contract: `configs/central_bidirectional_v4.yaml`.

## Training

- Algorithm: recurrent MAPPO with centralized training and decentralized execution.
- Initialization: `results/central_v4/bc_shapeaware_seed660701/checkpoint.pt`.
- Seed: `660701`.
- Environment steps: `65,536` (64 updates).
- Device: NVIDIA RTX 4060 Laptop GPU, PyTorch `2.7.1+cu126`.
- Observation interface: shape-aware 63-dimensional policy input.
- Action scale: full velocity range (`5.0 m/s`).
- Both training runs completed without NaN, OOM, or runtime error.

## Fixed evaluation

Twenty episodes per condition were run with the same seed blocks used for the BC baseline.

| Scene | Execution | Cooperative safe capture | Collision | Boundary violation | Transit |
|---|---:|---:|---:|---:|---:|
| S1 cylinder | raw | 100% | 0% | 0% | 100% |
| S1 cylinder | CBF | 100% | 0% | 0% | 100% |
| S1 box | raw | 45% | 55% | 0% | 100% |
| S1 box | CBF | 100% | 0% | 0% | 100% |
| S1 wall | raw | 100% | 0% | 0% | 100% |
| S1 wall | CBF | 100% | 0% | 0% | 100% |
| S2 mixed | raw | 15% | 85% | 20% | 100% |
| S2 mixed | CBF | 40% | 0% | 0% | 100% |

The first unregularized MAPPO pilot regressed materially on fixed S2. The shape-aware BC baseline reached 100% cooperative safe capture on the same S2 block in both raw and CBF evaluation.

## BC-retention retry

The retry kept the same V4 task, seed, 65,536 environment steps, recurrent architecture, and curriculum. It changed only the learning rate (`5e-5`) and added an auxiliary MSE term on the audited 650-sequence V4 expert archive (`coefficient=1.0`). The archive was shape-checked and recorded with SHA-256:

`7666a03c446ca04ecb802664bd92817547aac1736ed041ebd8a76d0050085f48`

| Scene | Execution | Cooperative safe capture | Collision | Boundary violation | Transit |
|---|---:|---:|---:|---:|---:|
| S1 cylinder | raw | 100% | 0% | 0% | 100% |
| S1 cylinder | CBF | 100% | 0% | 0% | 100% |
| S1 box | raw | 100% | 0% | 0% | 100% |
| S1 box | CBF | 100% | 0% | 0% | 100% |
| S1 wall | raw | 100% | 0% | 0% | 100% |
| S1 wall | CBF | 100% | 0% | 0% | 100% |
| S2 mixed | raw | 100% | 0% | 0% | 100% |
| S2 mixed | CBF | 100% | 0% | 0% | 100% |

The retry therefore passes the fixed V4 C4 gate. On S2, raw mean time-to-capture was `3.36 s` with worst observed clearance `0.104 m`; CBF mean time-to-capture was `3.60 s` with worst observed clearance `0.448 m`. These are simulation metrics, not physical-flight guarantees.

## Decision

1. The original unregularized pilot is rejected because it failed fixed S2.
2. The BC-retention retry passes fixed S1/S2 and is the current MAPPO candidate.
3. Do not claim final multi-seed generalization yet; proceed to S3 validation and retain S1/S2 as regression tests.
4. Keep the shape-aware BC checkpoint as the fallback formal model and comparison baseline.
5. The next learning task is randomized central-obstacle validation, followed by three-seed locked testing only if S3 does not materially regress.

This result is a 3D kinematic simulation result only. It does not claim physical contact capture, real-flight validity, or hardware deployment readiness.
