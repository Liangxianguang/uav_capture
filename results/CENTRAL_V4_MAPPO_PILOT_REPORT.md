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
- Training completed without NaN, OOM, or runtime error.

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

The shape-aware BC baseline reached 100% cooperative safe capture on fixed S2 in both raw and CBF evaluation. Therefore the MAPPO pilot regresses materially on the main mixed-obstacle task, even though its independent Transit test remains 100%.

## Decision

1. The MAPPO pilot training pipeline and checkpoint format are valid.
2. The pilot does **not** pass the fixed S2 validation gate.
3. Do not launch three independent MAPPO seeds yet.
4. Keep the shape-aware BC checkpoint as the current formal model for capture demonstrations and regression tests.
5. The next learning task is diagnosis: compare MAPPO rollout/action statistics against the BC initialization, then correct curriculum, rollout horizon, reward balance, or recurrent-state handling before another pilot.

This result is a 3D kinematic simulation result only. It does not claim physical contact capture, real-flight validity, or hardware deployment readiness.
