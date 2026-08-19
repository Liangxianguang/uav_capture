# External Source Record

## gym-pybullet-drones

- Canonical repository: https://github.com/utiasDSL/gym-pybullet-drones
- Retrieved release: `v1.0.0`, 2026-08-18
- Immutable source commit: `7688e7208a1572b1680736a3c0c9b93c379db3fe`
- Local source directory:
  `third_party/gym-pybullet-drones-7688e7208a1572b1680736a3c0c9b93c379db3fe`
- Archived release file: retained outside Git in the local experiment archive
  (`F:\\uav_capture_archive_20260819\\three_d_encirclement`).
- Release archive SHA-256:
  `d6bc0954061ed8b36cd4be8f8103864f7b18118969bbb4a790a8bedb815cb948`
- Licence: MIT, copyright (c) 2020 Jacopo Panerati
- Runtime: Conda-forge `pybullet=3.25`, pip `gym==0.26.2`
- Interfaces used: `VelocityAviary` for the direct velocity-interface
  diagnostic, and `CtrlAviary` plus `DSLPIDControl` for the current default
  hierarchical position-PID backend, all with `DroneModel.CF2X`. The physics
  rate is 240 Hz and each high-level action spans 24 physics steps per 0.1 s.
  The wrapper recomputes the low-level PID every 1/240 s and holds the
  high-level command over those 24 substeps.

The pinned release predates Python 3.10 and refers to `collections.Mapping`.
`src/encirclement3d/pybullet_env.py` provides an import-time compatibility
alias without modifying the vendored source. Its setup file declares optional
RL dependencies that are not installed: only the files required for the
VelocityAviary simulation are imported directly from the fixed source tree.

Reference: J. Panerati et al., "Learning to Fly---A Gym Environment with
PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter
Control," IEEE/RSJ IROS 2021, doi:10.1109/IROS51168.2021.9635857.

## Reproduction Boundary

The project previously used only a custom kinematic environment and a separate
inertial diagnostic. The PyBullet wrapper now makes defender motion a
Crazyflie-rigid-body simulation driven by the simulator's PID velocity
interface; the target remains the benchmark's kinematic non-cooperative actor.
Results from this backend are simulator-transfer evidence only. They are not
real-flight validation and must never be merged with the kinematic success rate
in a single headline number.
