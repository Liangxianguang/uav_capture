# DN-MPC + Strict CBF Current Closed-Loop Result

**Date:** 2026-09-08
**Phase:** development-only frozen G5 replay
**Controller:** retained V5 actor -> reachable route candidates -> candidate-level CBF probe -> distributed minimax DN-MPC -> final strict Joint CBF -> execute first step and replan

## Result

| Metric | Current H=5 result |
|---|---:|
| Safe capture | **4/4 = 100.0%** |
| Collision | 0/4 |
| Defender boundary violation | 0/4 |
| Pairwise violation | 0/4 |
| Raw-unverified execution | 0 steps |
| Controlled abort | 0 steps |
| CBF fallback | 0 steps |
| Mean capture time | 19.1 s |
| Route switches | 96 total |
| Worst physical minimum clearance | 0.35015 m |
| Target-boundary diagnostic | 1/4 |

The target-boundary diagnostic is reported separately by the evaluator and does
not invalidate the UAV safe-capture contract. The minimum clearance remains
above the physical-plus-operational `0.35 m` CBF buffer in all four episodes.

## Comparison with the earlier S1 run

The earlier H=3 development replay was `3/4 = 75.0%` with one controlled
abort. With the same frozen G5 manifest, H=5 reduces the controlled-abort count
from one to zero and reaches `4/4`; this is a `+25` percentage-point
development improvement. It is not yet a formal benchmark or a multi-seed
claim.

The route planner still switched 96 times over the four episodes, so route
selection persistence and computational cost remain open engineering issues.
The route probe evaluated 5,569 candidate checks and accepted 5,065. The
average cycle latency is dominated by route generation and CBF probing, not by
the actor or final CBF solve.

## Provenance

- Output: `results/dn_mpc_cbf_g5_dev_current_h5_20260908/`
- TensorBoard: `results/dn_mpc_cbf_g5_tensorboard_current_h5_20260908/`
- Actor: `models/v5_development_exact_reactive_seed661606.pt`
- CBF horizon: 5; route-probe horizon: 5; route chunk: 5
- GPU: NVIDIA GeForce RTX 5050; PyTorch `2.7.1+cu128`
- `development_only=true`; `locked_test_opened=false`

## What this proves

This run proves that the executable DN-MPC + strict CBF chain can complete the
four frozen G5 scenarios without a safety event or controlled abort under the
current H=5 contract. It does **not** yet prove that JEPA improves capture:
JEPA was disabled in this run. The next direct experiment is a paired replay
on the same manifest with JEPA ranking enabled, followed by L0-L3 and multiple
seeds only if the paired run is at least as safe as this baseline.
