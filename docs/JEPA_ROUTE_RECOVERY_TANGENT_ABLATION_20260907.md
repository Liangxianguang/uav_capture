# JEPA Route-Recovery Tangent Ablation (2026-09-07)

## Decision

The runtime nearest-tangent selector is **not accepted** for the current
JEPA + reliability ledger + CBF contract.  G5 remains the best fixed-scene
development configuration.  The tangent mechanism is retained as an offline
route-label and candidate-ranking feature for a future JEPA checkpoint, not as
an unverified execution policy.

This decision is based on the paired four-episode validation manifest, not on
the target capture-time metric alone.  A candidate is rejected if it loses
safe capture, adds a timeout, or changes the safety contract.

## Fixed Contract

- Manifest: `results/jepa_route_recovery_dev_g1b_seed20260911/scene_manifest.jsonl`
- Manifest SHA-256:
  `6ef41f43d4b37e894559a4ee78604d1f3dc0c08fe48a069e7b4140ad646d327b`
- Actor: `models/v5_development_exact_reactive_seed661606.pt`
- JEPA: `results/jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt`
- Ledger: `results/jepa_route_identity_ledger_actor_train40_seed661606/reliability_ledger.json`
- CBF obstacle, boundary, and pairwise margins: `0.35 m`
- Strict buffer, stale/OOD gates, controlled abort, reachable projection, and
  raw-unverified execution prohibition were unchanged.
- All runs were recorded with the project GPU conda environment and wrote a
  TensorBoard event file.

## Results

| Run | Runtime change | Safe capture | Mean capture time (s) | Recovery fallback steps | Route switches | Timeout | Drone safety violations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G4 | Soft barrier, two-step confirmation | 4/4 | 16.725 | 402 | 461 | 0 | 0 |
| G5 | G4 + target escape alignment `0.5` | **4/4** | **13.775** | **336** | 350 | 0 | 0 |
| G6 | G5 + nearest tangent on all barrier risk | 2/4 | 15.750 | 593 | 480 | 2 | 0 |
| G7 | Nearest tangent on obstacle risk only | 0/4 | n/a | 741 | 688 | 4 | 0 |
| G8b | G5 + nearest tangent, target age <= 4 | 2/4 | 19.600 | 618 | 580 | 2 | 0 |
| G9 | G8b + persisted side preference | 2/4 | 19.600 | 618 | 580 | 2 | 0 |
| G10 | G9 + generic recovery side lock | 0/4 | n/a | 731 | 433 | 4 | 0 |
| G11 | G5 + finite two-step tangent hold | 2/4 | 19.350 | 555 | **329** | 2 | 0 |

The target did leave the simulated world in some runs, but defender collision,
defender boundary, pairwise, raw-unverified, and CBF controlled-abort counts
remained zero for all G8b-G11 traces.  The failure mode is therefore progress
loss, not a safety breach.

## Failure Diagnosis

The nearest-tangent rule was applied after a barrier-imminence trigger using
the current public geometry.  With delayed target observations, the two
verified tangent routes were often nearly tied while their route identities
changed with the active obstacle.  The selector consequently alternated
between `left_detour` and `right_detour` (the G8b traces contain hundreds of
switches).  A permanent side lock then kept a stale side across obstacle
changes and degraded capture to `0/4` (G10).  A two-step lock reduced switches
but still produced `2/4` (G11).

This is not evidence that tangential planning is wrong.  It shows that route
side must be a stateful, obstacle-conditioned latent variable and must be
predicted/calibrated by JEPA before it is used for execution.  A geometric
shortest-route choice by itself is insufficient under stale/noisy target
beliefs.

## Implementation Status

The evaluator now exposes the tangent selector and finite hold as explicit
development-only flags.  They default to disabled, so the accepted G5 path is
unchanged.  The route-recovery unit tests cover verified shortest-tangent
selection, side hysteresis, and hold behavior.

The next model-side experiment should train an auxiliary route-identity target
from the independently CBF-verified left/right candidates.  The target should
include active obstacle identity, route feasibility over the probe horizon,
target-observation age, and a `keep/switch/abort` label.  A new checkpoint and
new reliability ledger must be calibrated before any runtime tangent policy is
re-enabled.

## TensorBoard Artifacts

- G8b: `results/jepa_route_recovery_dev_g8b_seed20260911_tensorboard`
- G9: `results/jepa_route_recovery_dev_g9_seed20260911_tensorboard`
- G10: `results/jepa_route_recovery_dev_g10_seed20260911_tensorboard`
- G11: `results/jepa_route_recovery_dev_g11_seed20260911_tensorboard`

These are development runs and must not be reported as locked-test results.
