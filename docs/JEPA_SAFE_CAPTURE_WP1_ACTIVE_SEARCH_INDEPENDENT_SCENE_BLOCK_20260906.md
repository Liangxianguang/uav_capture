# WP1 Independent Active-Search Scene Block

**Date:** 2026-09-06  
**Status:** development-only; `locked_test_opened=false`  
**Scope:** one independent three-episode validation scene block  
**Hardware:** NVIDIA GeForce RTX 5050

## Decision

The bounded active-search route reproduces its gain on an independent scene
seed block. The same actor, route JEPA checkpoint, active-search contract,
horizon-5 strict-buffer CBF, and hash-bound Ledger were evaluated first with
M0 to generate a manifest, then with M3 on that exact manifest.

| Variant | Safe capture | Timeout | CBF abort | Collision | Boundary | Pairwise | Raw-unverified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0: actor + CBF | `2/3` | `1` | `0` | `0` | `0` | `0` | `0` |
| M3: JEPA + Ledger + active search + CBF | `3/3` | `0` | `0` | `0` | `0` | `0` | `0` |

The paired M3 minus M0 delta is `+33.3 percentage points`. M3 had `24/24`
accepted cautious-reacquisition steps, `690/690` accepted independent CBF
probes, and zero CBF infeasibility or solver timeout. The M0 manifest was
generated before M3 and was reused without modification.

## Causal Observation Audit

Episode `650101` was audited from evaluator public pre-action snapshots. The
simulator target truth was used only as an offline label.

| Metric | Value |
| --- | ---: |
| Never-received zero-belief rate | `1.0` |
| Cautious reacquisition steps | `12` |
| Non-zero search-action steps | `12` |
| Mean follow-up motion | `0.18 m` |
| First visible step | `15` |
| Public snapshot completeness | `true` |
| Safety hard gates | all zero |

This independently reproduces the mechanism observed in episode `646102`:
stationary hold did not move the defenders, while the bounded public-geometry
search restored visibility and allowed the normal closed loop to continue.

## Additional Diagnostic

The M3 block contains one `target_boundary_violation` in the task settlement
diagnostic (`1/3`). This is not a UAV collision/boundary/pairwise/raw-unverified
event and does not alter the UAV safety hard-gate result, but it must remain
reported. The M0 block has zero target-boundary events. The next task should
therefore inspect target-escape/task-settlement sensitivity before claiming a
general improvement.

## Provenance

- protocol: `configs/central_random_mixed_obstacle_s3_route_v1_reacquisition_v4_independent_scene_block_development_protocol.yaml`, SHA-256 `5fc57476361b016f474600900de1887849a853b30ae776f3b51476be7041e119`;
- Ledger: `results/jepa_route_identity_ledger_reacquisition_v4_independent_scene_seed661606/reliability_ledger.json`, SHA-256 `aadc901ef67a214503e614e91a1f58d05a22b6ca84b89b7053d5c5a086ed3b20`;
- M0 summary: `results/wp1_active_search_v4_independent_m0_seed20260911/summary.json`, SHA-256 `e16586aa3ea76af2b97bc9fcff403f7e8d2d5e6030a4b3f15fc1682eadd91ec7`;
- M3 summary: `results/wp1_active_search_v4_independent_m3_seed20260911/summary.json`, SHA-256 `05cd4b9436ff3634cea2200685b68a89f6b36ed233703e3fa58b76a9b904fede`;
- observation audit: `results/wp1_active_search_v4_observation_audit_episode650101/observation_contract.json`, SHA-256 `28c566f77c166669c5bc1cfe0fcfb34f78ac4e295ae05af0c5a611f958d55b73`.

TensorBoard event hashes:

- Ledger calibration: `b4c1aa85818242cc08ff2313191093dd36a2032dd68c76c505f7065a250b13eb`;
- M0: `71796d3d38a70baadd3cf5e301fed3f148cdcad067355ebbf5d783b2462f5f02`;
- M3: `188e3d68dced577023d154a8a7438426f5a109567313fffdc9f5f7ec85785e42`;
- observation audit: `be44aef60708e7596c569c9ba8d38a5e4f4653c861fb0725fa1fdf9e8b4e5c20`.

## Continue / Stop Rule

The active-search route now has two small scene blocks with positive paired
M3 deltas and no UAV safety hard-gate regressions. This authorizes a bounded
three-training-seed development smoke on the same contract, not a locked test
or L1-L3 expansion. The smoke must keep target-boundary diagnostics separate
and must stop if the paired delta becomes non-positive or any UAV safety gate
is violated.
