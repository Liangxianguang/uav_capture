# WP1 Active-Search Reacquisition Development Report

**Date:** 2026-09-06  
**Status:** development-only; `locked_test_opened=false`  
**Scope:** the frozen WP1 three-scene validation manifest only  
**Hardware:** NVIDIA GeForce RTX 5050  
**Primary metric:** episode-level `safe_capture`

## Decision

The bounded public-observation active-search route passes the WP1 micro-scene
gate and repairs the previously diagnosed reacquisition failure. On the same
actor, route JEPA checkpoint, scene manifest, `horizon=5`, strict-buffer CBF,
and three episode seeds, M3 improves from `2/3` to `3/3 safe_capture`.

This is a development result from one training/model seed, not a locked
benchmark or a three-training-seed claim. L1-L3 expansion and JEPA retraining
remain paused until a paired regression and multi-seed development block is
approved.

## What Changed

The prior `visibility_hold` route used the defender centroid when no target
belief had ever been initialized. Its action was therefore zero, so the
Ledger preserved safety but could not restore visibility.

The new protocol keeps the existing 12-way route-identity index for checkpoint
compatibility, but explicitly changes the route behavior only when the public
observation reports no received target belief:

- mode: `lateral_interior_scan_v1`;
- bounded search offset: `1.5 m`;
- maximum cautious reacquisition duration: `12` control steps;
- direction selection uses finite public world bounds, lateral alternatives,
  and public obstacle clearance only;
- every first action is reachable-dynamics projected and independently checked
  by selected, nominal, and safe-hold CBF probes;
- stale/OOD/non-finite Ledger gates remain unchanged and the route is never
  relabeled `trusted`.

No CBF margin, solver tolerance, controlled-abort rule, or raw-action path was
changed. No simulator target truth is read online.

## Paired Replay

| Contract | Safe capture | Timeout | Collision | Boundary | Pairwise | Raw-unverified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous M3 + stationary hold | `2/3` | `1` | `0` | `0` | `0` | `0` |
| M3 + active-search v3 | `3/3` | `0` | `0` | `0` | `0` | `0` |

The active-search run also recorded `0` CBF infeasible steps, `0` CBF solver
timeouts, `0` controlled abort steps, `12/12` accepted cautious-reacquisition
steps, and `906/906` accepted independent selected/nominal/safe-hold probes.
The three-scene block therefore preserves all safety hard gates while removing
the prior timeout.

A paired M0 (`legacy actor + horizon-5 CBF`, no JEPA and no Ledger) on the same
manifest reached `1/3 safe_capture`, with two controlled aborts and no
collision, boundary, pairwise, or raw-unverified event. The M3 minus M0 paired
delta is therefore `+66.7 percentage points` on this small development block.
M0's two CBF-infeasible/controlled-abort episodes are retained as the baseline
failure evidence; they are not hidden by the active-search result.

## Episode 646102 Observation Audit

The audit reads only the evaluator's causal pre-action public snapshots. Target
truth is used only as an offline label.

| Metric | Stationary v2 | Active-search v3 |
| --- | ---: | ---: |
| Trace steps | `250` | `132` |
| Never-received zero-belief rate | `1.0` | `1.0` |
| Cautious steps | `3` | `12` |
| Non-zero search-action steps | `0` | `12` |
| Mean follow-up motion | `0.0 m` | `0.18 m` |
| First visible step | `null` | `17` |
| Termination | timeout | safe capture |

The result supports the causal attribution: the previous failure was an
observation/reacquisition capability gap, not an unsafe CBF rejection. The
active search is still only a bounded route proposal; the CBF remains the
execution boundary.

## Calibration and Provenance

The route JEPA Ledger was recalibrated against the independent calibration
archive and bound to the v3 protocol. Calibration route-head metrics were:

- route identity accuracy: `1.0`;
- route side accuracy: `1.0`;
- route geometry accuracy: `0.9331`;
- route termination accuracy: `0.8038`;
- global credit by horizon: `0.8135, 0.8107, 0.8110, 0.7797, 0.7796`.

Inputs and hashes:

- protocol: `configs/central_random_mixed_obstacle_s3_route_v1_reacquisition_v3_active_search_development_protocol.yaml`  
  SHA-256 `61dc1cca5e1d42447a50e77b0f34cb9aaaf827572ed069e756561620c300a943`;
- Ledger: `results/jepa_route_identity_ledger_reacquisition_v3_active_search_seed661606/reliability_ledger.json`  
  SHA-256 `de4349c172e71ce7239a408a8b44a06766e680cf45cf46bcbf43ca0e35fe54b2`;
- replay summary: `results/wp1_active_search_v3_m3_seed20260911/summary.json`  
  SHA-256 `2bae4f054b269a930be14373f971adb3c8292ac60ead496160e40f6a1add5db0`;
- observation audit: `results/wp1_active_search_v3_observation_audit_episode646102/observation_contract.json`  
  SHA-256 `3de253aa3a813a66c34650dd6c2a049a20f18984238297cbc56d1e086c026288`.
- paired M0 summary: `results/wp1_active_search_v3_m0_seed20260911/summary.json`  
  SHA-256 `ddbadd683f92e895ad1b7db1fa5a9d0fcd577cf09336ba906c522fdc1c1bf99`;

TensorBoard event files:

- Ledger calibration: `results/jepa_route_identity_ledger_reacquisition_v3_active_search_seed661606_tb`  
  event SHA-256 `709bfb7890af8b088df6ee1556ce983d33f9224758e3e79d089587716835254b`;
- replay: `results/wp1_active_search_v3_m3_seed20260911_tensorboard`  
  event SHA-256 `139800283f1344836fbb995537dcc69d8fced0840b8bf99b2f9a82b79b4feab`;
- observation audit: `results/wp1_active_search_v3_observation_audit_episode646102_tensorboard`  
  event SHA-256 `c83e1721d11429b8a2e8d8cc6cd1f4f960db72f48d08735e9ce841058f34a8bc`.
- paired M0: `results/wp1_active_search_v3_m0_seed20260911_tensorboard`  
  event SHA-256 `c0e57dc98d8b99ef1af3db1adff5d15b88505c20217e4cfe0d0c41b3a1dbde13`.

## Continue / Stop Rule

Continue only with a paired M0/M3 replay on the same three-scene manifest and a
second independent scene block if the active-search behavior remains causal,
bounded, and safe. Do not increase the search horizon or relax Ledger gates
without a separate protocol.

If the next paired block loses the gain while all safety gates remain zero,
stop and attribute the limitation to search-direction coverage or actor/route
interaction. Do not respond by adding model capacity or random data alone.
