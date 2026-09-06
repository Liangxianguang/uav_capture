# WP1 Route Sampling and Paired Replay Report

**Date:** 2026-09-06  
**Status:** development-only; `locked_test_opened=false`  
**Scope:** public-observation obstacle routes, corridor sampling, and a three-scene M0/M3 replay  
**Hardware:** NVIDIA GeForce RTX 5050

## Decision

WP1 passes the geometry gate at corridor sampling densities 65, 17, and 9. No
sampling density produced a high-density geometric false accept in the four
fixed public-observation micro-scenes. The 9-point setting is therefore
eligible for the next bounded route replay, but it is not yet a replacement
for the historical 65-point contract.

The three-scene runtime replay shows a positive but small development signal:
M3 (`obstacle_route_v1 + JEPA + Ledger + horizon-5 CBF`) reaches `2/3`, while
the paired M0 (`legacy + CBF + horizon-5`) reaches `1/3`. All safety hard gates
remain zero. This is not a locked benchmark result and is too small for a
formal claim.

The one M3 failure is attributed to Ledger over-abstention under stale/OOD
observations, not to CBF infeasibility or route geometry. A one-scene M1
diagnostic (same JEPA/route/CBF, Ledger disabled) captures successfully. The
next task is a Ledger stale/OOD routing audit; broad retraining and larger
route matrices are paused until that audit is resolved.

## WP1 Fixed Micro-Scene Audit

The audit uses only public obstacle records. `left_blocked` and `right_blocked`
use long public wall segments that reach the world boundary, so the blocked
side cannot be repaired into another in-bounds corridor. The opposite side
remains available.

Each run uses the unchanged Joint CBF probe and an independent 4097-sample
continuous-corridor verification. A `geometry_false_accept` means that the
selected sampling density marked a route geometrically feasible while the
high-density verification rejected it.

| Corridor samples | `central_single` | `left_blocked` | `right_blocked` | `wall_single_gap` | False accepts | CBF verified routes | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 65 | 8/12 | 7/12 | 7/12 | 10/12 | 0 | 8, 7, 7, 10 | PASS |
| 17 | 8/12 | 7/12 | 7/12 | 10/12 | 0 | 8, 7, 7, 10 | PASS |
| 9 | 8/12 | 7/12 | 7/12 | 10/12 | 0 | 8, 7, 7, 10 | PASS |

For all three densities:

- left/right waypoints are distinct in the central scene;
- the public wall rejects the blocked side and preserves the unblocked side;
- the wall gap keeps the nominal route valid;
- every scene has at least one reachable route and one CBF-verified route;
- no selected route is a high-density geometric false accept.

Artifacts:

- [65-point summary](../results/jepa_safe_capture_wp1_routes_samples65_cbf_v3/summary.json)
- [17-point summary](../results/jepa_safe_capture_wp1_routes_samples17_cbf_v3/summary.json)
- [9-point summary](../results/jepa_safe_capture_wp1_routes_samples9_cbf_v3/summary.json)
- TensorBoard event directories are the `tensorboard/` subdirectory of each
  result directory.

The audit implementation is
[`audit_jepa_safe_capture_wp1_obstacle_routes.py`](../scripts/audit_jepa_safe_capture_wp1_obstacle_routes.py).
It now records `--corridor-samples`, independent verification density, route
counts, CBF counts, and acceptance metrics in TensorBoard.

## Three-Scene Paired Replay

The M0 and M3 runs use the same generated validation manifest and therefore
the same scene hashes and episode seeds: `646101`, `646102`, and `646103`.
Both use the same actor checkpoint, `horizon=5`, strict-buffer CBF contract,
first-step execution, reachable projection, and route sampling 9. M3 also
uses the actor-matched route-identity JEPA checkpoint and its hash-bound
Ledger.

| Episode seed | M0 result | M3 result | M3 primary observation |
| ---: | :--- | :--- | :--- |
| 646101 | controlled abort | safe capture | no safety violation |
| 646102 | safe capture | timeout | Ledger safe-hold over-abstention |
| 646103 | controlled abort | safe capture | no safety violation |
| **Aggregate** | **1/3 (33.3%)** | **2/3 (66.7%)** | **+33.3 pp** |

Aggregate hard gates:

| Variant | Collision | Boundary | Pairwise | Raw-unverified | CBF abort | Timeout |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 0 | 0 | 0 | 0 | 2 | 0 |
| M3 | 0 | 0 | 0 | 0 | 0 | 1 |

Artifacts:

- [M0 summary](../results/wp1_route_replay_m0_samples9_seed20260911/summary.json)
- [M3 summary](../results/wp1_route_replay_m3_samples9_seed20260911/summary.json)
- [M0 TensorBoard](../results/wp1_route_replay_m0_samples9_seed20260911_tensorboard)
- [M3 TensorBoard](../results/wp1_route_replay_m3_samples9_seed20260911_tensorboard)

The replay is a development comparison only. It does not revise V4/V5
locked results and does not justify a three-training-seed claim: it is one
actor/model/Ledger contract evaluated on three independent scene seeds.

## Failure Attribution: Episode 646102

M3 timed out after 250 steps without a safety violation. The trace shows:

- `safe_hold_steps=250` and no controlled abort;
- `mean_visible_fraction=0.0`;
- mean message age approximately `53.6` steps and mean observation age
  approximately `125.5` steps;
- the Ledger recorded trusted steps early, then routed the remainder to
  `safe_hold` because stale/OOD evidence remained unresolved;
- CBF candidate probes and independent selected/nominal/safe-hold probes were
  feasible, with no solver timeout or infeasibility.

The M1 diagnostic uses the same actor, route candidates, JEPA checkpoint, CBF
contract, sampling density, and three-scene manifest but no Ledger. It reaches
`3/3 safe_capture`; specifically, episode `646102` reaches safe capture with
all UAV safety hard gates at zero. One separate M1 episode (`646103`) has a
target-boundary diagnostic, which is reported separately from the UAV safety
hard gate and does not affect the `646102` attribution. This isolates the
current capability loss in episode `646102` to Ledger stale/OOD
over-abstention rather than a geometry false accept, JEPA route ranking
failure, or CBF safety failure.

Artifact:

- [M1 no-Ledger three-scene diagnostic](../results/wp1_route_replay_m1_noledger_samples9_seed20260911/summary.json)
- [M1 TensorBoard](../results/wp1_route_replay_m1_noledger_samples9_seed20260911_tensorboard)

This does **not** authorize disabling stale/OOD gates. The correct follow-up
is to distinguish target-prediction staleness from public geometry and
kinematic route validity, then prove a cautious route policy with independent
CBF verification before changing any Ledger state transition.

## Stop / Continue Rule

The current route expansion is stopped here. The route geometry gate passed
and M3 improved on this small paired block, but the M3 timeout and Ledger
safe-hold concentration are unresolved. No L1-L3 route expansion, large
archive generation, or JEPA retraining is started until the following bounded
checks pass:

1. Audit Ledger state transitions and calibration bins for episode 646102,
   including stale, OOD, candidate separation, and queue-age reasons.
2. Run an offline settled counterfactual with selected, nominal, and safe-hold
   actions kept independent; report whether any independently CBF-verified
   route exists while Ledger is in each state.
3. Define a new cautious Ledger state only if it preserves stale/OOD
   detection, requires independent CBF verification, and improves the paired
   failure without raw or unverified execution.
4. Re-run the same three-scene M0/M3/M1 block. If M3 does not improve without
   increasing any safety event, stop and attribute the remaining limitation
   to route/ranker or observation coverage rather than adding more data.

No margin, stale/OOD gate, controlled-abort behavior, or historical locked
contract was changed in this stage.

## Reproducibility

The fixed micro-scene commands use:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_with_tensorboard_compat.py `
  scripts/audit_jepa_safe_capture_wp1_obstacle_routes.py `
  --output-dir results/jepa_safe_capture_wp1_routes_samples9_cbf_v3 `
  --with-cbf --corridor-samples 9 --verification-corridor-samples 4097
```

The runtime commands and complete checkpoint, protocol, Ledger, TensorBoard,
and scene-manifest hashes are stored in each replay's `summary.json` and
`provenance.json`.
