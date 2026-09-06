# WP2 Route Runtime Smoke (2026-09-06)

This is a development-only execution-contract check for the new
`obstacle_route_v1` profile. It is not a locked-test result and must not be
reported as a final V5 improvement.

## Contract

- Environment: `uav-encirclement-gpu`, RTX 5050, CUDA 12.8.
- Variant: `m3` (JEPA + reliability ledger + Joint CBF).
- Profile: `obstacle_route_v1`, 12 geometry-conditioned route proposals.
- Execution: reachable projection, per-route primary CBF prefilter, independent
  `selected`/`nominal`/`safe_hold` probes, then one normal CBF filter and one
  executed first step before replanning.
- Safety gates: CBF margin, stale/OOD gates, non-finite gate, and
  `controlled_abort` were unchanged.
- Split: validation development scenes only; `locked_test_opened=false`.

## Verification

The focused regression suite passed:

```text
93 passed
```

The tests cover route-to-ranker adaptation, action preservation, CBF
eligibility, timeout/infeasible/fallback rejection, independent counterfactual
labels, malformed actions, public-observation belief handling, and legacy CBF
regressions.

## Smoke outcome

Command output is stored in:

- `results/wp2_route_runtime_smoke_v4_m3_seed20260911/summary.json`
- `results/wp2_route_runtime_smoke_v4_m3_seed20260911/episodes.csv`
- `results/wp2_route_runtime_smoke_v4_m3_seed20260911/step_traces/`
- `results/wp2_route_runtime_smoke_tb_v4_m3_seed20260911/`

For two development episodes:

| Metric | Result |
|---|---:|
| `safe_capture` | `0/2` |
| collision | `0/2` |
| defender boundary violation | `0/2` |
| pairwise violation | `0/2` |
| raw-unverified execution | `0` steps |
| route candidates generated | `3180` |
| geometry-valid route candidates | `2774` |
| primary CBF accepted route probes | `2764` |
| independent CBF probes | `795` (`792` accepted) |
| controlled abort | `1` episode |
| transit success | `2/2` |

The earlier smoke before belief/parallel-corridor fixes reached `1/2` safe
capture with all safety gates green. The later smoke is not directly
comparable as a performance claim because route eligibility and the selected
action set changed. The final `0/2` run exposed a legitimate failure mode:
after 15 cycles in one scene, all current first-step requests were infeasible
under the existing acceleration and obstacle barriers. The primary CBF and
safe-hold probe both failed, so `controlled_abort` terminated the episode.
No unverified command was executed.

## Changes validated

1. Public target beliefs marked `never_received` are excluded from the route
   goal consensus; an entirely uninitialized belief falls back to the defender
   centroid rather than the origin.
2. A lateral bypass blocked by a second public obstacle searches bounded
   parallel corridors before being rejected.
3. The route runtime adapter keeps route chunks unchanged and adds explicit
   CBF diagnostics without replacing actions.
4. The paired evaluator records route metadata, CBF prefilter results, and
   independent counterfactuals in traces, episode rows, aggregate summaries,
   and TensorBoard.

## Decision and next step

Do not expand this profile to a multi-seed or 40/60-episode block yet. The next
development step is to replay the controlled-abort trace and classify whether
the cause is route selection, late obstacle approach, or CBF recovery
infeasibility. Only after that trace-level contract is stable should a new
route-identity counterfactual archive and a fresh route-bound JEPA calibration
ledger be generated.
