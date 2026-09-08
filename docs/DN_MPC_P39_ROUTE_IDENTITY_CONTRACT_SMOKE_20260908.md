# DN-MPC P39 Route-Identity Contract Smoke

**Date:** 2026-09-08
**Phase:** development-only contract smoke
**Decision:** field contract passes; full seed-disjoint archive rebuild remains pending

## Change

P39 separates the analytic planner's state from the frozen actor's executed
route. The collector now records both identities:

| Decision chain | Previous field | Current field | Switch field |
|---|---|---|---|
| Frozen actor after final CBF | `previous_executed_route_index` | `executed_route_index` | `route_switch_outcome` |
| Analytic DN-MPC selection | `previous_selected_candidate_index` | `planner_selected_candidate_index` | `planner_route_switch_outcome` |

The first planner decision and a no-feasible-candidate abstention use `-1` for
the planner switch outcome. They are not silently converted to a route hold.
The planner selected candidate is computed from public route geometry and
first-step CBF eligibility; this archive collector never executes that request.

The utility audit now uses the planner identity when all three P39 fields are
present across train/calibration/validation. Legacy P36 archives are explicitly
reported as `legacy_fallback` and continue using the executed-route field.

## Smoke result

Command used one development train episode with the frozen actor, strict
five-step CBF horizon, independent selected/nominal/safe-hold probes and
`sample_stride=16`. The output is retained at:

- Archive: `results/dn_mpc_jepa_safe_capture_dev/p39_route_identity_contract_smoke_20260908b/`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p39_route_identity_contract_smoke_20260908b/`

| Quantity | Result |
|---|---:|
| Total rows | 156 |
| Runtime rows | 144 |
| Offline boundary-shadow rows | 12 |
| Geometry-valid runtime fraction | 83.33% |
| First-step CBF-feasible fraction | 100.00% |
| Executed-route match fraction | 100.00% |
| Branch failures within horizon | 0 |
| Planner current-known rows | 144/144 |
| Planner previous-known rows | 96/144 |
| Planner switch rows | 48 |
| Planner unknown switch rows | 48 |

The 48 unknown planner-switch rows are the first sampled decision for the
episode, not hidden planner switches. Runtime execution and CBF safety metrics
remain separate from planner identity metrics.

## Verification

- `26 passed` for the route archive, DN-MPC and utility regression subset.
- Python compilation passed for the collector, trainer, utility audit and tests.
- No CBF margin, stale/OOD/non-finite gate, controlled-abort behavior or raw
  action execution rule was changed.
- No JEPA checkpoint was retrained, no Ledger-Lite was created and no locked
  test was opened.

## Remaining P39 work

1. Regenerate seed-disjoint train/validation/calibration archives with the new
   fields and independent provenance hashes.
2. Audit `lower_detour` eligibility and reachable-dynamics projection before
   treating it as a training negative.
3. Re-run utility calibration using planner selected identity, then compare
   corrected model-vs-planner agreement against the promotion gate.

