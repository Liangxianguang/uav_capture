# Distinct-Seed Route Ranking Audit

**Date:** 2026-09-07
**Scope:** read-only development audit of the negative independent model-seed gate
**Locked test:** not opened

## Findings

The audit did not re-simulate candidates and therefore makes no settled-best
route claim. It compares the recorded M3 score order and route execution with
the paired M0/M3 outcomes.

| Episode | M0 | M3 | M3 termination | Score-argmin match | Route switches | Target clearance |
|---:|:---:|:---:|---|---:|---:|---:|
| 650101 | success | success | safe_capture | 1.000 | 11 | 5.87 -> 7.13 m |
| 650102 | success | failure | timeout | 1.000 | 130 | 6.40 -> 10.77 m |
| 650103 | failure | failure | timeout | 1.000 | 73 | 4.68 -> 9.79 m |

For the two M3 failures, the correlation between predicted selected-route
progress and the next-step target-clearance improvement was `0.010` and
`-0.157`. The M3 score consistently selected its own argmin, but the selected
score was not a reliable proxy for closing the target distance. This is a
ranking-objective / target-motion generalization failure, not evidence that the
CBF filter rejected a safe route.

## Safety and Ledger separation

- M3 candidate CBF probes: `5939/5939` accepted.
- M3 independent selected/nominal/safe-hold probes: `1755/1755` accepted.
- M3 CBF infeasible, timeout, controlled-abort, and raw-unverified steps: all `0`.
- M3 Ledger was trusted on `246/250` and `248/250` steps in the two failed
  episodes; the remaining steps were the explicit `observation_never_received`
  safe-hold state.
- Cautious active-search attempts: `0`.

The failure is therefore not a permission to disable stale/OOD gates or to
execute unverified actions. It is a signal that the JEPA score needs a better
task-linked target-escape and route-progress objective, with calibration that
penalizes repeatedly switching between safe but ineffective routes.

## Artifact

The audit is reproducible with
`scripts/audit_jepa_safe_capture_distinct_seed_route_ranking.py`. It writes:

- `results/wp1_active_search_v4_distinct_seed_route_ranking_audit/ranking_audit.json`;
- `results/wp1_active_search_v4_distinct_seed_route_ranking_audit/report.md`;
- `results/wp1_active_search_v4_distinct_seed_route_ranking_audit_tensorboard/`.

The TensorBoard event contains the per-episode score-order, eligibility,
route-switch, and clearance-correlation scalars. `settled_best_claim_available`
is explicitly logged as `0`.

## Next gate

Keep the expansion gate closed. The next bounded change may add target-boundary
and escape-direction labels, route-progress calibration, and a switch penalty,
then rerun the same independent L0 paired manifest. CBF margins, horizon,
stale/OOD policy, controlled-abort semantics, and raw-action guards stay frozen.
