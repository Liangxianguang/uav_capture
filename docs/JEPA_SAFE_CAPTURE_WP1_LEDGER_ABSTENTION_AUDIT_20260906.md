# WP1 Ledger Abstention Audit

**Date:** 2026-09-06  
**Status:** development-only; `locked_test_opened=false`  
**Input:** M3 route replay, episode seed `646102`, `horizon=5`, strict-buffer CBF, route sampling `9`

## Finding

The M3 timeout is a Ledger/ranker routing failure, not a CBF safety failure.
The frozen trace contains safe, independently verified action opportunities,
but the current contract maps stale or never-received target information and an
invalid nominal route to `safe_hold` for the complete episode.

| Metric | Value |
| --- | ---: |
| Trace steps | 250 |
| Safe-hold steps | 250 |
| Actionable safe-hold steps | 208 |
| Mean valid candidates | 3.13 |
| Mean eligible candidates | 0.34 |
| Mean CBF-prefilter verified candidates | 3.13 |
| Independent selected/nominal/safe-hold CBF probes all accepted | 250/250 |
| Collision / boundary / pairwise / raw-unverified | 0 / 0 / 0 / 0 |

An actionable safe-hold is defined here as a step with zero eligible ranker
candidates while at least one route candidate passed the independent CBF
prefilter. It is an offline diagnostic label, not permission to execute that
candidate.

## Fallback Attribution

| Fallback reason | Steps |
| --- | ---: |
| `observation_never_received` | 4 |
| `nominal_infeasible` | 246 |

The first four steps are blocked by the explicit never-received gate. For the
remaining trace, the route generator usually exposes only braking,
visibility-hold, and verified-hold as geometrically valid candidates; the
nominal route is invalid. The ranker currently requires a trusted nominal
anchor before it can select an alternative, so it reports
`nominal_infeasible` even when a CBF-verified candidate exists.

The trace also shows a feedback loop: safe-hold prevents the defenders from
moving toward a reacquisition route, so visibility remains zero and message
age remains saturated. The M1 no-Ledger paired run moves, reacquires the
target, and captures the same episode. This explains the M3/M1 gap without
indicating that the stale/OOD gate itself is incorrect.

## Safety and Provenance Gates

All audit gates pass:

- every trace row is structurally valid;
- all 250 selected/nominal/safe-hold counterfactual sets are present;
- all three CBF probes are accepted on all 250 steps;
- raw-unverified execution is zero;
- collision, boundary, and pairwise violations are zero.

The complete machine-readable output is
[ledger_abstention.json](../results/wp1_ledger_abstention_episode646102/ledger_abstention.json).
TensorBoard is recorded in
`results/wp1_ledger_abstention_episode646102_tensorboard`.

The audit implementation is
[`audit_jepa_safe_capture_wp1_ledger_abstention.py`](../scripts/audit_jepa_safe_capture_wp1_ledger_abstention.py).

## Required Next Contract

No current gate is disabled. Before another runtime matrix or JEPA retraining,
define and test a new development-only `cautious_reacquisition` route state:

1. stale/never-received/OOD remains visible in the Ledger decision and is
   never relabeled as `trusted`;
2. the state can select only a named reacquisition/visibility route, never an
   arbitrary JEPA-ranked route;
3. the candidate must pass reachable projection, primary CBF prefilter,
   selected/nominal/safe-hold independent CBF probes, and residual safety
   verification;
4. the route has bounded duration and a deterministic return to `safe_hold`;
5. the contract records target reacquisition progress, message age, visibility,
   and every fallback reason in TensorBoard;
6. if the cautious state does not improve the paired `646102` failure without
   any safety event, stop and attribute the remaining limitation to route
   coverage or observation design instead of adding more data.

The next experiment is therefore a one-episode deterministic counterfactual
contract test, followed by the same three-scene M0/M3/M1 replay only if the
contract test passes. No L1-L3 expansion or large-scale JEPA retraining is
authorized by this audit.
