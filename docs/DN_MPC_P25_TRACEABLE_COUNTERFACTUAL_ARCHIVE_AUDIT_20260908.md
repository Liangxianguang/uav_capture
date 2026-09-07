# DN-MPC P25 Traceable Counterfactual Archive Audit

**Status:** development-only; no online route action, Ledger, or locked test
was opened.

P25 added an explicit archive contract for independent CBF probes. For every
runtime route state, the collector now records the selected analytic DN-MPC
candidate (when one exists) and independently probes its requested first
action, the nominal anchor, and the verified safe-hold request. These probes
are read-only and do not replace the normal CBF filter or execute unverified
actions.

## Artifacts

| Split | Archive | Samples | Runtime rows | Trace groups | Missing trace groups |
|---|---|---:|---:|---:|---:|
| train | `results/dn_mpc_jepa_safe_capture_dev/p25_trace_train8` | 8,768 | 6,576 | 136 | 1 |
| validation | `results/dn_mpc_jepa_safe_capture_dev/p25_trace_validation8` | 11,008 | 8,256 | 172 | 0 |
| calibration | `results/dn_mpc_jepa_safe_capture_dev/p25_trace_calibration8` | 13,696 | 10,272 | 213 | 1 |

Each archive has a matching TensorBoard run under
`results/dn_mpc_jepa_safe_capture_tensorboard/p25_trace_*`. The P25 ranking
audit and route-progress audit are under
`results/dn_mpc_jepa_safe_capture_dev/p25_trace_candidate_ranking_audit_v3`
and `p25_trace_route_progress_audit_v2`, with corresponding TensorBoard logs.

## Counterfactual Contract

- Validation has independent trace coverage for all `172/172` runtime groups.
- Train has `136/137` groups with a selected route; one group has no candidate
  that survives the first-step CBF eligibility mask.
- Calibration has `213/214` groups with a selected route; one group has no
  eligible candidate.
- The arrays include selected candidate index and independent feasible,
  verified-feasible, infeasible, timeout, correction, and minimum-slack fields
  for `selected`, `nominal`, and `safe_hold`.
- The trace is selected by analytic DN-MPC over the first-step CBF-eligible
  route set. It is not a JEPA-selected action and is not online deployment
  evidence.

## P24 Checkpoint Re-audit on P25 Archives

The P24 checkpoint remains unchanged:

`results/dn_mpc_jepa_safe_capture_checkpoints/p24_route_jepa_listwise_temp005_seed232302/checkpoint.pt`

| Split | exact top-1 | pairwise agreement | geometry accuracy | zero-eligible groups |
|---|---:|---:|---:|---:|
| train | 59.56% | 94.88% | 89.54% | 1 |
| validation | 36.63% | 88.73% | 83.37% | 0 |
| calibration | 39.44% | 88.01% | 83.03% | 1 |

The validation informative-group top-1 remains `80.00%`, tie-aware top-1 is
`75.00%`, and the exact top-1 promotion threshold remains unmet at `36.63%`.
The ranking result is therefore unchanged by adding the trace fields, as
expected; P25 closes an auditability gap rather than changing the model.

## Promotion Decision

| Gate | Result |
|---|---|
| route/action/archive contract | pass |
| finite outputs | pass |
| score direction | pass (`88.73%` validation pairwise) |
| candidate eligibility on all splits | fail (one train and one calibration zero-eligible group) |
| exact top-1 | fail (`36.63% < 50%`) |
| selected/nominal/safe-hold traceability | partial: validation complete, train/calibration have one no-route group |
| OOD/disagreement binding | fail; fresh calibration ledger is not present |

P25 is a useful infrastructure milestone, not an online promotion. Keep the
analytic DN-MPC + strict joint CBF path as the executable authority. Do not
start a three-seed paired replay, create Ledger-Lite, alter CBF margins, or
disable stale/OOD/non-finite gates.

## Next Action

P26 should explain and handle the two no-eligible groups as an explicit
planner-abstention outcome, without fabricating a selected route. It should
also construct a fresh OOD/rollout-disagreement calibration archive bound to
these exact P25 provenance hashes. Only after those gates and the ranking gate
are independently satisfied should a JEPA-assisted closed-loop comparison be
considered.
