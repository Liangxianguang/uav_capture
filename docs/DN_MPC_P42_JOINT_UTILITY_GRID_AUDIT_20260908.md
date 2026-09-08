# DN-MPC P42 Joint Utility-Weight Grid Audit

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** utility-weight boundary audit passed; independent-seed gate remains

## Scope

P42 closes the bounded scale check started in P40. It evaluates the unchanged
P37 pairwise JEPA checkpoint on the unchanged P39 train, calibration and
validation archives. The active finite grid is:

```text
length = {0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0}
switch = {0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0}
cbf    = {0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0}
escape = {0, 0.1, 0.2, 0.5}
```

No action was executed. CBF margins, stale/OOD/non-finite gates,
controlled-abort behavior and route eligibility were unchanged. No Ledger or
locked benchmark was opened.

## Provenance

- Checkpoint: P37 pairwise JEPA, seed `373701`
- Checkpoint SHA-256: `51a8d1f86e7182fe3fc2f42a6fa92adba0baa7a0378bf5135f5499e46496ff72`
- P39 train archive SHA-256: `a90090332b1a4fa7eb97771c96b10fc72a45677a87972503d73accbfc6bb2a4b`
- P39 calibration archive SHA-256: `de37a54d1800b4bc8352cc20712e9dcf0676dbf60a4d44300f96441326a64ceb`
- P39 validation archive SHA-256: `72898cccc407a107820201b54cdb0152a62220df16d13d7fd89c0737e5cadcd4`
- Planner identity: `previous_selected_candidate_index`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p39_route_utility_audit_v5_cbfgrid/`

## Result

Calibration selected:

```text
length=1.5, escape=0.0, cbf=1.5, switch=1.5
```

All four selected coordinates are strictly inside their active finite grids;
the earlier P40/P41 boundary artifacts are therefore cleared for this bounded
audit.

| Split | Exact top-1 | Informative top-1 | Pairwise | Model vs planner-selected |
|---|---:|---:|---:|---:|
| Train | 98.13% | 100.00% | 98.92% | 76.78% |
| Calibration | 97.14% | 100.00% | 98.34% | 76.85% |
| Validation | 99.41% | 100.00% | 97.87% | 80.42% |

Relative to P39's original selected weights, validation agreement changes from
`80.71%` to `80.42%` (`-0.29 pp`) while pairwise agreement changes from `97.42%`
to `97.87%` (`+0.45 pp`). The corrected planner/route identity contract
therefore remains the dominant explanation for the P38-to-P39 improvement;
weight expansion does not create a new safety or capture claim.

## Gate decision

- [x] Switch, route-length and CBF utility scales audited on a bounded grid.
- [x] Selected utility coordinates are interior to all active grids.
- [x] Validation selected-route agreement remains above the development gate.
- [ ] Independent evaluator seed reproduces the agreement using calibration-only weight selection.
- [ ] Selected/nominal/safe-hold trace, OOD and disagreement gates are promoted.
- [ ] Any online JEPA override, Ledger-Lite or locked benchmark is authorized.

The next task is an independent JEPA evaluator seed trained from the same P39
train archive, with weights selected only on P39 calibration and evaluated once
on P39 validation. Until that result passes the stability gate, keep the
analytic DN-MPC + CBF planner authoritative and keep JEPA offline-only.

## Artifacts

- Script: `scripts/audit_dn_mpc_p36_route_utility.py`
- JSON: `results/dn_mpc_jepa_safe_capture_dev/p39_route_utility_audit_v5_cbfgrid/audit.json`
- Markdown: `results/dn_mpc_jepa_safe_capture_dev/p39_route_utility_audit_v5_cbfgrid/report.md`
- Candidate details: `results/dn_mpc_jepa_safe_capture_dev/p39_route_utility_audit_v5_cbfgrid/details.csv`
