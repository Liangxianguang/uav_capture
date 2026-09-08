# DN-MPC P40 Expanded Switch-Weight Audit

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** switch-weight boundary cleared; utility promotion remains closed

## Purpose

P39 selected `switch=0.5`, the largest value in its tested grid. P40 reruns
the same P39 P37 checkpoint and the same seed-disjoint train, calibration and
validation archives with an expanded switch grid:

```text
{0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0}
```

The audit is offline only. It executes no action, changes no CBF margin, and
does not create a Ledger or open a locked benchmark.

## Contract and provenance

- Checkpoint: P37 pairwise JEPA, seed `373701`
- Checkpoint SHA-256: `51a8d1f86e7182fe3fc2f42a6fa92adba0baa7a0378bf5135f5499e46496ff72`
- Train archive SHA-256: `a90090332b1a4fa7eb97771c96b10fc72a45677a87972503d73accbfc6bb2a4b`
- Calibration archive SHA-256: `de37a54d1800b4bc8352cc20712e9dcf0676dbf60a4d44300f96441326a64ceb`
- Validation archive SHA-256: `72898cccc407a107820201b54cdb0152a62220df16d13d7fd89c0737e5cadcd4`
- Planner route-switch identity: `previous_selected_candidate_index`
- CBF horizon: strict five-step; route projection and independent selected/nominal/safe-hold traces retained
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p39_route_utility_audit_v2_switchgrid/`

## Calibration result

Calibration selected:

```text
length=1.0, escape=0.0, cbf=0.5, switch=0.75
```

The `switch` weight is no longer at the active grid boundary. The `length`
weight is now at its maximum (`1.0`), so the full utility calibration is still
not boundary-free.

| Split | Exact top-1 | Informative top-1 | Pairwise | Model vs planner-selected |
|---|---:|---:|---:|---:|
| Train | 96.25% | 100.00% | 98.80% | 71.54% |
| Calibration | 96.90% | 100.00% | 98.12% | 76.37% |
| Validation | 99.11% | 99.69% | 97.92% | 80.12% |

P39 corresponding validation values were `99.11% / 100.00% / 97.42% /
80.71%`. Thus the expanded grid keeps selected-route agreement effectively
stable (`-0.59 pp`) and improves pairwise utility direction (`+0.50 pp`), but
it does not establish a safe-capture improvement or online route-selection
qualification.

## Interpretation

The original P39 conclusion about the corrected identity contract remains
valid: the large P38-to-P39 improvement was caused by separating planner
identity from frozen-actor execution identity, not by relaxing CBF. P40 adds
that switch regularization is not simply increasing without bound; calibration
settles at `0.75` on the expanded range. However, route length is still
boundary-selected, which means the utility label needs another bounded audit
before promotion.

The `lower_detour` geometry issue is unchanged. It remains a solid-ground
`lower_face_blocked` contract case and is not converted into a learned negative
without a separately verified ground-clearance route.

## Gate decision and next step

- [x] Expanded switch grid executed on all three P39 archives.
- [x] Actual grid and provenance written to JSON and TensorBoard.
- [x] Switch boundary artifact removed (`switch=0.75 < 1.0`).
- [ ] Full utility boundary cleared (`length=1.0` remains at the maximum).
- [ ] Independent evaluator seed stability demonstrated.
- [ ] Selected/nominal/safe-hold, OOD and disagreement gates promoted.

Remain offline-only. The next bounded task is to expand or otherwise audit the
route-length utility range using the same calibration contract, while reporting
whether the selected length prior is physically meaningful rather than merely
absorbing a route-label scale mismatch. Do not start three-seed paired replay,
create Ledger-Lite, or enable online JEPA route override until that audit and
an independent evaluator seed pass the promotion gates.

## Artifacts

- Script: `scripts/audit_dn_mpc_p36_route_utility.py`
- JSON: `results/dn_mpc_jepa_safe_capture_dev/p39_route_utility_audit_v2_switchgrid/audit.json`
- Markdown: `results/dn_mpc_jepa_safe_capture_dev/p39_route_utility_audit_v2_switchgrid/report.md`
- Candidate details: `results/dn_mpc_jepa_safe_capture_dev/p39_route_utility_audit_v2_switchgrid/details.csv`
