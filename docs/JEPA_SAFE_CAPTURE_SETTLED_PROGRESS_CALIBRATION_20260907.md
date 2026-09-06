# Settled Progress Calibration Audit

**Date:** 2026-09-07
**Scope:** development-only offline audit; one-step CBF counterfactual labels
**Locked test:** not opened
**Online contract:** unchanged

## Result

The audit joined the three existing route-regret reports with their source
traces and evaluated the existing JEPA heads against the settled next-step
distance after the same `strict_buffer`, `horizon=5` CBF branch. It used
`4,428` trusted settled candidate rows from episode seeds `650101`, `650102`
and `650103`.

| Head | Direction | Settled top-1 | Spearman vs after-distance |
|---|---|---:|---:|
| `ranker_score` | cost minimize | `36.67%` | `0.5615` |
| `target_cost_m` | cost minimize | `2.51%` | `0.6710` |
| `uncertainty_cost_m` | cost minimize | `0.00%` | `0.8201` |
| `clearance_cost_m` | cost minimize | `9.79%` | `-0.2281` |
| `ttc_cost` | cost minimize | `64.01%` | not identifiable (constant) |
| `visibility_cost` | cost minimize | `15.72%` | `0.1073` |
| `cbf_risk_cost` | cost minimize | `2.51%` | `-0.3217` |
| `candidate_separation_m` | cost minimize | `10.71%` | `0.1127` |
| `predicted_route_progress_m` | progress maximize | `27.33%` | `-0.3465` |
| `target_escape_progress_m` | progress maximize | `60.82%` | `-0.2602` |

The existing ranker is therefore internally consistent with its score argmin,
but its selected route agrees with the settled best route in only `36.67%` of
trusted steps. Agreement falls across the three scene seeds from `63.75%` to
`47.75%` to `22.98%`. This reproduces the earlier aggregate audit and confirms
that the capability bottleneck is score/label alignment, not candidate
execution order or selected-route CBF failure.

Some heads have a monotonic correlation with settled distance, but this is not
enough for a controller claim: `target_cost_m` and `uncertainty_cost_m` have
very low or zero top-1 agreement, and the clearance/risk heads have the wrong
aggregate direction. The `ttc_cost` top-1 result is not evidence of useful
prediction because it is constant in this archive.

The calibration gate required the existing ranker to reach at least `50%`
settled top-1 agreement overall and in every episode, with a nonnegative
per-episode cost correlation. It observed `36.67%` overall and
`63.75% / 47.75% / 22.98%` by seed, so the gate **failed**.

## Gate decision

This phase is a **prediction-signal diagnostic with no control gain**. It does
not authorize:

- changing online score weights or adding an escape switch;
- reducing CBF margins or disabling stale/OOD/non-finite gates;
- training a new JEPA checkpoint;
- expanding to L1-L3 or a larger scene matrix;
- opening a locked test.

The next permitted action is a new disjoint calibration archive with settled
progress, clearance, and CBF-feasibility labels, followed by a hash-bound
Ledger. A fresh paired L0 gate is allowed only if that offline calibration is
stable under held-out scenes and the score/settled top-1 agreement improves.

## Provenance

- Audit script: [audit_jepa_safe_capture_settled_progress_calibration.py](../scripts/audit_jepa_safe_capture_settled_progress_calibration.py)
- Unit test: [test_audit_jepa_safe_capture_settled_progress_calibration.py](../tests/test_audit_jepa_safe_capture_settled_progress_calibration.py)
- Source manifest SHA-256: `130f9602a87bcc392b10b2fa8bddaeaa88dab21bae4d932f95ddfeb7647da644`
- Input audit SHA-256: `71365e501dff22ff21b68d9022b7790bbbfb6c0af539bcd09d6700d36590bbeb`, `23497d018c1629b3a7291eae742e7ca7bfc8419201898a6372bf06fa9e3d7972`, `12aef50f7ef5b682ae5f70f5ab3ed2b1825a2d6b8753590d764ec0075c2db971`
- Calibration JSON: `results/wp1_active_search_v5_settled_progress_calibration_20260907_final/calibration.json`
- Calibration JSON SHA-256: `6f8735091d9b0150ebabc4a47f722dd43181aa7f7cc6528e4a2e2133d9c5e192`
- TensorBoard event: `results/wp1_active_search_v5_settled_progress_calibration_20260907_final_tensorboard/events.out.tfevents.1788717440.PC-20250926.33120.0`
- TensorBoard event SHA-256: `283c855b6f269863781626b33932063473fd1d0dd9379f682a1b88f8a33761f0`
