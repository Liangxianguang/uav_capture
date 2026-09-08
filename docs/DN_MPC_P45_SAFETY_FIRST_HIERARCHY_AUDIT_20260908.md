# DN-MPC P45 Safety-First Hierarchy Audit

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** hierarchy selection gate failed; retain analytic planner authority

## Objective

P45 evaluates a safety-first route selector on the P39 archives. Candidate
geometry and verified first-step CBF feasibility are hard gates. Among eligible
candidates, predicted progress minus target escape is the primary score; route
length and candidate-relative switching are lexicographic tie-breaks inside a
calibration-selected primary tie band. No action is executed.

Checkpoints:

- P37 seed `373701`, SHA-256 `51a8d1f86e7182fe3fc2f42a6fa92adba0baa7a0378bf5135f5499e46496ff72`
- Independent P43 seed `393702`, SHA-256 `2417bd3d65b72c94e96708d29fc4827f2ff31dbda9f8cba8ae616dcbd68290b1`

Calibration and validation archives are the unchanged P39 seed-disjoint files.
TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p45_safety_first_hierarchy_audit_v2/`.

## Calibration contract

Shared calibration-only scales were:

```text
progress P95 absolute = 0.0758545
target-escape P95 absolute = 1.0886667
route-length P95 = 16.014567 m
```

The selector grid was `escape_weight={0,0.25,0.5,1,2}` and
`tie_band={0,0.005,0.01,0.02,0.05}` in normalized primary-score units.

## Result

| Model | Selected `(escape weight, tie band)` | Calibration exact | Validation exact | Validation model vs planner |
|---|---|---:|---:|---:|
| P37 `373701` | `(0, 0.05)` | 61.81% | 60.83% | 40.65% |
| P43 `393702` | `(0, 0.05)` | 65.87% | 61.72% | 41.54% |

Both seeds choose the same parameters, so the hierarchy parameter choice is
stable in this narrow sense. However, the tie band is at the largest tested
value and the escape weight collapses to zero. More importantly, planner
agreement falls to about `41%`, compared with `80.42%` for the bounded weighted
utility audit. The hierarchy therefore does not reproduce the analytic
DN-MPC route contract and cannot be promoted to route selection.

## Interpretation

The hard eligibility tier remains correct and should stay in the runtime
candidate pipeline. The failed part is the learned primary ranking: the
progress head does not contain enough calibrated signal to choose the planner's
route, and the large tie band makes route length/switch tie-breaks dominate.
This is a model/label calibration limitation, not evidence that CBF margins
should be loosened.

P45 is also not a safe-capture experiment. It provides no capture-rate or
collision claim and does not authorize JEPA route override, Ledger-Lite,
paired replay or locked testing.

## Gate decision

- [x] Hard geometry/first-step CBF eligibility is enforced before ranking.
- [x] Calibration-only scale and tie-band selection completed.
- [x] P37/P43 comparison and TensorBoard provenance recorded.
- [ ] Hierarchy planner-agreement gate: **failed** (`40.65%/41.54%`).
- [ ] Online JEPA route override or Ledger-Lite: **closed**.
- [ ] Additional evaluator seeds: **stopped**.

## Next task

Do not tune the hierarchy grid or add more seeds. Keep the analytic DN-MPC +
CBF planner as the executable authority and use JEPA only for offline risk and
trajectory evaluation. Before another ranking experiment, redesign the route
label contract around planner actions or collect explicit planner-distillation
targets with per-route counterfactual capture outcomes, then recalibrate on a
new independent archive. CBF margins and stale/OOD/non-finite gates remain
unchanged.

## Artifacts

- Script: `scripts/audit_dn_mpc_p45_safety_first_hierarchy.py`
- JSON: `results/dn_mpc_jepa_safe_capture_dev/p45_safety_first_hierarchy_audit_v2/report.json`
- Markdown: `results/dn_mpc_jepa_safe_capture_dev/p45_safety_first_hierarchy_audit_v2/report.md`
- Candidate details: `results/dn_mpc_jepa_safe_capture_dev/p45_safety_first_hierarchy_audit_v2/details.csv`
