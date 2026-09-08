# DN-MPC P36 Route-Utility Calibration

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** promotion rejected; no online JEPA override and no Ledger-Lite

## Scope

This experiment applies one shared route-utility contract to the P36 train,
calibration, and validation archives and to the two frozen P28/P29 JEPA
checkpoints. The utility includes:

```text
progress
- route length
- target escape cost
+ multi-step CBF feasibility
- route-switch penalty
```

The switch penalty is now measured from `previous_executed_route_index` and the
candidate route identity. Unknown route rows are explicit CBF abstentions and
are excluded from ranking; no future truth is used to infer an executed route.

## Inputs

| Item | Value |
|---|---|
| Train archive | `p36_route_switch_contract_train8` (`13,936` rows) |
| Calibration archive | `p36_route_switch_contract_calibration8` (`21,840` rows) |
| Validation archive | `p36_route_switch_contract_validation8` (`17,628` rows) |
| Runtime rows | `12,864 / 20,160 / 16,272` |
| Checkpoints | P28 seed `282801`, P29 seed `282802` |
| Weight grid | length `{0,.1,.2,.3,.5,1}`, escape `{0,.1,.2,.5}`, CBF `{0,.1,.2,.5,1}`, switch `{0,.01,.05,.1,.2,.5}` |
| TensorBoard | `results/dn_mpc_jepa_safe_capture_tensorboard/p36_route_utility_audit8_v2` |

## Results

Weights were selected on the independent calibration split and then held fixed
for validation.

| Model | Selected `(length, escape, cbf, switch)` | Validation exact top-1 | Validation informative top-1 | Validation pairwise | Model vs selected route |
|---|---|---:|---:|---:|---:|
| P28 seed `282801` | `(1.0, 0.5, 0.5, 0.5)` | 99.11% | 99.69% | 98.92% | 34.72% |
| P29 seed `282802` | `(1.0, 0.1, 0.5, 0.5)` | 98.81% | 99.38% | 98.84% | 34.72% |

The calibrated utility improves ranking against offline labels, but both
checkpoints still disagree with the analytic planner's selected route in about
two thirds of comparison groups. `length=1.0` and `switch=0.5` are at the top
of the tested grid for both models; P28 also selects `escape=0.5`. This is a
boundary-of-grid diagnostic, not evidence that those weights are physically
optimal.

## Interpretation

The route identity contract is now traceable and the utility direction is
measurable, but there remains a contract mismatch between the learned evaluator
and the analytic DN-MPC selector. The high truth-ranking score cannot authorize
online route replacement because the runtime system would not execute the
route used to define that score. The result also does not establish a
safe-capture improvement.

## Gate decision

P36 passes archive and traceability gates but fails the candidate-agreement
promotion gate. Keep the JEPA evaluator offline-only, do not create or update a
Ledger-Lite, do not run a three-seed paired replay, and do not open a locked
test. The next bounded task is to diagnose the route-label/planner contract
using route-family and route-switch stratified agreement, then either align the
analytic selector labels with the learned candidate interface or stop expanding
the learned model. CBF margins and stale/OOD/non-finite gates remain unchanged.

Artifacts:

- Audit script: `scripts/audit_dn_mpc_p36_route_utility.py`
- Machine-readable audit: `results/dn_mpc_jepa_safe_capture_dev/p36_route_utility_audit8_v2/audit.json`
- Candidate details: `results/dn_mpc_jepa_safe_capture_dev/p36_route_utility_audit8_v2/details.csv`
- Archive contract audit: `results/dn_mpc_jepa_safe_capture_dev/p36_route_switch_contract_audit8.json`
