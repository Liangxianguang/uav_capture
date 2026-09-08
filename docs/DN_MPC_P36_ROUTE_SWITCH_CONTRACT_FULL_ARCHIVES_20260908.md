# DN-MPC P36 Full Route-Switch Archives

**Date:** 2026-09-08
**Phase:** development-only archive and label contract
**Decision:** archive gate passed; JEPA promotion and locked testing remain closed

## What was completed

P36 extends the route archive so that route utility can be tied to what the
closed-loop system actually did after CBF filtering. Each runtime row records:

- the previous and current executed route identity;
- route-switch outcome (`1` switch, `0` hold, `-1` unknown/CBF abstention);
- action-match residual under the fixed `1.0 m/s` tolerance;
- offline target-escape cost;
- independent selected, nominal and safe-hold CBF counterfactual trace fields.

An unknown route is accepted only when the first-step CBF is infeasible. It is
not treated as a route hold and no unverified action is executed.

## Disjoint archive results

| Split | episode seeds | all rows | runtime rows | first-step CBF feasible | route matched | unknown route | switch / known previous | residual P95 | independent trace |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train8-v2 | 658101-658108 | 13,936 | 12,864 | 99.63% | 99.63% | 0.37% | 4.62% | 0.356 m/s | 99.63% |
| validation8-v2 | 659101-659108 | 17,628 | 16,272 | 99.41% | 99.41% | 0.59% | 5.74% | 0.381 m/s | 99.41% |
| calibration8-v2 | 657101-657108 | 21,840 | 20,160 | 99.76% | 99.76% | 0.24% | 5.58% | 0.337 m/s | 99.76% |

The archive audit confirms that all unknown rows coincide with first-step CBF
infeasibility. This is a useful safety result: the route identity contract is
not silently inventing a route during controlled abstention. Geometry-valid
fractions are `77.24%`, `71.78%`, and `73.77%` for train, validation and
calibration respectively; they are archive label coverage statistics, not
capture rates.

## Artifacts

- Train archive: `results/dn_mpc_jepa_safe_capture_dev/p36_route_switch_contract_train8_v2/`
- Validation archive: `results/dn_mpc_jepa_safe_capture_dev/p36_route_switch_contract_validation8_v2/`
- Calibration archive: `results/dn_mpc_jepa_safe_capture_dev/p36_route_switch_contract_calibration8_v2/`
- Full audit: `results/dn_mpc_jepa_safe_capture_dev/p36_route_switch_contract_train8_validation8_calibration8_v2_audit.json`
- TensorBoard runs: `results/dn_mpc_jepa_safe_capture_tensorboard/p36_route_switch_contract_{train8_v2,validation8_v2,calibration8_v2}/`
- Configs: `configs/jepa_safe_capture_route_identity_archive_dn_mpc_chunk5_p36_route_switch_{train,validation,calibration}.yaml`

## Interpretation

This experiment did **not** run an online JEPA or safe-capture evaluation, so
it does not establish a capture-rate improvement. It establishes the data
contract needed to measure route switching and target escape without confusing
CBF abstention with route persistence. The observed switch rates are moderate
and similar across splits, while route matching remains above `99%` among
CBF-eligible runtime rows.

The next step is to collect the explicit pairwise action-conditioned feature
contract and train a new listwise evaluator. Its route utility must be
calibrated on the disjoint calibration split before any online route override,
Ledger-Lite construction, multi-seed replay or locked benchmark is considered.
