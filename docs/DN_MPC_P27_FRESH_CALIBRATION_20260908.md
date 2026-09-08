# DN-MPC P27 Fresh Calibration and Disagreement Audit

**Status:** development-only; offline-only; no online JEPA route action, no
Ledger-Lite, and no locked test were opened.

## Purpose

P27 binds a fresh calibration profile to the P25 traceable train/validation/
calibration archives and the unchanged P24 listwise route-JEPA checkpoint. It
measures public input/action OOD distance, route-progress error, five-step
rollout disagreement, candidate disagreement, and the two explicit P26
planner-abstention states.

The audit is read-only. It does not execute an action, alter CBF margins,
disable stale/OOD/non-finite gates, fabricate a selected route, or change
`controlled_abort` semantics.

## Artifacts

| Artifact | Path |
|---|---|
| Audit JSON | `results/dn_mpc_jepa_safe_capture_dev/p27_fresh_calibration/audit.json` |
| Audit Markdown | `results/dn_mpc_jepa_safe_capture_dev/p27_fresh_calibration/AUDIT.md` |
| Group details | `results/dn_mpc_jepa_safe_capture_dev/p27_fresh_calibration/groups.csv` |
| TensorBoard | `results/dn_mpc_jepa_safe_capture_tensorboard/p27_fresh_calibration` |
| Script | `scripts/audit_dn_mpc_p27_fresh_calibration.py` |

Checkpoint SHA-256:
`f72b851f4c7f89e8f41b7cbac6d37f72b55186f4a6443daf03126f9d510ab797`.

The OOD profile is fitted on the P25 train runtime rows (`6,576` rows), while
thresholds are estimated only on the independent P25 calibration runtime rows
(`10,272` rows). The validation block contains `8,256` runtime rows and
`172/172` groups with independent selected/nominal/safe-hold CBF traces.

## Fresh Thresholds

| Signal | Calibration rule | Threshold |
|---|---:|---:|
| Input/action OOD distance | p99 | `1.793742` |
| Route-progress absolute error | p95 | `0.083589` |
| Five-step rollout disagreement | p95 | `0.423982 m` |

## Results

| Split | OOD rate | route-error coverage | rollout coverage | model vs selected | model vs truth | zero-eligible groups |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.71% | 90.09% | 99.92% | 45.59% | 59.56% | 1 |
| calibration | 1.00% | 94.996% | 94.996% | 25.35% | 39.44% | 1 |
| validation | 1.93% | 94.925% | 95.252% | 19.19% | 36.63% | 0 |

The finite-output gate passed. Validation OOD rate is below the 10% review
limit, and both calibrated error-coverage gates passed. The candidate
agreement gate failed: only `19.19%` of validation runtime groups agree with
the analytic DN-MPC selected route. This agrees with the P25/P24 ranking audit
and is not evidence that CBF should be relaxed.

## Abstention Binding

The two P26 states remain explicit abstentions:

- train: `scenario=6, time=47`, all 12 candidates first-step CBF-infeasible;
- calibration: `scenario=5, time=47`, all 12 candidates first-step CBF-infeasible;
- validation: no zero-eligible group, minimum eligible candidate count is 2.

No synthetic selected route was created for either state. The JSON and group
CSV retain `selected_candidate_index=-1` for these rows.

## Promotion Decision

| Gate | Result |
|---|---|
| checkpoint/archive contract | PASS |
| finite JEPA outputs | PASS |
| fresh thresholds finite | PASS |
| abstention samples bound | PASS |
| validation OOD coverage | PASS |
| validation route-error coverage | PASS |
| validation rollout coverage | PASS |
| validation candidate agreement | **FAIL** (`19.19%`) |
| online promotion | **STOP** |

P27 therefore authorizes no online JEPA routing, no Ledger-Lite creation, and
no three-seed paired replay. The next permitted work is P28: collect
anticipatory braking, nearest-tangent, and boundary-rescue hard negatives
around the two P26 abstention states, then rebuild independent selected/
nominal/safe-hold traces and rerun the fresh calibration gate.
