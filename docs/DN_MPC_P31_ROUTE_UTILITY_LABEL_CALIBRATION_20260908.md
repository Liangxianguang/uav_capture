# DN-MPC P31 Route-Utility Label Calibration

**Date:** 2026-09-08
**Status:** development-only; offline-only; promotion rejected

## 1. Scope

P31 implements the next plan item after P29/P30: define and independently
calibrate a route-utility label using public route geometry and offline outcome
labels, then apply the same calibration to the P28 and P29 checkpoints. It
does not modify the online planner, CBF, Ledger, candidate contract, or
locked-test protocol.

The utility contains four measured terms:

```text
progress / 0.3
- lambda_length * route_length_m / 10
- lambda_escape * target_relative_distance / 2
+ lambda_cbf * mean(CBF-feasible[steps 0..2])
```

The archive does not contain the previous executed route identity. Therefore a
route-switch penalty is explicitly marked **unavailable**, rather than inferred
from the current selected candidate.

## 2. Protocol and provenance

- Checkpoints: P28 seed `282801` and P29 seed `282802`.
- Train archive: `p28_augmented_train8_v2`.
- Calibration archive: `p28_augmented_calibration8_v2`.
- Validation archive: frozen `p25_trace_validation8`.
- Calibration grid: length `{0, .1, .2, .3, .5, 1}`, escape `{0, .1, .2, .5}`, CBF `{0, .1, .2, .5, 1}`.
- Eligibility: runtime row, valid route geometry, and first-step CBF feasible.
- Device: RTX 5050 CUDA, PyTorch `2.7.1+cu128`.
- No action was executed; no raw-unverified branch was opened.

The complete archive/checkpoint hashes and per-candidate details are recorded
in `results/dn_mpc_jepa_safe_capture_dev/p31_route_utility_label_audit_v2/audit.json`
and `details.csv`.

## 3. Calibration-selected results

Weights are selected separately on calibration and then held fixed for
validation.

| Model | Selected `(length, escape, cbf)` | Validation exact | Validation informative | Validation pairwise | Validation vs selected |
|---|---|---:|---:|---:|---:|
| P28 seed `282801` | `(1.0, 0.1, 0.5)` | `62.21%` | `80.65%` | `96.71%` | `41.28%` |
| P29 seed `282802` | `(1.0, 0.1, 0.2)` | `43.02%` | `87.10%` | `96.89%` | `29.65%` |

Relative to each model's zero-weight baseline, P31 improves exact top-1 from
`47.09%` to `62.21%` for P28 and from `41.28%` to `43.02%` for P29. The larger
informative and pairwise gains show that the public route-length and safety
terms correct route ordering, but the low model-vs-selected values show that
the analytic planner's selected route is a different contract and is not yet
reproduced by the learned evaluator.

The P29 seed remains substantially weaker than P28 on exact and selected
agreement despite similar pairwise agreement. This confirms that the utility
label improves the scoring direction but does not remove seed sensitivity.

## 4. Decision

P31 is a successful offline label/calibration diagnostic, not a deployment
promotion. Keep JEPA offline-only and do not create Ledger-Lite, run a three-
seed paired replay, relax CBF margins, disable gates, or execute unverified
actions.

The next data-contract change is to add `previous_executed_route_id` and an
explicit route-switch outcome to a new archive. Only then can the switch term
be trained and audited. After that archive is independently calibrated, P28
and P29 must be reevaluated with one shared utility contract and complete
selected/nominal/safe-hold CBF traces.

## 5. Artifacts

- Script: `scripts/audit_dn_mpc_p31_route_utility_label.py`
- Tests: `tests/test_audit_dn_mpc_p31_route_utility_label.py` (`2 passed`)
- Audit JSON/Markdown/details: `results/dn_mpc_jepa_safe_capture_dev/p31_route_utility_label_audit_v2/`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p31_route_utility_label_audit_v2/`
