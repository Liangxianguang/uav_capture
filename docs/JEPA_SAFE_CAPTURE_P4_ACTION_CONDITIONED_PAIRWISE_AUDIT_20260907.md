# Action-Conditioned Pairwise Interaction Audit

**日期：** 2026-09-07
**状态：** development-only offline audit
**Locked test：** not opened

## 1. Purpose and boundary

This bounded audit tests the next JEPA prerequisite: whether a candidate first
step action adds useful pairwise interaction information beyond the current
public observation. For each archive row, the focal defender's current velocity
is replaced by the candidate first-step velocity. Teammate public velocities are
held fixed, so no simulator future state or target truth is used. Relative
velocity, short-horizon pairwise distance, and formation-topology features are
then compared with the recorded five-step pairwise TTC label.

The audit is diagnostic only. It does not execute actions, modify CBF margins,
disable stale/OOD/non-finite gates, train a JEPA, build a Ledger-Lite, or open a
closed-loop evaluation.

## 2. Inputs and split integrity

| split | archive | episodes | runtime samples | episode seeds |
|---|---|---:|---:|---|
| train | `jepa_safe_capture_p3_hard_negative_archive_train_smoke4_v2_20260907` | 4 | 1,728 | 645101--645104 |
| validation | `jepa_safe_capture_p3_hard_negative_archive_validation_smoke4_v2_20260907` | 4 | 1,392 | 646101--646104 |
| calibration | `jepa_route_identity_hard_negative_actor_calibration20_seed661606` | 20 | 24,528 | 648101--648120 |

The script verifies that all three seed sets are disjoint. The calibration
dataset SHA-256 is:

```text
0b8e0da8529a425c08198a6eca4a7fb98168b2ca5559238057194359b3000ec8
```

## 3. Calibration result

The diagnostic hazard is `minimum future pairwise TTC <= 1.0 s`, evaluated on
runtime rows. A threshold is selected only for reporting the best F1 trade-off;
it is not a deployed safety threshold.

| feature | runtime AUC | best precision | best recall |
|---|---:|---:|---:|
| current observed TTC | 0.7428 | 0.8245 | 0.5167 |
| candidate projected TTC | 0.6839 | 0.6307 | 0.4830 |
| candidate min distance at 0.3 s | 0.7695 | 0.4703 | 0.7858 |
| candidate min distance at 0.5 s | 0.8055 | 0.5071 | 0.7768 |
| **candidate min distance at 1.0 s** | **0.8246** | **0.5426** | **0.7555** |
| candidate topology edges at 1.0 s | 0.7727 | 0.4170 | 0.9522 |
| candidate topology edge delta | 0.5758 | 0.2985 | 0.9520 |

The best action-conditioned candidate feature improves AUC over the current
observation TTC (`0.8246 > 0.7428`), but it does not satisfy the pairwise gate:

```text
required recall >= 0.80 and precision >= 0.50
observed best candidate: recall 0.7555, precision 0.5426
gate: FAIL
```

The topology feature has high recall but insufficient precision. The result is
therefore not evidence that a learned hazard head is ready for runtime use.

## 4. Train/validation sanity check

The same feature is not uniformly stable across splits:

| split | candidate min distance at 1.0 s AUC | best precision | best recall |
|---|---:|---:|---:|
| train | 0.8646 | 0.5525 | 0.8139 |
| validation | 0.9056 | 0.7108 | 0.7905 |
| calibration | 0.8246 | 0.5426 | 0.7555 |

The calibration drop is exactly why the gate remains closed. It suggests
distribution shift and/or insufficiently targeted action-conditioned hard
negatives, rather than a reason to loosen CBF constraints.

## 5. Artifacts and TensorBoard

The audit implementation and tests are:

- `scripts/audit_jepa_pairwise_action_conditioned.py`
- `tests/test_audit_jepa_pairwise_action_conditioned.py`

The reproducible outputs are:

- `results/jepa_safe_capture_pairwise_action_conditioned_audit_20260907_v3.json`
- `results/jepa_safe_capture_pairwise_action_conditioned_audit_20260907_v3_tensorboard/`

TensorBoard records split sample counts, runtime positive fractions, per-feature
AUC values, the contract, archive hashes, and the gate decision. The contract
records `teammate_future_state_used=false`, `raw_unverified_execution_allowed=false`,
and `cbf_margin_changed=false`.

## 6. Decision

- [x] Candidate action projection is finite and reproducible.
- [x] Train/validation/calibration episode seeds are disjoint.
- [x] Candidate projected-distance signal improves calibration AUC over the
  current TTC feature.
- [ ] Pairwise recall/precision gate passes.
- [ ] Full JEPA training is authorized.
- [ ] New Ledger-Lite calibration is authorized.
- [ ] Closed-loop JEPA evaluation is authorized.

The next bounded action is targeted data collection/relabeling for the
calibration tail: candidate actions that separate safe formation contraction,
pairwise closing, and false-positive topology edges. Until a new disjoint
calibration archive passes the gate, retain analytic DN-MPC + strict Joint CBF
as the active development system and do not use this audit as a safety claim.
