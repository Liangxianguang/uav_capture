# DN-MPC P44 Utility Label and Scale Diagnosis

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** weighted utility is underdetermined; replace with safety-first hierarchy before more training

## Scope

P44 audits the P39 calibration archive without a model and without executing
actions. It measures the exact normalized terms used by the P40-P43 route
utility audit: route progress, route length, target escape cost, multi-step CBF
feasibility and planner route-switch penalty.

Archive: `p39_route_identity_calibration8`
Dataset SHA-256: `de37a54d1800b4bc8352cc20712e9dcf0676dbf60a4d44300f96441326a64ceb`
Eligible candidate rows: `3,709`
TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p44_utility_scale_diagnosis_v2/`

## Normalized distributions

| Component | Min | P05 | Median | P95 | Max | Mean | Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| progress (`label / 0.3`) | -0.3204 | -0.1300 | 0.0051 | 0.2220 | 0.4990 | 0.0208 | 0.1067 |
| route length (`m / 10`) | 0.0000 | 0.0000 | 0.0000 | 1.3351 | 2.4502 | 0.3458 | 0.4726 |
| escape (`label / 2`) | 0.0496 | 0.0906 | 0.1958 | 0.4834 | 0.7809 | 0.2264 | 0.1207 |
| CBF feasibility (`mean first 3`) | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9996 | 0.0134 |
| switch (known previous only) | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8918 | 0.3106 |

The previous planner route is known for `98.95%` of candidate rows. The
`89.18%` switch rate is not the actual planner switch frequency: each group
contains many alternative candidates, so all candidates except the held route
receive a switch penalty. It is therefore a highly imbalanced binary feature.

## Main findings

1. **CBF term has almost no ranking variance.** Its median and P95 are both
   `1.0`, with standard deviation `0.0134`. CBF should be a hard eligibility
   gate or lexicographic first tier, not a freely tuned smooth reward.
2. **Route length is sparse and much larger than progress.** Half of eligible
   rows have zero route length, while P95 normalized length is `1.3351` versus
   P95 progress `0.2220`. This explains why calibration moves the length weight
   to the grid edge: the weighted sum is compensating for a scale/definition
   mismatch, not discovering a physical constant.
3. **Switch is candidate-relative, not event-relative.** The switch penalty is
   useful for tie-breaking against unnecessary changes, but its row-level
   distribution cannot be interpreted as a route-switch prior and is unstable
   under seed-dependent score perturbations.
4. **Progress and length are not linearly redundant in this archive.** Their
   Pearson correlation is `-0.0153`; progress and escape are only `0.2537`
   correlated. The instability is therefore primarily term scale and contract
   semantics, not simple duplicate progress labels.

## Required contract change

Before another evaluator seed, define and freeze a safety-first hierarchy:

```text
Tier 0: remove geometry-invalid or first-step CBF-infeasible candidates
Tier 1: reject non-finite/stale/OOD candidates and explicit abstention states
Tier 2: maximize predicted capture progress minus target-escape cost
Tier 3: minimize normalized route length
Tier 4: minimize candidate-relative route switch, only within a tie band
```

CBF is no longer calibrated as an additive weight when it is already enforced
by the verified candidate gate. Route length and switch are tie-break costs,
with normalization declared in physical units and a fixed tie band. The
analytic DN-MPC planner remains the executable authority during this change.

## Gate decision

- [x] Calibration-only range and correlation audit completed.
- [x] Unknown planner-history rows excluded from switch statistics.
- [x] TensorBoard and archive hashes recorded.
- [ ] New hierarchy implemented and independently calibrated.
- [ ] Existing P37/P43 checkpoints stable under the fixed hierarchy.
- [ ] Online JEPA override, Ledger-Lite, three-seed replay or locked test authorized.

The P43 instability is now attributable to an underdetermined weighted utility
contract. Stop adding seeds and do not tune CBF margins. The next bounded task
is to implement the hierarchy above as an offline evaluator mode and compare
P37/P43 on the same calibration and validation archives.

## Artifacts

- Script: `scripts/diagnose_dn_mpc_p44_utility_scale.py`
- JSON: `results/dn_mpc_jepa_safe_capture_dev/p44_utility_scale_diagnosis_v2/report.json`
- Markdown: `results/dn_mpc_jepa_safe_capture_dev/p44_utility_scale_diagnosis_v2/report.md`
