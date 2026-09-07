# DN-MPC P12 Pairwise TTC Label-Semantics Audit

**Date:** 2026-09-08
**Status:** development-only; stop the current pairwise-hazard training branch
**Locked test:** not opened

## Scope

P12 checks whether `labels_pairwise_ttc <= 1.0 s` is a suitable proxy for an
actual strict pairwise-margin violation or CBF infeasibility. It is a read-only
audit of the original and fresh disjoint calibration archives. It does not
change the CBF margin, train a checkpoint, or make a model runtime eligible.

The archive stores physical inter-agent clearance after subtracting the two
vehicle radii. Therefore the strict operational safety condition is
`inter_agent_clearance >= 0.35 m`; `clearance < 0.35 m` is the reported margin
violation. `boundary_shadow` rows are synthetic offline negatives and are
excluded from real branch-failure intersections while remaining visible as a
separate class.

## Inputs and provenance

| Block | Rows | Archive SHA-256 | Metadata SHA-256 |
| --- | ---: | --- | --- |
| Original calibration | 32,704 | `f8f0a337752a2382876f6c45c3aadf0e4c2d07bd46c718f353636d6e94e03d84` | `6c071c46476a5ad7fa014e1c58ec505a74753f69ab4e2fa192f33cce48acd803` |
| Fresh calibration | 32,896 | `b27b9ca6064dc34600eda64c3671a11e41274a9c07339f477426d097b8a84b11` | `dbdee7d18f3ea78ca41b11b05e395ef669fe3a2d4764453159baa0dcab5bb587` |

Artifacts:

- JSON audit: `results/dn_mpc_jepa_safe_capture_dev/DN_MPC_P12_PAIRWISE_TTC_LABEL_AUDIT_20260908_v2.json`
- TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p12_pairwise_ttc_label_audit_20260908_v2`
- Script: `scripts/audit_dn_mpc_pairwise_ttc_labels.py`

## Results

The table reports candidate-level rates. A candidate is positive when any of
its five rollout horizons satisfies the condition. `safe|hazard` and
`CBF-feasible|hazard` are conditional on a TTC hazard. Cell-level results are
also retained in the JSON and TensorBoard; they are the correct view for
per-horizon feasibility (`97.9%`--`98.5%` CBF feasible among runtime hazards).

### Original calibration

| Sample type | TTC hazard | Safe clearance given hazard | CBF feasible given hazard | Early branch failure given hazard | Strict margin violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| runtime | 25.34% | 100.00% | 100.00% | 3.01% | 0.00% |
| near_pass | 33.07% | 100.00% | 100.00% | 2.07% | 0.00% |
| formation_crossing | 51.71% | 100.00% | 100.00% | 2.46% | 0.00% |
| split_merge | 26.32% | 100.00% | 100.00% | 1.86% | 0.00% |

### Fresh calibration

| Sample type | TTC hazard | Safe clearance given hazard | CBF feasible given hazard | Early branch failure given hazard | Strict margin violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| runtime | 19.16% | 100.00% | 100.00% | 2.26% | 0.00% |
| near_pass | 28.40% | 100.00% | 100.00% | 3.42% | 0.00% |
| formation_crossing | 47.96% | 100.00% | 100.00% | 2.43% | 0.00% |
| split_merge | 25.10% | 100.00% | 100.00% | 3.10% | 0.00% |

The `boundary_shadow` class is intentionally different: all of its rows are
offline negative probes with TTC hazard and infeasible labels. It contains
2,044 original and 2,056 fresh rows and is not evidence of an executed branch
failure.

## Interpretation

1. The TTC calculation is not detecting observed pairwise violations in these
   archives. For valid runtime and interaction branches, every TTC hazard is
   still at or above the strict `0.35 m` operational margin.
2. The label is an anticipatory constant-velocity warning: it asks whether the
   current relative velocity would cross the operational set without future
   braking or CBF correction. The subsequent CBF-protected rollout usually
   remains feasible and safe.
3. Treating this warning as a binary failure target explains the P8--P11
   precision/recall trade-off. Increasing the positive BCE weight cannot make
   the target simultaneously represent “predicted hazard” and “actual CBF
   failure”.
4. The result does **not** justify weakening CBF, reducing the margin, or
   executing a TTC-positive route without verification. It identifies a label
   contract mismatch instead.

## Decision and next action

- Stop further scalar pairwise-hazard weight sweeps and do not attach any P8 or
  P11 checkpoint to runtime ranking.
- Keep strict CBF, reachable projection, non-finite/OOD/stale gates and
  controlled abort unchanged.
- Design a disjoint replacement label contract before retraining. The next
  candidate should separate at least:
  - `predicted_ttc_hazard` (warning under a stated motion assumption);
  - `strict_margin_violation` (actual clearance below `0.35 m`);
  - `cbf_infeasible` (the verified primary solve fails);
  - `branch_failure` (the counterfactual cannot advance under the execution
    contract).
- Recollect or relabel a calibration block with explicit counterfactual
  intervention outcomes. A new checkpoint may be trained only after the new
  label definition, calibration archive hash, and gate are independently
  recorded.

This stop is a positive diagnostic result: the safety filter is preventing the
potential TTC hazard from becoming a physical pairwise violation, while the
current predictor target conflates those two events.
