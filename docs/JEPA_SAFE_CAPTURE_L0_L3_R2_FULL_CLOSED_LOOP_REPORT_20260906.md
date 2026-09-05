# JEPA Safe-Capture L0-L3 R2 Full Closed-Loop Report

Date: 2026-09-06  
Status: development-only evaluation; no locked test was opened.

## 1. Scope

This report records the completed L0-L3 paired evaluation for the action-conditioned, interaction-aware JEPA trajectory evaluator with reliability-ledger routing, joint CBF filtering, and rolling-horizon execution. The purpose of this run is to measure safety-preserving capture behavior across increasing observation, delay/noise, obstacle, and S3 stress conditions.

The primary metric is `safe_capture`. Collision, boundary violation, pairwise violation, and raw-unverified execution are hard safety metrics. `A3` is a raw/no-CBF diagnostic only and is excluded from every safety decision.

## 2. Protocol

- Training seeds: `20260911`, `20260912`, `20260913`.
- Variants: `M0`, `M1`, `M2`, `M3`, `A1`, `A2`, `A3`.
- Eight scenario families per run, eight episodes per family, `64` episodes per run.
- The same scene manifest and episode seeds were used for paired comparisons.
- Configuration: `configs/jepa_safe_capture_l0_l3_collection_v2.yaml`.
- Canonical scene-manifest SHA-256: `748d706ae6c2a064c92620200fe4c125a5be6f357c8bf85d7642235efee7a520`.
- Device: NVIDIA GeForce RTX 5050, CUDA execution.
- Aggregate artifacts: `results/l0_l3_r2_full_aggregate/`.

No CBF margin was reduced, no stale/OOD gate was disabled, and no controlled-abort path was removed.

## 3. Overall results

| Variant | Safe capture mean +/- std | Collision | Boundary | Pairwise | Raw-unverified steps |
|---|---:|---:|---:|---:|---:|
| M0 | 46.9% +/- 0.0% | 0 | 0 | 0 | 0 |
| M1 | 48.4% +/- 3.1% | 0 | 0 | 0 | 0 |
| M2 | 40.1% +/- 5.0% | 0 | 0 | 0 | 0 |
| M3 | 45.8% +/- 5.5% | 0 | 0 | 0 | 0 |
| A1 | 52.6% +/- 6.3% | 0 | 0 | 0 | 0 |
| A2 | 47.4% +/- 5.0% | 0 | 0 | 0 | 0 |
| A3 (raw/no-CBF) | 0.0% +/- 0.0% | 180 | 12 | 87 | 3192 |

All safety-preserving variants achieved a 100% transit-success rate. Their zero safety-event counts are preserved by ledger routing, CBF filtering, and controlled aborts; they must not be interpreted as evidence that the underlying policy is capture-complete.

## 4. Level findings

The aggregate contains 24 paired episodes per level (three seeds). Selected M0/M3 results are:

| Level | M0 | M3 |
|---|---:|---:|
| L0 open | 0.0% | 0.0% |
| L0 single obstacle | 75.0% | 70.8% |
| L1 nominal | 62.5% | 58.3% |
| L1 S-curve | 37.5% | 45.8% |
| L2 delayed/noisy | 37.5% | 20.8% |
| L2 partial observation | 75.0% | 70.8% |
| L3 mixed obstacle | 62.5% | 62.5% |
| L3 S3 stress | 25.0% | 37.5% |

The best safety-preserving auxiliary variant in this development matrix was A1 at 52.6% overall. It reached 91.7% on L2 partial observation and 83.3% on L3 mixed obstacle, but only 41.7% on L3 S3 stress and 25.0% on L2 delayed/noisy. The requested 90%/85%/80% targets are therefore not yet met across the full level definitions.

## 5. Paired M3 decision

Compared with M0 on identical episodes, M3 deltas were:

| Seed | M0 | M3 | Delta |
|---:|---:|---:|---:|
| 20260911 | 30/64 | 29/64 | -1.6 pp |
| 20260912 | 30/64 | 26/64 | -6.3 pp |
| 20260913 | 30/64 | 33/64 | +4.7 pp |

Mean paired delta: `-1.0 pp`; non-negative seeds: `1/3`. The automated classification is `inconclusive_development_evidence`, not a claim of improvement or regression. The safety hard gate and reliability/provenance gate both passed.

## 6. Interpretation

The experiment confirms the safety contract and the end-to-end execution path: all safety variants transit the scenarios without recorded collision, boundary, pairwise, or raw-unverified events, while A3 exposes the expected unsafe behavior when CBF filtering is removed. The current bottleneck is capture opportunity, especially in delayed/noisy and S3 stress families. The ledger is conservative enough to route many candidate rollouts to safe-hold/abort, so the next optimization target is verified candidate coverage and capture-oriented action diversity, not a relaxed safety margin.

These are development results on the frozen L0-L3 matrix. They do not replace the formal V4 locked test and do not justify reporting a final improvement over the prior `75.3%` V4 result.

## 7. Reproduction

Run the evaluator with the RTX 5050 environment and then aggregate:

```powershell
& 'D:\download\anaconda3\envs\traj_pred_prep\python.exe' scripts/run_with_tensorboard_compat.py scripts/evaluate_jepa_safe_capture_l0_l3_paired.py --config configs/jepa_safe_capture_l0_l3_collection_v2.yaml --output-root results/l0_l3_r2_full_runs --tensorboard-root results/l0_l3_r2_full_tensorboard --development-only

& 'D:\download\anaconda3\envs\traj_pred_prep\python.exe' scripts/run_with_tensorboard_compat.py scripts/aggregate_jepa_safe_capture_l0_l3_paired.py --input-root results/l0_l3_r2_full_runs --output-dir results/l0_l3_r2_full_aggregate --episodes-per-scenario 8 --development-only
```

The aggregate JSON/CSV, generated report, and TensorBoard event file in `results/l0_l3_r2_full_aggregate/` are the machine-readable provenance for this report.

## 8. Next action

Keep the hard safety gates unchanged. Before another training cycle, inspect the L2 delayed/noisy and L3 S3 failure index, verify message-age and candidate eligibility accounting, and add reachable, CBF-certified braking, tangential, radial, formation, and verified safe-hold candidate blocks. Retrain JEPA with clearance, visibility, CBF-feasibility, and action-consistency auxiliaries, then rebuild calibration and the reliability ledger before rerunning the same paired matrix.
