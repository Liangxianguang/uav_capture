# JEPA Safe-Capture P6-R2 Full Closed-Loop Report

Date: 2026-09-05
Phase: development-only; `locked_test_opened=false`

## Scope

This phase evaluates the new three-seed JEPA checkpoints in a paired rolling-horizon S3 validation matrix. The same 40-episode scene manifest and episode seeds are used for every variant and seed. Each control cycle executes only the first action step, re-observes, ranks candidate action chunks, routes through the reliability ledger when enabled, and applies the Joint CBF-QP before execution.

The matrix contains 3 training seeds (`20260911`, `20260912`, `20260913`) and seven variants:

| Variant | JEPA | Ledger | Joint CBF | Role |
|---|---:|---:|---:|---|
| M0 | no | no | yes | nominal actor + CBF baseline |
| M1 | yes | no | yes | JEPA candidate ranking |
| M2 | yes | yes | yes | JEPA + reliability routing |
| M3 | yes | yes | yes | complete proposed pipeline with auxiliary score |
| A1 | yes | no | yes | auxiliary-score ablation |
| A2 | yes | yes | yes | clearance/visibility score ablation |
| A3 | no | no | no | raw/no-CBF safety diagnostic only |

The canonical scene-manifest hash is `d709914ffd877671129ee843aa369c48a3dd4013e219a5a1b3dd2143bd1a30b8`. The full run artifacts are under `results/p6_r2_full_runs/`; per-run TensorBoard logs are under `results/p6_r2_full_tensorboard/`.

## Safe-Capture Results

Rates below are mean +/- sample standard deviation across the three training seeds. All safe variants have zero collision, boundary, and pairwise violations.

| Variant | Safe capture | Collision | Boundary | Pairwise | CBF infeasible steps |
|---|---:|---:|---:|---:|---:|
| M0 | 0.325 +/- 0.000 | 0 | 0 | 0 | 69 |
| M1 | 0.342 +/- 0.029 | 0 | 0 | 0 | 76 |
| M2 | 0.317 +/- 0.058 | 0 | 0 | 0 | 78 |
| M3 | 0.383 +/- 0.052 | 0 | 0 | 0 | 71 |
| A1 | 0.375 +/- 0.050 | 0 | 0 | 0 | 72 |
| A2 | 0.417 +/- 0.080 | 0 | 0 | 0 | 68 |
| A3 (diagnostic) | 0.000 +/- 0.000 | 117 | 0 | 45 | 0 |

Transit success is `100%` for every run. The raw A3 diagnostic is excluded from the safety decision: it intentionally bypasses CBF and demonstrates why the safety filter is a hard system boundary.

## M3 Paired Comparison Against M0

| Training seed | M0 | M3 | Delta | Improved | Degraded | Tied | McNemar exact p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 13/40 (32.5%) | 17/40 (42.5%) | +10.0 pp | 8 | 4 | 28 | 0.3877 |
| 20260912 | 13/40 (32.5%) | 13/40 (32.5%) | +0.0 pp | 5 | 5 | 30 | 1.0000 |
| 20260913 | 13/40 (32.5%) | 16/40 (40.0%) | +7.5 pp | 6 | 3 | 31 | 0.5078 |
| **Mean** | **32.5%** | **38.3%** | **+5.83 pp** | **19** | **12** | **89** | - |

The episode-pair bootstrap (4,000 resamples, seed `20260903`) gives a 95% interval of `[-2.5, +15.0] pp`. Therefore the result supports positive development evidence and satisfies the current non-inferiority decision rule (3/3 seeds non-negative), but it does not justify a locked-test or statistically significant improvement claim.

## Runtime and Safety Observations

- `raw_unverified_executed_steps=0` for every safety-preserving run.
- `controlled_abort` is counted separately from collision and does not silently become a successful fallback.
- The maximum observed CBF p95 solve latency remains well below the 100 ms contract.
- Ledger-enabled variants retain zero unsafe rate in trusted, fallback-nominal, and safe-hold routes on the independent calibration replay.
- A2 exceeds M3 on this development split, suggesting the current clearance/visibility score weights may be conservative. This is a hypothesis for the next ablation, not a reason to weaken CBF margins or stale/OOD gates.

## Checkpoints and Ledger Provenance

| Seed | Checkpoint SHA-256 | S3-bound ledger |
|---:|---|---|
| 20260911 | `5cde74db6f46e00f473fc06ee7617267daf34b5417df01d53704cc41f06225b4` | `results/jepa_safe_capture_l0_l3_v1_ledger_s3bound_seed20260911_r2.json` |
| 20260912 | `af073c81bcea7a840f847ae8e622836e65e75a642d5e06d16c3de624f2993b7a` | `results/jepa_safe_capture_l0_l3_v1_ledger_s3bound_seed20260912_r2.json` |
| 20260913 | `f3a273ebbf6ff1cce3911fd3262a478fe8256fb4c402ea89ef95b66828eb5a0f` | `results/jepa_safe_capture_l0_l3_v1_ledger_s3bound_seed20260913_r2.json` |

The ledger builder now records the calibration protocol and an explicit runtime `evaluation_protocol` binding. The evaluator accepts the runtime binding while preserving the original calibration provenance; a protocol mismatch still fails closed.

## Reproduction

The authoritative aggregate is `results/p6_r2_full_aggregate/summary.json`, with CSV episode/run tables and TensorBoard at the same output root. Re-run the matrix with `scripts/evaluate_jepa_safe_capture_v2_paired.py` through `scripts/run_with_tensorboard_compat.py`, then aggregate with:

```powershell
& 'D:\download\anaconda3\envs\traj_pred_prep\python.exe' scripts/run_with_tensorboard_compat.py scripts/aggregate_jepa_safe_capture_v2_paired.py --input-root results/p6_r2_full_runs --output-dir results/p6_r2_full_aggregate --stage full --development-only
```

The 40-episode validation split is development-only. The locked-test split was not opened and must remain a separate future audit.

## Next Work

1. Diagnose the 12 M3 regressions and 19 improvements at episode level, grouped by obstacle count, motion mode, and observation condition.
2. Recalibrate score weights using a new development-only archive; preserve CBF margins, stale/OOD routing, and controlled-abort semantics.
3. Add the requested L0/L1/L2 scenario-specific closed-loop evaluator. The present P6 matrix is the S3 stress contract and should not be relabeled as the full L0-L3 endpoint.
4. Only after the above fixes, run a fresh paired matrix and consider opening a preregistered locked test.
