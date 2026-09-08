# DN-MPC + Strict CBF: L0-L3 64-Episode Development Result

Date: 2026-09-08  
Status: development-only; not a locked benchmark result

## Scope

This replay completed the remaining L2/L3 slices of the frozen 64-episode
L0-L3 paired manifest. All eight difficulty groups use the same contract:

- actor: `models/v5_development_exact_reactive_seed661606.pt`
- DN-MPC horizon: 5 steps; execute first step then replan
- CBF horizon: 5 steps; `strict_buffer`
- route-probe horizon: 5 steps
- minimum route hold: 3 steps; tangent-route hold: 6 steps
- no JEPA and no Reliability Ledger
- no CBF margin reduction, stale/OOD gate bypass, or raw-unverified action
- RTX 5050, PyTorch `2.7.1+cu128`, CUDA 12.8

The evaluator did not open the V4 locked test.

## Results

| Difficulty | Episodes | Safe capture | Rate | Mean capture time (s) | Collision | UAV boundary | Pairwise | Raw-unverified | Abort | Target boundary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 open | 8 | 8 | 100.0% | 5.90 | 0 | 0 | 0 | 0 | 0 | 0 |
| L0 single obstacle | 8 | 8 | 100.0% | 8.73 | 0 | 0 | 0 | 0 | 0 | 0 |
| L1 nominal | 8 | 8 | 100.0% | 11.19 | 0 | 0 | 0 | 0 | 0 | 0 |
| L1 S-curve | 8 | 6 | 75.0% | 15.38* | 0 | 0 | 0 | 0 | 0 | 2 |
| L2 partial observation | 8 | 8 | 100.0% | 11.96 | 0 | 0 | 0 | 0 | 0 | 1 |
| L2 delayed/noisy | 8 | 7 | 87.5% | 16.61* | 0 | 0 | 0 | 0 | 0 | 1 |
| L3 mixed obstacle | 8 | 5 | 62.5% | 14.20 | 0 | 0 | 0 | 0 | 0 | 0 |
| L3 S3 stress | 8 | 5 | 62.5% | 18.66* | 0 | 0 | 0 | 0 | 0 | 3 |
| **All groups** | **64** | **55** | **85.94%** | **12.27*** | **0** | **0** | **0** | **0** | **0** | **7** |

`*` Mean capture time is calculated over successful episodes only.

Route-probe totals over all 64 episodes were 74,133 checks, 63,752 accepted
and 10,381 rejected. The observed route-switch count was 1,248. The minimum
UAV clearance was at or above the 0.35 m operational buffer in every group.

## Failure analysis

There were 9 failed episodes, all terminating by `timeout`; no failure was a
collision, UAV boundary violation, pairwise violation, CBF controlled abort,
fallback execution, or raw-unverified action.

- L1 S-curve: 2 timeouts (episodes 24 and 25).
- L2 delayed/noisy: 1 timeout (episode 40).
- L3 mixed obstacle: 3 timeouts (episodes 48, 51 and 54).
- L3 S3 stress: 3 timeouts (episodes 56, 59 and 63).

The L3 traces show valid geometric detours and successful CBF verification,
but repeated progress loss around visibility-hold, detour and braking phases.
This is a route-selection/task-progress limitation, not evidence that the CBF
is too permissive. The next optimization should therefore improve route-side
commitment, terminal progress and obstacle-exit logic while preserving the
strict safety contract.

The seven target-boundary events are reported separately from UAV safety
violations. They occur in L1/L2/L3 stress conditions and should be retained as
an explicit adversarial-target outcome in later analyses.

## Reproduction outputs

Each group has an independent `summary.json`, `provenance.json`, and
`step_traces/` directory:

- `results/dn_mpc_cbf_l0_l3_scan_seed20260908_l0_open/`
- `results/dn_mpc_cbf_l0_l3_scan_seed20260908_l0_single/`
- `results/dn_mpc_cbf_l0_l3_scan_seed20260908_l1_nominal/`
- `results/dn_mpc_cbf_l0_l3_scan_seed20260908_l1_scurve/`
- `results/dn_mpc_cbf_l0_l3_scan_seed20260908_l2_partial/`
- `results/dn_mpc_cbf_l0_l3_scan_seed20260908_l2_delayed_noisy/`
- `results/dn_mpc_cbf_l0_l3_scan_seed20260908_l3_mixed_obstacle/`
- `results/dn_mpc_cbf_l0_l3_scan_seed20260908_l3_s3_stress/`

TensorBoard event files are in the matching
`results/dn_mpc_cbf_l0_l3_scan_tb_seed20260908_*` directories. Each run logs
the full provenance contract and aggregate/episode metrics.

## Decision for the next stage

Do not promote this single-seed result to a final claim and do not train a new
JEPA checkpoint yet. First diagnose and repair L3 route progress using the
existing traces, then run a paired JEPA evaluation on exactly the same 64
scenes. Promotion requires preserving all hard safety counters and reporting
multi-seed confidence intervals.

