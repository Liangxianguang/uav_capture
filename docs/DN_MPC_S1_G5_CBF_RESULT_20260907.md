# S1 DN-MPC + Strict Joint CBF Development Result

## Scope

This is a development-only paired evaluation on the frozen four-episode G5
manifest. It uses the retained V5 actor and the same environment/protocol
contract as the previous G5 JEPA + Ledger + CBF run, but disables JEPA and the
reliability ledger. The controller is:

```text
public belief and obstacle geometry
-> reachable obstacle-route candidates
-> independent primary Joint CBF probe for every candidate
-> distributed minimax DN-MPC selection
-> independent selected/nominal/safe-hold probes
-> final strict Joint CBF filter
-> execute first step and replan
```

No locked-test split was opened. The output is not a formal benchmark claim.

## Result

Run directory: `results/dn_mpc_cbf_g5_dev_seed20260907_v2/`

| metric | value |
|---|---:|
| safe capture | `3/4 = 75.0%` |
| collision | `0/4` |
| defender boundary violation | `0/4` |
| pairwise violation | `0/4` |
| raw-unverified execution | `0` steps |
| controlled abort | `1` step |
| target boundary diagnostic | `1/4` |
| route probe accepted | `4762/4996` |
| route switches | `112` |
| mean capture time | `18.733 s` |
| worst physical minimum clearance | `0.3502 m` |

The target-boundary count is reported as a diagnostic and does not invalidate
safe capture, matching the established G5 contract. The one controlled abort
does invalidate its episode; it is not counted as raw-unverified execution.

## Interpretation

S1 reaches the historical G5 safe-capture level (`75%`) without JEPA or the
ledger, while preserving zero UAV collision, boundary, and pairwise violations.
This is sufficient to pass the S1 safety gate for continued development, but
not sufficient for a multi-seed or locked conclusion. The principal remaining
failure is one strict-CBF infeasibility/controlled-abort episode. The planner
also switches routes frequently (`112` transitions over four episodes), so
route persistence and proactive braking remain the next reliability targets.

The current implementation is computationally expensive: route generation and
candidate probing dominate cycle latency. This is an engineering issue for the
next pass, not a reason to relax the safety contract.

## Reproduction

```powershell
D:\miniconda3\Scripts\conda.exe run -n uav-encirclement-gpu `
  python scripts/evaluate_dn_mpc_cbf_g5.py `
  --development-only --episodes 4 `
  --output-dir results/dn_mpc_cbf_g5_dev_seed20260907_v2 `
  --tensorboard-dir results/dn_mpc_cbf_g5_tensorboard_seed20260907_v2 `
  --cbf-horizon 3 --route-probe-horizon 3 --device auto
```

TensorBoard contains aggregate and per-episode scalars under
`Aggregate/*` and `Episode/*`; `tensorboard --inspect` confirmed all expected
tags and four episode points.

## Next gate

Do not train or attach JEPA yet. First audit the episode-2 trace, especially
the selected route, the three independent counterfactuals, the active CBF
constraints, and why every eligible progress route disappeared before the
controlled abort. Then reduce route switching with the already implemented
tangent-side persistence and rerun this same four-scene paired evaluation.
