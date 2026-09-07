# DN-MPC P26 Planner-Abstention Audit

**Status:** development-only; read-only diagnosis.

P26 inspects the two P25 runtime groups that have no selected candidate. It
does not change CBF margins, candidate generation, fallback behavior, or
`controlled_abort` semantics.

## Results

| Split | runtime groups | zero-eligible groups | minimum eligible candidates | missing trace groups |
|---|---:|---:|---:|---:|
| train | 137 | 1 | 0 | 1 |
| validation | 172 | 0 | 2 | 0 |
| calibration | 214 | 1 | 0 | 1 |

The complete JSON and TensorBoard records are:

- `results/dn_mpc_jepa_safe_capture_dev/p26_abstention_audit/audit.json`
- `results/dn_mpc_jepa_safe_capture_dev/p26_abstention_audit/AUDIT.md`
- `results/dn_mpc_jepa_safe_capture_tensorboard/p26_abstention_audit`

## Root Cause

Both abstention groups contain all 12 route candidates, but every candidate is
rejected by the first-step CBF feasibility check. Geometry is invalid for only
some detours; the decisive common failure is first-step CBF infeasibility,
including nominal, braking, verified safe-hold, and the detour candidates.

| Split | scenario index | time index | geometry-invalid candidates | first-step-CBF-infeasible candidates |
|---|---:|---:|---:|---:|
| train | 6 | 47 | 3 | 12 |
| calibration | 5 | 47 | 3 (left/right/lower) | 12 |

This is a genuine planner abstention state: the current state has already
entered a joint CBF-infeasible region. It is not evidence that JEPA selected a
bad route, and it must not be “fixed” by fabricating a selected candidate or
loosening the safety contract.

## Decision

- Preserve `controlled_abort` when the eligible set is empty.
- Use the group as a hard-negative diagnostic for anticipatory route planning,
  especially earlier braking and tangent commitment.
- Keep the strict CBF margin, stale/OOD/non-finite gates, and raw-unverified
  prohibition unchanged.
- Do not open online JEPA promotion or a three-seed replay yet; exact top-1
  and fresh OOD/disagreement calibration remain unmet.

