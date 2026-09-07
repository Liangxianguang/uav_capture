# DN-MPC P2 Multi-Cycle Replay

**Date:** 2026-09-07

**Status:** development-only; not a safe-capture benchmark

**Config:** `configs/dn_mpc_jepa_safe_capture_development.yaml`

**Output:** `results/dn_mpc_multicycle_replay_dev_seed20260907_v2/`

**TensorBoard:** `results/dn_mpc_multicycle_replay_tensorboard_seed20260907_v2/`

## Purpose

This stage validates the analytic distributed minimax planner over multiple
replanning cycles before connecting it to JEPA or the Joint CBF execution
boundary. Each cycle uses only public target belief, public obstacle geometry,
the previous selected action, and the first action of the selected route. No
simulator future state is read and no action is executed in the environment.

The planner now exposes and records:

- route phase (`approach`, `pre_brake`, tangent, `encircle`, `intercept`, or
  `safe_hold`);
- active obstacle ID and route identity;
- terminal progress cost;
- stopping-distance cost;
- route age, switch reason, and selected score.

## Protocol

| Item | Value |
|---|---:|
| Scenarios | 4 (`open`, `single_cylinder`, `single_wall`, `mixed`) |
| Cycles per scenario | 18 |
| Total cycles | 72 |
| Planner horizon | 5 steps |
| Route candidates | 12 per cycle |
| CBF | not executed |
| JEPA | disabled |
| Ledger | disabled |
| Simulator action execution | disabled |

## Results

| Scenario | Valid-route cycles | Route switches | Switch rate |
|---|---:|---:|---:|
| open | 18/18 (100%) | 0 | 0.0% |
| single cylinder | 18/18 (100%) | 2 | 11.8% |
| single wall | 18/18 (100%) | 1 | 5.9% |
| mixed | 18/18 (100%) | 5 | 29.4% |

All 72 cycles selected a route whose reachable-dynamics and geometric checks
passed. The open case held `safe_intercept`; the single-obstacle cases moved
from `visibility_hold` to one tangent side and then held that side. The mixed
case switched between `visibility_hold`, right/upper detours, and back to
`visibility_hold`, which is too unstable for the P3 acceptance gate.

## Interpretation

The planner contract is executable and auditable, and the explicit terminal
progress/stopping-distance terms are present in both JSONL traces and
TensorBoard. However, the replay does **not** establish safe capture and does
not justify JEPA integration. The mixed-scene switch rate identifies the next
engineering task: add an explicit route state machine with active-obstacle
identity, tangent-side persistence, phase-aware switching, and a scan of
minimum hold steps and hysteresis margin. CBF margins and stale/OOD gates must
remain unchanged.

## Artifacts and audit

- `summary.json`: aggregate gate and per-scenario switch statistics.
- `cycle_traces.jsonl`: one causal public-belief record per cycle.
- `provenance.json`: source/config hashes and development-only contract.
- TensorBoard event file: route score, phase, age, switch, progress, stopping
  distance, and valid-candidate metrics.

The failed first direct invocation was isolated to an import-path issue and is
kept in the separate `_v1` directory; `_v2` is the complete replay. No
historical V3/V4/V5/G5 result or archive was modified.
