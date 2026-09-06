# WP1 Cautious Reacquisition Stop Report

**Date:** 2026-09-06
**Scope:** development-only, fixed WP1 three-scene replay
**Locked test:** not opened

## Decision

The bounded `cautious_reacquisition` contract is implemented and passed its
single-step/unit safety contract, but it is **stopped at the development gate**.
It did not improve episode-level `safe_capture` on the frozen replay, so no
larger route matrix, new archive, or JEPA retraining is started from this
change.

## Contract Added

When the target is not visible and the normal nominal anchor is unavailable,
the ranker may select only the explicitly named `visibility_hold` route. The
route remains Ledger-`safe_hold`; it is not relabeled `trusted`. It is bounded
to three steps and must pass reachable projection, the primary candidate CBF
prefilter, and independent selected/nominal/safe-hold CBF probes. Any probe
failure returns to explicit `safe_hold`. Existing stale/OOD/non-finite gates,
CBF margins, and `controlled_abort` are unchanged.

Protocol: `configs/central_random_mixed_obstacle_s3_route_v1_reacquisition_v2_development_protocol.yaml`

## Paired Replay

The same actor, route JEPA checkpoint, scene manifest, `horizon=5`, strict
buffer, route sampling `9`, and episode seeds `646101/646102/646103` were used.

| Run | Safe capture | Timeout | Collision | Boundary | Pairwise | Raw-unverified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous M3 | `2/3` | `1` | `0` | `0` | `0` | `0` |
| M3 + bounded reacquisition | `2/3` | `1` | `0` | `0` | `0` | `0` |

The important episode `646102` remained a timeout. It executed exactly three
`visibility_hold` steps, all three independent probe sets passed, then returned
to `safe_hold` for the remaining 247 steps. Target visibility remained zero;
there was no observation reacquisition and no capture. This attributes the
remaining failure to **visibility/reacquisition route effectiveness and the
observation contract**, not to a CBF safety rejection.

Aggregate safety checks in the new run were strict: independent probes
`1260/1260` accepted, candidate prefilter `2327/2327` accepted, and all four
hard safety counters were zero. The run therefore demonstrates safety
preservation but no control-capability gain (`prediction_signal_no_control_gain`).

## TensorBoard and Provenance

- Replay TensorBoard: `results/wp1_route_replay_m3_reacquisition_v2_seed20260911_tensorboard`
- Recalibration TensorBoard: `results/jepa_route_identity_ledger_reacquisition_v2_seed661606_tb`
- Replay summary: `results/wp1_route_replay_m3_reacquisition_v2_seed20260911/summary.json`
- Episode trace: `results/wp1_route_replay_m3_reacquisition_v2_seed20260911/step_traces/episode_0001.jsonl`
- Recalibrated Ledger: `results/jepa_route_identity_ledger_reacquisition_v2_seed661606/reliability_ledger.json`

The generated result directories are local development artifacts and remain
ignored; their hashes and paths are recorded in the replay `provenance.json`.

## Next Action

Do not increase the reacquisition duration or add data yet. The next permitted
investigation is an observation-contract audit: determine whether the public
observation can provide a causal bearing/occupancy cue for a target that has
never been received. Only after a deterministic micro-scene shows that the
route can restore visibility should this contract be reconsidered.
