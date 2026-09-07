# DN-MPC P3 Route Hysteresis G5 Result

**Status:** development-only; fixed G5 manifest; locked test closed

## Scope

This stage evaluates route persistence before any JEPA or Ledger-Lite
integration. The controller remains:

```text
retained actor nominal action
-> reachable obstacle-route candidates
-> independent Joint CBF probes
-> analytic DN-MPC route selection with hysteresis
-> selected/nominal/safe-hold counterfactuals
-> final strict Joint CBF
-> execute first step and replan
```

Only the planner's route-hold parameters changed. CBF margins, strict-buffer
semantics, acceleration limits, stale/OOD gates, controlled abort, and raw
unverified rejection were unchanged.

## Fixed Contract

| Item | Value |
| --- | ---: |
| Manifest | `jepa_route_recovery_dev_g1b_seed20260911/scene_manifest.jsonl` |
| Episodes | 4 |
| CBF horizon | 3 |
| Route probe horizon | 3 |
| Planner horizon | 5 |
| `minimum_hold_steps` | 3 |
| `switch_improvement_m` | 0.35 |
| `tangent_route_hold_steps` | 6 |
| JEPA / Ledger-Lite | disabled |
| Protocol | development-only |

The evaluator now accepts these three values as CLI parameters and writes them
into `summary.json`, `provenance.json`, and TensorBoard metadata. The default
values remain the v2 contract (`2`, `0.25`, `5`).

## Results

| Run | Safe capture | Route switches | Controlled abort | Fallback | Collision / boundary / pairwise | Mean capture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 baseline | 3/4 = 75.0% | 112 | 1 | 1 | 0 / 0 / 0 | 18.733 s |
| hold=5, hysteresis=0.5, tangent=8 | 2/4 = 50.0% | 80 | 0 | 0 | 0 / 0 / 0 | 18.4 s |
| **hold=3, hysteresis=0.35, tangent=6** | **4/4 = 100.0%** | **75** | **0** | **0** | **0 / 0 / 0** | **17.1 s** |

The successful v7 episodes captured in `17.9`, `20.3`, `13.6`, and `16.6`
seconds. The worst physical clearance was `0.35018 m`; target-boundary,
defender-boundary, pairwise, and raw-unverified counters were all zero.

The aggressive v6 setting reduced switches but lost two captures, so it is
rejected. The v7 setting is retained as a development candidate because it
improves both the task metric and route stability relative to v2 on the same
frozen manifest.

## Artifacts

- Output: `results/dn_mpc_cbf_g5_dev_seed20260907_v7/`
- TensorBoard: `results/dn_mpc_cbf_g5_tensorboard_seed20260907_v7/`
- Manifest SHA-256: `6ef41f43d4b37e894559a4ee78604d1f3dc0c08fe48a069e7b4140ad646d327b`
- Actor SHA-256: `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`

TensorBoard inspection confirmed aggregate and per-episode tags:
`Aggregate/safe_capture_rate`, `Aggregate/route_switch_steps`,
`Aggregate/controlled_abort_steps`, `Episode/safe_capture`,
`Episode/route_switch_steps`, and `Provenance/metadata`.

## Gate Interpretation

This passes the fixed-G5 S1/P3 development gate and justifies moving to the
next audit: replay the selected route state on additional independent scene
manifests/seeds. It does **not** justify JEPA training, a locked test, or a
claim of generalization. If the three-seed paired replay does not preserve the
direction, revert to analytic DN-MPC + strict CBF and analyze the route traces
before adding any learned evaluator.
