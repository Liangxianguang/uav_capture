# JEPA Safe Capture V5 Escape/Switch Bounded Gate Stop Report

**Date:** 2026-09-07  
**Scope:** development-only, independent L0 paired replay  
**Locked test:** not opened  
**Hardware:** NVIDIA GeForce RTX 5050

## Decision

The bounded ranking-contract experiment is stopped. It did not improve
`safe_capture` and introduced a CBF controlled abort, so no further weight
tuning, seed expansion, L1-L3 evaluation, or retraining is authorized from
this branch.

| Variant | Safe capture | Timeout | CBF controlled abort | Collision | Defender boundary | Pairwise | Raw-unverified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous M3 route ranker | 1/3 (33.3%) | 2 | 0 | 0 | 0 | 0 | 0 |
| V5 escape + switch ranker | 1/3 (33.3%) | 1 | 1 | 0 | 0 | 0 | 0 |

The M0 reference on the same manifest remains `2/3 (66.7%)`. The new M3
therefore has no capability gain and a worse execution outcome than the
previous M3 contract.

## Bounded change

The checkpoint, actor, scene manifest, CBF horizon (`5`), strict-buffer
margin (`0.35 m`), stale/OOD policy, Ledger states, and raw-action guard were
unchanged. The new protocol only enabled:

- a public-observation target escape-alignment score term (`0.75`);
- a route-switch cost (`0.15 m`) in the ranker score.

The term is computed from public `target_belief_positions` and
`target_belief_velocities`; no simulator target truth is used online.

## Result and attribution

The same three validation episodes were replayed with the same manifest hash
`130f9602a87bcc392b10b2fa8bddaeaa88dab21bae4d932f95ddfeb7647da644`.

- Episode `650101`: M0 captured; new M3 ended at step `85` with
  `cbf_controlled_abort`. The active constraints included
  `pairwise_1_3` and all four acceleration bounds; the final minimum
  constraint value was `-0.0922`. There was no collision or boundary
  violation.
- Episode `650102`: new M3 captured safely (`1/3` success block); route
  switches fell to `18` from the previous `130` in the corresponding failed
  replay.
- Episode `650103`: both M0 and M3 timed out; this remains a tied failure.

The new audit reports route-switch counts `7`, `18`, and `18`, while the
predicted-progress/next-step clearance correlations are `-0.087`, `0.289`,
and `-0.144`. The public escape term and switch penalty did not establish a
reliable capture-progress signal. The first episode also shows that the
bounded score change can steer the system into a late pairwise-feasibility
failure even though the final CBF remains the only execution authority.

## Reproducibility artifacts

- Protocol: `configs/central_random_mixed_obstacle_s3_route_v1_reacquisition_v5_escape_switch_development_protocol.yaml`
- Protocol SHA-256: `22dd92fcf4732fc887c238c8e4a10bfa3beeaf674fc61e4a345a4aa64120571f`
- Recalibrated Ledger: `results/jepa_route_identity_ledger_v5_escape_switch_seed20260912/reliability_ledger.json`
- Ledger SHA-256: `3513dda292bf841eeaa1ea13302abece5606a9cedbd509e3a6aee8b1540860c7`
- M3 summary: `results/wp1_active_search_v5_escape_switch_m3_seed20260912/summary.json`
- M3 summary SHA-256: `25724ac7825d59b3d0117b4fb4b3fb199eb0c71fa8c3bbce60c04799dc0784fb`
- TensorBoard: `results/wp1_active_search_v5_escape_switch_m3_seed20260912_tensorboard/`
- Route audit: `results/wp1_active_search_v5_escape_switch_route_ranking_audit/`
- Route-audit TensorBoard: `results/wp1_active_search_v5_escape_switch_route_ranking_audit_tensorboard/`

The M3 TensorBoard event SHA-256 is
`cd5213ca9ece4e801e105411ffb5aef095e225a3b6c6fb96d0fac7e29f7e15c9`.

## Next permitted work

Keep the expansion gate closed. The next work must be offline only:

1. Build a settled counterfactual route-regret audit for episode `650101`.
2. Separate route choice from late braking/pairwise viability using the saved
   step traces.
3. Do not enable another online score term until its offline signal is
   calibrated and its CBF feasibility impact is demonstrated on a fixed
   micro-scene.
4. If a new checkpoint is trained later, rebuild its calibration archive and
   hash-bound Ledger before any paired gate.

CBF margins, stale/OOD gates, `controlled_abort`, independent counterfactual
probes, and raw-unverified guards remain unchanged.
