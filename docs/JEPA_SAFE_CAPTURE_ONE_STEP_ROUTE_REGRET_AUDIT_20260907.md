# One-Step Settled Route-Regret Audit

**Date:** 2026-09-07  
**Scope:** development-only offline counterfactual audit  
**Episode:** `650101` from the V5 escape/switch bounded gate  
**Locked test:** not opened

## Purpose

The previous paired gate showed that JEPA selected its own score argmin while
M3 still regressed against M0. This audit replays the saved environment to
each decision state and branches every geometrically valid, primary-CBF-
accepted route for one environment step. Simulator target state is used only
to settle the offline label; it is not available to the online controller.

This is a **one-step** settled route comparison. It is not a claim that the
best one-step action is the best full-episode route.

## Result

Only steps whose online ranker was actually in `trusted` mode are included in
the ranking comparison. Ledger `safe_hold` steps are kept as a separate
routing outcome and are not mislabeled as route-ranking failures.

| Metric | Value |
| --- | ---: |
| Trusted comparable steps | `80` |
| Selected route = one-step settled best | `63.75%` |
| Mean selected regret | `0.0204 m` |
| Maximum selected regret | `0.1047 m` |
| JEPA score argmin = online selected route | `100%` |
| Selected route primary CBF failure | `0` steps |
| Final no-primary-candidate state | `1` step (step `85`) |

The ranker is therefore internally consistent but externally misaligned: it
faithfully executes its score argmin, while that argmin is not a reliable
proxy for immediate target-distance progress. The largest sustained mismatch
was steps `46-69`, where `formation_split` was selected while `nominal` or
`safe_intercept` was the one-step settled best, with regret up to about
`0.10 m`.

At step `85`, the online state was `safe_hold`; no candidate passed the
primary CBF probe. The saved trace reports simultaneous `pairwise_1_3` and
four acceleration constraints with a minimum constraint value of `-0.0922`.
This is a late feasibility failure, not an unverified route execution.

## Interpretation

The evidence supports the following failure decomposition:

1. **JEPA/ranker objective mismatch:** `63.75%` one-step route agreement and
   `100%` score-argmin agreement show that the learned score is not aligned
   with the settled target-progress label.
2. **CBF late feasibility:** the final abort occurs after the route-selection
   stage has lost all primary-feasible candidates; relaxing CBF or executing a
   raw action would violate the safety contract.
3. **Ledger separation:** the four initial `safe_hold` steps are information
   routing outcomes and are not counted as route-regret samples.

No new online score term, checkpoint, seed, or training run should be enabled
until this label is calibrated on additional fixed micro-scenes. The next
permitted change remains offline route-progress/escape calibration followed
by a new paired L0 gate.

## Reproducibility artifacts

- Script: [audit_jepa_safe_capture_one_step_route_regret.py](../scripts/audit_jepa_safe_capture_one_step_route_regret.py)
- Source run: `results/wp1_active_search_v5_escape_switch_m3_seed20260912/`
- Source trace SHA-256: `45ac71c2f13e3de8193b7b8ab240dba639200696dede83702c510209cf3340f3`
- Source manifest SHA-256: `130f9602a87bcc392b10b2fa8bddaeaa88dab21bae4d932f95ddfeb7647da644`
- Settled audit JSON: `results/wp1_active_search_v5_escape_switch_one_step_route_regret_e0_v2/route_regret.json`
- Settled audit JSON SHA-256: `71365e501dff22ff21b68d9022b7790bbbfb6c0af539bcd09d6700d36590bbeb`
- TensorBoard event: `results/wp1_active_search_v5_escape_switch_one_step_route_regret_e0_v2_tensorboard/`
- TensorBoard event SHA-256: `5975c02b13546cf6263a17e5e31903ea651635838ec690a1cd11e6c5c2b699b0`

The audit directory contains the complete per-step, per-candidate branch
outcomes and provenance source hashes.
