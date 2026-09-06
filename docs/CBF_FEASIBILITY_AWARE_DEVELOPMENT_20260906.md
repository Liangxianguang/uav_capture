# Feasibility-Aware CBF Development Audit (2026-09-06)

## Scope

This report evaluates whether the current Joint CBF-QP is rejecting
physically recoverable pursuit states because an operational clearance buffer
and a first-order discrete CBF condition are treated as the same hard set.
This is a development-only audit. It does not modify the historical V4/V5
locked contracts and it does not open a locked test.

The safety requirements remain unchanged at the physical level:

- no defender collision or world-boundary violation;
- no pairwise violation;
- no execution of raw or unverified actions;
- non-finite, stale/OOD, timeout, and infeasible paths remain explicit;
- `controlled_abort` remains available and is still counted as unsuccessful.

## Literature Check

The supplied references support separating mission progress from the safety
invariant, but they do not justify disabling a safety filter:

| Reference | Verified contribution | Relevance |
| --- | --- | --- |
| Yin et al., *Safe Cooperative Pursuit for Multi-UAV Systems Based on Control Barrier Functions and Neurodynamic Optimization*, IEEE TII (2025), DOI `10.1109/TII.2025.3598518` | Cooperative pursuit with CBF constraints and an optimization layer | Supports task-aware cooperative pursuit with a separate safety layer |
| Lv et al., *Control Barrier Function-Based Collision Avoidance Guidance Strategy for Multi-Fixed-Wing UAV Pursuit-Evasion Environment*, Drones 8(8):415 (2024), DOI `10.3390/drones8080415` | CBF guidance for pursuit-evasion and collision avoidance | Supports using CBF for the inner guidance loop |
| Peng et al., *Design of Safe Optimal Guidance With Obstacle Avoidance Using Control Barrier Function-Based Actor-Critic Reinforcement Learning*, IEEE TSMC (2023), DOI `10.1109/TSMC.2023.3288826` | Learned guidance command followed by CBF-based safety assurance | Matches the JEPA/ranker -> CBF decomposition |
| Zhong et al., *A Feasibility-Enhanced Control Barrier Function Method for Multi-UAV Collision Avoidance*, arXiv:2603.13103 (2026) | Internal-compatibility analysis, sign-consistency constraints, worst-case estimates and slack variables | Directly relevant to the observed all-candidates-infeasible QP failure |
| Sorensen et al., *A Temporal Barrier Framework for Collision Avoidance in Multi-Agent Autonomous Aerial Vehicles*, arXiv:2608.14239 (2026) | Adversarial time-to-collision barrier and a differentiable real-time surrogate | Motivates time/viability quantities instead of distance-only early rejection |
| Acharya et al., *Capture, Shield, or Neutralize: Engagement-Aware Pursuit-Evasion*, arXiv:2607.10986 (2026) | Minimax receding-horizon planning with an inner discrete-time CBF | Supports engagement-aware planning outside the hard safety filter |

One supplied identifier needs correction: arXiv:2605.12735 is *The Unified
Autonomy Stack: Toward a Blueprint for Generalizable Robot Autonomy*, not a
paper titled *Composite Control Barrier Functions for Last Resort Safety
Filtering*. That title/identifier pairing should not be cited without a
separate source check. Likewise, Crossref did not resolve
`10.1109/TII.2025.3496271` for the supplied Yin title; the exact-title
Crossref record is `10.1109/TII.2025.3598518`.

## Implementation

`JointCBFQPSafetyFilter` now has an explicit `barrier_mode`:

- `strict_buffer`: historical behavior. The configured `safety_margin` is a
  hard CBF margin for obstacle, boundary, and pairwise rows.
- `physical_feasibility`: the physical invariant uses only the actual drone
  radius and world/obstacle geometry. The configured `safety_margin` remains
  an operational buffer for diagnostics and future ranking costs; it is not
  silently removed from the contract.

The default remains `strict_buffer`, so existing evaluations are unchanged.
The new mode is selected explicitly with
`--cbf-barrier-mode physical_feasibility` in the development L0-L3 evaluator.
Every selected action still passes the same reachable-dynamics projection,
Joint CBF-QP, residual verification, and fallback policy.

## Counterfactual Result

The comparison uses the same scene manifest
`results/l0_l3_r2_full_runs/jepa_safe_capture_l0_l3_paired_full_seed20260911_m0/scene_manifest.jsonl`,
the same recovered actor checkpoint, and `recurrent_reset_interval=1`.
Neither run is a locked test.

Artifacts:

- strict-buffer recovery baseline: `results/l0_recovery_paired_m0_seed661606_reset1/`;
- physical-feasibility run: `results/l0_physical_cbf_m0_seed661606/`;
- physical-feasibility TensorBoard: `results/l0_physical_cbf_tb/m0_seed661606/`.

| Level | strict-buffer M0 | physical-feasibility M0 |
| --- | ---: | ---: |
| L0 open | 4/8 (50%) | 6/8 (75%) |
| L0 single obstacle | 0/8 (0%) | 0/8 (0%) |
| L1 nominal | 1/8 (12.5%) | 1/8 (12.5%) |
| L1 S-curve | 0/8 (0%) | 0/8 (0%) |
| L2 delayed/noisy | 0/8 (0%) | 0/8 (0%) |
| L2 partial observation | 0/8 (0%) | 0/8 (0%) |
| L3 mixed obstacle | 0/8 (0%) | 0/8 (0%) |
| L3 S3 stress | 0/8 (0%) | 1/8 (12.5%) |
| **Overall** | **5/64 (7.8%)** | **8/64 (12.5%)** |

For the physical-feasibility run:

- collision count: `0`;
- defender boundary violation count: `0`;
- pairwise violation count: `0`;
- raw-unverified executed steps: `0`;
- controlled-abort steps: `17`;
- transit success: `100%`.

The run's minimum reported clearance reaches the physical boundary (`0`)
and pairwise physical contact (`approximately 0`), while remaining
non-negative in the environment safety accounting. This confirms that the
new mode is less conservative with respect to the operational buffer; it is
not evidence that the complex scenarios are solved.

## Diagnosis

The original L0 open trace reached a state with a defender at approximately
`x=-7.35 m` and `v_x=-4.84 m/s`. With the 0.35 m operational buffer, the
one-step CBF bound required approximately `v_x >= -3.23 m/s`, while the
6 m/s^2 acceleration limit could only change the velocity to about `-4.24
m/s`. The QP was therefore correctly infeasible for the *buffered* set, but
the physical boundary still left enough stopping distance. The strict
condition converted this recoverable situation into `controlled_abort`.

The new mode removes this category of false mission failure while preserving
the actual physical invariant. It does not fix cases where the actor has
collapsed the formation to near contact: those remain genuine pairwise
feasibility failures and require earlier interaction-aware formation/routing
actions.

## Required Next Experiments

1. Keep `strict_buffer` as the historical baseline and report
   `physical_feasibility` as a separately named development ablation.
2. Add buffer-clearance and time-to-boundary diagnostics to every trace; do
   not report physical-feasibility results as if the 0.35 m operational
   buffer were guaranteed.
3. Add a feasibility-aware candidate objective: maximize predicted capture
   progress subject to the physical CBF, while penalizing buffer incursions
   and time-to-contact. The ranker must never execute an unverified action.
4. Add interaction-aware formation actions before the pairwise barrier becomes
   active. Candidate-level CBF probes should record whether failure is due to
   boundary, obstacle, pairwise, or motion-ball compatibility.
5. Rebuild calibration and the reliability ledger for every actor/JEPA/CBF
   contract. Do not reuse a strict-buffer ledger for physical-feasibility
   rows.
6. Only after L0 clean M0 is stable should the complete JEPA + ledger +
   rolling-horizon M3 matrix be rerun. No locked test is opened by this audit.
