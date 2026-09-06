# JEPA Safe-Capture Extended Prefilter Diagnosis

Date: 2026-09-06  
Scope: development-only; `locked_test_opened=false`  
Hardware: NVIDIA GeForce RTX 5050, CUDA 13.0, PyTorch 2.9.1+cu130

## Executive conclusion

The latest result is not a GPU or TensorBoard problem, and it is not evidence
that the CBF safety margin is too large. The current JEPA path improves a
prediction signal, but the control score is not yet coupled to the failure that
ends the episode. The dominant endpoint is a late CBF infeasibility followed by
the required `controlled_abort`.

The latest paired development smoke used one identical eight-scene manifest:

| Variant | Safe capture | CBF aborts | Collision | Boundary | Pairwise | Raw-unverified |
|---|---:|---:|---:|---:|---:|---:|
| A1: extended JEPA + CBF + auxiliary score + candidate prefilter | 2/8 (25.0%) | 5 | 0 | 0 | 0 | 0 |
| M0: nominal actor + CBF | 3/8 (37.5%) | 5 | 0 | 0 | 0 | 0 |

Outputs:

- `results/jepa_safe_capture_v3_extended_prefilter_a1_seed20260911/`
- `results/jepa_safe_capture_v3_extended_prefilter_m0_seed20260911/`
- TensorBoard: `results/jepa_safe_capture_v3_extended_tensorboard/prefilter_a1_seed20260911/`
- TensorBoard: `results/jepa_safe_capture_v3_extended_tensorboard/prefilter_m0_seed20260911/`

The prefilter performed 6,696 independent probes and rejected 60 (0.9%); no
probe timed out. Thus it was active, but it changed almost none of the normal
decision cycles and could not increase task progress.

## What the traces show

### The failure is late feasibility loss

In A1, the five abort episodes terminate at steps 13, 32, 35, 37 and 49. In
the preceding cycles, the selected candidate is still reported as trusted and
the final CBF solve succeeds, often with a growing correction. At the abort
cycle, all twelve candidates fail the independent primary probe with
`solver_failure`; the final CBF diagnostic reports a negative constraint slack
and enters `controlled_abort`.

Representative A1 values immediately before abort include predicted minimum
clearance of approximately 0.68--0.81 m, followed by a CBF correction of
0.13--0.61 m/s and then a negative boundary or obstacle slack. The predicted
clearance and the final CBF barrier are therefore measuring different risks.

The failing constraints are mostly:

- `boundary_upper_defender_*_axis_0` in the open, single-obstacle, nominal and
  S3 episodes;
- obstacle constraints in the partial-observation episode.

This also occurs in L0/open scenes, so scene difficulty contributes but is not
the primary explanation. The controller can drive toward a world boundary even
when there is no obstacle.

### The prefilter is a safety gate, not a progress optimizer

`_prefilter_candidate_batch_with_cbf` checks only the requested first action at
the current observation. It does not roll the three-step candidate chunk through
future observations. At the final abort cycle, all candidates are already
infeasible; removing them cannot create a feasible action or recover capture.
The final `filter` call remains the execution boundary, as required.

### The extended candidates are out of distribution

The extended runtime protocol has twelve labels (`extended_v1`), while the
checkpoint was trained under the V3 contract with `candidate_count: 5` (the
training protocol allows five or nine). The model has no training evidence for
the new label semantics. The candidates are also constant three-step chunks
with a 0.10 m/s perturbation around one nominal actor action, so adding labels
does not provide a learned, horizon-aware planner.

### The training labels cannot teach boundary/future-feasibility avoidance

The training archive used by the hard-replay checkpoint reports:

- `labels_cbf_qp_feasible`: mean 1.0 in train and approximately 0.9998 in
  validation;
- `labels_boundary`: mean 0.0 and nonzero fraction 0.0;
- `labels_collision`: mean 0.0 and nonzero fraction 0.0;
- hard-replay `collision_or_boundary_samples`: 0.

The current predictor has obstacle and inter-agent clearance heads, but no
boundary-clearance head. The ranker computes `predicted_min_clearance_m` from
only obstacle and inter-agent outputs. It also does not use a horizon-level
CBF feasibility rollout. Consequently, a candidate may receive a good learned
clearance score while its physical trajectory is approaching the boundary.
The CBF then correctly refuses the action, but that refusal is counted as a
failed safe capture.

### The latest run is not the full ledger pipeline

The reported A1 variant has `use_ledger=false`. It is the no-ledger ablation,
not the complete `JEPA + Reliability Ledger + CBF` variant. A full extended
pipeline cannot reuse the old ledger because the candidate protocol and
checkpoint/calibration hashes would not match. The older three-seed full-flow
evidence already showed M3 45.8% versus M0 46.9%, so enabling the ledger alone
is not expected to fix the missing boundary/future-feasibility signal.

## Root-cause ranking

1. **Primary:** late CBF infeasibility caused by missing boundary and future
   feasibility terms in the learned ranking objective.
2. **Primary:** training distribution has no negative feasibility/boundary
   examples, making the feasibility auxiliary signal non-discriminative.
3. **Secondary:** learned ranking remains mismatched to settled episode
   outcomes; prior V21 audits measured high selected-not-best rates and
   negative score/outcome correlation.
4. **Secondary:** twelve untrained extended candidates are being evaluated by a
   checkpoint trained for the five-candidate contract.
5. **Not primary:** RTX 5050 execution. A1 is near the 100 ms cycle p95 budget
   in some episodes, but the failure is a safety abort, not an inference
   crash or raw-action execution.

## Required next revision

The next revision should not lower CBF margins, disable stale/OOD/non-finite
gates, remove `controlled_abort`, or expand the seed matrix yet.

1. Generate a new train-only counterfactual archive from earliest-abort and
   near-boundary states. Include requested-action outcomes, negative primary
   CBF feasibility, future boundary clearance, obstacle/inter-agent clearance,
   TTC, visibility and correction magnitude. Keep target truth offline-only.
2. Add a boundary-clearance head and a horizon-level CBF-feasibility output;
   train and validate them on the new archive. A deterministic boundary penalty
   from the online geometry may be used in ranking, but it must not replace the
   final Joint CBF-QP.
3. Make candidate probing horizon-aware: independently evaluate each candidate
   over the fixed chunk/replan horizon and record the minimum slack and earliest
   solver failure. Keep the final CBF filter as the sole execution boundary.
4. Freeze a new protocol with a candidate profile represented in training,
   retrain three seeds, rebuild calibration archives and hash-bound ledgers,
   then run a new M3-versus-M0 paired smoke on a fresh shared manifest.
5. Promote only if the safety hard gate passes, at least two of three paired
   seed deltas are non-negative, and settled ranking no longer shows systematic
   reversal. Otherwise retain the negative result as
   `prediction_signal_no_control_gain`.

This diagnosis is development evidence only; it does not alter any V4/V5
locked result and does not open a locked split.
