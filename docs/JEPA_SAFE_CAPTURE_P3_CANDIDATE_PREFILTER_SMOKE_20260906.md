# P3 Candidate-CBF Prefilter Development Smoke

Date: 2026-09-06
Status: development-only evidence; not a locked-test result

## Purpose

This smoke evaluates the extended 12-candidate profile after wiring an
independent primary Joint CBF-QP feasibility probe before JEPA ranking. The
probe is read-only: it does not execute an action and does not invoke nominal,
safe-hold, or controlled-abort fallback. The selected first step still passes
through the ordinary execution-time CBF filter.

The A1 and M0 runs use the same eight-scene manifest:

`results/jepa_safe_capture_v3_extended_smoke_a1_seed20260911/scene_manifest.jsonl`

Both runs use the development-only extended collection protocol, the frozen
actor checkpoint, CUDA on the NVIDIA GeForce RTX 5050, and
`locked_test_opened=false`.

## Results

| Variant | Safe capture | Transit | CBF abort steps | Collision | Boundary | Pairwise | Raw-unverified | Prefilter checks | Prefilter rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0, CBF actor path | 3/8 (37.5%) | 87.5% | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| A1, JEPA + CBF, extended-v1 + prefilter | 2/8 (25.0%) | 87.5% | 5 | 0 | 0 | 0 | 0 | 6,696 | 60 (0.90%) |

A1 termination reasons were five `cbf_controlled_abort`, two
`safe_capture`, and one `timeout`. M0 terminated with five
`cbf_controlled_abort` and three `safe_capture` episodes. A1's candidate
prefilter had no timeout and accepted 6,636 of 6,696 checked candidates.

The prefilter therefore did not improve task success in this smoke. The
observed difference is -12.5 percentage points for A1 versus M0 on this
manifest, while the hard safety counters remain unchanged. This is a
development diagnostic, not evidence against the architecture in general.

## Interpretation

The extended action library and independent CBF eligibility gate are operating
as specified, but only 0.9% of otherwise valid candidates are removed. They
cannot explain the dominant failure mode. The shared five controlled aborts
show that the bottleneck remains execution-contract feasibility/termination or
candidate ranking, rather than physical collision protection. The A1 run also
has a higher candidate-generation cost because twelve candidates are probed
at every control cycle.

No safety contract was relaxed: CBF margins, stale/OOD/non-finite gates, and
`controlled_abort` are unchanged; no raw-unverified action was executed.

## Earliest-abort audit

The five A1 abort cycles were `episode_0000/step 32`,
`episode_0001/step 37`, `episode_0002/step 35`, `episode_0004/step 13`, and
`episode_0007/step 49`. At each cycle all 12 probes returned
`solver_status=solver_failure` (zero probe timeouts), so the prefilter exposed
an already infeasible CBF state rather than creating a new unsafe action. The
execution-time CBF then followed the existing `safe_hold -> nominal ->
controlled_abort` path. The corresponding A1 run without prefilter had the
same five abort episode indices and steps, but selected a different candidate
before the final CBF abort. This is evidence that the prefilter is diagnostic,
not the cause of the aborts.

The ranker trace now preserves `cbf_infeasible` and `cbf_timeout` rejection
reasons in `candidate_eligibility_reasons`, so future failure indexes can
distinguish candidate-generation invalidity from JEPA score eligibility.

## Artifacts

- A1 output: `results/jepa_safe_capture_v3_extended_prefilter_a1_seed20260911/`
- M0 output: `results/jepa_safe_capture_v3_extended_prefilter_m0_seed20260911/`
- A1 TensorBoard: `results/jepa_safe_capture_v3_extended_tensorboard/prefilter_a1_seed20260911/`
- M0 TensorBoard: `results/jepa_safe_capture_v3_extended_tensorboard/prefilter_m0_seed20260911/`
- Extended protocol: `configs/jepa_safe_capture_l0_l3_collection_v3_extended_candidates.yaml`

Each output contains `summary.json`, `provenance.json`, `episodes.csv`, the
scene manifest, step traces, and a TensorBoard event file with candidate-CBF
checks, rejections, timeouts, and accepted counts.

## Next gate

Do not expand this candidate profile to a three-seed claim yet. First index the
earliest abort and ranking mismatch for the A1/M0 paired traces, verify the
nominal anchor and recurrent-reset contract, and determine whether a
task-directed candidate score or a reachable-action block can reduce aborts
without changing safety gates. Only after that diagnosis should a new
calibration archive and hash-bound reliability ledger be generated.
