# WP-6 CPU/RTX 5050 Rolling-Horizon Replay Audit

**Date:** 2026-09-04  
**Status:** development-only; `locked_test_opened=false`  
**CUDA run:** `results/jepa_safe_capture_v3_wp6_replay_m3_seed20260911_cuda/`  
**CPU run:** `results/jepa_safe_capture_v3_wp6_replay_m3_seed20260911_cpu/`  
**Audit:** `results/jepa_safe_capture_v3_wp6_device_replay_audit_current_next/`

## Rollout Evidence

Both runs used the same validation scene manifest, actor checkpoint, JEPA
checkpoint, reliability ledger, protocol, and environment configuration. Each
ran 20 episodes with M3 (`Interaction-aware JEPA + ledger + auxiliary score +
Joint CBF`) and recorded complete summaries, episode tables, step traces,
provenance, and TensorBoard events.

CUDA and CPU both produced `8/20` safe captures (40.0%), zero collision,
defender-boundary, target-boundary, and pairwise violations, 11 CBF controlled
aborts, and one timeout. Maximum per-run CBF p95 latency was 27.37 ms (CUDA)
and 41.29 ms (CPU), below the 100 ms contract.

## Equivalence Result

All 20/20 settled safety outcomes were identical, CBF verification counts were
identical, all actions were finite, and raw/unverified execution was zero in
both runs. The new `candidate_rejection_reasons` field was present at all 820
paired ranking steps.

Candidate decisions differed at 9/820 steps across 5 episodes, caused by
floating-point score ties near the ranking margin. This did not change any
settled safety outcome, but it is a reproducibility risk for task performance.
The result is therefore classified
`cpu_cuda_safety_settlement_equivalent_decision_drift`, not exact decision
equivalence. Before a final block, either a deterministic score tie policy or a
documented score quantization rule must be frozen and evaluated in a new
protocol revision. No ranking weight was changed based on this audit.

Machine-readable evidence is in
`results/jepa_safe_capture_v3_wp6_device_replay_audit_current_next/device_replay_audit.json`,
with TensorBoard provenance and a hash manifest in the same output family.
