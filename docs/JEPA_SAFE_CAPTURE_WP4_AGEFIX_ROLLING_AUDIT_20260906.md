# JEPA Safe-Capture WP4 Age-Fix Rolling Audit

Date: 2026-09-06
Status: development-only; no locked test was opened.

## Scope

This audit reruns the rolling-horizon and Joint CBF execution contract after
the explicit `message_age` state-machine fix in commit `bca003a`. The frozen
20-episode validation manifest, actor checkpoint, JEPA checkpoint, ledger and
protocol are unchanged. The two repeated runs for each device use identical
inputs and are compared after removing wall-clock latency fields.

## Inputs

- Protocol: `configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml`
- Scene manifest SHA-256: `6a5fa0905a6b8391993fba3335452d1f0f3f1b8670749b45346a5ff71e3470ba`
- JEPA checkpoint SHA-256: `2317a9464f8001f27a5c028bb6b4c431c904af7bfc33bf43b3a1d05a5a9c6154`
- Ledger SHA-256: `30e65cb060176482c5f47f1e76bdacbf34aa0379b7757e7d25be60491b65e407`
- Code revision: `bca003a36d92e3ad0124e723895b862b85c4dd87`
- Contract: five reachable candidates, three-step chunks, first-step-only
  execution, same Joint CBF-QP, and no raw-unverified execution.

## Results

| Device | Repeats | Episodes/run | Control cycles/run | Safe capture | Collision | Boundary | Pairwise | Raw unverified | Trace differences | Cycle p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 2 | 20 | 1746 | 10/20 (50.0%) | 0 | 0 | 0 | 0 | 0 | 14.52 ms |
| RTX 5050 CUDA | 2 | 20 | 1746 | 10/20 (50.0%) | 0 | 0 | 0 | 0 | 0 | 23.84 ms |

The CPU comparison is in
`results/jepa_safe_capture_v21_wp4_agefix_rolling_audit_seed20260911/`.
The CUDA comparison is in
`results/jepa_safe_capture_v21_wp4_agefix_cuda_rolling_audit_seed20260911/`.
Both audits report `all_gates_pass=true`, `field_difference_count=0`, and
`repeat_decision_trace_equal=true`. Each evaluation run also recorded eight
CBF controlled-abort steps with verified fallback, while keeping
`raw_unverified_executed_steps=0`.

## Message-age state audit

The repaired state machine audit at
`results/jepa_safe_capture_v21_wp2_message_age_state_machine_bca003a/`
reports `all_cases_passed=true`. A stream that has never received a packet is
now represented as `never_received`, while an expired accepted packet is
represented as `saturated`; the bounded numeric age remains available for the
frozen actor input. The corresponding ledger route is explicit
`observation_never_received -> safe_hold`.

## Interpretation and boundary

WP4 is passed for the tested 100/500-cycle coverage, first-step replan
semantics, deterministic replay, CBF fallback ordering, zero safety events,
and the 100 ms cycle-latency gate. The total-cycle count is not evidence that
all hard-context categories were covered. The `50.0%` safe-capture rate is a
development observation only; this report makes no JEPA improvement claim and
does not replace the V4 locked test.

The age fix changes the current reproducible M3 result from the prior stale
trace's `60.0%` to `50.0%` on this paired manifest. This is why all subsequent
smoke and training evidence must use the new code revision and newly generated
provenance rather than mixing old traces with repaired traces.

## TensorBoard

The CPU and CUDA rolling audits each write a dedicated TensorBoard event file
under `results/jepa_safe_capture_v21_tensorboard/`, including configuration,
provenance, replay equality, safety counters and p50/p95/p99 latency tags.

Next permitted step: run the new three-seed paired 20-episode smoke with fresh
outputs and the repaired message-age fields. Do not open a locked split or
change CBF margins, stale/OOD gates, abstention semantics or controlled abort.
