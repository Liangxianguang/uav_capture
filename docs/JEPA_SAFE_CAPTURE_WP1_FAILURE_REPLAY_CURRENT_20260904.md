# WP-B2 Deterministic Failure Replay

**Status:** development-only; `locked_test_opened=false`  
**Input:** `results/jepa_safe_capture_v3_wp1_failure_index_current/`  
**Output:** `results/jepa_safe_capture_v3_wp1_failure_replay_current/`  
**TensorBoard:** `results/jepa_safe_capture_v3_tensorboard/wp1_failure_replay_current/`

This is a read-only trace audit. It does not retrain, regenerate an environment
rollout, or open the locked split. The implementation validates the V3 index,
source summary/provenance/scene-manifest hashes, training and episode seeds,
trace identity, finite action arrays, candidate masks/scores, ledger state and
credit, CBF verification, and termination semantics.

Six failure categories were selected deterministically, three episodes per
category, with no shortage: candidate capture regression (45 available), high
credit failure (121), nominal fallback (35), candidate oscillation (47), stale
or noisy observation (151), and timeout (8). The 18 selected episodes contain
14 CBF controlled aborts and 4 timeouts.

Every reduced trace was emitted twice as canonical sorted-key JSON. All 18/18
episode pairs have identical SHA-256 values; the full list is in
`results/jepa_safe_capture_v3_wp1_failure_replay_current/hash_manifest.json`.
TensorBoard reload found 74 scalar tags and the required configuration,
failure-index provenance, source-run provenance, and selection-policy text
tags. Target-boundary diagnostics remain separate from defender safety.

The replay is an audit of recorded evidence, not a new success-rate estimate or
a causal proof of future target drift. Any unresolved cause must be addressed
with offline labels in the next calibration stage.
