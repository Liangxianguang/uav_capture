# JEPA Safe-Capture WP2 Execution Report

Date: 2026-09-06
Status: development-only; `locked_test_opened=false`

## Scope

WP2 validates the execution contract between the action-conditioned JEPA
ranker, the reliability ledger, and the rolling-horizon safety path.  It does
not claim an episode-level capture improvement and it does not alter any
frozen V4/V5 result.  The audit inputs remain read-only.

## Evidence and provenance

| Audit | Output | TensorBoard | Result |
|---|---|---|---|
| Synthetic monotonic score suite | `results/jepa_safe_capture_v21_wp2_monotonic_r1/` | `results/jepa_safe_capture_v21_wp2_monotonic_tensorboard_r1/` | 7/7 cases pass |
| Candidate ranking diagnosis | `results/jepa_safe_capture_v21_wp2_ranking_diagnosis_r1/` | `results/jepa_safe_capture_v21_wp2_ranking_tensorboard_r1/` | completed; ranking unresolved |
| Abstention counterfactual | `results/jepa_safe_capture_v21_wp2_abstention_r1/` | `results/jepa_safe_capture_v21_wp2_abstention_tensorboard_r1/` | completed; offline-only |
| Age state machine | `results/jepa_safe_capture_v21_wp2_message_age_state_machine_r3/` | `results/jepa_safe_capture_v21_wp2_message_age_tensorboard_r3/` | all synthetic transitions pass |

The V21 protocol used by the ranking/abstention audits has SHA-256
`278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014`.
The monotonic suite and candidate-separation revision retain their own
recorded protocol hashes in their output provenance.  Every result is
development-only and no locked split was opened.

## Results

### Score contract

The monotonic suite passed task progress, uncertainty, clearance eligibility,
visibility, TTC, CBF-risk ordering, and fixed-point tie breaking.  The result
proves the synthetic score direction and deterministic tie rule only; it is
not evidence that the learned score matches settled episode outcomes.

### Ranking diagnosis

The three frozen V21 seed traces contain 1,167, 935, and 1,133 decision rows.
The all-candidate-ineligible rates are 60.4%, 42.0%, and 21.6% respectively.
The recorded policy selected a settled-best candidate in only about 9.5%,
6.9%, and 7.0% of rows.  These values identify eligibility, nominal-anchor,
and fallback pressure as the immediate control bottleneck; they do not imply
that the score sign is inverted.

### Abstention counterfactual

Across 3,235 rows, 1,878 were multi-eligible.  The recorded route and the
eligible score argmin agreed on 10.9% of rows.  Recorded selected-not-best was
92.4%, while the score-argmin counterfactual reduced it to 42.0%; score argmin
matched the settled-best oracle on 58.0% of rows.  Removing abstention in this
offline diagnostic did not improve settled safe-capture precision and reduced
local settled safety, so abstention must not be disabled online.

### Message-age state machine

The previous implementation used the numeric ceiling `60` for both
“never received” and “received but old”, and did not advance age while a
visible target's delayed packet was pending.  The repaired environment now
retains the bounded numeric field for frozen actor compatibility and exposes:

- `message_received` and `message_age_state`;
- `target_observation_received` and `target_observation_age_state`;
- explicit `never_received`, `fresh`, `delayed`, and `saturated` states;
- packet-delivery-based age increments, including visible-but-undelivered
  steps;
- episode-level received/never-received/saturated fractions in evaluator rows
  and TensorBoard.

The synthetic audit covers reset/unknown, delayed delivery, visible pending
delivery, saturation, and recovery.  All cases pass.  A saturated compatibility
value is no longer labeled as stale when no packet has ever been accepted.
The explicit state is also routed by the ledger to
`safe_hold(observation_never_received)`; legacy contexts without the field
remain backwards compatible.

## Gates

### Passed

- score monotonicity and fixed-point tie suite: 7/7;
- age-state reset, delivery, visibility-gap, saturation, and recovery;
- targeted environment/ranker/ledger/audit regression: 82 passed;
- `raw_unverified_executed=0` in the existing V21 evidence;
- TensorBoard event files and provenance for every new audit;
- no CBF margin, stale/OOD threshold, or controlled-abort rule was relaxed.

### Not passed / not yet demonstrated

- settled ranking mismatch remains high;
- candidate eligibility and nominal-anchor fallback still suppress task
  progress;
- CPU/CUDA fixed-point replay at the abstention boundary is not yet proven;
- the repaired age fields have not yet been regenerated through a new
  three-seed paired smoke or calibration archive;
- no new safe-capture performance claim is made.

## Decision and next work

WP2 is accepted as a contract/debugging stage, not as a control-performance
stage.  The next execution block is:

1. replay candidate, reachable-nominal, and safe-hold CBF counterfactuals on
   each earliest-abort cycle;
2. run CPU/CUDA fixed-point/tie/abstention/nominal-anchor replay and a
   100/500-cycle rolling deterministic check;
3. generate a new calibration archive and checkpoint-bound ledger only after
   the age fields pass through the real trace path;
4. run a new three-seed x 20 development smoke only when the preceding gates
   pass; do not open the locked test or expand to 40/60 episodes beforehand.

The existing L0-L3 full evaluation remains the authoritative current
episode-level result.  This WP2 report only narrows the failure mechanism and
provides the executable evidence needed for the next revision.
