# Interaction-Aware Action-Conditioned JEPA + CBF
# P5 action-chunk development report

**Date:** 2026-09-03  
**Status:** `admitted_to_p6`  
**Scope:** development only; this document does not open, replace, or reinterpret a V4/V5 locked test.

## Decision

The precision-contract-corrected v2 implementation passed the strict
zero-perturbation regression and completed the prespecified seed-11
development smoke and diagnostic.  It is therefore admitted to the frozen,
three-training-seed P6 evaluation.  This is **not** evidence that the model
improves the final system: one development seed cannot establish that claim.

The decision follows the post-smoke safe-capture-first amendment in
`docs/JEPA_V3_P5_SAFE_CAPTURE_PRIMARY_AMENDMENT_20260903.md`:

1. safe capture and no new collision/boundary event are primary;
2. capture time, path length, clearance, CBF correction, fallback, and
   latency remain mandatory disclosures, but capture time alone is not an
   automatic rejection rule;
3. the amendment is dated after the smoke, so the report does not represent it
   as preregistered.

## Frozen inputs and provenance

| Item | Value |
|---|---|
| Frozen V5 actor | `models/v5_development_exact_reactive_seed661606.pt` |
| Actor SHA-256 | `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba` |
| v2 train archive SHA-256 | `0d165646db5f0545115fa5f8cdb2bc6fd44b9ab2db5981e8de5b96963e84787c` |
| v2 validation archive SHA-256 | `1c04b9556b95fbcc050678fc4ee3a1b62b45c9185bc928d904be18745ddfe51c` |
| JEPA training seed | `20260911` |
| JEPA checkpoint SHA-256 | `57741bbfdffb806d14043bc8620024f602eb412f7907f81e762e3d6af5b48c4f` |
| Training | CUDA, 40 epochs, replay disabled |
| Candidate contract | `K=5`, perturbation `0.10 m/s`, constant chunk length `3`, execute first step then replan |
| Reliability ledger | checkpoint-hash-bound validation-only ledger; horizon `0.5 s`, minimum credit `0.65` |
| Environment | `configs/capture_radius_pursuit_central_v4_flee.yaml` |
| Scenario block | frozen S3 validation scenes from `results/jepa_v2_control_baseline60/` |
| Safety filter | CBF enabled for both baseline and candidate, applied last |
| Locked test | `locked_test_opened=false` |

The v2 data contract, three-seed training audit, offline prediction gate,
action-following audit, and ledgers are recorded in:

- `docs/JEPA_V3_P5_V2_ARCHIVE_AUDIT_20260903.md`
- `docs/JEPA_V3_P5_V2_THREE_SEED_TRAINING_AUDIT_20260903.md`
- `docs/JEPA_V3_P5_V2_OFFLINE_GATES_AND_LEDGER_20260903.md`

All seed-11 online output contains its complete invocation and source paths in
`evaluation_metadata.json`.  The runtime did not update the actor, JEPA, or
ledger online.

## Precision and zero-perturbation gate

The earlier v1 action-chunk evidence is excluded because a float32 candidate
path differed from the float64 nominal CBF path even at zero perturbation.
The corrected runtime has an explicit identity bypass only when an enabled
JEPA has `perturbation=0.0`: it sends the frozen nominal action directly to
the baseline CBF path.  Nonzero candidate handling is unchanged.

The corrected seed-11 run
`results/jepa_v3_p5_chunk3v2p1_seed20260911_zero20/` passed:

| Check | Result |
|---|---:|
| Paired episodes | `20 / 20` |
| Frozen scenes byte-identical | yes |
| Non-JEPA fields compared | `87` |
| Field differences | `0` |
| Collision/boundary/capture/path/time differences | `0` |

The full comparator output is retained with the run.  See
`docs/JEPA_V3_P5_ZERO_PERTURBATION_REGRESSION_20260903.md`.

## 20-episode smoke

The smoke used the 20 paired frozen scenes in
`results/jepa_v3_p3_zero_baseline20/`.  Neither side had collision or boundary
violation, and transit was successful in all episodes.

| Metric | V5 + CBF baseline | JEPA + CBF | Paired delta / note |
|---|---:|---:|---|
| Safe capture | `18/20` (`90.0%`) | `19/20` (`95.0%`) | 1 improved, 0 degraded, 19 tied |
| Collision | `0` | `0` | no new event |
| Boundary violation | `0` | `0` | no new event |
| Transit success | `100.0%` | `100.0%` | unchanged |
| Mean total defender path | `85.3673 m` | `84.2018 m` | `-1.1655 m` |
| Mean CBF correction norm | `0.22354` | `0.23057` | `+0.00703` |
| Mean capture time among captures | `6.1667 s` | `6.7895 s` | reported, not a safety failure |

The changed sample is episode 17 (seed `646118`): baseline timeout became
safe capture at `24.8 s`.  It is a positive paired outcome, but this small
smoke is not used as effectiveness evidence.

## 60-episode seed-11 diagnostic

The diagnostic reused the same sixty frozen S3 validation scenes as
`results/jepa_v2_control_baseline60/`, with no parameter change.  Outputs are
in `results/jepa_v3_p5_chunk3v2p1_seed20260911_dev60/`.

| Metric | V5 + CBF baseline | JEPA + CBF | Delta / interpretation |
|---|---:|---:|---|
| Safe capture | `57/60` (`95.0%`) | `58/60` (`96.7%`) | +1 episode net |
| Collision | `0` | `0` | safety condition retained |
| Boundary violation | `0` | `0` | safety condition retained |
| Transit success | `100.0%` | `100.0%` | retained |
| Pairing | - | - | 2 improved, 1 degraded, 57 tied |
| Mean total defender path | `84.3609 m` | `83.6327 m` | `-0.7282 m` (`-0.86%`) |
| Mean minimum clearance | `0.3549 m` | `0.3475 m` | `-0.0074 m`; worst: `0.1859` -> `0.0912 m` |
| Mean CBF correction norm | `0.21435` | `0.22271` | `+0.00837` |
| Mean capture time among captures | `6.7649 s` | `7.1741 s` | reported, not a safety failure |
| Mean ledger credit | - | `0.69374` | above the `0.65` admission threshold on average |
| Mean nominal fallback fraction | - | `60.34%` | conservative fallback is frequent |
| Mean global/OOD fallback fraction | - | `5.92%` | ledger diagnostics |
| Mean selected candidate index | - | `0.9704` | candidate-ranking diagnostic |

### Outcome changes and failure classification

| Episode | Scene seed | Baseline | JEPA + CBF | Classification |
|---:|---:|---|---|---|
| 17 | `646118` | timeout | safe capture | improved; 4 obstacles, `cylinder1+box2+wall1`, nominal observation |
| 48 | `646149` | safe capture | timeout | degraded; 4 obstacles, `cylinder1+box2+wall1`, delayed/noisy observation |
| 56 | `646157` | timeout | safe capture | improved; 3 obstacles, `cylinder1+box1+wall1`, delayed/noisy observation |

Episode 48 is the important retained negative result.  Both runs were
collision-free and in-bounds, but the candidate timed out where the baseline
captured at `23.4 s`.  Its minimum clearance fell from `0.3656 m` to
`0.2365 m`, despite a ledger credit of `0.7574` and a nominal fallback fraction
of `34.8%`.  This is consistent with a ranking/reliability limitation under a
delayed noisy, high-obstacle flee-persistence scene.  It must be checked across
the two remaining training seeds; it is not hidden by the aggregate capture
gain.

No collision or boundary failure occurred, so the predeclared safety stop
condition was not triggered.  The clearance reduction and CBF increase are
cost signals to disclose and analyse in P6, not grounds for post hoc tuning.

## Interpretation and next step

Seed-11 establishes only that the v2 runtime is contract-correct, preserves
CBF safety on this block, and warrants the frozen three-seed test.  It does
not establish a reliable capture improvement, prove that the ledger predicts
safety, or authorize a locked evaluation.

P6 will use the other two independently trained v2 checkpoints and their own
immutable ledgers on the identical reference scenes, with unchanged
`K=5`, `0.10 m/s`, chunk length `3`, score weights, CBF and fallback policy.
It will report safe capture first, then collision/boundary, transit, paired
improvements and regressions, clearance, path, CBF correction, capture time,
fallback, and difficulty buckets for all three seeds.

