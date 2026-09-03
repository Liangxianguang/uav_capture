# Interaction-Aware Action-Conditioned JEPA + CBF
# P7 Reproducibility and Locked-Readiness Audit

**Date:** 2026-09-03
**Final development decision:** `prediction_improvement_no_control_gain`
**Locked-test decision:** `do_not_open_locked_test`

## Audit Result

P5-P7 execution is reproducible as a development-only evidence chain, but it does not meet the threshold for a new locked evaluation. The reason is not capture time: safe capture is the primary metric. The reason is that three trained JEPA seeds produce a zero mean paired safe-capture delta against the frozen V5 + CBF baseline, with six improvements and six regressions. The current configuration preserves zero collision/boundary events but has not shown a reliable safe-capture gain.

## Required Checks

| Requirement | Result | Evidence |
|---|---|---|
| v2 data contract | pass | train/validation archives audited; no development or locked data in training |
| Three CUDA training runs | pass | 40 epochs per seed, provenance and TensorBoard complete |
| Offline prediction/action gate | pass with mechanism warning | finite outputs and candidate separation; weak action antisymmetry remains disclosed |
| Checkpoint-bound ledger | pass | three validation-only ledgers with immutable hash bindings |
| Zero-perturbation regression | pass | 20/20 exact pairs, 87 non-JEPA fields, zero differences |
| P6 smoke | pass | all three seeds zero collision/boundary and 100% transit |
| P6 final pairing | pass | 3 x 60 exact paired episode IDs and immutable scenario specifications |
| P6 safe capture | no reliable gain | `+1.667`, `0.000`, `-1.667` percentage points by seed |
| New collision/boundary | none | `0/180` candidate episodes for both categories |
| Locked block opened | no | all P5/P6 runs have `split=validation`, `locked_test=false` |

## Reproduction Instructions

Use the GPU Conda environment already used for training and evaluation:

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'
& $py -m pytest -q `
  tests\test_jepa_v3_counterfactual_dataset.py `
  tests\test_jepa_v3_multitask.py `
  tests\test_jepa_v3_multitask_aggregate.py `
  tests\test_jepa_v3_reliability.py `
  tests\test_jepa_v3_zero_perturbation.py `
  tests\test_jepa_v3_p6_aggregate.py
```

Then regenerate the final development aggregation with `scripts/aggregate_jepa_v3_p6_development.py` and the three P6 result directories. The aggregator rejects a run if it is not validation-only, lacks CBF, has a changed candidate contract, contains a nonmatching episode pair, uses a different frozen scene specification, or uses a ledger whose checkpoint hash does not match the candidate checkpoint.

## Artifact Map

| Phase | Primary artifact |
|---|---|
| Plan and protocol amendment | `docs/JEPA_V3_P5_TO_P7_EXECUTION_PLAN_20260903.md`, `docs/JEPA_V3_P5_SAFE_CAPTURE_PRIMARY_AMENDMENT_20260903.md` |
| v2 data/training/ledger audits | `docs/JEPA_V3_P5_V2_ARCHIVE_AUDIT_20260903.md`, `docs/JEPA_V3_P5_V2_THREE_SEED_TRAINING_AUDIT_20260903.md`, `docs/JEPA_V3_P5_V2_OFFLINE_GATES_AND_LEDGER_20260903.md` |
| Runtime identity evidence | `docs/JEPA_V3_P5_ZERO_PERTURBATION_REGRESSION_20260903.md` |
| Seed-11 diagnostic | `docs/JEPA_V3_P5_ACTION_CHUNK_REPORT_20260903.md` |
| Final P6 result | `docs/JEPA_V3_P6_SAFE_CAPTURE_FIRST_DEVELOPMENT_REPORT_20260903.md` |
| Aggregation code and tests | `scripts/aggregate_jepa_v3_p6_development.py`, `tests/test_jepa_v3_p6_aggregate.py` |

All checkpoints, counterfactual archives, TensorBoard event files, episode CSVs, scene JSONLs, and generated aggregate JSON/Markdown remain in `results/` and are intentionally not added to Git. The source code and reports provide the commands, hashes, and path contract required to regenerate them locally.

## Locked-Readiness Decision

The following P7 condition from the execution plan is not met:

```text
Three training seeds show a consistent, non-single-seed, positive paired
safe-capture benefit with no new collision or boundary event.
```

It fails because seed 13 is negative and the mean paired delta is exactly zero. No additional run on this development block should be used to select new score weights, chunk size, perturbation, or ledger threshold. Such tuning would convert the frozen development scenes into training feedback and would weaken the evidence.

A future variant may be considered only after a separate protocol freezes:

1. an independent validation archive for revising the reliability/abstention rule, particularly high-credit delayed/noisy failure states;
2. a safe-capture primary endpoint and the same zero collision/boundary safety requirement;
3. the candidate generator, score weights, fallback policy, sample size, statistical test, and failure handling before a new development block is evaluated; and
4. a new locked block, opened only after explicit user authorization.

Until then, the defensible result is: **CBF keeps this JEPA reranker collision-free and in-bounds on the tested development scenes, but the current reranker has not earned a safe-capture-improvement claim.**
