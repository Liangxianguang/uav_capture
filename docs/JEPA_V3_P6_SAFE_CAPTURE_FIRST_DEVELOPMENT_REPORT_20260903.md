# Interaction-Aware Action-Conditioned JEPA + CBF
# P6 Safe-Capture-First Three-Seed Development Report

**Date:** 2026-09-03
**Classification:** `prediction_improvement_no_control_gain`
**Locked-test status:** `locked_test_opened=false`; do not open a new locked block.

## Primary Conclusion

All three frozen JEPA training seeds preserved the CBF safety envelope on the frozen S3 development block: all `180` candidate episodes had zero collision, zero boundary violation, and successful transit. This is necessary but not sufficient for the safe-capture-first objective. Safe capture was not reliably improved over the frozen V5 + CBF baseline.

| Primary outcome | Frozen V5 + CBF | JEPA + CBF, three training seeds |
|---|---:|---:|
| Safe capture | `57/60 = 95.0%` | `95.0% +/- 1.67%` (`58/60`, `57/60`, `56/60`) |
| Collision | `0/60` | `0/180` |
| Boundary violation | `0/60` | `0/180` |
| Transit success | `100.0%` | `100.0% +/- 0.0%` |
| Paired safe-capture changes | - | `6` improved, `6` degraded, `168` tied |
| Mean paired safe-capture delta | - | `0.000 +/- 1.667` percentage points |
| Hierarchical paired-bootstrap 95% CI | - | `-4.444` to `+3.889` percentage points |
| Exact two-sided McNemar | - | `p=1.000000` |

The `180` candidate outcomes are three trained models on the same sixty scenes, not 180 independent scenes. The primary seed-level estimate and the hierarchical bootstrap resample training seeds and then full paired episodes; they do not treat control steps or candidate actions as independent samples.

The implementation has preserved CBF-protected execution safety on this development block but has not demonstrated a reliable net safe-capture gain. This development result must not be described as a V4, V5, or locked-test improvement.

## Frozen Contract and Provenance

| Item | Fixed value |
|---|---|
| Baseline actor | `models/v5_development_exact_reactive_seed661606.pt` |
| Actor SHA-256 | `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba` |
| Development split | `validation`; no locked data read |
| Scenes and baseline rows | `results/jepa_v2_control_baseline60/scenes.jsonl` and `episodes.csv` |
| Frozen scene-spec SHA-256 | `e05217ae316c6e8e6a6c63250358390200b219e16de2a5787fb0aa9fbb9bc0b8` |
| Environment | `configs/capture_radius_pursuit_central_v4_flee.yaml` |
| Candidate contract | `K=5`, perturbation `0.10 m/s`, constant action chunk `3`, execute first step then replan |
| Safety execution order | frozen actor -> candidate rerank or nominal fallback -> CBF last |
| Online adaptation | none: actor, JEPA checkpoint, scorer parameters, and ledger were immutable |

Each candidate run was checked for `locked_test=false`, `split=validation`, 60 exact episode IDs/seeds, identical frozen scene specifications, CBF enabled, and the expected candidate contract. The `scenes.jsonl` files include online outcomes, so their full byte hashes differ by method as expected; the audit canonically hashes only immutable `episode_index`, `spec`, and `scenario` fields. All three canonical scene hashes match the frozen baseline.

| JEPA seed | Checkpoint SHA-256 | Ledger SHA-256 |
|---:|---|---|
| 20260911 | `57741bbfdffb806d14043bc8620024f602eb412f7907f81e762e3d6af5b48c4f` | `750c8d6349f5f25c9e7da454078fc2f86f4aed7cea5d9f89a727365ba4289e08` |
| 20260912 | `df9813a49db73216a336d3321ed7b96d8b0c8bddd83f4f786185a1445a6ed31f` | `9ce2e1eb66087ece9abaece2e561a8b96258de485ea30a3ae97599263a79e2cc` |
| 20260913 | `1318f9b62bc29e287b00e0dd4ded81208f4c00260d165c80b615204f0c1f0118` | `fe6617309b4d8add37383e8ffe0f423690d6c1ca88cb9081e85e3ff0abb64725` |

Every ledger is validation-only and checkpoint bound, with horizon `0.5 s`, minimum sample count `128`, and minimum credit `0.65`. The detailed v2 archive, CUDA training, TensorBoard, prediction-gate, action-following, and ledger audits remain in the P5 reports.

## P6 Safety Smoke Gate

All smoke runs used the first twenty rows of the same frozen reference block and final P6 paths. They are interface/safety checks, not effectiveness claims.

| JEPA seed | Safe capture | Collision | Boundary | Transit |
|---:|---:|---:|---:|---:|
| 20260911 | `19/20` (`95.0%`) | `0` | `0` | `100.0%` |
| 20260912 | `18/20` (`90.0%`) | `0` | `0` | `100.0%` |
| 20260913 | `19/20` (`95.0%`) | `0` | `0` | `100.0%` |

No smoke triggered the collision/boundary, pairing, provenance, or non-finite-output stop rules. No settings were changed before the 60-episode runs.

## Final Paired Development Results

| JEPA seed | Safe capture | Improved / degraded / tied | Paired delta | Mean total path | Mean min. clearance | Mean CBF correction | Ledger credit | Nominal fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `57/60` (`95.0%`) | - | - | `84.361 m` | `0.3549 m` | `0.21435` | - | - |
| 20260911 | `58/60` (`96.7%`) | `2 / 1 / 57` | `+1.667 pp` | `83.633 m` | `0.3475 m` | `0.22271` | `0.6937` | `60.34%` |
| 20260912 | `57/60` (`95.0%`) | `1 / 1 / 58` | `0.000 pp` | `81.270 m` | `0.3586 m` | `0.21500` | `0.6751` | `68.85%` |
| 20260913 | `56/60` (`93.3%`) | `3 / 4 / 53` | `-1.667 pp` | `78.286 m` | `0.3503 m` | `0.23633` | `0.7365` | `8.21%` |

Efficiency and intervention quantities remain mandatory disclosures but are not rejection gates under the post-smoke amendment.

| Quantity, JEPA minus baseline | Mean | Hierarchical 95% CI |
|---|---:|---:|
| Total defender path | `-3.298 m` | `-10.409` to `+3.585 m` |
| Minimum clearance | `-0.0028 m` | `-0.0101` to `+0.0038 m` |
| Mean CBF correction | `+0.01033` | `-0.00142` to `+0.02251` |

The ledger is highly conservative for seeds 11 and 12 but much less so for seed 13. Mean local nominal fallback is `45.80% +/- 32.83%`; mean global/OOD fallback is `5.65% +/- 0.32%`. A credit above the threshold is not a safety certificate and did not reliably distinguish ranking mistakes from safe ranking opportunities.

## Failure Analysis

All six capture regressions were `safe_capture -> timeout`, not collision or boundary events. Five occurred under delayed/noisy observations; four used flee-persistence target motion, and three had five obstacles. Each had ledger credit above `0.73`, so the current ledger did not abstain in these cases.

| JEPA seed | Episode seed | Baseline -> candidate | Scenario pattern | Candidate ledger credit |
|---:|---:|---|---|---:|
| 20260911 | 646149 | safe capture -> timeout | 4 obstacles, flee persistence, delayed/noisy | `0.7574` |
| 20260912 | 646153 | safe capture -> timeout | 5 obstacles, flee persistence, delayed/noisy | `0.7629` |
| 20260913 | 646107 | safe capture -> timeout | 3 obstacles, s-curve, nominal observation | `0.7356` |
| 20260913 | 646146 | safe capture -> timeout | 5 obstacles, s-curve, delayed/noisy | `0.7635` |
| 20260913 | 646151 | safe capture -> timeout | 3 obstacles, flee persistence, delayed/noisy | `0.7322` |
| 20260913 | 646155 | safe capture -> timeout | 5 obstacles, flee persistence, delayed/noisy | `0.7513` |

The six positive changes are retained in `results/jepa_v3_p6_chunk3v2_aggregate.json`; they balance the six regressions rather than supporting a net claim. Episode seed `646157` improved in all three training seeds, whereas the regressions vary by seed and scene. This is a diagnostic for a future separately preregistered reliability study, not a license to tune against this development block.

## Decision and Reproduction

The safe-capture-first requirement has two parts: prevent unsafe termination, and achieve safe capture. The first part is retained by CBF in all evaluated runs. The second part is not improved consistently: seed-11 is positive, seed-12 is neutral, and seed-13 is negative. The primary decision is therefore `prediction_improvement_no_control_gain`, with `eligible_to_open_locked_test=false`.

The machine-readable aggregate and regenerated Markdown are `results/jepa_v3_p6_chunk3v2_aggregate.json` and `results/jepa_v3_p6_chunk3v2_aggregate.md`. They are generated by `scripts/aggregate_jepa_v3_p6_development.py`, which verifies the development-only split, scenario specification hash, pairing, CBF, fixed candidate contract, and checkpoint-bound ledger before calculating the report.
