# JEPA Safe-Capture L0-L3 Dataset Stage Report

**Date:** 2026-09-05  
**Phase:** development-only; `locked_test_opened=false`  
**Hardware:** NVIDIA GeForce RTX 5050  
**Scope:** dataset collection, shard merge, provenance and archive audit

## 1. What was produced

The new curriculum contains eight scenario families:

| Level | Families | Contract |
|---|---|---|
| L0 | open, single obstacle | 0-1 obstacle, nominal observation |
| L1 | three-obstacle flee, three-obstacle S-curve | 3 obstacles, nominal observation in v2 |
| L2 | dropout, delayed noisy S-curve | 3 obstacles with partial observation |
| L3 | four-obstacle mixed, five-obstacle delayed S-curve | mixed 3-5 obstacles, delayed/noisy stress |

Each archive uses five counterfactual candidates, eight history steps, a three-step constant action chunk, and offline-only target truth for labels. No locked split was opened.

## 2. Archive inventory

| Split | Archive | Episodes | Samples | SHA-256 |
|---|---|---:|---:|---|
| train shard 0 | `results/jepa_safe_capture_l0_l3_v1_train_shard0` | 240 | 292,800 | `0709f9a04ce75d8174d4972fe0b46219ee081cfdbc342744f6c6eeb08b87bc21` |
| train shard 1 | `results/jepa_safe_capture_l0_l3_v1_train_shard1` | 240 | 292,800 | `7acb004fb08a77804e6080594c77d91adc4091a9d8345e7c5d6f0665a1eb7467` |
| train shard 2 | `results/jepa_safe_capture_l0_l3_v1_train_shard2` | 240 | 292,800 | `9441022a8f95bdd8128e72982b9f92b38dc52dc0ea0cb7a0aaa19dbac24b67c7` |
| merged train | `results/jepa_safe_capture_l0_l3_v1_train_merged` | 720 | 878,400 | `87ceb1a5e9866bde94ceab38bde7576c7b62679094d9d33370d93db07b0961b5` |
| validation | `results/jepa_safe_capture_l0_l3_v2_validation` | 64 | 78,080 | `44b80b144cc674c50886eb62ba34320e36805f33835da886971c15fc391500b6` |
| calibration | `results/jepa_safe_capture_l0_l3_v2_calibration` | 64 | 78,080 | `16b0dc0e1e365eef78adf5d5d5fa876786f42dabb82f89cb0974374afd136719` |

The three training shards use disjoint seed offsets (`0`, `1000`, `2000`). Validation and calibration use separate seed blocks (`272...` and `273...`) and have zero episode-seed overlap with the merged train archive and with each other.

## 3. Important configuration correction

The first training collection was generated from `jepa_safe_capture_l0_l3_collection_v1.yaml`. Its L1 overrides accidentally included mild dropout, observation noise and message delay. Those files are retained and labelled `v1 mixed-noise training archive`; their hashes must not be silently reinterpreted as nominal L1 data.

`configs/jepa_safe_capture_l0_l3_collection_v2.yaml` corrects both L1 families to zero dropout, zero observation noise, zero message delay and zero message dropout. The independent validation and calibration archives were generated from v2. This makes the current training/validation boundary explicit; a fully nominal-L1 training rerun remains a separate development action if the mixed-noise training distribution proves limiting.

## 4. Audit evidence

Audits:

- `results/jepa_safe_capture_l0_l3_v1_train_merged_audit.json`
- `results/jepa_safe_capture_l0_l3_v2_validation_audit.json`
- `results/jepa_safe_capture_l0_l3_v2_calibration_audit.json`

All audited arrays are finite. Every state-agent group has exactly five candidates and the nominal fraction is `0.2`. Collision and boundary labels are zero in the offline archives. The validation archive reports a `0.0001793` CBF-QP-infeasible label fraction; calibration reports zero in this sample. These are offline label statistics, not online safe-capture results.

Each archive writes TensorBoard data and provenance text under its corresponding `*_tensorboard` directory, including sample/episode counts, finite coverage, label histograms, protocol text, collection text and source hashes.

## 5. Next gates

1. Finish the three-seed JEPA training runs using the merged train archive and v2 validation/calibration metadata.
2. Audit checkpoint history, required TensorBoard tags, configuration hashes and finite parameters.
3. Build a checkpoint-bound Reliability Ledger from v2 calibration only.
4. Evaluate prediction/ranking on v2 validation, then run the paired M0/V5-CBF/M3 closed-loop development smoke.
5. Keep all `safe_capture`, controlled-abort and safety-hard-gate results separate from archive label statistics.

No L0-L3 success-rate claim is made by this report. The archive stage is complete; model and closed-loop performance remain to be measured.
