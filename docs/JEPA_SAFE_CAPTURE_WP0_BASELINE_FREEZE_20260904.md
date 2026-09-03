# WP0 Baseline Freeze Report

**Date:** 2026-09-04
**Status:** development-only; `locked_test_opened=false`
**Protocol:** `configs/jepa_safe_capture_v3_next_phase.yaml`
**Manifest:** `results/jepa_safe_capture_v3_wp0_baseline_freeze_20260904/manifest.json`
**TensorBoard:** `results/jepa_safe_capture_v3_tensorboard/wp0_baseline_freeze/`

## Purpose

WP0 freezes the post-P7 starting point before any new model, ledger, candidate,
or CBF change. It records the parent P7 result, all declared runtime inputs,
the RTX 5050 environment, and the immutable execution contract. The freeze
script does not alter any existing run and refuses a locked-test parent.

## Frozen Contract

- Primary endpoint: `safe_capture`; capture time is secondary.
- Five candidate action chunks, three steps per chunk, execute only the first
  step, then reobserve and replan.
- JEPA is a candidate trajectory evaluator/ranker only.
- All baseline and candidate actions pass through the same Joint CBF-QP.
- Online target ground truth, online actor updates, and online ledger updates
  are forbidden.
- QP infeasible, timeout, stale, OOD, non-finite, and unverified routes use an
  explicit safe-hold or nominal-CBF fallback and cannot be counted as safe
  capture.
- Development evaluation uses training seeds `20260911`, `20260912`, and
  `20260913`; the new final block has 40 episodes per seed.

## Parent P7 Evidence

The parent is `positive_development_evidence`, with the safety hard gate and
TensorBoard provenance recorded as passed. M3 versus M0 had a mean paired delta
of `+4.17 pp`, but its bootstrap 95% interval was `[-4.17 pp, +12.50 pp]`.
That interval crossing zero is preserved as a reason not to claim a robust
general improvement or open a locked test.

## Generated Evidence

The freeze command completed with the RTX 5050 environment:

```text
PyTorch: 2.7.1+cu128
CUDA: 12.8
Device: NVIDIA GeForce RTX 5050
TensorBoard: 2.19.0
```

The generated manifest records SHA-256 values for the actor checkpoint,
environment config, scenario protocol, parent P7 summary/report, and this
next-phase protocol. TensorBoard contains three provenance text records and
contract/safety scalars. The manifest is intentionally under `results/` and is
not a Git-tracked dataset or locked-test artifact.

## Acceptance

- [x] Parent P7 is development-only and did not open locked test.
- [x] Primary endpoint is `safe_capture`.
- [x] JEPA evaluator, first-step replan, and CBF execution-boundary invariants
      are validated by the freeze script.
- [x] Declared inputs exist and are hash-recorded.
- [x] TensorBoard provenance was written and reloaded with EventAccumulator.
- [x] Existing experiment directories were not overwritten.

The next allowed work package is WP1 failure indexing and deterministic replay.
No new final block may start until its scene manifest, seed list, thresholds,
and provenance are frozen separately.
