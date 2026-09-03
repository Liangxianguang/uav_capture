# WP1 Failure Index and Causal Replay Audit

**Date:** 2026-09-04
**Status:** development-only; `locked_test_opened=false`
**Input:** P7 full development runs (`21` runs, `840` episodes)
**Freeze:** `results/jepa_safe_capture_v3_wp0_baseline_freeze_20260904/manifest.json`
**Output:** `results/jepa_safe_capture_v3_wp1_failure_index_20260904_rerun/`
**TensorBoard:** `results/jepa_safe_capture_v3_tensorboard/wp1_failure_index_rerun/`

## Purpose

WP1 turns the existing P7 step traces into an episode-level failure index. It
does not change, retrain, or re-evaluate a policy. The audit joins the
aggregator-validated outcome fields with the full source `episodes.csv` and
each JSONL step trace, preserving layout, obstacle, target-motion, visibility,
ledger, ranking, CBF, and termination context for the next engineering cycle.

The audit is strictly development-only and did not read the 100-episode locked
split. Existing run directories were read-only inputs.

## Aggregate Findings

| Quantity | Count |
|---|---:|
| P7 full runs | 21 |
| Indexed episodes | 840 |
| Safe capture | 244 (29.0%) |
| Failed episodes | 596 |
| CBF controlled-abort primary outcomes | 447 |
| Timeout primary outcomes | 32 |
| Collision primary outcomes | 117 |

The `117` collision outcomes are the deliberate A3 raw/no-CBF diagnostic path.
All safety-preserving P7 variants retained zero collision, boundary, and
pairwise violation outcomes in the P7 aggregate. The raw diagnostic therefore
remains evidence that JEPA/ranking cannot be the final safety boundary.

## Primary Outcome Causes

| Primary cause | Episodes |
|---|---:|
| `cbf_controlled_abort` | 447 |
| `collision` | 117 |
| `timeout` | 32 |

The primary label follows a fixed priority: safety violation, verified/unverified
CBF termination, timeout, candidate regression, reliability/fallback signal,
and finally unresolved non-capture. Multiple secondary labels are retained in
the CSV; the primary label is not intended to erase those signals.

## Diagnostic Signals

| Signal | Episodes |
|---|---:|
| stale observation (`age > 3` or message age `> 3`) | 575 |
| CBF controlled abort | 447 |
| CBF infeasible/unverified trace | 447 |
| degraded visibility (`mean visible fraction < 0.5`) | 425 |
| high-credit failure (trusted ledger state on failed episode) | 392 |
| candidate oscillation (`switch rate > 0.25`) | 143 |
| nominal/low-credit fallback on failed episode | 136 |
| candidate capture regression against paired M0 | 101 |
| pairwise violation | 45 |
| clearance prediction gap diagnostic | 19 |
| visibility prediction gap diagnostic | 7 |

These counts are overlapping episode labels, not independent samples. In
particular, a high-credit failure is an abstention-calibration warning, not a
proof that the ledger caused the termination. CBF labels are execution facts;
prediction-gap labels are timestamp-aligned diagnostics only.

## Variant Summary

| Variant | Episodes | Safe capture | High-credit failures | Fallback episodes | Mean candidate switch rate |
|---|---:|---:|---:|---:|---:|
| M0 | 120 | 32.5% | 0 | 0 | 0.000 |
| M1 | 120 | 33.3% | 80 | 0 | 0.154 |
| M2 | 120 | 34.2% | 77 | 45 | 0.178 |
| M3 | 120 | 36.7% | 74 | 43 | 0.216 |
| A1 | 120 | 30.8% | 83 | 0 | 0.215 |
| A2 | 120 | 33.3% | 78 | 48 | 0.204 |
| A3 | 120 | 2.5% | 0 | 0 | 0.000 |

The M3 and M2 high-credit failure counts show why improving prediction MAE
alone is insufficient: the ledger must distinguish trustworthy context from
high-confidence but unsuccessful rollouts and abstain before ranking can hurt
capture. A1 provides the corresponding no-ledger diagnostic.

## Causal Chain Retained Per Episode

`failure_index.csv` retains, for every episode:

```text
training seed / episode seed / layout / obstacle count / target motion
-> observation condition and age
-> JEPA candidate predictions and candidate rank
-> ledger states, credits, and fallback reasons
-> CBF solver status, active fallback, correction and latency
-> executed trace length and termination reason
```

The script validates that each trace row belongs to the source episode and that
the complete source episode table agrees with the aggregator-checked episode
identity. It also reduces the five candidate predictions to one conservative
per-step clearance prediction and one per-step visibility prediction before
computing diagnostic gaps, avoiding candidate/time misalignment.

## Known Evidence Limit

`target_drift_observable=false`. The P7 JSONL traces do not contain offline
future target labels, so this WP1 audit does not infer target drift from target
clearance or ranking cost proxies. WP2/WP6 must add a separately labelled,
offline-only future target outcome channel before target-drift calibration can
be claimed. Online target ground truth remains forbidden.

## Reproducibility and TensorBoard

The audit command was:

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
& $py scripts/index_jepa_safe_capture_failures.py `
  --input-root results `
  --output-dir results/jepa_safe_capture_v3_wp1_failure_index_20260904_rerun `
  --tensorboard-logdir results/jepa_safe_capture_v3_tensorboard/wp1_failure_index_rerun `
  --freeze-manifest results/jepa_safe_capture_v3_wp0_baseline_freeze_20260904/manifest.json `
  --stage full --development-only
```

Generated local artifacts:

- `failure_index.json` with all rows, grouped summaries, input run hashes and
  the target-drift evidence limit;
- `failure_index.csv` for episode-level filtering and replay triage;
- `provenance.json` with source hashes, freeze manifest hash, Git revision and
  environment;
- `report.md` and a TensorBoard event file.

TensorBoard contains configuration, run provenance, evidence limits, total
safe-capture/failure scalars, primary/diagnostic label counts, and per-variant
safe-capture/fallback scalars. EventAccumulator reload verified all required
provenance text tags.

## WP1 Exit Decision

WP1 passes its indexing gate: all `840/840` episodes have source metadata and a
non-empty trace, all `21/21` runs remain development-only, and all outputs are
reproducible from the frozen P7 inputs. The next work package is to add
offline-only future target labels and train/calibrate a new JEPA/ledger version
against the hard-case buckets. No locked test is opened by this report.
