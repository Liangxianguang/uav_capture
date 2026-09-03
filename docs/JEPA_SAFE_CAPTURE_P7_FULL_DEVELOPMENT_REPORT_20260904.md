# JEPA Safe-Capture v2 P7 Full Development Report

Date: 2026-09-04
Status: development-only; locked_test_opened=false
Hardware: NVIDIA GeForce RTX 5050, PyTorch 2.7.1+cu128
Protocol: central_random_mixed_obstacle_s3_protocol.yaml, validation split
Scope: 7 variants x 3 training seeds x 40 validation episodes = 840 episodes

This report evaluates the Interaction-Aware Action-Conditioned JEPA as a
candidate trajectory evaluator, with Reliability Ledger gating, joint
multi-agent CBF-QP filtering, and receding-horizon first-step execution. The
validation split has 40 episodes. The 100-episode locked_test split was not
opened.

## Frozen Execution Contract

- Five candidates, three-step constant action chunks, execute only the first
  filtered step, then replan.
- Candidate requests are projected into the reachable speed/slew envelope.
- Every non-diagnostic action goes through the same Joint CBF-QP.
- CBF uses obstacle, boundary, pairwise, kinematic, and three-step
  anticipatory braking constraints.
- Infeasible, timeout, non-finite, stale, OOD, and unverified paths remain
  explicitly logged; raw action is never used as a safety fallback.
- Safe capture requires capture before timeout and no collision, boundary,
  pairwise, target-obstacle, or unverified-CBF failure.

The canonical paired scene-manifest hash (excluding only per-run
training_seed provenance) is:

d709914ffd877671129ee843aa369c48a3dd4013e219a5a1b3dd2143bd1a30b8

## Results

| Variant | Safe-capture rate across 3 seeds | Total collision | Boundary | Pairwise | CBF infeasible steps |
|---|---:|---:|---:|---:|---:|
| M0 frozen nominal + CBF | 32.5% +/- 0.0% | 0 | 0 | 0 | 69 |
| M1 JEPA target/uncertainty + CBF | 33.3% +/- 3.8% | 0 | 0 | 0 | 78 |
| M2 JEPA + Ledger + CBF | 34.2% +/- 8.8% | 0 | 0 | 0 | 78 |
| M3 JEPA + Ledger + auxiliary safety ranking + CBF | 36.7% +/- 10.1% | 0 | 0 | 0 | 72 |
| A1 M3 without Ledger | 30.8% +/- 9.5% | 0 | 0 | 0 | 77 |
| A2 M3 without clearance/visibility terms | 33.3% +/- 14.2% | 0 | 0 | 0 | 77 |
| A3 raw/no CBF diagnostic | 2.5% +/- 0.0% | 117 | 0 | 45 | 0 |

The +/- values above are the sample standard deviation of the three
training-seed rates, not a confidence interval.

## M3 Paired Comparison

Each M3 episode is paired with the M0 episode at the same episode index,
episode seed, layout, target motion, and observation schedule.

| Training seed | M0 | M3 | Paired delta | Improved | Degraded | McNemar exact p |
|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 13/40 | 17/40 | +10.0 pp | 7 | 3 | 0.3438 |
| 20260912 | 13/40 | 10/40 | -7.5 pp | 3 | 6 | 0.5078 |
| 20260913 | 13/40 | 17/40 | +10.0 pp | 6 | 2 | 0.2891 |

Across all 120 paired episode units:

- improved/degraded/tied: 16/11/93;
- mean paired delta: +4.17 pp;
- fixed-seed bootstrap 95% CI: [-4.17 pp, +12.50 pp];
- non-negative training seeds: 2/3.

The predeclared development classification is
positive_development_evidence: safety and reliability hard gates pass, the
mean paired delta is positive, and at least two of three seed deltas are
non-negative. The bootstrap interval crosses zero and the per-seed exact tests
are not significant, so this is not evidence of a statistically reliable
general improvement and must not be promoted to a locked-test claim.

## Safety Interpretation

All safety-preserving variants had zero collision, boundary, and pairwise
violation counts in these development runs. A3 deliberately removes CBF and
exposes the expected failure mode: 117/120 episodes collided and 45 had a
pairwise violation. This supports the architectural requirement that the
world-model/ranker can only propose or order candidates; the CBF remains the
hard execution boundary.

The CBF infeasible counts are not silently converted into successes. Verified
nominal-CBF or hold fallbacks are distinguished from unverified controlled
aborts, and all such routes are present in the per-episode traces.

## Reproducibility Artifacts

Generated locally under results/ (ignored by Git):

~~~text
results/jepa_safe_capture_v2_p7_readiness_full_20260904_rerun/summary.json
results/jepa_safe_capture_v2_p7_readiness_full_20260904_rerun/report.md
results/jepa_safe_capture_v2_p7_readiness_full_20260904_rerun/paired_comparison.json
results/jepa_safe_capture_v2_p7_readiness_full_20260904_rerun/run_metrics.csv
results/jepa_safe_capture_v2_p7_readiness_full_20260904_rerun/m3_seed_comparisons.csv
results/jepa_safe_capture_v2_p7_readiness_full_20260904_rerun/tensorboard/
~~~

Every run also contains summary.json, provenance.json, scene_manifest.jsonl,
scenes.jsonl, episodes.csv, and per-step traces. The evaluator writes
configuration, checkpoint hashes, ledger hashes, environment details,
TensorBoard event names, and locked_test_opened=false.

The full regression suite after the implementation changes was:

~~~text
270 passed, 17 warnings
~~~

## Decision

The complete development experiment supports the intended safety architecture:
JEPA may improve candidate ordering in some seeds, while the Ledger and CBF
make uncertainty and infeasibility explicit and prevent raw-action execution.
The current evidence is sufficient to prioritize reliability-ledger analysis,
hard-episode replay, and improved candidate/action-block training. It is not
sufficient to claim a robust JEPA control gain or to open a new locked test.
