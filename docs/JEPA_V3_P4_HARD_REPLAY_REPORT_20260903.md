# JEPA-v3 P4 Hard-Example Replay Report

**Status:** completed; replay-on is rejected for P5/P6 runtime control
**Scope:** development-only; V4/V5 locked blocks remain closed
**Decision:** retain the P3 reliability-ledger path, but do not carry this P4 replay weighting configuration into closed-loop evaluation.

## Question

P4 asks whether train-only mixed replay improves the multitask interaction-aware JEPA on hard counterfactual situations. It does not ask whether lower training loss or a selected single seed improves capture. CBF remains the final safety filter in all later candidate-action paths.

## Fixed Replay Contract

The replay manifest is derived only from the P1 training archive. A sample is hard when any settled rollout horizon has low obstacle/inter-agent clearance (`< 0.30 m`), high CBF correction (`> 0.25 m/s`), or a collision/boundary label. Every epoch retains exactly `50%` uniform draws; the remaining `50%` is sampled with weight `3.0` for hard samples and `1.0` otherwise. Validation, S3 development scenes, and all locked artifacts remain excluded from replay training.

| Training replay property | Value |
| --- | ---: |
| P1 training samples | 146,400 |
| Train hard samples | 14,559 (9.94%) |
| Train low-clearance / collision-or-boundary samples | 0 / 0 |
| Train high-CBF-correction samples | 14,559 |
| Uniform draw fraction | 50.0% |
| Hard sample weight | 3.0 |
| Replay weights SHA-256 | `d84e516bbc687615ec07bb0484024e34adf96ba1b76e79694f50a67fe31c2467` |
| Manifest SHA-256 | `5282a70a1cb4881c70d39bef0e2503d214240e60a2eaf5e7c5936d3b38a67e4d` |

The replay predicate is implemented once in `scripts/build_jepa_v3_hard_replay_weights.py` and reapplied read-only to independent validation labels by `scripts/evaluate_jepa_v3_replay_subsets.py`. The evaluator never creates validation weights or changes the replay policy.

## Matched Training

Both models use seed `20260911`, 40 epochs, batch size 512, the same architecture, optimizer, multitask losses, CUDA device, train/validation data, and selection procedure. The mixed sampler is the only changed factor.

| Run | Best epoch | Best validation loss | Checkpoint SHA-256 |
| --- | ---: | ---: | --- |
| replay-off | 7 | -3.522987 | `1454f4c492eaf97fa6d83ed69a3fb408ec494caf91d0a1de7cd67597024709d2` |
| replay-on | 7 | -3.430099 | `b0f613e571a6913efe1502a4163580f15c80808e6e62ca845c95cbb6ebb9ff15` |

Both held-out prediction gates have all-finite outputs and pass the existing broad development-smoke gate. Replay-off is better than constant velocity at all four horizons; replay-on is better at `0.2/0.3/0.5 s` but not at `0.1 s`. This broad gate is intentionally insufficient to admit replay into closed-loop control.

## TensorBoard Provenance

Each run has an independent event file under `results/jepa_v3_tensorboard/`. The TensorBoard audit found all required scalar/text artifacts, 40 points for both `Loss/train` and `Loss/validation`, and 149 histogram tags in each run. Replay-on additionally has 40 points each for `Replay/uniform_draw_fraction` and `Replay/hard_draw_fraction`, plus the immutable replay-manifest text artifact.

| TensorBoard run | Required provenance | Loss epochs | Histograms | Replay scalars |
| --- | --- | ---: | ---: | --- |
| `multitask_replay_off_seed20260911` | complete | 40 | 149 | not applicable |
| `multitask_replay_on_seed20260911` | complete | 40 | 149 | 40 / 40 |

## Held-Out Hard-Subset Evaluation

The validation archive contains 146,400 samples. Applying the fixed training predicate read-only gives 12,486 hard samples (8.53%) and 133,914 non-hard samples. Of hard validation samples, 12,486 are high-CBF correction, 20 meet the low-clearance predicate, and none has collision/boundary labels. P4 can therefore assess short-horizon error, CBF labels, and candidate ranking, but cannot establish a collision-classifier benefit.

The table gives `replay-off error - replay-on error`; positive values favour replay-on. Intervals are 95% paired bootstrap intervals using complete `(episode_seed, time_index, agent_id)` blocks, so the five candidates from one state-agent snapshot are not treated as independent rollouts.

| Horizon | Target MAE reduction (m) | Obstacle-clearance MAE reduction (m) | Inter-agent-clearance MAE reduction (m) | CBF-correction MAE reduction (m/s) |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 s | -0.0262 [-0.0327, -0.0201] | -0.0182 [-0.0396, 0.0031] | -0.0171 [-0.0232, -0.0116] | +0.0106 [-0.0033, +0.0253] |
| 0.2 s | -0.0257 [-0.0334, -0.0185] | -0.0064 [-0.0308, +0.0177] | -0.0313 [-0.0389, -0.0239] | +0.0105 [-0.0031, +0.0240] |
| 0.3 s | -0.0078 [-0.0143, -0.0012] | +0.0176 [-0.0042, +0.0386] | -0.0214 [-0.0291, -0.0134] | +0.0248 [+0.0090, +0.0408] |
| 0.5 s | -0.0370 [-0.0451, -0.0299] | +0.0301 [+0.0102, +0.0497] | -0.0186 [-0.0249, -0.0122] | +0.0248 [+0.0100, +0.0397] |

Replay-on gains hard-sample CBF-intervention AUROC by `+0.0281`, `+0.0075`, `+0.0025`, and `+0.0365` at `0.1/0.2/0.3/0.5 s`, and improves some longer-horizon CBF-correction/obstacle values. Those gains do not offset target-MAE and inter-agent-clearance degradation, whose intervals are negative at all four horizons.

The degradation is not isolated to hard data. On the non-hard subset, target-MAE reduction is `-0.0101`, `-0.0093`, `-0.0086`, and `-0.0159 m`; obstacle and inter-agent clearance MAE also worsen at every horizon. Thus the sampler is not a useful long-tail specialization that preserves nominal operation.

The bootstrap comparison JSON SHA-256 is `87de07365cc1b09367c0e7378805d994890c0c9da703c91365da51337ddedfd7`.

## Complete-Candidate Ranking

Ranking is evaluated on complete five-candidate groups. A group is hard when at least one candidate is hard; no candidate is removed before selecting a predicted winner. There are 2,569 hard groups and 26,711 non-hard groups.

| Horizon | Hard credit, off -> on | Hard win rate, off -> on | Non-hard credit, off -> on | Non-hard win rate, off -> on |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 s | 0.8846 -> 0.8895 | 64.19% -> 64.81% | 0.9505 -> 0.9451 | 71.67% -> 70.11% |
| 0.2 s | 0.9499 -> 0.9477 | 76.45% -> 76.88% | 0.9702 -> 0.9681 | 80.68% -> 78.93% |
| 0.3 s | 0.9518 -> 0.9552 | 77.70% -> 78.94% | 0.9786 -> 0.9771 | 85.43% -> 84.17% |
| 0.5 s | 0.9414 -> 0.9381 | 78.05% -> 78.16% | 0.9833 -> 0.9806 | 88.18% -> 87.10% |

Hard-group win rate has small gains, but ranking credit is mixed and non-hard ranking regresses consistently. Together with the sample-level errors, this is not a reliable ranking benefit.

## Action-Following Audit

The existing sensitivity audit uses the same 4,096 validation samples and `0.02` normalized final-action perturbation for both checkpoints.

| Run | All finite | Mean candidate separation | Mean antisymmetry |
| --- | --- | ---: | ---: |
| replay-off | yes | 0.00076257 | 0.00001180 |
| replay-on | yes | 0.00075099 | 0.00001318 |

Replay-on still responds non-trivially to candidate actions and has no numerical failure. This is a model-behaviour check, not a safety guarantee. The audit JSON SHA-256 is `af0aa7951b3c2fe06532c7dbc90b2abd353806c3a369bac145622d81a04273ab`.

## Gate Decision

| Gate | Result | Rationale |
| --- | --- | --- |
| Train-only provenance | pass | manifest/data hashes validate; uniform share is 50% |
| TensorBoard training record | pass | required scalar/text artifacts and histograms are present |
| Finite/shape/action-history contract | pass | both gates and action audit are finite |
| Hard-subset target/clearance benefit | fail | target and inter-agent-clearance errors worsen at every horizon |
| Non-hard preservation | fail | target and both clearance errors regress systemically |
| Candidate-ranking benefit | fail | hard gains are mixed and non-hard credit/win regress |
| Closed-loop smoke / S3 development | not run by design | the offline admission gate failed; no S3 result was used for tuning |

**P4 decision:** the replay implementation is retained as an auditable ablation, but replay-on is rejected for P5/P6. The next candidate-action work starts from the accepted P3 reliability-ledger model, not this reweighted checkpoint. This is a negative development result, not evidence against the broader interaction-aware JEPA + CBF hypothesis.

## Reproduction

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'

& $py scripts/evaluate_jepa_v3_replay_subsets.py `
  --checkpoint replay-off results/jepa_v3_multitask_replay_off_seed20260911/checkpoint.pt `
  --checkpoint replay-on results/jepa_v3_multitask_replay_on_seed20260911/checkpoint.pt `
  --replay-manifest results/jepa_v3_hard_replay/train_replay_manifest.json `
  --dataset results/jepa_v3_counterfactual_validation/counterfactual_multitask_dataset.npz `
  --metadata results/jepa_v3_counterfactual_validation/metadata.json `
  --bootstrap-replicates 1000 --bootstrap-seed 20260903 `
  --output results/jepa_v3_hard_replay/replay_off_on_validation_comparison_bootstrap1000.json `
  --device cuda
```

```text
19 passed: P1/P2/P3/P4 relevant JEPA tests
```
