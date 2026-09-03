# JEPA-v3 Action-Scale Contract Fix

**Status:** completed implementation and smoke verification

**Scope:** development-only counterfactual data collection. This report does
not change the V4/V5 locked benchmarks or authorize a JEPA control run.

## Defect

The frozen V5 actor checkpoint
`models/v5_development_exact_reactive_seed661606.pt` specifies
`action_scale: 5.0`. At deployment time,
`ActionConditionedCandidateHistory` divides each historical physical action
and candidate action by that scale before passing them to the predictor.

The original JEPA-v3 counterfactual collector stored raw m/s values. That
would have trained the model on action inputs five times larger than the
runtime inputs. The interrupted full P1 collection used that invalid contract
and is not a usable data artifact.

## Corrected Contract

The collector now:

1. resolves the actor checkpoint only from the frozen v3 protocol;
2. loads and verifies its positive `action_scale` before collection;
3. rejects an explicit `--action-scale` that does not exactly match it;
4. divides the seven executed historical actions and the final desired
   counterfactual action by that common scale;
5. preserves `candidate_action_norm_mps` in physical m/s for reliability and
   outcome analysis; and
6. records the checkpoint path, checkpoint SHA-256, scale, and normalization
   declaration in every `metadata.json`.

The dataset auditor now rejects missing/mismatched scale metadata and any
action history whose maximum absolute normalized component exceeds `1.05`.

## Smoke Evidence

Two newly generated, isolated one-episode-per-scenario data sets were audited:

| Split | Samples | Dataset SHA-256 | Max abs normalized action | Seed overlap |
| --- | ---: | --- | ---: | ---: |
| train | 4,880 | `adf462620e37d39f9d93caeb82acf5ccc185d71f167b95e39dfd5e6bd5992fd1` | `0.960788` | `0` |
| validation | 4,880 | `bc94d70bfb1f7771ab557c411222a28e82d25e7924eefb907c88252d7c14e662` | `0.937343` | `0` |

Both have a fixed five candidates per state-agent group, a `0.2` nominal
candidate fraction, finite values, balanced scenario sample counts, and the
same frozen actor checkpoint hash:
`535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`.

## TensorBoard Training Smoke

The corrected data was consumed by a two-epoch CUDA training smoke using seed
`20260911`. The run wrote its event file to:

```text
results/jepa_v3_tensorboard/multitask_normalized_smoke_seed20260911/
```

Event inspection confirmed scalar groups for training and validation loss,
target prediction, clearance, visibility, risk, calibration, and learning
rate; 149 parameter/gradient histograms; and text records for the protocol,
model, optimizer configuration, both data metadata files, and source hashes.

The resulting two-epoch checkpoint was finite but did **not** beat the
constant-velocity target baseline at any prediction horizon. It is therefore
marked `accepted_for_development_control_smoke: false`. This is expected from
a two-epoch plumbing smoke and must not be reported as a negative result for
the planned full three-seed training.

## Verification

```powershell
& D:\miniconda3\envs\uav-encirclement-gpu\python.exe -m pytest `
  tests/test_jepa_v3_counterfactual_dataset.py `
  tests/test_jepa_v3_multitask.py `
  tests/test_prediction.py -q
```

Result: `11 passed`.

The next admissible step is full P1 train/validation collection using this
corrected contract, followed by audit before any full training run.
