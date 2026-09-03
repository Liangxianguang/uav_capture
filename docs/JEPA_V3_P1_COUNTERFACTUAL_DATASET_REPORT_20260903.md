# JEPA-v3 P1 Counterfactual Dataset Report

**Status:** accepted for development-only multitask training

**Protocol:** `configs/jepa_v3_development_protocol.yaml`

**Collection source commit:** `8e3dcdb`

This report documents train/validation data for JEPA-v3 development. It does
not open a locked test, modify V4/V5 artifacts, or support a control claim.

## Collection Contract

Each retained policy-safe state contains five candidates: one nominal V5
controller action and four deterministic local alternatives. For every
candidate, the collector rolls a cloned kinematic environment through
`[0.1, 0.2, 0.3, 0.5]` seconds with CBF enabled and records offline-only target
relative displacement, obstacle and inter-agent clearance, visibility, CBF
correction/intervention, collision, and boundary labels.

The model inputs remain local 63-D policy-safe observations plus eight causally
aligned action tokens. All action history values are divided by the frozen
actor's verified `action_scale=5.0`; physical candidate magnitude remains a
separate m/s label.

## Reproduction Commands

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'

& $py scripts/generate_jepa_v3_counterfactual_dataset.py `
  --collection-config configs/action_conditioned_jepa_pilot.yaml `
  --output results/jepa_v3_counterfactual_train `
  --split train --candidate-count 5 --perturbation-mps 0.10 `
  --sample-stride 4 --chunk-length-steps 1 --action-scale 5.0

& $py scripts/generate_jepa_v3_counterfactual_dataset.py `
  --collection-config configs/action_conditioned_jepa_pilot.yaml `
  --output results/jepa_v3_counterfactual_validation `
  --split validation --candidate-count 5 --perturbation-mps 0.10 `
  --sample-stride 4 --chunk-length-steps 1 --action-scale 5.0
```

## Data Audit

| Property | Train | Validation |
| --- | ---: | ---: |
| Samples | 146,400 | 146,400 |
| Dataset SHA-256 | `6e8609484139fad93b427d8069f8f2517c472383962c036f2b6de9bd03c4b895` | `2176f09196a6c88787271ee7ee3f375311176163622c4df032fc078d107514d4` |
| Metadata SHA-256 | `ee70acd63a280205228cbf26267e6374b7ee6e1ea1c9a8be72944377c4cbb3e3` | `5c82070a8b2c30867d6afa699d793a0894acb026c7e52fb954247285fcd0373e` |
| State-agent groups | 29,280 | 29,280 |
| Candidates per group | exactly 5 | exactly 5 |
| Nominal candidate fraction | 0.200000 | 0.200000 |
| Scenario samples (each of 4) | 36,600 | 36,600 |
| Maximum abs normalized action | 1.019549 | 1.019972 |
| Frozen actor action scale | 5.0 | 5.0 |
| Episode-seed overlap | \- | 0 |
| Non-finite values | none | none |

The frozen actor checkpoint SHA-256 recorded by both collections is
`535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`.

## Label Coverage

| Label diagnostic | Train | Validation |
| --- | ---: | ---: |
| Minimum normalized obstacle clearance | 0.033158 | 0.010081 |
| Minimum normalized inter-agent clearance | 0.040595 | 0.042332 |
| Target visible fraction | 0.789185 | 0.789650 |
| CBF intervention fraction | 0.069027 | 0.061474 |
| Counterfactual collision fraction | 0.0 | 0.0 |
| Counterfactual boundary fraction | 0.0 | 0.0 |

The positive CBF-intervention coverage gives the risk heads valid labels while
the zero collision/boundary labels mean those rare terminal outcomes cannot be
learned as a useful classifier from this first P1 collection. Clearance and
CBF correction remain the appropriate short-horizon safety supervision.

## Admission Decision

The data meets the P1 contract:

- train and validation seed blocks are disjoint;
- no S3 development or locked data is included in training;
- candidate coverage and scenario coverage are balanced;
- all arrays are finite and have the frozen `8 x 63` / `8 x 3` model shape;
- action normalization matches the frozen runtime contract; and
- the dataset can be reproduced using the commands above.

It is admitted for P2 full training only. P2 checkpoints must still pass the
held-out prediction and action-following gates before any control evaluation.
