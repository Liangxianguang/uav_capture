# DN-MPC P28 Hard-Negative Replay and Augmented JEPA Audit

**Status:** development-only; offline-only; no online JEPA route action,
Ledger-Lite, or locked test was opened.

## Scope

P28 targets the two explicit P26 abstention states without relaxing safety:

- train: `scenario=6, time=47`;
- calibration: `scenario=5, time=47`.

For the preceding window (`time=39..47`), four read-only branches were
recorded: `braking`, `left_detour`, `right_detour`, and `boundary_rescue`.
They use sample types `5..8`, keep `route_candidate_index=-1`, pass through
reachable dynamics and the unchanged five-step CBF counterfactual, and are
never sent to `env.step()`.

## Archive and TensorBoard Artifacts

| Item | Train | Calibration |
|---|---:|---:|
| hard-negative archive rows | 144 | 144 |
| rows per route | 36 | 36 |
| runtime rows retained from P25 window | 432 | 432 |
| first-step CBF feasibility, each route | 88.89% | 88.89% |
| branch-failure fraction, each route | 22.22% | 11.11% |
| augmented archive rows | 8,912 | 13,840 |

Replay audit:
`results/dn_mpc_jepa_safe_capture_dev/p28_hard_negative_replay_audit/`.
Augmented archive provenance and TensorBoard runs are under
`results/dn_mpc_jepa_safe_capture_dev/p28_augmented_*_v2/` and
`results/dn_mpc_jepa_safe_capture_tensorboard/p28_augmented_*_v2/`.

The audit confirms all four hard-negative route types have candidate index
`-1`, finite arrays, unchanged runtime contract, and
`raw_unverified_execution_allowed=false`.

## Augmented Training

The P25 train archive was augmented only with the 144 P28 hard-negative rows.
The P25 validation archive remained frozen. A new listwise route-JEPA was
trained with the P24 structure and temperature `0.005`:

- seed: `282801`;
- device: RTX 5050, CUDA (`torch 2.7.1+cu128`);
- epochs completed: `21` (early stop after 8 stale epochs);
- best epoch: `13`;
- best validation loss: `2.210235354512237`;
- checkpoint:
  `results/dn_mpc_jepa_safe_capture_checkpoints/p28_route_jepa_augmented_seed282801/checkpoint.pt`;
- checkpoint SHA-256:
  `ca4d23299697052dfbb653a5ba83b6aad659eea1977b147df16176b717430749`;
- TensorBoard:
  `results/dn_mpc_jepa_safe_capture_tensorboard/p28_route_jepa_augmented_seed282801`.

## Ranking Comparison

The comparison uses the same frozen P25 validation block and the same strict
candidate eligibility mask.

| Model | validation exact top-1 / model-vs-truth | model-vs-selected | pairwise agreement |
|---|---:|---:|---:|
| P24 listwise temperature 0.005 | 36.63% | 19.19% | 88.73% |
| P28 augmented checkpoint | **47.09%** | **39.53%** | 87.74% |

The hard-negative augmentation produces a real offline improvement in direct
route agreement: `+10.46 pp` versus truth and `+20.35 pp` versus the analytic
selected route. Pairwise agreement is slightly lower by `0.99 pp`, so the
improvement is not a uniform gain across every metric.

Fresh P27 calibration still passes finite/OOD/error-coverage checks, but the
candidate-agreement gate remains below the required `50%` (`39.53%`). The
standard P18 audit also reports one train and one calibration zero-eligible
abstention group, and selected/nominal/safe-hold traceability is incomplete on
those explicit abstentions.

## Decision

P28 is a useful training-data and failure-mode milestone, not an online
promotion. Do not create Ledger-Lite, run three-seed paired replay, lower CBF
margins, or execute raw-unverified actions. The next step is a bounded
reproducibility check of the augmented archive/checkpoint on an independent
seed or a targeted label/route utility audit; promotion remains blocked until
candidate eligibility, exact top-1, fresh calibration, and complete
selected/nominal/safe-hold trace gates pass together.
