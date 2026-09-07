# DN-MPC P24 Listwise Temperature-0.005 Candidate-Ranking Audit

**Status:** development-only; no online route action or CBF policy was changed.

P24 retrained the interaction-aware action-conditioned route-JEPA with listwise
utility ranking at temperature `0.005`, using seed `232302` and the same P17
train, P20 validation, and P15 calibration archives. The checkpoint is:

`results/dn_mpc_jepa_safe_capture_checkpoints/p24_route_jepa_listwise_temp005_seed232302/checkpoint.pt`

Checkpoint SHA-256:

`f72b851f4c7f89e8f41b7cbac6d37f72b55186f4a6443daf03126f9d510ab797`

The training TensorBoard run is preserved at
`results/dn_mpc_jepa_safe_capture_tensorboard/p24_route_jepa_listwise_temp005_seed232302`.
The audit TensorBoard run is preserved at
`results/dn_mpc_jepa_safe_capture_tensorboard/p24_listwise_temp005_candidate_ranking_audit`.

## Ranking Results

| Split | groups | exact top-1 | pairwise agreement | route identity | geometry accuracy | zero-eligible groups |
|---|---:|---:|---:|---:|---:|---:|
| train | 137 | 60.63% | 94.91% | 100.00% | 89.45% | 10 |
| validation | 172 | 36.63% | 88.73% | 100.00% | 83.37% | 0 |
| calibration | 214 runtime / 200 ranked | 40.50% | 88.01% | 100.00% | 82.10% | 14 |

Compared with P23, validation exact top-1 increased from `29.65%` to
`36.63%`, and pairwise agreement increased from `83.47%` to `88.73%`.
The improvement is real but the registered exact top-1 promotion threshold is
`50%`, so this checkpoint does not qualify for direct route selection.

## Promotion Gates

| Gate | Result | Reason |
|---|---|---|
| contract | pass | `[N,5,3]` route chunks and `[N,5,9]` interaction chunks are intact |
| finite outputs | pass | all split outputs are finite |
| score direction | pass | validation pairwise agreement is `88.73%` |
| candidate eligibility | fail | train and calibration still contain zero-eligible groups |
| exact top-1 | fail | validation exact top-1 is `36.63% < 50%` |
| selected/nominal/safe-hold traceability | fail | selected independent CBF trace is absent |
| OOD/disagreement binding | fail | no fresh calibration-ledger binding is present |

## Decision

P24 is a positive offline result and a better candidate evaluator than P23, but
it remains offline-only. Do not connect it to online route override, a
reliability ledger, a locked test, or a three-seed paired replay. Do not change
CBF margins, stale/OOD gates, or `controlled_abort` semantics.

The next work item is a traceability/calibration contract: record independent
CBF counterfactuals for the selected candidate, nominal anchor, and verified
safe-hold branch, then bind OOD and rollout disagreement to a fresh calibration
archive. Further temperature sweeps should wait until that contract is
available; otherwise improvements cannot be distinguished from an audit gap.
