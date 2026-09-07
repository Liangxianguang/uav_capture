# Route-Identity Hard-Negative Training (2026-09-07)

## Decision

This is a development-only offline result. The hard-negative route-JEPA
checkpoint is **not promoted** to a new Reliability Ledger or online runtime
policy because the held-out pairwise-hazard gate failed. The existing G5
runtime baseline and strict CBF contract remain unchanged.

## Contract

- Train split: `645101-645120`, 20 episodes, `24,856` samples.
- Validation split: `646101-646120`, 20 episodes, `21,424` samples.
- Calibration split: `648101-648120`, 20 episodes, `26,572` samples.
- Train/validation/calibration seeds are disjoint.
- Actor: `models/v5_development_exact_reactive_seed661606.pt`.
- Base JEPA: `results/jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt`.
- CBF: strict buffer, five-step archive rollout, 0.35 m obstacle/boundary/pairwise margins.
- Stale/OOD/non-finite gates, controlled abort, reachable projection and
  raw-unverified prohibition were unchanged.
- No locked split was opened and no online episode was executed.

## Archive audit

The three archives passed the read-only label and provenance audit:

| split | samples | runtime rows | geometry valid | first-step CBF feasible | horizon failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 24,856 | 22,944 | 81.40% | 100.00% | 7,388 |
| validation | 21,424 | 19,776 | 82.36% | 100.00% | 6,368 |
| calibration | 26,572 | 24,528 | 81.43% | 99.90% | 6,364 |

All required risk arrays are finite, visibility has both classes, runtime
boundary clearance remains non-negative, and the train/validation/calibration
episode seed sets are disjoint.

## Training

The model was initialized from the frozen route-identity v1 checkpoint and
fine-tuned with route-progress ranking plus stopping-distance, TTC, CBF
feasibility, clearance and intervention heads. Pairwise pooling was not used
because the preceding pooling smoke had already failed its calibration gate.

- Output checkpoint:
  `results/jepa_route_identity_hard_negative_actor_train20_v2_seed20260907/checkpoint.pt`
- Checkpoint SHA-256:
  `cf419b94c09bf6afa48047434780a1c0a9a9ab228d7ccfad38357929f175c51d`
- Device: CUDA, NVIDIA GeForce RTX 5050, PyTorch 2.7.1+cu128.
- 26 epochs completed; best validation loss at epoch 18; early stop after
  eight epochs without improvement.
- TensorBoard:
  `results/jepa_route_identity_hard_negative_actor_train20_v2_seed20260907_tb`

## Held-out prediction

Metrics use the default 0.5 hazard cutoff; `<=1 s` is the primary gate.

| split | pairwise recall | pairwise precision | obstacle recall | obstacle precision | boundary recall | boundary precision | stopping-distance correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 80.0% | 51.7% | 96.8% | 78.4% | 91.4% | 73.7% | 0.954 |
| validation | 68.4% | 51.3% | 86.1% | 76.4% | 82.6% | 68.0% | 0.945 |
| calibration | **66.3%** | **47.2%** | 91.8% | 73.1% | 83.2% | 70.0% | 0.955 |

The declared pairwise gate is `recall >= 80%` and `precision >= 50%` on
calibration. It fails on both dimensions (`66.3%`, `47.2%`). Increasing the
calibration block from four to twenty episodes did not change this decision,
so the result is not attributable only to a tiny calibration sample.

Prediction audit JSON:

`results/jepa_route_identity_hard_negative_actor_train20_v2_prediction_audit20_20260907.json`

Prediction TensorBoard:

`results/jepa_route_identity_hard_negative_actor_train20_v2_prediction_audit20_20260907_tb`

## Interpretation and stop rule

The hard-negative objective learns single-agent obstacle/boundary and braking
risk substantially better than the pairwise interaction tail. Pairwise TTC
remains poorly calibrated under split shift, so using it for Ledger routing
would either miss unsafe interactions or over-abstain and reduce capture. No
CBF margin or safety gate should be relaxed to compensate.

Stop the following for this checkpoint:

- Reliability Ledger construction or threshold tuning;
- nearest-tangent online execution;
- L0-L3 or multi-seed closed-loop evaluation.

The next permitted experiment is targeted, disjoint pairwise-transition data:
sample relative teammate velocity/topology changes under each candidate action
chunk, train or calibrate only the pairwise interaction head, and require the
same calibration gate before any runtime use. If that targeted archive still
fails, retain the current G5 runtime baseline and stop adding model capacity.
