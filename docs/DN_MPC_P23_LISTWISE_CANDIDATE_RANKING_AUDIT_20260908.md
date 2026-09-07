# DN-MPC P19 P18 Candidate-Ranking Audit

- Audit type: `dn_mpc_p19_p18_candidate_ranking_audit`
- Protocol: development-only; no action was executed
- Decision: **STOP BEFORE ONLINE INTEGRATION**
- Checkpoint SHA-256: `ff47f60dd715db133b61cf8a354caa0ec3afb20ca4a95466ddeb8ea14dad66a9`
- Git revision: `c0817207f5faff1fe45965badb4fec29e1d4e21e`

## Contract

- Model type: `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2`
- Route chunk shape contract: `[N,5,3]`; pairwise interaction: `[N,5,9]`
- Contract gate: `True`
- Finite output gate: `True`

## Ranking Results

| Split | groups | top-1 agreement | pairwise agreement | route identity | geometry accuracy |
|---|---:|---:|---:|---:|---:|
| train | 137 | 44.88% | 92.35% | 100.00% | 88.87% |
| validation | 172 | 29.65% | 83.47% | 100.00% | 84.58% |
| calibration | 214 | 31.00% | 84.47% | 100.00% | 83.64% |

The validation route-progress top-1 gate requires >= 50%; observed `0.29651162790697677`.
The validation informative pairwise gate requires >= 70%; observed `0.8346698113207547`.

## Counterfactual Traceability

- Nominal rows present: `True`
- Verified safe-hold rows present: `True`
- Selected-candidate independent CBF trace present: `False`
- Three-way counterfactual gate: `False`

## Promotion Gates

- `contract`: `True`
- `finite_outputs`: `True`
- `candidate_eligibility`: `False`
- `score_direction`: `True`
- `top1_ranking`: `False`
- `ood_disagreement_binding`: `False`
- `counterfactual_traceability`: `False`

The checkpoint remains offline-only until selected/nominal/safe-hold are recorded as independent CBF counterfactuals and OOD/disagreement fields are bound to a fresh calibration ledger.
