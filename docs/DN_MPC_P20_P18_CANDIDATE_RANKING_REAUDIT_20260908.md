# DN-MPC P20 P18 Candidate-Ranking Re-audit

- Audit type: `dn_mpc_p20_p18_candidate_ranking_reaudit`
- Protocol: development-only; no action was executed
- Decision: **STOP BEFORE ONLINE INTEGRATION**
- Checkpoint SHA-256: `474e4178a41d3765e2b978c0eb1411f759c52ce36b5c0d31f1c682785f938848`
- Git revision at re-audit: `5d3e38a477b2865d7f642475138fa5919b1f5cbe`

## Geometry repair

P19 had 10/172 validation groups with zero candidates that were both
geometrically valid and first-step CBF-feasible. The route generator was
applying the obstacle-corridor test to `braking` and `verified_safe_hold`, so
the current centroid's low clearance rejected the very fallback that the
downstream CBF should verify. P20 preserves the geometric diagnostic but lets
those low-motion fallbacks reach the independent CBF check. Margins, stale/OOD
gates, acceleration limits, and `controlled_abort` were unchanged.

## Contract

- Model type: `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2`
- Route chunk shape contract: `[N,5,3]`; pairwise interaction: `[N,5,9]`
- Contract gate: `True`
- Finite output gate: `True`

## Ranking Results

| Split | groups | top-1 agreement | pairwise agreement | route identity | geometry accuracy |
|---|---:|---:|---:|---:|---:|
| train | 137 | 29.13% | 88.82% | 100.00% | 89.69% |
| validation | 172 | 19.19% | 84.32% | 100.00% | 84.23% |
| calibration | 214 | 20.50% | 84.25% | 100.00% | 83.53% |

The validation route-progress top-1 gate requires >= 50%; observed `0.19186046511627908`.
The validation informative pairwise gate requires >= 70%; observed `0.8431603773584906`.

The geometry eligibility result changed from P19's `10/172` zero-eligible
groups to `0/172`; the minimum eligible count is now 2. This repairs candidate
availability but does not repair the P18 route-progress top-1 ranking.

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

## Decision

P20 remains a stop before online integration and three-seed replay. Pairwise
direction is useful as a diagnostic, but 19.19% top-1 agreement is inadequate
for route selection. The next task is to audit the route-progress labels and
score direction, then collect selected/nominal/safe-hold CBF counterfactuals
and fresh uncertainty/disagreement calibration.
