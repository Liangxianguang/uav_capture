# DN-MPC P24 Listwise Temperature-0.005 Route-Progress Audit

**Status:** development-only; no online action or CBF policy was changed.

This audit consumes the P24 ranking details at
`results/dn_mpc_jepa_safe_capture_dev/p24_listwise_temp005_candidate_ranking_audit/ranking_details.csv`
with the registered tie and informative margins of `0.005`.

| Split | groups | exact top-1 | tie-aware top-1 | informative top-1 | pairwise agreement | top-gap <= margin |
|---|---:|---:|---:|---:|---:|---:|
| train | 127 | 60.63% | 94.49% | 100.00% | 94.91% | 74.80% |
| validation | 172 | 36.63% | 75.00% | 80.00% | 88.73% | 79.65% |
| calibration | 200 | 40.50% | 75.50% | 89.29% | 88.01% | 86.00% |

The validation informative-group result improved from P23's `51.43%` to
`80.00%`, while tie-aware top-1 improved from `61.63%` to `75.00%`. This shows
that listwise supervision at a lower temperature is learning meaningful route
ordering on groups with a non-negligible best-route gap. It does not eliminate
the exact top-1 failure on near-tied groups: `79.65%` of validation groups
remain within the registered tie margin.

The validation prediction-on-label slope is `0.1014`, with prediction standard
deviation `0.00937` versus label standard deviation `0.03204`; the score is
therefore still compressed. Exact top-1 and the missing runtime trace contract
remain blocking gates.

## Decision

P24 is retained as an offline development checkpoint. The executable authority
remains analytic DN-MPC plus strict joint CBF. The next milestone is P25
traceable counterfactual collection and fresh OOD/disagreement calibration, not
online promotion or another untracked model sweep.
