# V5 S3 Failure Analysis (Expert Feasibility Validation)

This report audits a fresh V5 validation artifact generated with `rule expert + CBF`.
It is a map/dynamics feasibility check, not a learned-policy improvement claim.

- Episodes: `60`
- Cooperative Safe Capture: `59/60`
- Cooperative failures: `1` (1.67%)
- Transit failures: `0`
- Hard-example rows: `60`
- Source CSV SHA-256: `b88d52192e199a56e3d225d50a45bd43ec8efdf4a02b866712a117934021410f`

## Failure stages

| Stage | Count |
| --- | ---: |
| timeout | 1 |

## Episode-level failures

| Episode | Episode seed | Layout seed | Condition | Obstacles | Stage | Termination | Min clearance (m) |
| ---: | ---: | ---: | --- | ---: | --- | --- | ---: |
| 29 | 646130 | 1646130 | delayed_noisy | 5 | timeout | timeout | 0.349 |

## Interpretation boundary

The expert result establishes that the V5 validation maps have a safe route under the current kinematic contract. It does not explain retained-BC failures, because no policy checkpoint was evaluated in this run. The same tool must be run on policy + CBF `episodes.csv` before selecting hard examples for training.
