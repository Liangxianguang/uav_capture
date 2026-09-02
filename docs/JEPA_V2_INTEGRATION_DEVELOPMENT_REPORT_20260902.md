# JEPA v2 Paired Development Aggregate

> This is development evidence on the frozen V5 development scenes. It is not a V4/V5 locked test and does not change any formal conclusion.

## Baseline

- Frozen V5 baseline: `57/60 = 95.00%` safe capture.
- Collision/boundary: `0.00%` / `0.00%`; transit `100.00%`.

## Three-seed family summary

| Family | Safe capture | Collision | Boundary | Mean capture time (s) | Path (m) | Paired capture delta (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| interaction_aware | 96.11% +/- 0.96% | 0.00% +/- 0.00% | 0.00% +/- 0.00% | 7.308 +/- 0.225 | 87.910 +/- 0.892 | 1.11 +/- 0.96 |
| plain | 93.89% +/- 0.96% | 1.11% +/- 1.92% | 1.11% +/- 1.92% | 6.911 +/- 0.251 | 86.827 +/- 2.567 | -1.11 +/- 0.96 |

## Per-seed paired outcomes

| Run | Safe capture | Collision | Boundary | Improved / degraded / tied vs baseline | Path delta (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| interaction_aware_seed20260911 | 58/60 (96.67%) | 0.00% | 0.00% | 2 / 1 / 57 | 4.274 |
| interaction_aware_seed20260912 | 57/60 (95.00%) | 0.00% | 0.00% | 1 / 1 / 58 | 2.553 |
| interaction_aware_seed20260913 | 58/60 (96.67%) | 0.00% | 0.00% | 2 / 1 / 57 | 3.820 |
| plain_seed20260911 | 56/60 (93.33%) | 0.00% | 0.00% | 1 / 2 / 57 | 5.163 |
| plain_seed20260912 | 56/60 (93.33%) | 3.33% | 3.33% | 2 / 3 / 55 | 2.183 |
| plain_seed20260913 | 57/60 (95.00%) | 0.00% | 0.00% | 2 / 2 / 56 | 0.052 |

## Interpretation

- Interaction-aware JEPA is the primary candidate because it is the only family whose prediction gate passed all four horizons for all three seeds.
- Capture deltas are episode-paired development evidence, not a claim of a statistically significant or locked improvement.
- CBF remains enabled in every control run; predictor uncertainty is a ranking feature, not a safety proof.
- The next required audit is action-following sensitivity: candidate actions must change predicted futures in a way that tracks the corresponding rollout differences.
