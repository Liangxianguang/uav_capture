# Baseline freeze report

The original capture-radius baseline is frozen before the hard-benchmark
extension. Its authoritative multi-seed result remains in
`results/CAPTURE_RADIUS_FINAL_REPORT.md`.

## Frozen task

- Four defenders and one evasive target;
- 20 m x 20 m x 10 m kinematic world;
- partial local target observations, occlusion, dropout, delayed teammate
  messages, and cylindrical obstacles;
- `r_capture = 0.80 m`;
- Safe Capture requires no prior collision or boundary violation.

## Frozen evidence

Three independently trained warm-start MAPPO seeds were evaluated on the same
locked block of 300 episodes each (open, clutter, and occluded scenarios):

| Metric | Frozen result |
|---|---:|
| Safe Capture | 100% +/- 0% |
| Collision | 0% +/- 0% |
| Mean capture time | 1.366 +/- 0.023 s |
| Mean minimum clearance | 0.843 +/- 0.025 m |

The historical checkpoint directories and TensorBoard outputs are kept in the
external local archive rather than tracked in Git. Reproduction commands and
source-hash rules are documented in the final report and `README.md`.

The hard benchmark does not alter the frozen actor input dimension or the
capture event. It is a held-out diagnostic extension for the next stage.
