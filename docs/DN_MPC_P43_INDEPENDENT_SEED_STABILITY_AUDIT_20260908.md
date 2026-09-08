# DN-MPC P43 Independent Evaluator-Seed Stability Audit

**Date:** 2026-09-08
**Phase:** development-only, offline-only
**Decision:** independent-seed stability gate failed; stop seed expansion

## Objective

P43 trains one evaluator from scratch with seed `393702` using the same P39
route-identity train/validation archives and the same action-conditioned,
interaction-aware JEPA architecture as P37 seed `373701`. Utility weights are
selected independently on the P39 calibration split and evaluated on the same
held validation archive. This is an offline evaluator audit, not an online
safe-capture run.

## Training and provenance

| Item | Value |
|---|---|
| Independent seed | `393702` |
| Architecture | hidden/latent `128/64`, pairwise pooling and relational features |
| Ranking objective | listwise, temperature `0.005` |
| Requested/completed epochs | `40 / 21` |
| Best epoch | `13` |
| Best validation loss | `2.1401464` |
| Stop reason | early stop after 8 stale epochs |
| Checkpoint SHA-256 | `2417bd3d65b72c94e96708d29fc4827f2ff31dbda9f8cba8ae616dcbd68290b1` |
| TensorBoard | `results/dn_mpc_jepa_safe_capture_tensorboard/p42_route_jepa_pairwise_seed393702/` |

The checkpoint uses the P39 train archive SHA-256
`a90090332b1a4fa7eb97771c96b10fc72a45677a87972503d73accbfc6bb2a4b` and
validation archive SHA-256
`72898cccc407a107820201b54cdb0152a62220df16d13d7fd89c0737e5cadcd4`.

## Utility audit

Both checkpoints were evaluated with the same bounded P42 grid:

```text
length = {0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0}
switch = {0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0}
cbf    = {0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0}
escape = {0, 0.1, 0.2, 0.5}
```

| Checkpoint | Calibration-selected `(length, escape, cbf, switch)` | Boundary coordinates | Validation exact | Validation informative | Validation pairwise | Validation vs planner |
|---|---|---|---:|---:|---:|---:|
| P37 `373701` | `(1.5, 0, 1.5, 1.5)` | none | 99.41% | 100.00% | 97.87% | 80.42% |
| P43 `393702` | `(2.0, 0, 0.1, 2.0)` | length, switch | 99.11% | 99.69% | 97.87% | 80.42% |

P43's planner-selected agreement matches P37 numerically, but the selected
utility coordinates are not stable: P43 moves both route-length and
route-switch weights to the top of the tested range and reduces the CBF weight
to `0.1`. The agreement therefore does not demonstrate a reproducible physical
utility decomposition. It is consistent with route-identity contract repair
being useful while the utility label remains underdetermined or differently
scaled across seeds.

## Gate decision

- [x] Independent evaluator trained from scratch with disjoint provenance.
- [x] Calibration-only weight selection and held validation evaluation completed.
- [x] TensorBoard training and audit runs recorded.
- [ ] Utility coordinates stable across seeds: **failed**.
- [ ] Online JEPA route override: **closed**.
- [ ] Ledger-Lite or three-seed paired replay: **closed**.
- [ ] Locked benchmark: **not opened**.

No action was executed. CBF margins, stale/OOD/non-finite gates,
controlled-abort behavior, route geometry and candidate eligibility were not
changed.

## Next bounded task

Stop adding evaluator seeds. Audit the physical normalization and label
contract before any more training:

1. Report per-term utility ranges and correlations for progress, route length,
   CBF feasibility and switch penalty on calibration.
2. Check whether route length is double-counting progress or encoding a unit
   mismatch (`m` versus normalized progress).
3. Define a pre-registered safety-first utility hierarchy or normalized costs,
   then rerun one calibration-only audit for both existing checkpoints.
4. Only if that fixed contract is stable may a new evaluator seed or online
   paired replay be considered.

## Artifacts

- Training checkpoint: `results/dn_mpc_jepa_safe_capture_checkpoints/p42_route_jepa_pairwise_seed393702/`
- Utility JSON: `results/dn_mpc_jepa_safe_capture_dev/p43_independent_seed_utility_audit/audit.json`
- Utility Markdown: `results/dn_mpc_jepa_safe_capture_dev/p43_independent_seed_utility_audit/report.md`
- Utility TensorBoard: `results/dn_mpc_jepa_safe_capture_tensorboard/p43_independent_seed_utility_audit/`
