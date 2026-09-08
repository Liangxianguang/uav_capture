# DN-MPC P36 Route-Switch Contract Pilot

**Date:** 2026-09-08
**Phase:** development-only archive contract validation
**Status:** contract passed; no JEPA checkpoint promotion and no locked test opened

## Purpose

P36 closes a traceability gap identified in P28-P31. The route archive now records
the route that was actually executed after the final CBF filter, the previous
executed route, the observed route-switch outcome, and the action-match residual.
It also keeps independent selected/nominal/safe-hold CBF counterfactual traces
and an offline target-escape label. These fields are observational labels; they
do not authorize an unverified action or replace the CBF safety certificate.

## Protocol and provenance

| Item | Value |
|---|---|
| Dataset version | `dn_mpc_route_identity_chunk5_p36_route_switch_v1` |
| Candidate contract | 12 geometry-conditioned routes, chunk length 5 |
| CBF contract | strict buffer, anticipatory horizon 5 |
| Actor | `models/v5_development_exact_reactive_seed661606.pt` |
| Actor SHA-256 | `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba` |
| GPU environment | PyTorch `2.7.1+cu128`, NVIDIA GeForce RTX 5050 |
| TensorBoard | `results/dn_mpc_jepa_safe_capture_tensorboard/p36_route_switch_contract_*` |

The archive configuration, environment configuration, protocol, actor
checkpoint, dataset, metadata, and TensorBoard directory are recorded in each
archive's `provenance.json`. Both archives have `locked_test_opened=false` and
`development_only=true`.

## Pilot results

| Split | all rows | runtime rows | route match | previous route known | switch / known previous | route residual P95 | independent CBF trace | geometry valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train pilot | 2,392 | 2,208 | 100.00% | 95.65% | 4.55% | 0.356 m/s | 100.00% | 83.88% |
| calibration pilot | 6,344 | 5,856 | 100.00% | 98.36% | 5.00% | 0.368 m/s | 100.00% | 70.42% |

All arrays passed finite-value checks. Runtime route IDs are never inferred from
future truth: they are assigned only by matching the final CBF-filtered first
step against the candidate route action with a 1.0 m/s tolerance. Calibration
contains 100 branch failures within the five-step offline counterfactual
horizon, but its first-step CBF-feasible fraction is 100%; these failures are
retained as labels and are not executed.

The machine-readable audit is:

`results/dn_mpc_jepa_safe_capture_dev/p36_route_switch_contract_audit.json`

## Verification

The P36 archive audit passed for both train and calibration, including:

- route identity fields and switch outcomes;
- target-escape label presence and non-negativity;
- selected/nominal/safe-hold independent CBF trace coverage;
- action-chunk shapes `[N, 5, 3]` and pairwise action shapes `[N, 5, 9]`;
- finite arrays and TensorBoard scalar coverage;
- preservation of controlled abort, stale/OOD/non-finite gates, and the no-raw-unverified execution boundary.

Relevant tests: `65 passed` across the route-identity collector, candidate
contract, and failure-index regression suites.

## Decision and next gate

P36 passes the archive traceability gate and supports a larger disjoint train /
validation / calibration collection. It does **not** show a safe-capture-rate
improvement, does not justify online JEPA route override, and does not justify
reusing an old Ledger. The next experiment is a sufficiently sized disjoint
archive collection followed by fresh route-utility calibration; only if
candidate agreement and calibration gates pass should a new development
checkpoint be trained. Locked benchmark evaluation remains closed.
