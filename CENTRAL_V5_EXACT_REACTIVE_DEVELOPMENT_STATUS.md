# Central V5 Exact-Reactive Development Status

## Scope

This document records development evidence only. V5 locked seed block `647201`
has not been opened. The existing V4 locked-test result (`75.3% +/- 6.5%`
S3 Cooperative Safe Capture for the retained-BC + CBF stack) remains the only
formal result for that benchmark.

## Frozen Inputs And Controlled Difference

The exact-reactive candidates reuse the byte-identified V4 fixed archive
(`7666a03c446ca04ecb802664bd92817547aac1736ed041ebd8a76d0050085f48`),
the frozen V5 S3 archive
(`69f20885b12596ee3b33f52e4bca61c38720ac393f360221902936808b6249c5`),
equal sequence balance, V4 warm start, 128-dimensional actor, 64 epochs,
full-range action scale, and fixed CBF/environment protocol. The explicitly
audited difference is `sequence_length=1`, which matches deployment by
resetting the recurrent hidden state at every control step.

## Exact-Reactive Three-Seed Development Results

| Seed | Fixed CBF screen | S3 CBF (60 episodes) | Decision |
| --- | --- | --- | --- |
| 661602 | Cylinder/Box/Wall/S2: 20/20 | 53/60 (88.3%); collision 3.3%; boundary 3.3%; Transit 100%; paired scenes | Rejected: S3 execution-safety gate |
| 661604 | Cylinder 20/20; Box 20/20; Wall 19/20; S2 20/20 | Not run after fixed failure | Rejected: fixed Wall gate |
| 661606 | Cylinder/Box/Wall/S2: 20/20 | 57/60 (95.0%); collision 0%; boundary 0%; Transit 100%; paired scenes | One-seed development pass only |

All three training audits passed the frozen archive, source-balance,
initialization, loss-finiteness, and non-loader source-hash checks. The
approved single-step loader hash is recorded in each audit.

## Initialization And Determinism Forensics

The V4 warm-start checkpoint contains all 15 actor tensors. Before optimizer
updates, actor hashes for seeds `661602`, `661604`, and `661606` were identical
(`070d33d0a2df77be8912ebb54bf0ac83612a0b30fe592ed0bd6b0dbf927e5330`) and
matched the warm start exactly. The earlier fixed-scene regressions therefore
are not caused by partially random actor initialization.

Three executions of the `661606` exact-reactive contract produced the same
checkpoint SHA-256
(`535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`): the
historical run, a run with deterministic PyTorch kernels enabled, and a run
with the safety-weighting code present but disabled. The deterministic runs
recorded `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disabled TF32 and cuDNN
benchmarking, and used deterministic algorithms. CUDA numerical
nondeterminism is therefore not the source of the cross-seed variation.

## Rejected Low-LR Recoveries

1. `1e-5` for 64 epochs, pilot seed 661701: rejected in the fixed CBF screen
   (Cylinder 19/20, Box 19/20, Wall 20/20, S2 18/20). These were timeouts and
   the final imitation MSE was 0.17634, consistent with underfitting.
2. `1e-5` for 320 epochs, pilot seed 661801: audit passed and final action MSE
   was 0.09916, but Cylinder was 19/20 in the fixed CBF screen. It is rejected
   by its preregistered decision rule.
3. Fixed archive-sampling seed, pilot 661901: both archive hashes, 650:650
   sequence balance, and the fixed sampling seed 660901 passed audit, but S2
   was 19/20 in the fixed CBF screen. It is rejected without S3 evaluation.

## Rejected Safety-Critical BC Candidate

The preregistered safety-critical behavior-cloning run `662101` kept the two
frozen archives, 650:650 balance, V4 warm start, optimizer, and 64-epoch
contract, but assigned a fourfold normalized action-MSE weight to local action
frames within `0.85 m` of an observed obstacle. Its training provenance audit
passed, and 9.69% of action frames were weighted (mean action weight 1.291).

It failed the fixed CBF gate: Cylinder `18/20`, Box `20/20`, Wall `20/20`, and
S2 `18/20`. Cylinder had 10% collision and boundary-violation rates. Under
the preregistered decision rule, this method family is closed: seeds `662102`
and `662103` are not trained, and no S3 development or locked evaluation is
performed for it.

## Rejected Actor-SWA Candidate

The actor-SWA run `662201` retained uniform behavior cloning and the same
frozen data/optimizer contract, but saved the actor as the arithmetic mean of
epochs 48 through 64 (17 snapshots). Its provenance audit passed. Fixed CBF
results were Cylinder `20/20`, Box `20/20`, Wall `20/20`, and S2 `19/20`.
Although this isolates the remaining instability to the S2 scene, `19/20` is
below the preregistered all-pass gate. The actor-SWA family is therefore closed
without S3 evaluation; seeds `662202` and `662203` are not trained.

## Current Decision

Only one checkpoint (`exact_contract_reactive_seed661606`) passes the complete
development gate. The requirement for three independent complete passes is not
met; therefore no checkpoint is frozen for V5 locked testing and `647201` must
remain unopened. The high S3 development results from the earlier four-source
reactive experiment are not a formal replacement because all three models
regressed on at least one fixed CBF scenario.

Any next recovery must be preregistered as a new method or data-contract
change, evaluated on the same development protocol, and must produce three
independent full passes before a locked-test comparison is permitted.
