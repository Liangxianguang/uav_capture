# V5 P3-A Wall-Coverage Recovery: Pre-registration

Date: 2026-08-21
Status: frozen before P3-A pilot training

## Research question

Can one data-only extension of the fixed wall curriculum remove the observed
fixed-wall regression instability while preserving the V5 S3 development
contract?

This is a recovery experiment, not a claim of an OOD wall generalization
method. The fixed wall layout is an ID regression layout; evaluation seed and
motion noise are still never used as training samples.

## Prior evidence and hypothesis

The V5 shape-aware replication has only two passing development seeds out of
three. Seed 661606 failed only `s1_wall + CBF` at 19/20 (95%), below the frozen
98% requirement. Its failure seed 660514 ended because the target crossed the
upper x-boundary at step 35; the nearest pursuer stopped at 0.847717 m rather
than reaching the 0.80 m capture radius. The rule expert + identical CBF on the
same seed and geometry succeeded at 2.8 s. See
`WALL_FIXED_REGRESSION_DIAGNOSTIC.md`.

The paired fixed archive had 94 accepted wall demonstrations, all at a 5.5 m
initial side distance. The fixed S1 wall regression uses 5.0 m. The
pre-registered hypothesis is that a wall-only 5.0/5.5 m coverage extension
will make the learned policy approach/capture early enough on this ID
fixed-wall regime without altering safety execution.

## The one and only intervention

The P3-A fixed stage uses the original V4 shape-aware 640-episode curriculum,
with a single conditional sampling change in the stages that contain `wall`:

| Property | Previous fixed stage | P3-A wall-coverage fixed stage |
| --- | --- | --- |
| `wall.initial_side_distances` | `[5.5]` | `[5.0, 5.5]` |
| cylinder / box / cylinder_box / mixed distance | `[5.5]` | unchanged `[5.5]` |
| side, speed, target mode, crossing, zone entries | left / 0.45 / flee persistence / false / 2 | unchanged |
| fixed-stage requested accepted experts | 640 | unchanged 640 |
| rejection threshold, optimizer, network, epoch | 25%, Adam 3e-4, 128, 96 | unchanged |
| retained random archive, source balance, epoch | frozen archive, equal sequences, 64 | unchanged |
| CBF-default, task success definition, horizons | unchanged | unchanged |

The implementation only permits `layout_overrides` for listed showcase
layouts and only for sampler fields. No pursuit/CBF setting can be overridden
by this mechanism. The new P3-A fixed YAML is SHA-256
`83e80a2a175705e9325353901e78c29cb0a2b7d632b83b7ce28b64bd737f4e31`; the
retained YAML is SHA-256
`4e85967a3d4c1b85b8d68c6568da260e8764b7e0012c362f329df06619eb660f`.

## Seed and data rules

- Pilot fixed/retained training seeds are exactly `661701` / `661702`.
- The new accepted wall examples are generated only by the rule expert. They
  must all be safe and meet 2/4 central-zone entry.
- Training collection seeds are generated from the fixed seed and are excluded
  from all evaluation blocks: fixed `660501`–`660520` (including `660514`),
  development S3 `646101`–`646160`, and locked S3 `647201`–`647300`.
- The fixed geometry remains a known ID layout, but seed `660514`, the exact
  evaluation rollout, and its rendered trace are not added to training.
- The frozen random archive remains
  `results/central_v5/bc_baseline_seed661401/expert_sequence_dataset.npz`
  with SHA-256 `de5386e512e3458b902c438cf1ada94cc6ab81acfa1a1f591d2ccbaf6bcbbaac`.
- In the retained stage, the P3-A fixed archive and frozen random archive must
  contribute exactly the same selected sequence count.

## Training integrity gates

Before evaluation, require all of the following:

1. exactly 640 accepted fixed expert episodes; rejection rate <= 25%; all
   accepted demonstrations safe and cooperative;
2. both wall distance strata (5.0 and 5.5 m) are represented by at least 25
   accepted wall episodes each; this is a reject-only quality condition, not a
   prompt to change the seed or collect extra episodes;
3. no checkpoint is evaluated if the fixed stage lacks finite loss or if its
   archive/checkpoint integrity record is incomplete;
4. retained warm-start SHA equals the P3-A fixed checkpoint SHA; source
   balance is `equal_sequences`; all retained loss values are finite;
5. raw actor and policy + CBF are reported separately.

## Fixed development evaluation protocol

No protocol change is permitted:

- S1 cylinder / box / wall and S2 mixed, raw and CBF, 20 episodes each;
  fixed seed block begins at 660501.
- S3 development raw and CBF, 60 episodes each, seed block `646101`.
- Raw/CBF must use exactly paired static maps, initial states, target profiles,
  and episode seeds; static-scene SHA must match.
- All standard episode-level metrics and difficult-group failure index remain
  mandatory.

## Pilot decision rule

The P3-A pilot passes only if all conditions hold:

- each fixed CBF regression rate >= 98%;
- S3 + CBF Cooperative Safe Capture >= 85%; Collision <= 2%; Boundary <= 2%;
  Transit >= 99%;
- raw/CBF S3 pairing is exact; training integrity gates above pass.

If any condition fails, write `P3A_WALL_COVERAGE_REJECTION_REPORT.md`, retain
the artifacts, stop V5 tuning, and do not introduce a second factor. If all
conditions pass, freeze this candidate and run three independent development
training seeds before opening 647201.

## Frozen inputs and source identity

- Old fixed YAML SHA-256:
  `b2bff851d89722f5421addbfb6567e1934b6db7f61be0ef927ce1504218e79a9`.
- Source fixed archive SHA-256:
  `05e656d743eb774c690cde97ff2ff1acdbd8c3b131f53f85caea1a70abdb82c5`.
- User-owned `src/encirclement3d/pursuit_env.py` SHA-256:
  `ea15252dcfd793ae0e3583cd5d8bb5c442f47490900b63aa131f275d1e4998fe`.
  This file is not staged, committed, replaced, or reverted by this experiment.
- `showcase.py` sampler source SHA-256 at pre-registration:
  `2db09a7d7e7aa18978c8afa225dd7d0ca4894d68017c596ab43848f5d00c73a4`.
- Training script source SHA-256:
  `b263160a62afe46b1c6544cc010c4b1f06a60e7b7ce9781ac7f6dff537d334de`.
