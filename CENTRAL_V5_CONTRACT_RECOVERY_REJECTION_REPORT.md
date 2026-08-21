# V5 Initial Contract-Recovery Rejection

## Scope

This report records the development rejection of
'results/central_v5/contract_recovery_seed661502'. It is a one-seed
development result, not a locked test, and it does not open V5 locked block
'647201'.

The run mixed a newly collected fixed-scene expert archive with the frozen V5
random archive using equal selected sequence counts. It was the first P2-0
recovery attempt after the V4/V5 contract audit found that the V5 fresh
baseline lacked fixed-scene retained-BC coverage.

## Evidence

- Checkpoint SHA-256:
  'c3b87633ecfbd078451d39b777c3ceeeeb352aaf32f185c06d15c75ae90a9ee3'
- Expert archive SHA-256:
  '4196b43587743fec61cdd3902d6bacd05faeebc6625de41f101f44e4496bce27'
- Archive sources: '322 -> 457' fixed sequences and '457 -> 457' frozen V5
  random sequences, selected with 'equal_sequences'.
- All source demonstrations satisfy the recorded safe and cooperative quality
  flags; all 64 imitation-loss epochs are finite.
- Raw/CBF S3 validation scenes are exactly paired, with static-scene SHA-256:
  '066b690d766a919a6c452eaa9fc9bf2b5b80093a8a32e920c1425274f0d4c1ab'.

## Development Validation

| Evaluation | Cooperative Safe Capture | Collision | Boundary | Transit |
| --- | ---: | ---: | ---: | ---: |
| S1 cylinder + CBF | 20.0% | 0.0% | 0.0% | 100.0% |
| S1 box + CBF | 95.0% | 0.0% | 0.0% | 100.0% |
| S1 wall + CBF | 100.0% | 0.0% | 0.0% | 100.0% |
| S2 mixed + CBF | 90.0% | 0.0% | 0.0% | 100.0% |
| S3 raw, 60 episodes | 1.7% | 98.3% | 0.0% | 100.0% |
| S3 + CBF, 60 episodes | 50.0% | 0.0% | 0.0% | 100.0% |

All 30 S3 + CBF failures terminate by timeout. The raw actor is unsafe and
cannot be presented as a deployable policy without the CBF execution layer.

## Rejection Decision

The candidate is rejected because it fails both pre-registered development
requirements:

1. Fixed CBF regression is below 98% for cylinder, box, and S2.
2. S3 + CBF Cooperative Safe Capture is 50.0%, below the 85% threshold.

The run must not proceed to three-seed replication or V5 locked-test.

## Why This Is Not a Full V4 Retained-BC Reconstruction

This recovery used equal sequence balancing, but it did not reproduce the full
historical training order that supplied the V4 shape-aware policy prior:

1. It did not train the V4-style three-stage, 640-episode fixed shape-aware
   curriculum before random-scene retention.
2. It therefore had no V4-style fixed-stage checkpoint from which to warm
   start the retained phase.
3. The fixed archive contained 322 source sequences and was resampled to the
   457 random-source sequence count; equal selection alone does not replace a
   dedicated fixed-stage optimization pass.

Accordingly, this result isolates the inadequacy of archive mixing alone. It
does not show that the full retained-BC construction is ineffective.

## Next Pre-Registered Step

Run P2-0.2 from 'NEXT_EXPERIMENT_SUPPLEMENT_TODOLIST_V8.txt':

1. Train the 640-episode fixed shape-aware stage with seed '661601'.
2. Verify its expert quality, layout coverage, finite loss, and checkpoint
   provenance.
3. Warm-start the retained stage with seed '661602', using the fixed archive
   and frozen V5 random archive with equal selected sequence counts.
4. Evaluate fixed regression before the paired S3 development block.

Do not change capture radius, the 2/4 cooperative definition, episode horizon,
CBF defaults, or the locked seeds during this recovery.
