# P3-A Wall-Coverage Pilot Rejection Report

## Decision

**Rejected.** The pre-registered P3-A wall-only fixed-stage distance coverage
intervention does not pass its fixed-regression gate. V5 tuning stops here.
V5 locked block `647201` remains unopened; no MAPPO, CBF-margin change,
second curriculum intervention, or fresh re-run is authorized by this result.

This report retains a negative result. It is not evidence that the intervention
improves V5, nor a reason to select a better random seed.

## Pre-registered intervention

P3-A changed exactly one data-sampling factor before training:

- only `wall` examples in the 640-episode fixed stage sampled initial side
  distance from `[5.0, 5.5] m` rather than `[5.5] m`;
- non-wall layouts, 4+1 task definition, `r_capture=0.80 m`, 2/4 zone-entry
  rule, observation contract, CBF-default, fixed/random archive balance,
  network, learning rate, and epoch budgets were unchanged.

The pre-registration SHA-256 is
`a0624632ba19417408d0957ff6e531f0b66607041182e98c83c6d23d4b697f76`.

## Training integrity: passed

The new fixed/retained seed pair was `661701` / `661702`.

| Integrity check | Result |
| --- | --- |
| Fixed accepted / requested expert episodes | 640 / 640 |
| Fixed rejected / attempts | 0 / 640 |
| Fixed safe / cooperative demonstrations | True / True |
| Wall 5.0 m / 5.5 m accepted examples | 58 / 53 (each >= 25) |
| Fixed epochs / finite loss | 96 / True |
| Fixed checkpoint SHA-256 | `54b543343fff6a26b91c8d686ca63f714a7f54e674c080429aec5117d954554d` |
| Retained warm-start hash matches fixed checkpoint | True |
| Retained source selection | equal sequences: 646 / 646 |
| Retained epochs / finite loss | 64 / True |
| Retained checkpoint SHA-256 | `09284ff66b4297642d626ad3364999c5e2ed66d890f335fae6fcd234132c898b` |

The complete training audit is
`CENTRAL_V5_WALL_COVERAGE_PILOT_TRAINING_AUDIT.md`.

## Fixed regression stopping evidence

The frozen fixed sequence is evaluated before S3. The first two raw/CBF
artifacts for `s1_cylinder` completed:

| Scene | Execution | Cooperative Safe Capture | Collision | Boundary | Transit |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 cylinder | raw actor | 1/20 = 5.0% | 95.0% | 0.0% | 100.0% |
| S1 cylinder | policy + CBF | **19/20 = 95.0%** | 0.0% | 0.0% | 100.0% |

The CBF fixed-regression requirement is `>= 98%` in **every** fixed scene.
`19/20 = 95.0%` therefore fails before the next fixed scene. The one failed
CBF episode was seed `660508`: it was a safe timeout with all four pursuers
entering the central zone, not a collision or boundary violation. Its minimum
clearance was `0.374746 m`, and its maximum CBF correction was `0.738536`.

Fixed artifact hashes:

- raw `episodes.csv`: `f6e434136a67620a91005f71d9295628795579fae42d7245a8f4ec32112c314c`
- CBF `episodes.csv`: `f45b112207b63ef95e0be456146de94dc90c3006e71e3555769c79a61539dc0a`

## Protocol-conformant stop

Per `V5_WALL_COVERAGE_P3A_PREREGISTRATION.md`, a fixed CBF result below 98%
rejects the pilot. Consequently:

1. the in-progress next fixed artifact was interrupted before it wrote episode
   results and its empty directory was removed;
2. no box, wall, S2, or S3 artifact was run for this rejected candidate;
3. no `candidate_gate_passed` value is calculated, because the complete
   development protocol was correctly not opened after its mandatory first
   gate failed;
4. no locked seed, checkpoint selection, hyperparameter adjustment, CBF
   adjustment, or visual sample selection used `647201`.

## Interpretation and next work

P3-A has not established a stable wall-specific recovery: it failed to preserve
even the cylinder fixed regression under its independent pilot seed. This does
not invalidate the historical V4 locked baseline; it shows that the single
wall-distance data change is insufficient under V5's required joint fixed/S3
contract.

The planned V5 tuning loop is therefore closed. The defensible current result
remains V4 retained-BC + CBF on historical locked block `647001`. Future work
must be a separately pre-registered research question (for example an E1
dynamics transfer or E2 synthetic sensing transfer), with a fresh task split;
it must not be described as a continuation selected on this V5 development
failure.
