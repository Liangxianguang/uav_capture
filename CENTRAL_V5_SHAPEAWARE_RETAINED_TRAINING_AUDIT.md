# Central V5 Shape-Aware Retained-BC Training Audit

## Scope

This audit covers P2-0.2 training only. It does not contain a fixed-scene
regression, S3 development result, or locked-test result.

The run was executed with the committed shape-aware recovery script and frozen
configuration:

- Fixed stage: 'fixed_shapeaware_seed661601'
- Retained stage: 'shapeaware_retained_seed661602'
- Environment configuration: 'capture_radius_pursuit_central_v4_flee.yaml'
- Capture contract: 0.80 m radius, safe capture, and 2/4 defender central-zone
  entry requirement.

## Fixed Shape-Aware Stage

| Check | Result |
| --- | --- |
| Requested / accepted expert episodes | 640 / 640 |
| Rejected episodes / rejection rate | 0 / 0.00% |
| Safe demonstrations | All true |
| Cooperative demonstrations | All true |
| Expert sequences / frames | 651 / 20,857 |
| Training epochs | 96 |
| Finite action-MSE loss | True |
| First / final action MSE | 2.67210765 / 0.02867453 |
| Checkpoint SHA-256 | '7ff8f7b77cb99bce8c416a0134e9fab524557441bb923b191d0253dc7455a6fc' |

Fixed-stage archive coverage:

| Layout | Accepted expert episodes |
| --- | ---: |
| cylinder | 298 |
| box | 106 |
| wall | 89 |
| cylinder_box | 98 |
| mixed | 49 |

The fixed stage therefore satisfies the P1 collection-quality gate. Its
training-internal raw actor evaluation is not used as a fixed-scene regression
or checkpoint-selection criterion.

## Warm-Start Retained Stage

The retained-stage 'initialization.json' references the fixed-stage checkpoint
above, and its recorded SHA-256 equals the independently computed fixed-stage
SHA-256.

| Check | Result |
| --- | --- |
| Warm-start checkpoint hash matches | True |
| Archive source balance | equal_sequences |
| Fixed archive: original / selected sequences | 651 / 651 |
| Frozen V5 random archive: original / selected sequences | 457 / 651 |
| Selected sequences per source equal | True |
| Total sequences / frames | 1,302 / 41,664 |
| All source demonstrations safe / cooperative | True / True |
| Training epochs | 64 |
| Finite action-MSE loss | True |
| First / final action MSE | 0.18518275 / 0.09261915 |
| Retained checkpoint SHA-256 | '849b5fd97664e7d11d0d96d9a95c2e4c0179dcdcaf3d720b51a516b50e313ae4' |

Resampling the smaller frozen random archive to 651 sequences is intentional:
the configured 'equal_sequences' contract balances source contributions in
sequence count. It is distinct from the rejected P2-0.1 run because P2-0.2
first learned the V4-style shape-aware fixed curriculum and then warm-started
the retained phase from that checkpoint.

## Source Provenance

The artifacts record source hashes. In particular, the active
'src/encirclement3d/pursuit_env.py' hash during these runs was:

'ea15252dcfd793ae0e3583cd5d8bb5c442f47490900b63aa131f275d1e4998fe'

That workspace file has user-owned, uncommitted changes. It is not staged or
modified by this experiment. The added execution-settings defaults are disabled
under this frozen configuration; nevertheless, its hash is retained so the run
is reproducible against the exact source tree used.

## Decision

P1 passes. The retained checkpoint may now enter P2 development validation:

1. Fixed S1 cylinder/box/wall and S2 mixed, raw and CBF, 20 episodes each.
2. Paired S3 development block 646101, raw and CBF, 60 episodes each.
3. Aggregate episode-level artifacts and apply the pre-registered gates.

No V5 locked seed is opened by this audit.
