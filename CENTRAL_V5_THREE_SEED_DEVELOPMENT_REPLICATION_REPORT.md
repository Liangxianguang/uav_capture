# Central V5 Three-Seed Development Replication Audit

This report aggregates independent training seeds, not episodes. It uses only the V5 development block and does not open the locked block.

## Candidate Results

| Retained seed | Fixed CBF (cylinder / box / wall / S2) | S3 CBF | Collision | Boundary | Transit | Paired raw/CBF scenes | Gate |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 661602 | 100.0% / 100.0% / 100.0% / 100.0% | 91.7% (60) | 1.7% | 1.7% | 100.0% | True | True |
| 661604 | 100.0% / 100.0% / 100.0% / 100.0% | 96.7% (60) | 1.7% | 1.7% | 100.0% | True | True |
| 661606 | 100.0% / 100.0% / 95.0% / 100.0% | 98.3% (60) | 1.7% | 1.7% | 100.0% | True | False |

## Provenance

- Seed `661602`: checkpoint `849b5fd97664e7d11d0d96d9a95c2e4c0179dcdcaf3d720b51a516b50e313ae4`, summary SHA-256 `7c2168ca7d1e16a55f316dfc524bd3edc33765ccd015d238d2a5a7f23bdbd17f`.
- Seed `661604`: checkpoint `44424cd4d0579ab6c5771e38c8ee799060c5ab304bea71d5af29b64ccb587558`, summary SHA-256 `9ab1c274f7c541a4641c104eb2970bd2bc6c132a91004f7203ca56ff812b2c4e`.
- Seed `661606`: checkpoint `4fe54f86b033b1d5290ffdaa8d1fb097f7e8b8491071e64f7baf1f8dbbb36bf3`, summary SHA-256 `0835e79fad3eb6fd4ce60ca69e1282792b2eb71a810c6023ce9aa6600cb2d47e`.

## Decision

The three-seed replication requirement is not met; locked block 647201 remains unopened. Failed development evidence: seed 661606: all_fixed_cbf_at_least_98_percent.
