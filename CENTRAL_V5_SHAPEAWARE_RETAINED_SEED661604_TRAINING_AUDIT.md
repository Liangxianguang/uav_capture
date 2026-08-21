# Central V5 Shape-Aware Retained-BC Training Audit

This record audits training provenance only. It is neither a development evaluation nor a locked test.

## Fixed Shape-Aware Stage

| Check | Result |
| --- | --- |
| Requested / accepted expert episodes | 640 / 640 |
| Rejected episodes / rejection rate | 0 / 0.00% |
| Safe / cooperative accepted demonstrations | True / True |
| Expert sequences / frames | 647 / 20710 |
| Training epochs / finite action-MSE | 96 / True |
| First / final action MSE | 2.63087539 / 0.02858412 |
| Checkpoint SHA-256 | `019ecc2e8d5a92ff2b155a81711ccea20b315d44a1f84464326867635b120f52` |

Fixed-stage accepted expert coverage:

| Layout | Accepted expert episodes |
| --- | ---: |
| box | 112 |
| cylinder | 296 |
| cylinder_box | 84 |
| mixed | 39 |
| wall | 109 |

## Warm-Start Retained Stage

| Check | Result |
| --- | --- |
| Warm-start checkpoint hash matches fixed checkpoint | True |
| Archive source balance / selected sequence balance | equal_sequences / True |
| All source demonstrations safe / cooperative | True |
| Source 0: original / selected sequences | 647 / 647 |
| Source 1: original / selected sequences | 457 / 647 |
| Total sequences / frames | 1294 / 41408 |
| Training epochs / finite action-MSE | 64 / True |
| First / final action MSE | 1.28566858 / 0.09105545 |
| Retained checkpoint SHA-256 | `44424cd4d0579ab6c5771e38c8ee799060c5ab304bea71d5af29b64ccb587558` |

## Source Provenance

- `scripts/train_capture_radius_recurrent_behavior_cloning.py`: `b263160a62afe46b1c6544cc010c4b1f06a60e7b7ce9781ac7f6dff537d334de`
- `src/encirclement3d/learning.py`: `2cb49096c19d8e82966ff503f16e59151fa2d310b12168e44f3418b7779780fd`
- `src/encirclement3d/observation_encoding.py`: `7fc2328cabda73971f141bc60682c01899d2fcff4270a3bf49f883921a84f8f6`
- `src/encirclement3d/prediction.py`: `6fb10e1c0768140a6d62244c27f20e11071d04aa0a7622c176d0fedf60136423`
- `src/encirclement3d/pursuit_controllers.py`: `86a4b1e0e0ba49ebf69d1a78e17ba1d29caa06d58737366ee7fd1ac36d8fe680`
- `src/encirclement3d/pursuit_env.py`: `ea15252dcfd793ae0e3583cd5d8bb5c442f47490900b63aa131f275d1e4998fe`
- `src/encirclement3d/showcase.py`: `6820e1ca62b15e86e3381d0cbda0d02b43a0974d2ff387fbdeefc2710eba3092`

## Decision

Training integrity passes; a separate fixed/S3 development evaluation is still required.
