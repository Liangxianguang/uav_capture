# Central V5 Shape-Aware Retained-BC Training Audit

This record audits training provenance only. It is neither a development evaluation nor a locked test.

## Fixed Shape-Aware Stage

| Check | Result |
| --- | --- |
| Requested / accepted expert episodes | 640 / 640 |
| Rejected episodes / rejection rate | 0 / 0.00% |
| Safe / cooperative accepted demonstrations | True / True |
| Wall initial-distance coverage passes pre-registration | True |
| Expert sequences / frames | 646 / 20677 |
| Training epochs / finite action-MSE | 96 / True |
| First / final action MSE | 2.64874633 / 0.02755872 |
| Checkpoint SHA-256 | `54b543343fff6a26b91c8d686ca63f714a7f54e674c080429aec5117d954554d` |

Fixed-stage accepted expert coverage:

| Layout | Accepted expert episodes |
| --- | ---: |
| box | 95 |
| cylinder | 299 |
| cylinder_box | 97 |
| mixed | 38 |
| wall | 111 |

Wall initial-distance pre-registration check:

| Initial distance (m) | Accepted wall episodes | Required minimum |
| ---: | ---: | ---: |
| 5.0 | 58 | 25 |
| 5.5 | 53 | 25 |

## Warm-Start Retained Stage

| Check | Result |
| --- | --- |
| Warm-start checkpoint hash matches fixed checkpoint | True |
| Archive source balance / selected sequence balance | equal_sequences / True |
| All source demonstrations safe / cooperative | True |
| Source 0: original / selected sequences | 646 / 646 |
| Source 1: original / selected sequences | 457 / 646 |
| Total sequences / frames | 1292 / 41344 |
| Training epochs / finite action-MSE | 64 / True |
| First / final action MSE | 1.43201810 / 0.09192167 |
| Retained checkpoint SHA-256 | `09284ff66b4297642d626ad3364999c5e2ed66d890f335fae6fcd234132c898b` |

## Source Provenance

- `scripts/train_capture_radius_recurrent_behavior_cloning.py`: `b263160a62afe46b1c6544cc010c4b1f06a60e7b7ce9781ac7f6dff537d334de`
- `src/encirclement3d/learning.py`: `2cb49096c19d8e82966ff503f16e59151fa2d310b12168e44f3418b7779780fd`
- `src/encirclement3d/observation_encoding.py`: `7fc2328cabda73971f141bc60682c01899d2fcff4270a3bf49f883921a84f8f6`
- `src/encirclement3d/prediction.py`: `6fb10e1c0768140a6d62244c27f20e11071d04aa0a7622c176d0fedf60136423`
- `src/encirclement3d/pursuit_controllers.py`: `86a4b1e0e0ba49ebf69d1a78e17ba1d29caa06d58737366ee7fd1ac36d8fe680`
- `src/encirclement3d/pursuit_env.py`: `ea15252dcfd793ae0e3583cd5d8bb5c442f47490900b63aa131f275d1e4998fe`
- `src/encirclement3d/showcase.py`: `2db09a7d7e7aa18978c8afa225dd7d0ca4894d68017c596ab43848f5d00c73a4`

## Decision

Training integrity passes; a separate fixed/S3 development evaluation is still required.
