# V5 Policy Failure Analysis

This is a development-validation diagnostic for one training seed. It does not open the V5 next locked-test block.

## Paired S3 result

- Raw/CBF static scenes exactly paired: `True`
- Raw Cooperative Safe Capture: `1/60`; collision `98.3%`.
- CBF Cooperative Safe Capture: `30/60`; collision `0.0%`; boundary `0.0%`.
- Raw failure stages: `{'safety_failure': 59}`.
- CBF failure stages: `{'timeout': 30}`.

## Fixed-scene regression

- `s1_cylinder` CBF Cooperative Safe Capture: `20.0%`.
- `s1_box` CBF Cooperative Safe Capture: `95.0%`.
- `s1_wall` CBF Cooperative Safe Capture: `100.0%`.
- `s2` CBF Cooperative Safe Capture: `90.0%`.

## Decision

The raw actor fails before task-level pursuit in every S3 episode, while CBF removes collisions but leaves distributed timeouts. Together with the V4/V5 contract audit, this rejects the fresh V5 baseline as a candidate and selects P2-0 fixed-contract recovery: equal-sequence training on a newly collected fixed S1/S2 archive plus the frozen V5 random archive. Do not start MAPPO, change CBF margins, or open seed block 647201 before this data-only recovery passes fixed regression.
