# V5 Policy Failure Analysis

This is a development-validation diagnostic for one training seed. It does not open the V5 next locked-test block.

## Paired S3 result

- Raw/CBF static scenes exactly paired: `True`
- Raw Cooperative Safe Capture: `1/60`; collision `98.3%`.
- CBF Cooperative Safe Capture: `59/60`; collision `1.7%`; boundary `1.7%`.
- Raw failure stages: `{'safety_failure': 59}`.
- CBF failure stages: `{'safety_failure': 1}`.

## Fixed-scene regression

- `s1_cylinder` CBF Cooperative Safe Capture: `100.0%`.
- `s1_box` CBF Cooperative Safe Capture: `100.0%`.
- `s1_wall` CBF Cooperative Safe Capture: `95.0%`.
- `s2` CBF Cooperative Safe Capture: `100.0%`.

## Decision

The checkpoint passes neither the complete fixed-scene regression nor the one-seed development gate. Fixed CBF regression is below the pre-registered 98% threshold for: s1_wall. Treat this as a P3-A fixed-coverage failure: first diagnose the exact fixed-scene safety failure, then pre-register one data-coverage-only recovery. Do not rerun evaluation to select a favorable outcome, start MAPPO, change CBF margins, or open seed block 647201.
