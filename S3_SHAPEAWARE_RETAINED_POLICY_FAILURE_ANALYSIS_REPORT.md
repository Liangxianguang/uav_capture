# V5 Policy Failure Analysis

This is a development-validation diagnostic for one training seed. It does not open the V5 next locked-test block.

## Paired S3 result

- Raw/CBF static scenes exactly paired: `True`
- Raw Cooperative Safe Capture: `1/60`; collision `98.3%`.
- CBF Cooperative Safe Capture: `55/60`; collision `1.7%`; boundary `1.7%`.
- Raw failure stages: `{'safety_failure': 59}`.
- CBF failure stages: `{'safety_failure': 1, 'timeout': 4}`.

## Fixed-scene regression

- `s1_cylinder` CBF Cooperative Safe Capture: `100.0%`.
- `s1_box` CBF Cooperative Safe Capture: `100.0%`.
- `s1_wall` CBF Cooperative Safe Capture: `100.0%`.
- `s2` CBF Cooperative Safe Capture: `100.0%`.

## Decision

The shape-aware warm-start retained-BC checkpoint passes the one-seed development gate. Freeze the effective configuration, expert-archive provenance, checkpoint-selection rule, and CBF parameters, then train two additional independent seeds. Do not open seed block 647201 until all three checkpoints pass the same development gate.
