# E1 Rule-Expert Feasibility Rejection Report

## Decision

**Rejected early at P1.** The first pre-registered development profile, E0, did not satisfy the rule-expert feasibility gate. Therefore E1 policy development, model selection, and the E1 locked block `681201` are not opened.

This is an intentional protocol stop, not a request to relax the gate, alter the CBF, choose a different V4 checkpoint, or rerun until a favourable seed appears.

## Completed evidence

E0 uses the original command path (`0` delay, acceleration scale `1.00`, no execution noise) and execution-aware CBF. It was evaluated with the rule expert on all 60 pre-registered development cases from block `681101`.

| Metric | Pre-registered gate | Observed E0 result | Status |
| --- | ---: | ---: | --- |
| Cooperative Safe Capture | >= 95.0% | **56/60 = 93.3%** | FAIL |
| Collision | <= 5.0% | 1/60 = 1.7% | PASS |
| Boundary | <= 5.0% | 1/60 = 1.7% | PASS |
| Transit | diagnostic | 60/60 = 100.0% | — |

The three non-capture terminations were two timeouts and one safety failure. The artifact hashes are retained in [E1_RULE_EXPERT_FEASIBILITY_REJECTION_SUMMARY.json](E1_RULE_EXPERT_FEASIBILITY_REJECTION_SUMMARY.json):

- `episodes.csv`: `924f6c59412373daba5c38dbc2c4d58876e67be5c1433525cac2eb061fc4e3f5`
- `cases.jsonl`: `e12b1ed0e3202511e22e58fbf3f866710962eafb92b90fb73577ebfce8544ce7`
- `summary.json`: `ed7e75f4030cd845663170872d479a62a1636c92a94f6aae10534c001febb693`

## Conformant stop

The remaining E1–E6 rule-expert profiles were not opened after the E0 failure. No frozen V4 checkpoint was loaded, no policy-development case was evaluated, and no locked case was read. The source checkpoint artifacts are still absent locally; their absence is not used to explain or override this independent feasibility failure.

E1 is therefore a documented negative result for this exact protocol. The historical V4 motion-kinematic result remains unchanged. Any new execution robustness study must be E1-prime: it needs a newly stated feasibility question, a separate pre-registration, and new development/locked case blocks. It must not reuse `681101` or `681201` to choose a weaker profile.
