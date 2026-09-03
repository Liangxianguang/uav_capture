# JEPA Safe-Capture v2 P4 候选动作块与轨迹重排序

> Development-only implementation and synthetic audit. This report is not a
> closed-loop safe-capture result and does not open or replace any locked test.

## 1. P4 目标

P4 establishes the runtime boundary between a frozen V5 nominal actor and the
JEPA world-model evaluator:

```text
frozen nominal actor
    -> five dynamics-checked constant action chunks
    -> v2 JEPA target/clearance/visibility/risk prediction
    -> checkpoint-bound reliability decision
    -> trusted reranking or explicit nominal/safe-hold fallback
    -> downstream CBF/QP (P5)
    -> execute only the first control step
```

JEPA does not emit a final action and P4 does not claim that any candidate is
safe. The returned first action must be passed through the same CBF/QP as the
baseline.

## 2. Implemented artifacts

| Artifact | Purpose |
|---|---|
| `src/encirclement3d/jepa_safe_capture_candidates.py` | Fixed K=5 candidate labels, constant 3-step chunks, causal history window, finite/speed/acceleration/slew pre-checks |
| `src/encirclement3d/jepa_safe_capture_ranker.py` | v2 horizon-3 scoring, conservative clearance/TTC/visibility/risk penalties, ledger gating, nominal anchor, serializable trace |
| `scripts/audit_jepa_safe_capture_v2_candidate_ranking.py` | Real-checkpoint synthetic action-following/ranking audit with TensorBoard provenance |
| `tests/test_jepa_safe_capture_candidates.py` | Candidate contract, history, action-following, fallback and zero-perturbation tests |
| `tests/test_jepa_safe_capture_v2_candidate_audit.py` | Aggregate summary and TensorBoard audit tests |

Fixed candidate order:

1. `nominal` (exact frozen actor output)
2. `intercept`
3. `lateral_clearance`
4. `formation_clearance`
5. `visibility_hold`

All candidates are constant desired actions for three control steps. Only the
first step is returned; the caller must reobserve and replan.

## 3. Safety and reliability behavior

- Non-finite, speed-limit, acceleration-limit, non-constant, and slew-limit
  candidates are rejected before JEPA inference.
- `trusted` permits candidate ranking.
- `fallback_nominal` selects candidate 0 and records the reason.
- `safe_hold` selects candidate 0 with an explicit execution mode; P5 must
  route that mode through its deterministic hold/abort ladder.
- A candidate prediction is used only as a ranking feature. Obstacle,
  inter-agent, boundary, altitude, speed, acceleration, and capture safety
  constraints remain the responsibility of the downstream CBF/QP.
- The audit requires the ledger checkpoint hash to equal the loaded checkpoint
  SHA-256.

## 4. Tests and synthetic audit

Core P4 tests:

```text
7 passed
```

The combined v2 model, ledger, candidate, and audit regression subset passes
with `13 passed`.

The real checkpoint audit used:

```text
checkpoint: results/jepa_safe_capture_v2_p2_seed20260911/checkpoint.pt
checkpoint SHA-256: 3307c3935eabe0f6fb11a0dbe83ada0b4a4c610a1d96911a67c81cd6c66760e7
ledger: results/jepa_safe_capture_v2_p3_rerun_ledger_seed20260911/reliability_ledger.json
ledger SHA-256: 39e9e46344872978ee79238ec76f3978cb297c0940ec7ace95749e744a8a9d05
episodes/steps: 12 synthetic ranking steps
locked_test_opened: false
```

Aggregate audit observations:

| Quantity | Value |
|---|---:|
| valid candidate fraction | `1.0000` |
| eligible candidate fraction | `0.8333` |
| trusted decision fraction | `0.8333` |
| nominal fallback fraction | `0.1667` |
| safe-hold fraction | `0.0000` |
| non-nominal selection fraction | `0.6667` |
| mean action-following absolute prediction delta | `0.001193` normalized units |
| mean rank latency | `3.325 ms` on CPU |

The positive action-following delta shows that the checkpoint responds to
different candidate actions. The non-nominal selections show that the scorer
can change the candidate choice when the ledger permits it. The nominal
fallback rows show that low-credit decisions are observable and do not silently
use an untrusted candidate.

These observations are interface evidence only. They do **not** establish
collision-free execution, boundary safety, pairwise separation, CBF-QP
feasibility, or safe-capture improvement. Those claims require P5 and P6.

## 5. TensorBoard provenance

Audit logdir:

```text
results/jepa_safe_capture_v2_tensorboard/p4_candidate_audit_seed20260911_final
```

The log contains 14 scalar tags and three required text tags:

- `Config/audit/text_summary`
- `Provenance/sources/text_summary`
- `Provenance/environment/text_summary`

The JSON audit summary is:

```text
results/jepa_safe_capture_v2_p4_candidate_audit_seed20260911_final.json
SHA-256: 46e5cbe017b05f3271762d0db62ed624f63d0e693b624b97ae384e9b3c96d670
```

## 6. P4 decision and next gate

P4 candidate generation/ranking interface: **PASS**.

P4 closed-loop safety or task-performance claim: **NOT AUTHORIZED**.

The next implementation stage is P5: replace the existing feasibility proxy
with a real deterministic multi-agent CBF/QP, expose solver status/slack and
infeasible handling, and prove zero-perturbation identity before any paired
safe-capture development run.
