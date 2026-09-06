# WP1 Obstacle-Conditioned Route Audit

**Date:** 2026-09-06
**Status:** development-only; `locked_test_opened=false`
**Scope:** verify that route proposals react to public obstacle geometry before any JEPA retraining.

## Question

Can the route proposal layer use the online `observation["obstacles"]` records to produce different, interpretable left/right/up/down routes, reject a blocked side, and preserve at least one reachable route?

This audit is not a safe-capture evaluation. It does not train a model, change CBF margins, execute actions, or open a locked split.

## Implementation

- Route layer: `src/encirclement3d/obstacle_route_candidates.py`
- Unit tests: `tests/test_obstacle_route_candidates.py`
- Audit CLI: `scripts/audit_jepa_safe_capture_wp1_obstacle_routes.py`
- Route contract: `obstacle_route_v1`, 12 ordered candidates, three-step action chunks, reachable-dynamics projection, downstream Joint CBF execution boundary, first-step execution contract.
- Geometry: one signed-distance interface for cylinder, box, and wall; sampled route-corridor clearance; no simulator hidden target state.

## Deterministic Micro-Scenes

| Scene | Expected geometry condition |
| --- | --- |
| `central_single` | A central cylinder blocks the direct belief corridor; both horizontal detours and the upper detour are available. |
| `left_blocked` | A second public cylinder blocks the left bypass; the right bypass remains available. |
| `right_blocked` | A second public cylinder blocks the right bypass; the left bypass remains available. |
| `wall_single_gap` | Two public wall segments leave one nominal gap; the gap route remains available. |

## Result

The audit was run with the repository's existing GPU environment:

```powershell
$env:PYTHONPATH = "src"
& D:\miniconda3\envs\uav-encirclement-gpu\python.exe `
  scripts/audit_jepa_safe_capture_wp1_obstacle_routes.py `
  --output-dir results/jepa_safe_capture_wp1_obstacle_routes_cbf_verified_v2 `
  --with-cbf
```

| Scene | Valid routes | Joint CBF verified first-step routes | Left valid | Right valid |
| --- | ---: | ---: | ---: | ---: |
| `central_single` | 8 / 12 | 8 | yes | yes |
| `left_blocked` | 7 / 12 | 7 | no | yes |
| `right_blocked` | 7 / 12 | 7 | yes | no |
| `wall_single_gap` | 10 / 12 | 10 | yes | yes |

All ten acceptance checks passed:

- left/right waypoints are distinct;
- the blocked side is rejected using all public obstacle records;
- the unblocked side remains reachable;
- the wall gap keeps a valid nominal route;
- every scene has a reachable candidate;
- every scene has at least one first-step candidate accepted by the unchanged Joint CBF probe.

The machine-independent route-only output is in [summary.json](../results/jepa_safe_capture_wp1_obstacle_routes_cbf_verified_v2/summary.json), the complete candidate trace is in [route_candidates.json](../results/jepa_safe_capture_wp1_obstacle_routes_cbf_verified_v2/route_candidates.json), and the generated audit view is in [report.md](../results/jepa_safe_capture_wp1_obstacle_routes_cbf_verified_v2/report.md).

## Interpretation and Boundary

This is evidence that the new route proposal layer has the required obstacle-to-route causal behavior. It is **not** evidence that the complete JEPA + reliability ledger + rolling-horizon system improves `safe_capture`.

The existing historical `legacy` and `extended_v1` candidate contracts were not modified. The next implementation stage is to connect `obstacle_route_v1` to the development evaluator, add independent selected/nominal/safe-hold CBF counterfactuals, then generate and calibrate a new JEPA archive bound to this route contract.

## Provenance

- Git revision at audit: `e4c94e9706ce4dfa6c232ef8d5f3a40e37545092`
- Python: 3.11.16
- NumPy: 2.1.3
- Route module SHA-256: `bee11792416a65078c11aea3e58b3f048d69f1a1ba8d6438074ece1c921cd963`
- Audit script SHA-256: `04e9f73f3e4ec39253cc802d86a63e5b68aef1948945d1d848143a3979629616`
- Environment config SHA-256: `42bd4e158c5e314e0ece6add8038b32c384a7a2ca027e9387327656fccf751ad`
