# JEPA Action-Following Sensitivity Audit

> This diagnostic checks whether changing the final candidate action changes the predicted future. It is not a safety proof and does not open a locked test.

| Model | Axis | Horizon separations (normalized position units) | Non-trivial response |
| --- | ---: | --- | --- |
| jepa_v2_interaction_seed20260911 | 0 | 0.00197, 0.00219, 0.00209, 0.00194 | 100.0%, 99.9%, 99.9%, 99.5% |
| jepa_v2_interaction_seed20260911 | 1 | 0.00214, 0.00240, 0.00230, 0.00192 | 100.0%, 100.0%, 99.8%, 99.6% |
| jepa_v2_interaction_seed20260911 | 2 | 0.00215, 0.00234, 0.00225, 0.00189 | 100.0%, 100.0%, 100.0%, 99.7% |
| jepa_v2_interaction_seed20260912 | 0 | 0.00185, 0.00203, 0.00192, 0.00172 | 100.0%, 100.0%, 99.7%, 99.5% |
| jepa_v2_interaction_seed20260912 | 1 | 0.00189, 0.00205, 0.00193, 0.00160 | 100.0%, 100.0%, 99.7%, 99.3% |
| jepa_v2_interaction_seed20260912 | 2 | 0.00200, 0.00212, 0.00195, 0.00173 | 100.0%, 100.0%, 99.9%, 99.6% |
| jepa_v2_interaction_seed20260913 | 0 | 0.00201, 0.00219, 0.00207, 0.00188 | 100.0%, 100.0%, 99.8%, 99.6% |
| jepa_v2_interaction_seed20260913 | 1 | 0.00189, 0.00205, 0.00194, 0.00183 | 100.0%, 99.9%, 99.8%, 99.5% |
| jepa_v2_interaction_seed20260913 | 2 | 0.00195, 0.00214, 0.00202, 0.00174 | 100.0%, 100.0%, 99.8%, 99.1% |
| jepa_v2_baseline_seed20260911 | 0 | 0.00243, 0.00260, 0.00244, 0.00208 | 100.0%, 100.0%, 100.0%, 99.7% |
| jepa_v2_baseline_seed20260911 | 1 | 0.00246, 0.00262, 0.00251, 0.00209 | 100.0%, 100.0%, 100.0%, 99.8% |
| jepa_v2_baseline_seed20260911 | 2 | 0.00233, 0.00241, 0.00237, 0.00202 | 100.0%, 100.0%, 100.0%, 99.8% |
| jepa_v2_baseline_seed20260912 | 0 | 0.00216, 0.00227, 0.00215, 0.00191 | 100.0%, 100.0%, 99.9%, 99.5% |
| jepa_v2_baseline_seed20260912 | 1 | 0.00203, 0.00219, 0.00208, 0.00184 | 100.0%, 100.0%, 99.8%, 99.5% |
| jepa_v2_baseline_seed20260912 | 2 | 0.00244, 0.00258, 0.00243, 0.00199 | 100.0%, 100.0%, 100.0%, 99.7% |
| jepa_v2_baseline_seed20260913 | 0 | 0.00183, 0.00196, 0.00184, 0.00178 | 100.0%, 99.9%, 99.6%, 99.5% |
| jepa_v2_baseline_seed20260913 | 1 | 0.00190, 0.00205, 0.00198, 0.00187 | 100.0%, 99.9%, 99.8%, 99.6% |
| jepa_v2_baseline_seed20260913 | 2 | 0.00189, 0.00202, 0.00197, 0.00178 | 100.0%, 100.0%, 99.8%, 99.5% |

Interpretation: candidate separation should be clearly non-zero. The audit is a model-behavior check only; actual rollout correspondence and CBF safety remain separate evaluations.
