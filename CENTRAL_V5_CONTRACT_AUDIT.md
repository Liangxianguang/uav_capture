# V4/V5 Retained-BC Contract Audit

This is a read-only contract audit. It does not open or inspect the V5 locked-test block.

## Data-source comparison

| Field | V4 retained | V5 baseline |
| --- | --- | --- |
| Expert data source | `local collection` | `local collection` |
| Source balance | `None` | `None` |
| Training stages identical | `True` | `True` |
| Action scale mode | `full_range` | `full_range` |
| Required zone entries | `2` | `2` |

## Findings

- `fixed_archive_inherited`: `False`
- `v4_fixed_archive_declared`: `True`
- `v5_uses_local_collection`: `True`
- `equal_sequence_balance_preserved`: `False`
- `action_scale_mode_match`: `True`
- `hidden_dim_match`: `True`
- `zone_entry_requirement_match`: `True`
- `stage_contract_changed`: `False`
- `environment_config_sha256`: `2bbf3ca133fe13d58d4e4995b026624aacde111d0f0ce81983a65393b706244f`
- `v4_retention_report_available`: `True`
- `high_risk_data_contract_gap`: `True`

## Decision

V5 does not inherit the V4 fixed-scene archive and therefore cannot be treated as a faithful retained-BC reconstruction. Recover fixed-scene coverage before hard-example training.

## V5 run artifact snapshot

```json
{
  "run_dir": "F:\\uav_capture\\three_d_encirclement\\results\\central_v5\\bc_baseline_seed661401",
  "checkpoint.pt": {
    "present": true,
    "sha256": "4cfa2de38ce4cea8462f02a9e91b73ce9a2d4e0a4bf9b9aa356c5e4118cf51c5"
  },
  "expert_sequence_dataset.npz": {
    "present": true,
    "sha256": "de5386e512e3458b902c438cf1ada94cc6ab81acfa1a1f591d2ccbaf6bcbbaac"
  },
  "expert_dataset_manifest.json": {
    "present": true,
    "sha256": "28dd7dbcfda63cab753349170a74153234050029e614a4f99c2dc4b35b659882"
  },
  "config.yaml": {
    "present": true,
    "sha256": "6d35594b47f4d30dfef506333b58f7c5ad0fc7ddfe23b6dfbf4a9dd2a87f878b"
  },
  "manifest": {
    "accepted_episodes": 320,
    "rejected_episodes": 30,
    "collection_attempts": 350,
    "expert_rejection_rate": 0.08571428571428572,
    "expert_safe_capture_rate": 1.0,
    "expert_cooperative_requirement_rate": 1.0,
    "sequence_length": 32,
    "sequence_count": 457,
    "frame_count": 14643
  },
  "dataset_shapes": {
    "local_observations": [
      457,
      32,
      4,
      63
    ],
    "actions": [
      457,
      32,
      4,
      3
    ],
    "reset_masks": [
      457,
      32
    ]
  }
}
```
