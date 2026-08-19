from __future__ import annotations

from scripts.run_stage4_common_locked_evaluation import METHODS, checkpoint_path


def test_common_locked_replay_uses_p1_seed_and_f1_f2_configurations() -> None:
    assert METHODS["f1"]["name"] == "f1_time_aligned_belief"
    assert METHODS["f2"]["name"] == "f2_uncertainty_features"
    assert METHODS["f1"]["environment_config"].endswith("time_aligned_belief_dev.yaml")
    assert METHODS["f2"]["environment_config"].endswith("time_aligned_uncertainty_dev.yaml")


def test_common_locked_checkpoint_path_keeps_training_seed() -> None:
    path = checkpoint_path(__import__("pathlib").Path("archive"), "f1_time_aligned_belief", 521002)
    assert path.as_posix().endswith("archive/f1_time_aligned_belief/seed521002/recurrent_mappo/checkpoint.pt")
