import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "checkpoint_state", Path(__file__).parents[1] / "utils/checkpoint_state.py"
)
checkpoint_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint_state)


class CheckpointStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        checkpoint_state.prepare_checkpoint(self.directory)

    def create_snapshot(self, rollout_id, include_runtime=True):
        (self.directory / f"iter_{rollout_id:07d}").mkdir()
        (self.directory / "rollout").mkdir(exist_ok=True)
        for prefix in checkpoint_state.ROLLOUT_PREFIXES:
            if prefix == "runtime_state_" and not include_runtime:
                continue
            (self.directory / "rollout" / f"{prefix}{rollout_id}.pt").write_text(
                "state"
            )
        (self.directory / checkpoint_state.TRACKER_FILE).write_text(str(rollout_id))

    def test_partial_first_save_returns_to_initial_weights(self):
        self.create_snapshot(9, include_runtime=False)
        state = checkpoint_state.prepare_checkpoint(self.directory)
        self.assertEqual(state["rollout_id"], -1)
        self.assertFalse((self.directory / checkpoint_state.TRACKER_FILE).exists())
        self.assertFalse((self.directory / "iter_0000009").exists())

    def test_partial_later_save_restores_last_complete_checkpoint(self):
        self.create_snapshot(9)
        checkpoint_state.commit_checkpoint(self.directory, 9)
        checkpoint_state.mark_evaluated(self.directory, 9)
        self.create_snapshot(19, include_runtime=False)
        state = checkpoint_state.prepare_checkpoint(self.directory)
        self.assertEqual(state["rollout_id"], 9)
        self.assertEqual(state["evaluated_rollout_id"], 9)
        self.assertEqual(
            (self.directory / checkpoint_state.TRACKER_FILE).read_text(), "9"
        )
        self.assertFalse((self.directory / "iter_0000019").exists())

    def test_rejects_incomplete_commit(self):
        self.create_snapshot(9, include_runtime=False)
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            checkpoint_state.commit_checkpoint(self.directory, 9)
        self.assertEqual(
            checkpoint_state.read_checkpoint_state(self.directory)["rollout_id"], -1
        )

    def test_keeps_two_checkpoints_and_records_pending_evaluation(self):
        for rollout_id in [9, 19, 29]:
            self.create_snapshot(rollout_id)
            checkpoint_state.commit_checkpoint(self.directory, rollout_id)
        self.assertEqual(
            checkpoint_state.checkpoint_iterations(self.directory), [19, 29]
        )
        state = checkpoint_state.prepare_checkpoint(self.directory)
        self.assertLess(state["evaluated_rollout_id"], state["rollout_id"])
        checkpoint_state.mark_evaluated(self.directory, 29)
        checkpoint_state.mark_completed(self.directory)
        self.assertTrue(
            checkpoint_state.prepare_checkpoint(self.directory)["completed"]
        )

    def test_can_keep_only_one_committed_checkpoint(self):
        with patch.dict(os.environ, {"RLVE_CHECKPOINT_KEEP": "1"}):
            for rollout_id in [9, 19, 29]:
                self.create_snapshot(rollout_id)
                checkpoint_state.commit_checkpoint(self.directory, rollout_id)
                self.assertEqual(checkpoint_state.checkpoint_iterations(self.directory), [rollout_id])
        self.assertEqual(checkpoint_state.prepare_checkpoint(self.directory)["rollout_id"], 29)

    def test_global_dataset_checkpoint_does_not_require_rlve_manager(self):
        directory = self.directory / "global"
        checkpoint_state.prepare_checkpoint(directory, data_source="global_dataset")
        (directory / "iter_0000009").mkdir()
        (directory / "rollout").mkdir()
        for prefix in ("global_dataset_state_dict_", "runtime_state_"):
            (directory / "rollout" / f"{prefix}9.pt").write_text("state")
        (directory / checkpoint_state.TRACKER_FILE).write_text("9")
        checkpoint_state.commit_checkpoint(directory, 9)
        self.assertEqual(
            checkpoint_state.prepare_checkpoint(directory, data_source="global_dataset")["rollout_id"], 9
        )
        with self.assertRaisesRegex(RuntimeError, "data source differs"):
            checkpoint_state.prepare_checkpoint(directory)

    def test_early_stop_survives_restart_after_evaluated_checkpoint(self):
        self.create_snapshot(9)
        checkpoint_state.commit_checkpoint(self.directory, 9)
        with self.assertRaisesRegex(RuntimeError, "before checkpoint and evaluation"):
            checkpoint_state.mark_early_stopped(self.directory, 9, 0.96)
        checkpoint_state.mark_evaluated(self.directory, 9)
        checkpoint_state.mark_early_stopped(self.directory, 9, 0.96)
        state = checkpoint_state.prepare_checkpoint(self.directory)
        self.assertTrue(state["early_stopped"])
        self.assertEqual(state["early_stop_val_accuracy"], 0.96)
        self.assertEqual(state["early_stop_reason"], "accuracy")

    def test_plateau_progress_survives_restart_only_after_evaluation(self):
        self.create_snapshot(49)
        checkpoint_state.commit_checkpoint(self.directory, 49)
        with self.assertRaisesRegex(RuntimeError, "before checkpoint and evaluation"):
            checkpoint_state.mark_plateau_progress(self.directory, 49, 0.62, 3)
        checkpoint_state.mark_evaluated(self.directory, 49)
        checkpoint_state.mark_plateau_progress(self.directory, 49, 0.62, 3)
        state = checkpoint_state.prepare_checkpoint(self.directory)
        self.assertEqual((state["plateau_best_accuracy"], state["plateau_stale_evals"]), (0.62, 3))

    def test_never_overwrites_unmanaged_checkpoint(self):
        (self.directory / checkpoint_state.STATE_FILE).unlink()
        self.create_snapshot(9)
        with self.assertRaisesRegex(RuntimeError, "unmanaged"):
            checkpoint_state.prepare_checkpoint(self.directory)

    def test_atomic_state_is_json_and_monotonic(self):
        self.create_snapshot(9)
        checkpoint_state.commit_checkpoint(self.directory, 9)
        with self.assertRaisesRegex(RuntimeError, "advance"):
            checkpoint_state.commit_checkpoint(self.directory, 9)
        state = json.loads((self.directory / checkpoint_state.STATE_FILE).read_text())
        self.assertEqual(state["rollout_id"], 9)
        self.assertEqual(list(self.directory.glob(".*.tmp")), [])

    def test_final_only_retry_discards_partial_state_and_wandb_identity(self):
        run = self.directory / "run"
        run.mkdir()
        self.assertEqual(checkpoint_state.begin_final_only_attempt(run), (1, False))
        checkpoint_state.prepare_checkpoint(run / "ckpt")
        (run / "wandb_run_id.txt").write_text("old-id\n")
        (run / "ckpt" / "partial-model").write_text("incomplete")
        self.assertEqual(checkpoint_state.begin_final_only_attempt(run), (2, True))
        self.assertFalse((run / "ckpt").exists())
        self.assertFalse((run / "wandb_run_id.txt").exists())
        self.assertEqual((run / "attempt.txt").read_text(), "2\n")

    def test_final_only_completed_checkpoint_is_kept_for_export_retry(self):
        run = self.directory / "completed-run"
        run.mkdir()
        checkpoint_state.prepare_checkpoint(run / "ckpt")
        checkpoint_state.mark_completed(run / "ckpt")
        (run / "wandb_run_id.txt").write_text("finished-id\n")
        self.assertEqual(checkpoint_state.begin_final_only_attempt(run), (1, False))
        self.assertTrue((run / "ckpt" / checkpoint_state.STATE_FILE).exists())
        self.assertEqual((run / "wandb_run_id.txt").read_text(), "finished-id\n")


if __name__ == "__main__":
    unittest.main()
