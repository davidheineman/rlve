"""Recovery must preserve the experiment and only manage its recorded job."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import monitor_mopd_clustering as monitor


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.health = {
            "job_id": "123",
            "slurm": {"state": "NODE_FAIL", "node": "gpu1", "reason": "node failure"},
            "completed_teachers": 0,
            "sum_training_rollouts": 400,
            "evaluation_cells": {"selection": 0, "confirmation": 0},
            "errors": [],
        }

    def test_unknown_code_failure_does_not_retry(self):
        self.health["slurm"]["state"] = "FAILED"
        (self.root / "slurm-123.out").write_text("ValueError: invalid objective")
        with patch.object(monitor, "command") as command:
            self.assertFalse(monitor.recover(self.root, self.health, {}))
            command.assert_not_called()

    def test_manual_cancel_is_not_undone_by_an_older_error(self):
        self.health["slurm"]["state"] = "CANCELLED"
        (self.root / "slurm-123.out").write_text(
            "ActorDiedError from an earlier attempt"
        )
        with patch.object(monitor, "command") as command:
            self.assertFalse(monitor.recover(self.root, self.health, {}))
            command.assert_not_called()

    def test_requeues_only_the_recorded_job_without_changing_source_or_budget(self):
        monitor.write(self.root / "plan.json", {"teacher_rollouts": 8000})
        original = (self.root / "plan.json").read_bytes()
        with patch.object(monitor, "command") as command:
            self.assertTrue(monitor.recover(self.root, self.health, {}))
            self.assertEqual(
                [call.args[0] for call in command.call_args_list],
                [["scontrol", "requeuehold", "123"], ["scontrol", "release", "123"]],
            )
        self.assertEqual((self.root / "plan.json").read_bytes(), original)

    def test_full_tmp_excludes_the_failed_node(self):
        self.health["slurm"]["state"] = "FAILED"
        (self.root / "slurm-123.out").write_text(
            "No space left on device: /tmp/miles-lora-123"
        )
        with patch.object(monitor, "command") as command:
            self.assertTrue(monitor.recover(self.root, self.health, {}))
            updates = [
                call.args[0]
                for call in command.call_args_list
                if "update" in call.args[0]
            ]
        self.assertEqual(len(updates), 1)
        self.assertIn("JobId=123", updates[0])
        self.assertIn("gpu1", monitor.read(self.root / "full-tmp-nodes.json"))

    def test_retry_limit_uses_durable_progress(self):
        journal = {}
        with patch.object(monitor, "command"):
            for step in range(3):
                self.health["sum_training_rollouts"] += step * 8
                self.assertTrue(monitor.recover(self.root, self.health, journal))
            self.assertFalse(monitor.recover(self.root, self.health, journal))

    def test_releases_only_a_hold_that_the_monitor_recorded(self):
        self.health["slurm"] = {
            "state": "PENDING",
            "node": "",
            "reason": "job requeued in held state",
        }
        with patch.object(monitor, "command") as command:
            monitor.finish_interrupted_recovery(self.root, self.health, {})
            command.assert_not_called()
            journal = {
                "recovery_in_progress": {"job_id": "123", "reason": "node failure"}
            }
            monitor.finish_interrupted_recovery(self.root, self.health, journal)
            command.assert_called_once_with(["scontrol", "release", "123"])
        self.assertNotIn("recovery_in_progress", journal)

    def fixture_complete(self):
        teachers = [
            {"source_label": f"teacher-{i:02d}", "environment": name}
            for i, name in enumerate(["a", "b"])
        ]
        output = self.root / "plots"
        monitor.write(
            self.root / "plan.json",
            {
                "teachers": teachers,
                "teacher_rollouts": 8000,
                "teacher_validation_threshold": 0.98,
                "plot_output": str(output),
            },
        )
        monitor.write(self.root / "completed.json", {"stage": "complete"})
        for teacher in teachers:
            directory = self.root / "teacher" / teacher["source_label"]
            adapter = directory / "final/adapter"
            adapter.mkdir(parents=True)
            for name in ["adapter_config.json", "adapter_model.safetensors"]:
                (adapter / name).write_bytes(b"diagnostic")
            monitor.write(
                directory / "completed.json",
                {
                    "environment": teacher["environment"],
                    "training_rollouts": 8000,
                    "accuracy": 0.5,
                    "hf": str(adapter),
                },
            )
        monitor.write(
            output / "mopd-clusters.json", {"clusters": [{"environments": ["a", "b"]}]}
        )
        for name in [
            "results.json",
            "uncertainty.json",
            "transfer.json",
            "candidate-results.json",
            "individual_transfer.csv",
            "teacher_summary.csv",
            "README.md",
        ]:
            (output / name).write_text("diagnostic")
        for stem in monitor.PLOTS:
            (output / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 120)
            (output / f"{stem}.pdf").write_bytes(b"%PDF-" + b"0" * 120)
        return output

    def test_completion_requires_all_requested_artifacts(self):
        output = self.fixture_complete()
        self.assertTrue(monitor.validate_completion(self.root))
        (output / "teacher_curves.pdf").unlink()
        with self.assertRaises(FileNotFoundError):
            monitor.validate_completion(self.root)

    def test_completion_rejects_early_unqualified_teacher_stop(self):
        self.fixture_complete()
        path = self.root / "teacher/teacher-00/completed.json"
        record = monitor.read(path)
        record["training_rollouts"] = 400
        monitor.write(path, record)
        with self.assertRaises(ValueError):
            monitor.validate_completion(self.root)

    def test_balanced_completion_requires_confirmation_and_excludes_own_adapter(self):
        output = self.fixture_complete()
        plan = monitor.read(self.root / "plan.json")
        plan["clustering"] = {"mode": "balanced"}
        monitor.write(self.root / "plan.json", plan)
        for name in ("results.json", "transfer.json", "assignments.csv", "README.md"):
            path = output / "balanced" / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("diagnostic")
        for stem in ("k_selection", "balanced_transfer"):
            (output / "balanced" / f"{stem}.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + b"0" * 120
            )
            (output / "balanced" / f"{stem}.pdf").write_bytes(b"%PDF-" + b"0" * 120)
        confirmation = {
            "assignments": {
                "controls": {
                    key: [[1], [0]]
                    for key in ("selected-loo", "random-loo", "outside-matched")
                }
            }
        }
        monitor.write(output / "confirmation.json", confirmation)
        self.assertTrue(monitor.validate_completion(self.root))
        confirmation["assignments"]["controls"]["selected-loo"][0] = [0]
        monitor.write(output / "confirmation.json", confirmation)
        with self.assertRaisesRegex(ValueError, "own adapter"):
            monitor.validate_completion(self.root)


if __name__ == "__main__":
    unittest.main()
