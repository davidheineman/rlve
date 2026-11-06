"""The queued coordinator must not initialize GPU execution before headroom exists."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from async_branching.io import read_json, write_json
from async_branching.storage import wait_for_launch_storage


class AsyncStorageTest(unittest.TestCase):
    def test_wait_reports_headroom_and_heartbeat_until_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = []

            def waited(interval):
                self.assertEqual(interval, 60)
                samples.append(read_json(root / "storage-wait.json"))
                self.assertEqual(samples[-1]["status"], "waiting")
                self.assertTrue((root / "coordinator-lease.json").exists())
                self.assertFalse((root / "state.json").exists())
                self.assertFalse((root / "slurm_jobs.json").exists())

            with patch("async_branching.storage.shutil.disk_usage", side_effect=[
                    SimpleNamespace(free=n * 1024**3) for n in (318, 749, 750)]), \
                    patch("async_branching.storage.time.sleep", side_effect=waited):
                wait_for_launch_storage(root, 750)
            self.assertEqual(len(samples), 2)
            self.assertEqual(read_json(root / "storage-wait.json")["status"], "ready")

    def test_invalid_threshold_fails_before_waiting(self):
        for value in (0, -1, 750.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                wait_for_launch_storage("/unused", value)

    def test_monitor_recognizes_storage_wait_without_restart(self):
        from async_branching.monitor import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "submission.json", {"job": "1"})
            write_json(root / "storage-wait.json", {
                "status": "waiting", "free_gib": 318, "minimum_free_gib": 750})
            with patch("sys.argv", ["monitor", "--directory", temporary]), \
                    patch("async_branching.monitor.queue", return_value={"1": {"state": "RUNNING"}}), \
                    patch("async_branching.monitor.restart") as restart, \
                    patch("async_branching.monitor.time.sleep", side_effect=StopIteration):
                with self.assertRaises(StopIteration):
                    main()
            status = read_json(root / "monitor.json")
            self.assertEqual(status["monitor_status"], "waiting_for_storage")
            self.assertEqual(status["running_worker_gpus"], 0)
            restart.assert_not_called()

    def test_cli_waits_before_constructing_backend(self):
        from async_branching.cli import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "plan.json", {"minimum_launch_free_gib": 750})
            with patch("sys.argv", ["cli", "--directory", temporary, "--plan", str(root / "plan.json")]), \
                    patch("async_branching.storage.wait_for_launch_storage", side_effect=StopIteration) as wait, \
                    patch("async_branching.cli.initial_state") as initialize:
                with self.assertRaises(StopIteration):
                    main()
            wait.assert_called_once_with(root, 750)
            initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
