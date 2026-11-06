import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from async_branching.io import read_json, write_json
from async_branching.monitor import inspect, main, teacher_validation


class AsyncMonitorTest(unittest.TestCase):
    def test_startup_gate_requires_all_parser_modes_and_actual_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / "tasks/teacher-00-g0001"
            task.mkdir(parents=True)
            log = task / "worker.log"
            log.write_text('{"kind":"teacher","start_update":0}\n'
                           '{"kind":"student","start_update":0}\n'
                           '{"kind":"student","start_update":50}\n')
            self.assertEqual(teacher_validation(root, "1", {"1": {"state": "RUNNING"}}), "waiting")
            with log.open("a") as handle:
                handle.write("david 0: {'david/rollouts': 8}\n")
            self.assertEqual(teacher_validation(root, "1", {"1": {"state": "RUNNING"}}), "passed")
            self.assertEqual(teacher_validation(root, "1", {}), "missing")

    def test_startup_gate_accepts_teacher_already_solved_at_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / "tasks/teacher-00-g0001"
            task.mkdir(parents=True)
            (task / "worker.log").write_text('{"kind":"teacher","start_update":0}\n'
                                            '{"kind":"student","start_update":0}\n'
                                            '{"kind":"student","start_update":50}\n')
            write_json(task / "result.json", {"rollouts": 0, "accuracy": .99})
            self.assertEqual(teacher_validation(root, "1", {}), "passed")

    def test_health_reports_running_gpu_count_and_student_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "state.json", {"student_rollouts": 400, "clock": 4400})
            write_json(root / "slurm_jobs.json", {
                "teacher": {"job": "2", "kind": "teacher"},
                "student-service": {"job": "3", "kind": "student"}})
            status = inspect(root, {key: {"state": "RUNNING"} for key in ("1", "2", "3")}, "1")
            self.assertEqual(status["running_worker_gpus"], 3)
            self.assertEqual(status["student_rollouts"], 400)
            self.assertEqual(status["coordinator"]["state"], "RUNNING")

    def test_missing_failed_coordinator_is_restarted_once_after_two_polls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "submission.json", {"job": "1"})
            with patch("sys.argv", ["monitor", "--directory", temporary]), \
                    patch("async_branching.monitor.queue", return_value={}), \
                    patch("async_branching.monitor.subprocess.check_output", return_value="FAILED\n"), \
                    patch("async_branching.monitor.restart", return_value="4") as restart, \
                    patch("async_branching.monitor.time.sleep", side_effect=[None, StopIteration]):
                with self.assertRaises(StopIteration):
                    main()
            status = read_json(root / "monitor.json")
            self.assertEqual(status["coordinator_job"], "4")
            self.assertEqual(len(status["restarts"]), 1)
            restart.assert_called_once_with(root)

    def test_recovery_limit_prevents_retry_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "submission.json", {"job": "1"})
            with patch("sys.argv", ["monitor", "--directory", temporary, "--max-restarts", "0"]), \
                    patch("async_branching.monitor.queue", return_value={}), \
                    patch("async_branching.monitor.subprocess.check_output", return_value="FAILED\n"), \
                    patch("async_branching.monitor.restart") as restart, \
                    patch("async_branching.monitor.time.sleep", side_effect=[None, StopIteration]):
                with self.assertRaises(StopIteration):
                    main()
            self.assertEqual(read_json(root / "monitor.json")["monitor_status"], "alert_recovery_limit_reached")
            restart.assert_not_called()

    def test_explicit_cancellation_is_not_automatically_restarted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "submission.json", {"job": "1"})
            with patch("sys.argv", ["monitor", "--directory", temporary]), \
                    patch("async_branching.monitor.queue", return_value={}), \
                    patch("async_branching.monitor.subprocess.check_output", return_value="CANCELLED by 1234\n"), \
                    patch("async_branching.monitor.restart") as restart, \
                    patch("async_branching.monitor.time.sleep"):
                main()
            self.assertEqual(read_json(root / "monitor.json")["monitor_status"], "coordinator_cancelled_no_restart")
            restart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
