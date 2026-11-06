import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from async_branching.io import read_json, write_json
from async_branching.watchdog import ProgressWatchdog


class ProgressWatchdogTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = self.root / "tasks/teacher-00-g0001"
        self.task.mkdir(parents=True)
        self.log = self.task / "worker.log"
        self.log.touch()
        token = hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        self.worker = {"job": "12", "kind": "teacher", "state": "RUNNING",
                       "start_time": "2026-10-24T00:00:00",
                       "comment": f"rlve-async:{token}:{self.task.name}"}
        self.workers = {self.task.name: self.worker}
        self.watchdog = ProgressWatchdog(self.root)

    def queue(self):
        return {self.worker["job"]: dict(self.worker)}

    def check(self, now):
        return self.watchdog.check(self.workers, self.queue, now=now)

    def append(self, content):
        with self.log.open("a") as handle:
            handle.write(content)

    def start_training(self):
        self.check(0)
        self.append("david 0: {'david/rollouts': 8}\n")
        self.check(1)

    @patch("async_branching.watchdog.subprocess.run")
    def test_server_noise_and_repeated_steps_do_not_hide_training_stall(self, requeue):
        self.start_training()
        self.append("GET /health 200 OK\ndavid 0: {'david/rollouts': 8}\n")
        self.check(1700)
        requeue.assert_not_called()
        status = self.check(1801)
        requeue.assert_called_once_with(["scontrol", "requeue", "12"], check=True,
                                        capture_output=True, text=True, timeout=30)
        self.assertEqual(status["requeues"], 1)
        self.assertEqual(read_json(self.root / "watchdog.json")["recoveries"][0]["step"], 0)

    @patch("async_branching.watchdog.subprocess.run")
    def test_advancing_steps_reset_deadline_and_partial_lines_survive_poll(self, requeue):
        self.start_training()
        self.append("david 1: {'david/rollouts': ")
        self.check(1700)
        self.append("16}\n")
        self.check(1750)
        self.check(3500)
        requeue.assert_not_called()
        self.check(3550)
        requeue.assert_called_once()

    @patch("async_branching.watchdog.subprocess.run")
    def test_startup_has_45_minutes_and_ignores_old_attempt_log(self, requeue):
        self.append("david 499: {'david/rollouts': 4000}\n")
        self.check(0)
        self.check(1801)
        requeue.assert_not_called()
        self.check(2700)
        requeue.assert_called_once()
        self.assertIsNone(read_json(self.root / "watchdog.json")["recoveries"][0]["step"])

    @patch("async_branching.watchdog.subprocess.run")
    def test_pending_and_suspended_time_is_excluded(self, requeue):
        self.worker["state"] = "PENDING"
        self.check(0)
        self.check(100000)
        self.worker["state"] = "RUNNING"
        self.check(100001)
        self.check(101900)
        self.worker["state"] = "SUSPENDED"
        self.check(102000)
        self.worker["state"] = "RUNNING"
        self.check(200000)
        self.check(201900)
        requeue.assert_not_called()

    @patch("async_branching.watchdog.subprocess.run")
    def test_student_idle_and_completed_requests_are_excluded(self, requeue):
        key = "student-service"
        self.worker.update(kind="student", comment=self.worker["comment"].rsplit(":", 1)[0] + ":" + key)
        self.workers = {key: self.worker}
        self.check(0)
        self.check(100000)
        write_json(self.root / key / "request.json", {"task": str(self.task / "task.json")})
        self.check(100001)
        self.append("david 50: {'david/rollouts': 408}\n")
        self.check(100002)
        write_json(self.task / "result.json", {"rollouts": 400})
        self.check(200000)
        requeue.assert_not_called()
        (self.task / "result.json").unlink()
        self.check(200001)
        self.check(201802)
        requeue.assert_not_called()
        self.check(202701)
        requeue.assert_called_once()

    @patch("async_branching.watchdog.subprocess.run")
    def test_retry_budget_survives_monitor_restart_and_replaced_job_ids(self, requeue):
        self.start_training()
        for i in range(3):
            self.check(1 + (i + 1) * 4000)
            self.watchdog = ProgressWatchdog(self.root)
            self.worker.update(job=str(13 + i), start_time=str(i + 1))
            self.check(2 + (i + 1) * 4000)
            self.append("david 0: {'david/rollouts': 8}\n")
            self.check(3 + (i + 1) * 4000)
        self.assertEqual(requeue.call_count, 2)
        self.assertEqual(len(read_json(self.root / "watchdog.json")["recoveries"]), 2)

    @patch("async_branching.watchdog.subprocess.run")
    def test_ownership_attempt_and_terminal_commit_are_rechecked(self, requeue):
        self.start_training()
        for changed in ({"comment": "unrelated"}, {"start_time": "new"}, {"state": "COMPLETING"}):
            self.watchdog.check(self.workers, lambda: {"12": {**self.worker, **changed}}, now=2000)
        def committed():
            write_json(self.task / "result.json", {"rollouts": 4000})
            return self.queue()
        self.watchdog.check(self.workers, committed, now=2000)
        requeue.assert_not_called()

    @patch("async_branching.watchdog.subprocess.run", side_effect=subprocess.CalledProcessError(1, "scontrol"))
    def test_scheduler_failure_records_alert_and_consumes_bounded_attempt(self, requeue):
        self.start_training()
        status = self.check(1801)
        self.assertEqual(status["alerts"][0]["reason"], "requeue_failed")
        self.assertEqual(read_json(self.root / "watchdog.json")["recoveries"][0]["action"], "requeue_failed")
        self.check(1802)
        requeue.assert_called_once()

    @patch("async_branching.watchdog.subprocess.run")
    def test_requeued_attempt_accepts_steps_lower_than_previous_attempt(self, requeue):
        self.start_training()
        self.append("david 49: {'david/rollouts': 400}\n")
        self.check(2)
        self.check(1802)
        self.worker["start_time"] = "new-attempt"
        self.check(1803)
        self.append("david 0: {'david/rollouts': 8}\n")
        status = self.check(1804)
        self.assertEqual(status["workers"][self.task.name]["rollouts"], 8)
        self.check(3603)
        self.assertEqual(requeue.call_count, 1)


if __name__ == "__main__":
    unittest.main()
