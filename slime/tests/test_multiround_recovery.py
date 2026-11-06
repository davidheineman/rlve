"""Exercise actual recovery state transitions with a stateful fake scheduler."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import watch_data_scaling_multiround as recovery
import submit_data_scaling_multiround_resilient as submit


class Scheduler:
    def __init__(self):
        self.jobs = {}
        self.calls = []
        self.fail_update = False

    def add(self, job, state="PENDING", **extra):
        self.jobs[job] = dict(JobId=job, UserId=f"test({os.getuid()})", JobState=state,
                              Restarts="0", Priority="100", Dependency="(null)", **extra)

    def info(self, job):
        return dict(self.jobs[job])

    def command(self, args):
        self.calls.append(args)
        if args[0] == "sha256sum":
            return ""
        if args[:2] == ["scontrol", "requeuehold"]:
            job = self.jobs[args[2]]
            job.update(JobState="PENDING", Priority="0", Restarts=str(int(job["Restarts"]) + 1))
        elif args[:2] == ["scontrol", "update"]:
            if self.fail_update:
                self.fail_update = False
                raise subprocess.CalledProcessError(1, args)
            fields = dict(arg.split("=", 1) for arg in args[2:])
            self.jobs[fields.pop("JobId")].update(fields)
        elif args[:2] == ["scontrol", "release"]:
            self.jobs[args[2]]["Priority"] = "100"
        elif args[0] == "squeue":
            return "\n".join(job + "|" + info["comment"] for job, info in self.jobs.items()
                             if "comment" in info and info["JobState"] == "PENDING")
        elif args[0] == "sbatch":
            self.add("200", PriorityUnused="unused")
            self.jobs["200"]["Priority"] = "0"
            self.jobs["200"]["comment"] = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--comment="))
            return "200"
        else:
            raise AssertionError(args)
        return ""


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = [dict(label="a", kind="teacher", job="101", phase="train", gpus="1"),
                     dict(label="b", kind="teacher", job="102", phase="train", gpus="1"),
                     dict(label="n16-mopd", kind="mopd", job="103", phase="train", gpus="4")]
        for row in self.rows:
            row.update(training_job=row["job"], training_gpus=row["gpus"], dependency="none")
            (self.root / row["kind"] / row["label"]).mkdir(parents=True)
        self.plan = dict(method="multi-round", round=2, initial_hf="original-round1-model",
                         runs=[{**r, "updates": 500} for r in self.rows])
        (self.root / "run_plan.json").write_text(json.dumps(self.plan))
        with (self.root / "jobs.tsv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows)
        self.scheduler = Scheduler()
        for row in self.rows:
            self.scheduler.add(row["job"])
        self.patch_command = patch.object(recovery, "command", side_effect=self.scheduler.command)
        self.patch_info = patch.object(recovery, "job_info", side_effect=self.scheduler.info)
        self.patch_command.start()
        self.patch_info.start()
        self.addCleanup(self.patch_command.stop)
        self.addCleanup(self.patch_info.stop)

    def fail(self, label, text, state="FAILED"):
        row = next(row for row in self.rows if row["label"] == label)
        self.scheduler.jobs[row["job"]]["JobState"] = state
        (self.root / row["kind"] / label / "train.log").write_text("Final-model run current: start\n" + text)

    def export(self, label):
        row = next(row for row in self.rows if row["label"] == label)
        run = self.root / row["kind"] / label
        (run / "hf-final").mkdir(exist_ok=True)
        (run / "weights_only_complete.json").write_text(json.dumps(dict(format="hf-safetensors", weight_bytes=1)))
        (run / "hf-final/model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"w": "model.safetensors"})))
        (run / "hf-final/model.safetensors").write_bytes(b"fixture")

    def holds(self):
        return [call for call in self.scheduler.calls if call[:2] == ["scontrol", "requeuehold"]]

    def test_both_observed_failures_upgrade_gpu_and_repair_student_without_replacing_ids(self):
        self.fail("a", "CUresult error: 2 (out of memory)")
        self.fail("b", "ray.exceptions.ActorDiedError: The actor died unexpectedly")
        self.scheduler.jobs["103"]["Dependency"] = "afterok:101(failed),afterok:102(failed)"
        watchdog = recovery.Watchdog(self.root)
        watchdog.tick()
        self.assertEqual(len(self.holds()), 2)
        for job in ("101", "102"):
            self.assertEqual(self.scheduler.jobs[job]["Features"], "80G|141G")
            self.assertNotIn("jag", self.scheduler.jobs[job]["Partition"])
            self.assertEqual(self.scheduler.jobs[job]["Priority"], "100")
        self.assertEqual(self.scheduler.jobs["103"]["Dependency"], "afterok:101:102")
        self.assertEqual([event["stage"] for event in watchdog.state["events"]], ["requeued", "requeued"])
        self.assertEqual([row["job"] for row in watchdog.rows()], ["101", "102", "103"])

    def test_unknown_errors_cancellation_healthy_and_finished_runs_are_not_retried(self):
        self.fail("a", "ValueError: unrecognized configuration")
        self.fail("b", "ActorDiedError", state="CANCELLED")
        recovery.Watchdog(self.root).tick()
        self.assertEqual(self.holds(), [])
        self.export("a")
        self.scheduler.jobs["102"]["JobState"] = "RUNNING"
        recovery.Watchdog(self.root).tick()
        self.assertEqual(self.holds(), [])

    def test_old_attempt_error_is_not_used_to_classify_new_application_error(self):
        self.assertIsNone(recovery.retry_reason("FAILED", "ValueError: bad config"))
        run = self.root / "teacher/a"
        (run / "train.log").write_text("Final-model run old: CUDA out of memory\nFinal-model run new: ValueError\n")
        self.assertNotIn("out of memory", recovery.log_tail(run))

    def test_retries_are_bounded_across_monitor_restarts(self):
        for _ in range(3):
            self.fail("a", "ActorUnavailableError")
            recovery.Watchdog(self.root, max_retries=2).tick()
        self.assertEqual(len(self.holds()), 2)
        state = json.loads((self.root / "recovery_watchdog.json").read_text())
        self.assertIn("retry_limit", [alert["reason"] for alert in state["alerts"]])

    def test_interrupted_gpu_upgrade_resumes_held_retry_without_double_requeue(self):
        self.fail("a", "CUDA out of memory")
        self.scheduler.fail_update = True
        with self.assertRaises(subprocess.CalledProcessError):
            recovery.Watchdog(self.root).tick()
        self.assertEqual(self.scheduler.jobs["101"]["Priority"], "0")
        recovery.Watchdog(self.root).tick()
        self.assertEqual(len(self.holds()), 1)
        self.assertEqual(self.scheduler.jobs["101"]["Priority"], "100")

    def test_changed_training_inputs_stop_recovery_before_scheduler_mutation(self):
        self.fail("a", "CUDA out of memory")
        def fail_checksum(args):
            if args[0] == "sha256sum":
                raise subprocess.CalledProcessError(1, args)
            return self.scheduler.command(args)
        with patch.object(recovery, "command", side_effect=fail_checksum):
            with self.assertRaises(subprocess.CalledProcessError):
                recovery.Watchdog(self.root).tick()
        self.assertEqual(self.holds(), [])

    def test_monitor_reattachment_does_not_submit_duplicate_cpu_job(self):
        self.scheduler.add("300", "RUNNING")
        (self.root / "recovery_watchdog_job.json").write_text(json.dumps({"job": "300"}))
        args = ["watch", "--experiment", str(self.root), "--submit"]
        with patch.object(sys, "argv", args):
            recovery.main()
        self.assertFalse(any(call[0] == "sbatch" for call in self.scheduler.calls))

    def terminal_checkpoint(self):
        run = self.root / "teacher/a"
        model = run / "final-ckpt/iter_0000499"
        model.mkdir(parents=True)
        for name in ("common.pt", ".metadata", "__0_0.distcp"):
            (model / name).write_bytes(b"fixture")
        (run / "final-ckpt/latest_checkpointed_iteration.txt").write_text("499")
        (run / "train.log").write_text("david 499: terminal step\nActorUnavailableError during shutdown\n")
        self.scheduler.jobs["101"]["JobState"] = "FAILED"

    def test_finished_training_gets_cpu_export_instead_of_retraining(self):
        self.terminal_checkpoint()
        watchdog = recovery.Watchdog(self.root)
        watchdog.tick()
        self.assertEqual(self.holds(), [])
        row = next(row for row in watchdog.rows() if row["label"] == "a")
        self.assertEqual((row["job"], row["phase"], row["gpus"]), ("200", "finalize", "0"))
        self.assertEqual(row["training_job"], "101")
        self.assertEqual(self.scheduler.jobs["103"]["Dependency"], "afterok:200:102")
        self.assertEqual(self.scheduler.jobs["200"]["Priority"], "100")

    def test_interrupted_finalizer_publication_recovers_journal_without_duplicate_submission(self):
        self.terminal_checkpoint()
        watchdog = recovery.Watchdog(self.root)
        watchdog.finalize(watchdog.rows()[0])
        self.assertEqual(self.scheduler.jobs["200"]["Priority"], "0")
        recovery.Watchdog(self.root).tick()
        self.assertEqual(sum(call[0] == "sbatch" for call in self.scheduler.calls), 1)
        self.assertEqual(recovery.Watchdog(self.root).rows()[0]["job"], "200")

    def test_stale_failed_dependency_clears_when_teacher_exports_are_ready(self):
        self.export("a")
        self.export("b")
        self.scheduler.jobs["103"]["Dependency"] = "afterok:101(failed)"
        recovery.Watchdog(self.root).tick()
        self.assertEqual(self.scheduler.jobs["103"]["Dependency"], "")

    def test_missing_export_shard_is_never_treated_as_a_finished_teacher(self):
        self.export("a")
        (self.root / "teacher/a/hf-final/model.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing final export"):
            recovery.Watchdog(self.root).tick()
        self.assertEqual(self.holds(), [])


class ResilientSubmissionTest(unittest.TestCase):
    def test_new_submission_attaches_monitor_after_established_training_launcher(self):
        args = ["submit", "--parent-experiment", "/scratch/round1", "--experiment", "/scratch/round2",
                "--submit"]
        with patch.object(sys, "argv", args), patch.object(submit.subprocess, "run") as run:
            submit.main()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn("submit_data_scaling_multiround_2k.py", commands[0][1])
        self.assertIn("watch_data_scaling_multiround.py", commands[1][1])
        self.assertIn("/scratch/round2", commands[1])
        self.assertFalse(any("array" in part for command in commands for part in command))


if __name__ == "__main__":
    unittest.main()
