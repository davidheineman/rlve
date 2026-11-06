"""Optional GPU ownership, exact adapter admission and disappearance handling."""

import os
import shlex
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from async_lora.flex_manager import FlexManager
from async_lora.flex_protocol import (
    check_probe,
    engine_argv,
    unregister,
    validate_engine,
)
from async_lora.flex_worker import lease_valid
from async_lora.plan import build_async_plan, experiment_stem
from lora_teachers.plan import gateway_argv

from async_branching.io import read_json, write_json


class FlexProtocolTest(unittest.TestCase):
    def test_flex_plan_keeps_two_trainers_and_original_objective(self):
        original = build_async_plan(single_question=True, replay_percent=30)
        plan = build_async_plan(single_question=True, replay_percent=30, speed_profile="v3", flex_gpus=8)
        self.assertEqual((plan["train_gpus"], plan["sampling_gpus"], plan["max_physical_gpus"]), (2, 2, 12))
        self.assertEqual(plan["student_config"], original["student_config"])
        self.assertEqual(plan["opd_replay"], original["opd_replay"])
        self.assertEqual(plan["single_question"], original["single_question"])
        self.assertEqual(plan["teacher_config"]["eval-interval"], 100)
        self.assertEqual(plan["teacher_config"]["n-samples-per-eval-prompt"], 4)
        self.assertTrue(experiment_stem(plan).endswith("-speed-v3-flex8"))
        argv = gateway_argv(plan, model_path="model", checkpoint_root="root", port=6000, router_port=7000)
        self.assertEqual(argv[argv.index("--sglang-router-request-timeout-secs") + 1], "600")
        for count in (-1, 9, True):
            with self.assertRaises(ValueError):
                build_async_plan(flex_gpus=count)
        with self.assertRaises(ValueError):
            build_async_plan(speed_profile="v2", flex_gpus=8)

    def test_clone_preserves_backend_but_drops_core_network_and_launch_gate(self):
        argv = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "/model with spaces",
            "--lora-target-modules",
            "q_proj",
            "v_proj",
            "--enable-lora",
            "--max-lora-rank",
            "32",
            "--host",
            "core",
            "--port",
            "20000",
            "--base-gpu-id",
            "3",
            "--random-seed",
            "1234",
            "--dist-init-addr",
            "core:123",
            "--gated-launch-port",
            "20034",
            "--nccl-port",
            "20032",
            "--engine-info-bootstrap-port",
            "20033",
            "--skip-server-warmup",
        ]
        result = engine_argv(
            {"launch_command": shlex.join(argv), "random_seed": 1234}, host="remote", port=6500, slot=2
        )
        self.assertIn("/model with spaces", result)
        self.assertEqual(result[result.index("--base-gpu-id") + 1], "0")
        self.assertEqual(result[result.index("--host") + 1], "remote")
        self.assertEqual(result[result.index("--random-seed") + 1], "2236")
        self.assertNotIn("--gated-launch-port", result)
        self.assertNotIn("--dist-init-addr", result)
        self.assertNotIn("--engine-info-bootstrap-port", result)
        self.assertIn("--enable-lora", result)
        self.assertEqual(result[result.index("--lora-target-modules") + 1 :][:2], ["q_proj", "v_proj"])
        flags_only = engine_argv(
            {"launch_command": shlex.join(argv[3:]), "random_seed": 1234}, host="remote", port=6500, slot=2
        )
        self.assertEqual(result, flags_only)

    def test_base_and_nonzero_adapter_must_match_before_admission(self):
        info = {
            "model_path": "/model",
            "enable_lora": True,
            "max_lora_rank": 32,
            "max_loras_per_batch": 17,
            "tp_size": 1,
        }
        validate_engine(info, model_path="/model", slots=17, rank=32)
        with self.assertRaises(ValueError):
            validate_engine(info | {"enable_lora": False}, model_path="/model", slots=17, rank=32)
        probe = {"request": {"lora_path": "immutable-version"}, "logprobs": [None, -2, -3]}
        with patch("async_lora.flex_protocol.http") as request:
            request.return_value = {"meta_info": {"input_token_logprobs": [[None], [-2.01], [-2.99]]}}
            self.assertAlmostEqual(check_probe("remote", probe)["adapter_logprob_max_abs_difference"], 0.01)
            request.assert_called_with("remote/generate", probe["request"], timeout=120)
            for values in ([[None], [-2.2], [-3]], [[None], [float("nan")], [-3]], [[None], [-2]]):
                request.return_value = {"meta_info": {"input_token_logprobs": values}}
                with self.assertRaises(ValueError):
                    check_probe("remote", probe)

    def test_unregistration_never_removes_core_workers(self):
        with patch("async_lora.flex_protocol.http") as request:
            request.return_value = {"workers": [{"id": "core", "url": "core"}, {"id": "extra", "url": "extra"}]}
            unregister("router", "extra")
            self.assertEqual(request.call_args_list[-1].args, ("router/workers/extra",))
            self.assertEqual(request.call_args_list[-1].kwargs, {"method": "DELETE"})

    def test_old_or_expired_core_lease_stops_optional_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / "lease.json", {"generation": "current", "time": time.time()})
            self.assertTrue(lease_valid(directory, "current"))
            self.assertFalse(lease_valid(directory, "old"))
            write_json(directory / "lease.json", {"generation": "current", "time": time.time() - 61})
            self.assertFalse(lease_valid(directory, "current"))


class FlexManagerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.experiment = Path(self.temporary.name)
        write_json(self.experiment / "plan.json", {"elastic_generation": {"max_extra_gpus": 8}})
        self.args = SimpleNamespace(
            experiment=self.experiment, scratch=Path("/scratch"), image=Path("/image"), model_path=Path("/model")
        )
        self.environment = patch.dict(os.environ, {"SLURM_JOB_ID": "core-job", "USER": "test"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.manager = FlexManager(self.args)
        self.core = {"generation": "generation", "core_job_id": "core-job", "router_url": "router"}

    def prepare(self):
        write_json(self.experiment / "flex/core.json", self.core)
        write_json(self.experiment / "flex/probe.json", {"request": {}})

    def test_no_extra_allocations_until_core_and_adapter_probe_ready(self):
        with patch.object(self.manager, "submit") as submit, patch.object(self.manager, "jobs", return_value=[]):
            self.manager.tick()
            write_json(self.experiment / "flex/core.json", self.core)
            self.manager.last_poll = 0
            self.manager.tick()
            submit.assert_not_called()

    def test_pending_and_running_both_count_and_are_adopted_after_restart(self):
        self.prepare()
        jobs = [
            [str(slot), f"{self.manager.prefix}generation:{slot}", "PENDING" if slot % 2 else "RUNNING"]
            for slot in range(8)
        ]
        with patch.object(self.manager, "jobs", return_value=jobs), patch.object(self.manager, "submit") as submit:
            self.manager.tick()
            submit.assert_not_called()
        self.assertEqual(len(self.manager.owned), 8)
        self.assertEqual(len(read_json(self.experiment / "flex/status.json")["active_jobs"]), 8)

    def test_lost_extra_is_removed_and_only_its_slot_is_replaced(self):
        self.prepare()
        jobs = [[str(slot), f"{self.manager.prefix}generation:{slot}", "RUNNING"] for slot in range(7)]
        self.manager.owned["lost"] = {"slot": 7, "generation": "generation"}
        write_json(self.experiment / "flex/worker-7.json", {"job_id": "lost", "url": "lost-url"})
        with (
            patch.object(self.manager, "jobs", return_value=jobs),
            patch("async_lora.flex_manager.unregister") as remove,
            patch.object(self.manager, "submit") as submit,
        ):
            self.manager.tick()
            remove.assert_called_once_with("router", "lost-url")
            submit.assert_called_once_with(self.core, 7)

    def test_terminating_old_generation_still_counts_toward_eight_job_cap(self):
        self.prepare()
        jobs = [[str(slot), f"{self.manager.prefix}old:{slot}", "COMPLETING"] for slot in range(8)]
        with (
            patch.object(self.manager, "jobs", return_value=jobs),
            patch.object(self.manager, "submit") as submit,
            patch("async_lora.flex_manager.subprocess.run") as cancel,
        ):
            self.manager.tick()
            self.assertEqual(cancel.call_count, 8)
            submit.assert_not_called()

    def test_slurm_failure_never_triggers_unbounded_submissions(self):
        self.prepare()
        with (
            patch.object(self.manager, "jobs", side_effect=subprocess.TimeoutExpired("squeue", 20)),
            patch.object(self.manager, "submit") as submit,
        ):
            self.manager.tick()
            submit.assert_not_called()
            self.assertTrue((self.experiment / "flex/manager_error.json").is_file())

    def test_cleanup_cancels_only_owned_extra_jobs_and_expires_the_lease(self):
        self.prepare()
        with (
            patch.object(self.manager, "jobs", return_value=[["extra", "comment", "PENDING"]]),
            patch("async_lora.flex_manager.subprocess.run") as cancel,
        ):
            self.manager.close()
            self.assertEqual(cancel.call_args.args[0], ["scancel", "extra"])
            self.assertEqual(read_json(self.experiment / "flex/lease.json")["time"], 0)


if __name__ == "__main__":
    unittest.main()
