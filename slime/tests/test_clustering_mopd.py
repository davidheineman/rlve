"""Clustered MOPD protocol, causal scheduling, and checkpoint ownership checks."""

import asyncio
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from clustering_mopd.analysis import at_clock, score
from clustering_mopd.backend import prune_history
from clustering_mopd.monitor import check, recover_job
from clustering_mopd.monitor import state as job_state
from clustering_mopd.plan import (
    ASSIGNMENTS,
    ASSIGNMENTS32,
    REFERENCE,
    build_plan,
    configurations,
    metadata,
    partition,
)
from clustering_mopd.schedule import ClusterSimulation, ClusterState, initial_state
from submit_clustering_mopd import copy_base, submission_command, submit

from async_branching.io import read_json, write_json
from async_branching.replay import ReplayBackend
from async_branching.runner import Runner
from async_branching.schedule import Snapshot, Version


def toy_plan(count=16, horizon=2400, lanes=15):
    return {
        "clusters": [{"id": f"cluster-{i:02d}"} for i in range(count)],
        "student_rollouts": horizon,
        "simulated_clock_budget": horizon,
        "teacher_worker_count": min(count, lanes),
        "schedule_seed": 1234,
        "parent_policy": "latest-available",
        "teacher_device_rollout_horizon": None,
    }


def replay(plan, backend=None):
    state = initial_state(plan, {"hf": "base", "resume": "state"})
    sim = ClusterSimulation(state, teacher_cap=400, merge_cap=200, quantum=80)
    backend = backend or ReplayBackend(teacher_rollouts=400)
    with tempfile.TemporaryDirectory() as temporary:
        runner = Runner(sim, backend, Path(temporary))
        runner.run()
    return state, backend, runner.events


class ClusterProtocolTests(unittest.TestCase):
    def test_n32_sweep_has_fresh_baseline_and_joint_repeats_on_eight_devices(self):
        cases = configurations(32)
        self.assertEqual(len(cases), 7)
        self.assertIn((32, "singleton"), cases)
        old = read_json(REFERENCE / "plan.json")
        legacy = read_json(REFERENCE / "source/rlve" / old["id_eval"])
        joint = []
        for count, assignment in cases:
            plan, inputs = build_plan(
                count, assignment, size=32, devices=8, assignments=ASSIGNMENTS32
            )
            self.assertEqual(plan["environments"], old["environments"])
            self.assertEqual(plan["simulated_devices"], 8)
            self.assertEqual(plan["teacher_worker_count"], min(count, 7))
            self.assertLessEqual(plan["adapter_slots"], 8)
            self.assertEqual(inputs[plan["id_eval"]], legacy)
            self.assertEqual(len(plan["teachers"]), 32)
            self.assertEqual(plan["id_eval_prompts"], 32)
            self.assertEqual(plan["student_config"], old["student_config"])
            if count == 1:
                joint.append(plan)
            if assignment == "optimized" and count > 1:
                selected = read_json(ASSIGNMENTS32)
                expected = next(c for c in selected["candidates"] if c["k"] == count)[
                    "fit"
                ]["groups"]
                self.assertEqual(
                    [c["environment_indices"] for c in plan["clusters"]],
                    sorted(sorted(g) for g in expected),
                )
        self.assertEqual(
            joint[0]["clusters"][0]["environments"],
            joint[1]["clusters"][0]["environments"],
        )
        self.assertNotEqual(joint[0]["seed"], joint[1]["seed"])
        self.assertNotEqual(
            joint[0]["training_seed_offset"], joint[1]["training_seed_offset"]
        )
        self.assertEqual(
            joint[0]["initial_student_reference"], joint[1]["initial_student_reference"]
        )

    def test_eight_device_replay_enforces_seven_teacher_lanes(self):
        for count in (32, 8, 2, 1):
            state, backend, _ = replay(toy_plan(count, lanes=7))
            self.assertEqual(len(state.devices), min(7, count))
            self.assertEqual(state.clock, 2400)
            work = sum(state.teacher_rollout_counts().values()) + state.student_rollouts
            self.assertLessEqual(work, 8 * state.clock)
            self.assertTrue(all(t.get("lane", 0) < 7 for t in backend.tasks))

    def test_balanced_frozen_assignments_and_unique_endpoints(self):
        selected = read_json(ASSIGNMENTS)
        original = copy.deepcopy(selected)
        self.assertEqual(len(configurations()), 6)
        for count, assignment in configurations():
            groups = partition(count, assignment, selected, 1234)
            self.assertEqual(sorted(i for g in groups for i in g), list(range(64)))
            self.assertEqual({len(g) for g in groups}, {64 // count})
            if assignment == "optimized":
                old = next(r for r in selected["candidates"] if r["k"] == count)["fit"][
                    "groups"
                ]
                self.assertEqual(groups, sorted(sorted(g) for g in old))
        self.assertEqual(selected, original)
        self.assertNotEqual(
            partition(4, "random", selected, 1234),
            partition(4, "random", selected, 4321),
        )

    def test_matched_recipe_and_all_64_environment_evaluation(self):
        old = read_json(REFERENCE / "plan.json")
        for count, assignment in configurations():
            plan, inputs = build_plan(count, assignment)
            for field in (
                "teacher_branch_rollouts",
                "student_rollouts_per_teacher_version",
                "teacher_recovery_fraction",
                "student_quantum_rollouts",
                "lora_rank",
                "lora_alpha",
                "teacher_config",
                "student_config",
            ):
                self.assertEqual(plan[field], old[field], field)
            self.assertEqual(plan["physical_gpus"], 4)
            self.assertLessEqual(plan["adapter_slots"], 16)
            self.assertEqual(plan["student_project"], "rl-clustering-students")
            rows = inputs[plan["id_eval"]]
            self.assertEqual(len(rows), 64)
            self.assertEqual(
                {metadata(r)["environment"] for r in rows}, set(plan["environments"])
            )
            legacy = read_json(ASSIGNMENTS.parents[4] / plan["legacy_id_eval"])
            self.assertTrue(all(row in rows for row in legacy))
            for cluster in plan["clusters"]:
                gates = inputs[cluster["eval"]]
                self.assertEqual(len(gates), max(32, 64 // count))
                self.assertEqual(
                    {metadata(r)["environment"] for r in gates},
                    set(cluster["environments"]),
                )

    def test_pool_fifo_and_no_simultaneous_duplicate_clusters(self):
        _, backend, events = replay(toy_plan(64))
        teachers = [t for t in backend.tasks if t["kind"] == "teacher"]
        self.assertEqual(len({t["environment"] for t in teachers[:64]}), 64)
        self.assertEqual(len({t["id"] for t in teachers}), len(teachers))
        active, peak = {}, 0
        for event in events:
            if event["event"] == "branch":
                task = event["task"]
                self.assertNotIn(task["environment"], active.values())
                active[task["id"]] = task["environment"]
                peak = max(peak, len(active))
            elif event["event"] == "teacher_available":
                active.pop(event["task"]["id"])
        self.assertEqual(peak, 15)

    def test_causal_parents_and_fixed_common_deadline(self):
        for count in (1, 4, 16, 64):
            state, backend, _ = replay(toy_plan(count))
            self.assertEqual(state.clock, 2400)
            self.assertLessEqual(state.student_rollouts, state.clock)
            self.assertLessEqual(
                sum(state.teacher_rollout_counts().values()) + state.student_rollouts,
                16 * state.clock,
            )
            for task in backend.tasks:
                self.assertLessEqual(task["parent"]["clock"], task["start_clock"])
                self.assertLessEqual(task["start_clock"] + task["cap"], 2400)
                if task["kind"] == "teacher":
                    self.assertLessEqual(
                        task["parent"]["rollouts"], task["teacher_start_rollouts"]
                    )
            if count <= 4:
                for lane in range(count):
                    self.assertEqual(
                        len(
                            {
                                t["environment"]
                                for t in backend.tasks
                                if t["kind"] == "teacher"
                                and t["environment_index"] == lane
                            }
                        ),
                        1,
                    )

    def test_resume_with_active_tasks_matches_uninterrupted(self):
        plan = toy_plan(16)
        expected, _, _ = replay(plan)
        state = initial_state(plan, {"hf": "base", "resume": "state"})
        sim = ClusterSimulation(state, teacher_cap=400, merge_cap=200, quantum=80)
        sim.branches()
        restored = ClusterState.from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(restored.to_dict(), state.to_dict())
        with tempfile.TemporaryDirectory() as temporary:
            Runner(
                ClusterSimulation(restored, teacher_cap=400, merge_cap=200, quantum=80),
                ReplayBackend(teacher_rollouts=400),
                Path(temporary),
            ).run()
        self.assertEqual(restored.to_dict(), expected.to_dict())

    def test_zero_rollout_recovery_terminates_and_visits_every_cluster(self):
        state, backend, _ = replay(
            toy_plan(64), ReplayBackend(teacher_rollouts=0, recover_after=0)
        )
        teachers = [t for t in backend.tasks if t["kind"] == "teacher"]
        self.assertEqual(len(teachers), 64)
        self.assertEqual(state.student_rollouts, 0)
        self.assertEqual(state.stopped, "available_targets_recovered_or_exhausted")

    def test_zero_rollout_cluster_cannot_bounce_between_different_lane_parents(self):
        state = initial_state(toy_plan(16, lanes=2), {"hf": "base", "resume": "state"})
        state.clock = 800
        state.snapshots.append(Snapshot(400, 400, "later", "later-state"))
        state.devices[1].rollouts = 400
        state.cluster_queue = ["cluster-00"]
        sim = ClusterSimulation(state, teacher_cap=400, merge_cap=200, quantum=80)
        for expected in ("base", "later"):
            tasks = sim.branches()
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["parent"]["hf"], expected)
            sim.materialize(
                tasks[0]["id"], {"rollouts": 0, "accuracy": 0.99, "hf": expected}
            )
            sim.publish()
        self.assertEqual(sim.branches(), [])
        self.assertEqual(state.parked_parents["cluster-00"], ["base", "later"])

    def test_causal_curve_and_time_weighted_accuracy(self):
        rows = [
            {
                "async/simulated_clock": c,
                "david/eval/val_envs/pass@1": a,
                "david/rollouts": c,
            }
            for c, a in ((0, 0.1), (400, 0.3), (1200, 0.9))
        ]
        self.assertEqual(at_clock(rows, 399)["david/eval/val_envs/pass@1"], 0.1)
        result = score(rows, 800)
        self.assertAlmostEqual(result["mean_measured_id_over_clock"], 0.2)
        self.assertEqual(result["final_id_pass1"], 0.3)
        with self.assertRaises(ValueError):
            score(rows[1:], 800)

    def test_completed_analysis_exports_both_time_axes(self):
        from clustering_mopd.analysis import analyze

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            metrics = experiment / "workers/student/metrics.jsonl"
            metrics.parent.mkdir(parents=True)
            rows = [
                {
                    "async/simulated_clock": c,
                    "david/eval/val_envs/pass@1": a,
                    "david/rollouts": c,
                    "physical/elapsed_seconds": c * 2,
                }
                for c, a in ((0, 0.1), (400, 0.3), (800, 0.4))
            ]
            metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
            write_json(experiment / "completed.json", {"simulated_clock": 800})
            write_json(
                root / "sweep.json",
                {
                    "simulated_clock_budget": 800,
                    "experiments": [
                        {
                            "label": "synthetic-test",
                            "cluster_count": 4,
                            "assignment": "random",
                            "experiment": str(experiment),
                        }
                    ],
                },
            )
            result = analyze(root)
            self.assertEqual(result["records"][0]["final_id_pass1"], 0.4)
            for axis in ("simulated", "physical"):
                for suffix in ("png", "pdf"):
                    self.assertGreater(
                        (root / f"analysis/student_quality_vs_{axis}_time.{suffix}")
                        .stat()
                        .st_size,
                        1000,
                    )


class CheckpointAndSubmissionTests(unittest.TestCase):
    def test_launcher_bootstraps_repository_imports_outside_checkout(self):
        launcher = (
            Path(__file__).resolve().parents[2] / "tools/submit_clustering_mopd.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(launcher), "--help"],
                cwd=directory,
                env=os.environ | {"PYTHONPATH": ""},
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--simulated-devices", result.stdout)

    def test_prune_preserves_entire_future_parent_frontier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = initial_state(toy_plan(4), {"hf": "base", "resume": "state"})
            for d in state.devices:
                d.rollouts = 1600
            state.snapshots = []
            for rollouts in (0, 400, 800, 1200, 1600, 2000, 2400):
                directory = root / "tasks" / f"student-{rollouts}" / "final"
                (directory / "adapter").mkdir(parents=True)
                state.snapshots.append(
                    Snapshot(rollouts, rollouts, str(directory / "adapter"), "state")
                )
            state.devices[0].active = {
                "parent": vars(state.snapshots[2]).copy(),
                "cap": 400,
                "start_clock": 0,
            }
            final = root / "tasks/teacher-consumed/final"
            (final / "adapter").mkdir(parents=True)
            state.versions.append(
                Version(
                    "cluster-00", 1, 0, 400, 0.8, str(final / "adapter"), recovered=True
                )
            )
            prune_history(root, state, 1000)
            self.assertEqual(
                [s.rollouts for s in state.snapshots], [0, 800, 1600, 2000, 2400]
            )
            self.assertFalse((root / "tasks/student-400/final").exists())
            self.assertFalse(final.exists())
            self.assertEqual(
                read_json(root / "state.json")["snapshots"],
                [vars(s) for s in state.snapshots],
            )

    def test_prune_rejects_unowned_paths_before_mutating_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = initial_state(toy_plan(1), {"hf": "base", "resume": "state"})
            state.devices[0].rollouts = 400
            state.snapshots += [
                Snapshot(80, 80, "/outside/final/adapter"),
                Snapshot(400, 400, "latest"),
            ]
            before = state.to_dict()
            with self.assertRaises(ValueError):
                prune_history(root, state, 1000)
            self.assertEqual(state.to_dict(), before)
            self.assertFalse((root / "state.json").exists())

    def test_base_tensor_links_do_not_share_mutable_tenant_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for part in ("adapter", "state"):
                directory = root / "source" / part
                directory.mkdir(parents=True)
                write_json(
                    directory / "META.json", {"tenant_digest": "original", "rank": 32}
                )
                (directory / "tensors").write_bytes(b"immutable factors")
            copy_base(root / "source", root / "destination", "new-experiment")
            for part in ("adapter", "state"):
                self.assertEqual(
                    read_json(root / "source" / part / "META.json")["tenant_digest"],
                    "original",
                )
                self.assertNotEqual(
                    read_json(root / "destination" / part / "META.json")[
                        "tenant_digest"
                    ],
                    "original",
                )
                self.assertEqual(
                    (root / "source" / part / "tensors").stat().st_ino,
                    (root / "destination" / part / "tensors").stat().st_ino,
                )
                self.assertNotEqual(
                    (root / "source" / part / "META.json").stat().st_ino,
                    (root / "destination" / part / "META.json").stat().st_ino,
                )

    def test_submission_requests_four_gpus_on_the_approved_partitions(self):
        args = SimpleNamespace(
            scratch=Path("/scratch"), image=Path("/image"), model_path=Path("/model")
        )
        argv = submission_command(Path("/experiment"), args)
        self.assertIn("--gres=gpu:4", argv)
        self.assertIn(
            "--partition=jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio",
            argv,
        )

    def test_storage_guard_rejects_before_any_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"experiments": [], "required_free_scratch_gib": 204}
            with (
                patch("submit_clustering_mopd.verify_sweep", return_value=manifest),
                patch.dict("os.environ", {"WANDB_API_KEY": "test"}),
                patch(
                    "submit_clustering_mopd.shutil.disk_usage",
                    return_value=SimpleNamespace(free=0),
                ),
                patch("submit_clustering_mopd.subprocess.check_output") as sbatch,
            ):
                with self.assertRaisesRegex(RuntimeError, "Need 204 GiB"):
                    submit(root)
                sbatch.assert_not_called()

    def test_monitor_never_recovers_user_cancelled_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "owned"
            experiment.mkdir()
            write_json(
                root / "sweep.json",
                {"job_ids": ["123"], "experiments": [{"experiment": str(experiment)}]},
            )
            write_json(experiment / "job.json", {"job_id": "123"})
            (experiment / "slurm-123.out").write_text("ActorDiedError")
            with (
                patch(
                    "clustering_mopd.monitor.state",
                    return_value={"state": "CANCELLED", "node": "node"},
                ),
                patch("clustering_mopd.monitor.subprocess.run") as action,
            ):
                self.assertEqual(check(root, repair=True)["status"], "needs_attention")
                action.assert_not_called()


class ClusterMonitorTests(unittest.TestCase):
    def test_purged_job_uses_accounting_when_squeue_rejects_the_id(self):
        with (
            patch(
                "clustering_mopd.monitor.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 1, "", "Invalid job id specified"
                ),
            ),
            patch(
                "clustering_mopd.monitor.subprocess.check_output",
                return_value="123|FAILED|node-a|1:0\n",
            ) as accounting,
        ):
            self.assertEqual(
                job_state("123"), {"state": "FAILED", "node": "node-a", "reason": "1:0"}
            )
            self.assertEqual(accounting.call_args.args[0][0], "sacct")

    def test_scheduler_failure_is_not_misreported_as_a_purged_job(self):
        with (
            patch(
                "clustering_mopd.monitor.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 1, "", "Unable to contact slurm controller"
                ),
            ),
            patch("clustering_mopd.monitor.subprocess.check_output") as accounting,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                job_state("123")
            accounting.assert_not_called()

    def test_purged_recovery_reuses_experiment_and_records_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "owned"
            directory.mkdir()
            argv = ["sbatch", "--exclude=old-node", "--wrap", f"resume {directory}"]
            write_json(directory / "submit_argv.json", argv)
            write_json(
                directory / "job.json", {"job_id": "123", "experiment": str(directory)}
            )
            write_json(root / "sweep.json", {"job_ids": ["123", "456"]})
            checkpoint = directory / "checkpoint"
            checkpoint.write_bytes(b"saved Adam and adapter")
            with (
                patch(
                    "clustering_mopd.monitor.subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        [], 1, "", "Invalid job id specified"
                    ),
                ) as mutate,
                patch(
                    "clustering_mopd.monitor.subprocess.check_output",
                    return_value="789\n",
                ) as submit_job,
            ):
                self.assertEqual(
                    recover_job(root, directory, "123", exclude_node="full-node"), "789"
                )
            self.assertEqual(mutate.call_count, 1)
            submitted = submit_job.call_args.args[0]
            self.assertIn("--exclude=full-node,old-node", submitted)
            self.assertEqual(submitted[-1], f"resume {directory}")
            self.assertEqual(
                read_json(directory / "job.json")["previous_job_ids"], ["123"]
            )
            self.assertEqual(read_json(root / "sweep.json")["job_ids"], ["789", "456"])
            self.assertEqual(checkpoint.read_bytes(), b"saved Adam and adapter")

    def test_gateway_disk_error_triggers_recovery_and_node_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "owned"
            directory.mkdir()
            write_json(
                root / "sweep.json",
                {"job_ids": ["123"], "experiments": [{"experiment": str(directory)}]},
            )
            write_json(directory / "job.json", {"job_id": "123"})
            (directory / "slurm-123.out").write_text("Miles gateway exited")
            (directory / "gateway.log").write_text(
                "Failed flush to file /tmp/miles-123/log: No space left on device"
            )
            with (
                patch(
                    "clustering_mopd.monitor.state",
                    return_value={"state": "FAILED", "node": "full-node"},
                ),
                patch(
                    "clustering_mopd.monitor.recover_job", return_value="789"
                ) as recover,
            ):
                report = check(root, repair=True)
            recover.assert_called_once_with(
                root, directory, "123", exclude_node="full-node"
            )
            self.assertTrue(report["experiments"][0]["requeued"])
            self.assertEqual(report["experiments"][0]["job_id"], "789")


class ClusterWorkerTests(unittest.TestCase):
    def test_active_student_lease_prevents_teacher_export_reclamation(self):
        from clustering_mopd.worker import ClusterWorker

        with tempfile.TemporaryDirectory() as temporary:
            worker = object.__new__(ClusterWorker)
            worker.root = Path(temporary)
            uri = "tinker://teacher/sampler_weights/published"
            export = worker.root / "teacher/sampler_weights/published"
            export.mkdir(parents=True)
            (export / "adapter_model.safetensors").write_bytes(b"immutable teacher")
            with worker.sampler_lease(uri, shared=True):
                self.assertFalse(worker.retire_sampler(uri))
                self.assertEqual(
                    (export / "adapter_model.safetensors").read_bytes(),
                    b"immutable teacher",
                )
            self.assertTrue(worker.retire_sampler(uri))
            self.assertFalse(export.exists())

    def test_first_import_creates_an_absent_gateway_checkpoint_root(self):
        from async_lora.worker import Worker
        from clustering_mopd.worker import ClusterWorker

        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            worker = object.__new__(ClusterWorker)
            worker.root = p / "not-created-by-the-gateway-yet"
            durable = p / "durable/state"
            durable.mkdir(parents=True)
            write_json(
                durable.parent / "reference.json",
                {"state_uri": "tinker://model/weights/imported"},
            )

            async def load(path, *, optimizer):
                self.assertTrue(worker.root.is_dir())
                (worker.root / "model/weights/imported").mkdir(parents=True)

            with patch.object(Worker, "load", side_effect=load):
                asyncio.run(worker.load(str(durable), optimizer=False))
            self.assertTrue(worker.root.is_dir())
            self.assertFalse((worker.root / "model/weights/imported").exists())

    def test_imported_state_cache_is_reclaimed_after_successful_load(self):
        from async_lora.worker import Worker
        from clustering_mopd.worker import ClusterWorker

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = object.__new__(ClusterWorker)
            worker.root = root
            durable = root / "durable/state"
            durable.mkdir(parents=True)
            cache = root / "model/weights/imported"
            cache.mkdir(parents=True)
            write_json(
                durable.parent / "reference.json",
                {"state_uri": "tinker://model/weights/imported"},
            )
            with patch.object(Worker, "load", new=AsyncMock()) as load:
                asyncio.run(worker.load(str(durable), optimizer=True))
                load.assert_awaited_once_with(str(durable), optimizer=True)
            self.assertFalse(cache.exists())
            self.assertTrue(durable.exists())

    def test_finished_merge_reclaims_only_ephemeral_exports(self):
        from async_lora.worker import Worker
        from clustering_mopd.worker import ClusterWorker

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = object.__new__(ClusterWorker)
            worker.root = root
            worker.role = "student"
            worker.sampler_uri = "tinker://student/sampler_weights/old"
            for model, name in (
                ("student", "old"),
                ("student", "new"),
                ("teacher", "final"),
            ):
                (root / model / "sampler_weights" / name).mkdir(parents=True)
            durable = root / "durable/adapter"
            durable.mkdir(parents=True)
            write_json(
                durable.parent / "reference.json",
                {"sampler_uri": "tinker://teacher/sampler_weights/final"},
            )

            async def execute(task):
                worker.sampler_uri = "tinker://student/sampler_weights/new"
                return {"rollouts": 80}

            with patch.object(Worker, "execute", side_effect=execute):
                result = asyncio.run(worker.execute({"teacher_hf": str(durable)}))
            self.assertEqual(result["rollouts"], 80)
            self.assertTrue((root / "student/sampler_weights/new").is_dir())
            self.assertFalse((root / "student/sampler_weights/old").exists())
            self.assertFalse((root / "teacher/sampler_weights/final").exists())
            self.assertTrue(durable.exists())

    def test_panel_weights_environments_equally(self):
        from clustering_mopd.worker import ClusterWorker

        worker = object.__new__(ClusterWorker)
        worker.plan = {}
        worker.sampler = worker.tokenizer = None
        rows = [
            {"user_prompt": e, "metadata": {"environment": e, "config": {}}}
            for e in ("easy", "easy", "hard")
        ]

        async def sample(
            sampler, tokenizer, tokens, environment, config, settings, **kwargs
        ):
            return [
                SimpleNamespace(
                    accuracy=float(environment == "easy"), forced=False, truncated=False
                )
            ]

        with (
            patch("clustering_mopd.worker.prompt_tokens", return_value=[]),
            patch("clustering_mopd.worker.sample_group", side_effect=sample),
        ):
            result = asyncio.run(worker.panel(rows, {}))
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["samples"], 3)

    def test_initial_student_import_uses_common_factors_with_fresh_optimizer(self):
        from async_lora.worker import Worker

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = object.__new__(Worker)
            worker.plan = {
                "base_model": "base",
                "lora_rank": 32,
                "student_project": "project",
                "initial_student_reference": {"bundle": "initial-base"},
            }
            worker.experiment = worker.directory = worker.root = root
            worker.role = "student"
            worker.replay = None
            worker.service = SimpleNamespace(
                create_lora_training_client_async=AsyncMock(
                    return_value=SimpleNamespace(model_id="model")
                )
            )
            worker.load = AsyncMock()
            worker.open_run = Mock()
            (root / "baseline.json").touch()
            (root / "preflight").mkdir()
            (root / "preflight/passed.json").touch()
            with (
                patch(
                    "async_lora.worker.export",
                    new=AsyncMock(return_value=("sampler", "uri")),
                ),
                patch("async_lora.worker.uri_directory", return_value=root / "export"),
                patch(
                    "async_lora.worker.adapter_identity", return_value={"equal": True}
                ) as identity,
                patch(
                    "async_lora.worker.save_bundle",
                    new=AsyncMock(return_value={"hf": "hf", "resume": "state"}),
                ),
            ):
                asyncio.run(worker.initialize())
            worker.load.assert_awaited_once_with(
                str(root / "initial-base/state"), optimizer=False
            )
            identity.assert_called_once_with(
                root / "initial-base/adapter", root / "export"
            )
            self.assertTrue((root / "initial-student-identity.json").exists())


if __name__ == "__main__":
    unittest.main()
