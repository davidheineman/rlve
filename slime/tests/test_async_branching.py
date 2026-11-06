import json
import tempfile
import unittest
from pathlib import Path

from async_branching.schedule import Simulation, Snapshot, State, initial_state
from async_branching.replay import ReplayBackend
from async_branching.runner import Runner
from async_branching.segment import configure_args
from types import SimpleNamespace


class AsyncScheduleTest(unittest.TestCase):
    def make(self, names=("A", "B"), budget=16000, policy="latest-available"):
        return Simulation(initial_state(names, "base/hf", "base/ref", budget, policy))

    def complete_first(self, sim, accuracy=.5):
        tasks = sim.branches()
        for task in tasks:
            sim.materialize(task["id"], {"rollouts": 4000, "accuracy": accuracy, "hf": task["id"] + "/hf-final"})
        sim.state.clock = 4000
        sim.publish()
        return tasks

    def test_latest_parent_never_reads_future_student_rollouts_or_event_time(self):
        sim = self.make()
        sim.state.snapshots += [Snapshot(1200, 5200, "s1200"), Snapshot(4000, 8000, "s4000"),
                                Snapshot(4800, 8800, "s4800")]
        sim.state.student_rollouts = 4800
        sim.state.clock = 5200
        self.assertEqual(sim.parent(4000).hf, "s1200")
        sim.state.clock = 9000
        self.assertEqual(sim.parent(4000).hf, "s4000")
        sim.state.student_rollouts = 1200
        sim.state.clock = 5200
        self.assertEqual(sim.parent(4000).hf, "s1200")

    def test_teacher_is_not_published_early_and_idle_is_not_training(self):
        sim = self.make()
        task = sim.branches()[0]
        sim.materialize(task["id"], {"rollouts": 4000, "accuracy": .5, "hf": "teacher"})
        sim.state.clock = 3992
        self.assertEqual(sim.publish(), [])
        self.assertIsNone(sim.student_task())
        sim.state.clock = 4000
        sim.publish()
        self.assertEqual(sim.state.student_rollouts, 0)
        # Another unresolved teacher event must be resolved before any merge.
        with self.assertRaises(RuntimeError):
            sim.student_task()

    def test_90_percent_gate_and_exact_1000_rollout_per_version_cap(self):
        sim = self.make(names=("A",))
        self.complete_first(sim, .5)
        counts = []
        for accuracy in (.2, .3, .4):
            task = sim.student_task()
            counts.append(task["cap"])
            self.assertEqual(task["target"], .45)
            sim.merge(task, {"rollouts": task["cap"], "accuracy": accuracy, "hf": "s", "resume": "opt"})
        self.assertEqual(counts, [400, 400, 200])
        self.assertIsNone(sim.student_task())
        self.assertEqual(sim.state.versions[0].used, 1000)
        self.assertEqual(sim.state.student_rollouts, 1000)
        sim2 = self.make(names=("A",))
        self.complete_first(sim2)
        task = sim2.student_task()
        sim2.merge(task, {"rollouts": 0, "accuracy": .45})
        self.assertTrue(sim2.state.versions[0].recovered)
        self.assertEqual(sim2.state.student_rollouts, 0)

    def test_independent_device_counters_and_early_stops(self):
        sim = self.make()
        tasks = sim.branches()
        with self.assertRaises(ValueError):
            sim.materialize(tasks[0]["id"], {"rollouts": 400, "accuracy": .97, "hf": "bad"})
        sim.materialize(tasks[0]["id"], {"rollouts": 400, "accuracy": .99, "hf": "early"})
        sim.materialize(tasks[1]["id"], {"rollouts": 4000, "accuracy": .5, "hf": "late"})
        sim.state.clock = 400
        sim.publish()
        self.assertEqual([d.rollouts for d in sim.state.devices], [400, 0])
        again = sim.branches()
        self.assertEqual(again[0]["teacher_start_rollouts"], 400)
        self.assertEqual(again[0]["parent"]["rollouts"], 0)
        self.assertEqual(again[0]["start_clock"], 400)

    def test_new_version_does_not_discard_an_unfinished_old_merge(self):
        sim = self.make(names=("A",))
        self.complete_first(sim)
        old = sim.state.versions[0]
        task = sim.student_task()
        sim.merge(task, {"rollouts": 400, "accuracy": .1, "hf": "student", "resume": "opt"})
        branch = sim.branches()[0]
        sim.materialize(branch["id"], {"rollouts": 400, "accuracy": .99, "hf": "newteacher"})
        sim.state.clock = branch["start_clock"] + 400
        sim.publish()
        self.assertEqual(len(sim.state.versions), 2)
        self.assertEqual(old.used, 400)
        self.assertFalse(old.recovered)
        self.assertEqual(sim.student_task()["generation"], 2)

    def test_invalid_or_stale_backend_results_fail_closed(self):
        sim = self.make(names=("A",))
        self.complete_first(sim)
        task = sim.student_task()
        for result in ({"rollouts": 401, "accuracy": .9}, {"rollouts": 8, "accuracy": .1},
                       {"rollouts": 400, "accuracy": .2}):
            with self.assertRaises(ValueError):
                sim.merge(task, result)
        sim.merge(task, {"rollouts": 400, "accuracy": .2, "hf": "s400", "resume": "opt400"})
        with self.assertRaises(ValueError):
            sim.merge(task, {"rollouts": 400, "accuracy": .2, "hf": "s800", "resume": "opt800"})

    def test_state_roundtrip_and_checkpoint_watermark(self):
        sim = self.make(names=("A",))
        self.complete_first(sim)
        task = sim.student_task()
        sim.merge(task, {"rollouts": 400, "accuracy": .2, "hf": "s400", "resume": "opt400"})
        restored = State.from_dict(json.loads(json.dumps(sim.state.to_dict())))
        replay = Simulation(restored)
        self.assertEqual(replay.parent(4000).rollouts, 400)
        self.assertEqual(replay.state.versions[0].used, 400)
        self.assertEqual(replay.state.snapshots[-1].resume, "opt400")

    def test_zero_rollout_teacher_is_parked_on_an_unchanged_parent(self):
        sim = self.make(names=("A",))
        task = sim.branches()[0]
        sim.materialize(task["id"], {"rollouts": 0, "accuracy": .99, "hf": "already_solved"})
        sim.publish()
        self.assertEqual(sim.state.devices[0].rollouts, 0)
        self.assertEqual(sim.branches(), [])
        merge = sim.student_task()
        sim.merge(merge, {"rollouts": 0, "accuracy": .99})
        self.assertFalse(sim.advance_idle())
        self.assertEqual(sim.state.stopped, "available_targets_recovered_or_exhausted")

    def test_worker_configuration_preserves_student_optimizer_only(self):
        for kind, start, resume in (("teacher", 0, False), ("student", 0, False), ("student", 400, True)):
            args, update, count = configure_args(SimpleNamespace(), {
                "kind": kind, "start_rollouts": start, "cap": 4000 if kind == "teacher" else 400,
                "checkpoint": "checkpoint", "train_seed": 1000, "target": .45,
                "parent": {"resume": "previous_optimizer_checkpoint"}})
            self.assertEqual(not args.no_load_optim, resume)
            self.assertEqual(args.use_checkpoint_opt_param_scheduler, resume)
            self.assertFalse(args.override_opt_param_scheduler)
            self.assertEqual(not args.no_save_optim, kind == "student")
            self.assertEqual(update, start // 8)
            self.assertEqual(count, 500 if kind == "teacher" else 50)
            self.assertEqual(args.early_stop_eval_names, ["gate_envs"] if kind == "student" else ["val_envs"])

    def test_student_resume_restores_scheduler_for_different_segment_ends(self):
        from async_branching.segment import worker_arguments
        for start, cap in ((0, 400), (400, 400), (800, 200), (1000, 400)):
            task = {"kind": "student", "start_rollouts": start, "cap": cap,
                    "checkpoint": "checkpoint", "train_seed": 1000, "target": .45,
                    "parent": {"resume": "previous_optimizer_checkpoint"}}
            args, update, count = configure_args(SimpleNamespace(), task)
            flags = worker_arguments(task)
            self.assertEqual(args.num_rollout, (start + cap) // 8)
            self.assertEqual(update + count, args.num_rollout)
            self.assertEqual("--use-checkpoint-opt_param-scheduler" in flags, start > 0)
            if start:
                self.assertEqual(args.load, "previous_optimizer_checkpoint")
                self.assertFalse(args.finetune)
                self.assertFalse(args.no_load_rng)

    def test_zero_teacher_target_parses_then_recovers_at_measured_baseline(self):
        from async_branching.segment import worker_arguments
        task = {"kind": "student", "start_rollouts": 1200, "cap": 400,
                "checkpoint": "checkpoint", "train_seed": 1000, "target": 0.0,
                "parent": {"resume": "previous_optimizer_checkpoint"}}
        flags = worker_arguments(task)
        parsed_target = float(flags[flags.index("--early-stop-val-accuracy") + 1])
        self.assertGreater(parsed_target, 0)
        self.assertLessEqual(parsed_target, 1)
        configured, start, _ = configure_args(SimpleNamespace(early_stop_val_accuracy=parsed_target), task)
        self.assertEqual(configured.early_stop_val_accuracy, 0.0)
        self.assertEqual(start, 150)
        self.assertFalse(configured.no_load_optim)
        # Even a measured zero baseline satisfies the true recovery target.
        self.assertGreaterEqual(0.0, configured.early_stop_val_accuracy)

    def test_invalid_student_targets_are_rejected_instead_of_clamped(self):
        from async_branching.segment import worker_arguments
        for target in (-.01, 1.01, float("nan")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                worker_arguments({"kind": "student", "start_rollouts": 1200, "cap": 400,
                                  "checkpoint": "checkpoint", "train_seed": 1000, "target": target,
                                  "parent": {"resume": "previous_optimizer_checkpoint"}})

    def test_gpu_pool_lends_student_slots_while_it_waits(self):
        from async_branching.backend import GPUPool
        pool = GPUPool(["gpu0", "gpu1", "gpu2", "gpu3"])
        pool.reserve_student(True)
        with pool.claim(student=True) as student:
            self.assertEqual(student, ["gpu0", "gpu1"])
            with pool.claim() as first, pool.claim() as second:
                self.assertEqual(first + second, ["gpu2", "gpu3"])
        pool.reserve_student(False)
        with pool.claim() as teacher:
            self.assertEqual(teacher, ["gpu0"])
        self.assertEqual(len(pool.available), 4)

    def test_ray_socket_paths_fit_unix_limit(self):
        import re
        import shutil
        import subprocess
        source = (Path(__file__).parents[2] / "scripts/slurm/async-branching-worker.sh").read_text()
        assignment = re.search(r'^ray_tmp=.*$', source, re.MULTILINE).group()
        path = subprocess.check_output(["bash", "-c", assignment + '\nprintf "%s" "$ray_tmp"'], text=True)
        try:
            socket = path + "/session_2026-10-23_17-12-07_200828_3506399/sockets/plasma_store"
            self.assertLessEqual(len(socket.encode()), 107)
        finally:
            shutil.rmtree(path)

    def test_worker_passes_checkpoint_mode_before_argument_validation(self):
        from async_branching.segment import worker_arguments
        root = Path(__file__).parents[2]
        source = (root / "scripts/slurm/async-branching-worker.sh").read_text()
        self.assertIn('worker_arguments(json.load(open(sys.argv[1])))', source)
        probe = (root / "tools/check_async_branching.py").read_text()
        self.assertIn('*worker_arguments(task)', probe)
        for kind, start in (("teacher", 0), ("student", 0), ("student", 400)):
            task = {"kind": kind, "start_rollouts": start, "cap": 4000 if kind == "teacher" else 400,
                    "checkpoint": "checkpoint", "train_seed": 1234, "target": .45,
                    "parent": {"resume": "previous_checkpoint"}}
            arguments = worker_arguments(task)
            self.assertIn('--transactional-checkpoints', arguments)
            self.assertEqual(arguments[arguments.index('--save') + 1], 'checkpoint')
            self.assertEqual(arguments[arguments.index('--save-interval') + 1], '500' if kind == 'teacher' else '50')
            self.assertEqual(arguments[arguments.index('--num-rollout') + 1],
                             str(start // 8 + task['cap'] // 8))

    def test_release_counter_does_not_offset_fresh_training_or_hide_stale_resume(self):
        from async_branching.segment import validate_loaded_updates
        validate_loaded_updates([1], 0, finetune=True)
        validate_loaded_updates([0], 0, finetune=True)
        validate_loaded_updates([50], 50, finetune=False)
        for starts, expected, finetune in (([49], 50, False), ([50], 0, True), ([1], 0, False), ([0, 1], 0, True)):
            with self.assertRaises(RuntimeError):
                validate_loaded_updates(starts, expected, finetune)

    def test_runtime_source_freeze_is_independent_of_later_repo_edits(self):
        import importlib.util
        from unittest.mock import patch
        source = Path(__file__).parents[2] / "tools/submit_async_branching.py"
        spec = importlib.util.spec_from_file_location("async_submit_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            for name in ("async_branching", "Gym", "slime", "slime_plugins", "tools", "configs", "scripts/models", "data/DATA-SCALING-V1"):
                (root / name).mkdir(parents=True, exist_ok=True)
                (root / name / "fixture.txt").write_text("original")
            for name in ("train.py", "scripts/slurm/common.sh", "scripts/slurm/port_slot.sh", "scripts/slurm/async-branching-worker.sh",
                         "scripts/slurm/async-branching-elastic-worker.sbatch", "scripts/slurm/async-branching-elastic.sbatch",
                         "scripts/slurm/async-branching-monitor.sbatch",
                         "scripts/slurm/async-branching.sbatch"):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text("original")
            target = Path(directory) / "experiment"
            target.mkdir()
            with patch.object(module, "ROOT", root):
                runtime = module.freeze_runtime(target)
            (root / "async_branching/fixture.txt").write_text("edited while job runs")
            self.assertEqual((runtime / "async_branching/fixture.txt").read_text(), "original")
            self.assertTrue((target / "runtime_sha256.txt").stat().st_size)

    def test_old_progress_is_cleared_before_a_replayed_future_is_visible(self):
        from async_branching.backend import SlimeBackend
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            task = {"id": "teacher-00-g0001"}
            d = Path(directory) / "tasks" / task["id"]
            d.mkdir(parents=True)
            (d / "progress.json").write_text('{"next_validation_rollouts":4000}')
            backend = SlimeBackend({}, directory, ["gpu0", "gpu1", "gpu2", "gpu3"])
            try:
                with patch.object(backend, "execute", return_value={"rollouts": 400, "accuracy": .99, "hf": "new"}):
                    future = backend.teacher(task)
                    self.assertFalse((d / "progress.json").exists())
                    self.assertEqual(future.result()["rollouts"], 400)
            finally:
                backend.close()

    def test_finalization_keeps_only_final_models_and_is_idempotent(self):
        from async_branching.backend import SlimeBackend
        from async_branching.schedule import Version
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("student-old", "student-new", "teacher-old", "teacher-new"):
                (root / "tasks" / name / "hf-final").mkdir(parents=True)
                (root / "tasks" / name / "ckpt").mkdir()
            (root / "parents/ref").mkdir(parents=True)
            state = initial_state(["A"], "base/hf", "base/ref")
            student = str(root / "tasks/student-new/hf-final")
            teacher = str(root / "tasks/teacher-new/hf-final")
            state.snapshots.append(Snapshot(400, 4400, student, str(root / "tasks/student-new/ckpt")))
            state.versions.extend([Version("A", 1, 4000, 4000, .5, str(root / "tasks/teacher-old/hf-final")),
                                   Version("A", 2, 8000, 4000, .6, teacher)])
            result = SlimeBackend.finalize_storage(root, state)
            self.assertEqual(result, {"A": teacher})
            self.assertTrue(Path(student).is_dir())
            self.assertTrue(Path(teacher).is_dir())
            self.assertFalse((root / "tasks/student-old/hf-final").exists())
            self.assertFalse((root / "tasks/teacher-old/hf-final").exists())
            self.assertFalse((root / "tasks/student-new/ckpt").exists())
            self.assertFalse((root / "parents").exists())
            self.assertEqual(SlimeBackend.finalize_storage(root, state), result)

    def test_full_16_32_and_64_environment_replays(self):
        for size in (16, 32, 64):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                sim = self.make([f"env{i}" for i in range(size)])
                backend = ReplayBackend()
                runner = Runner(sim, backend, directory)
                state = runner.run()
                self.assertEqual(state.student_rollouts, 16000)
                self.assertEqual(state.stopped, "student_budget")
                for event in runner.events:
                    if event["event"] == "branch":
                        task = event["task"]
                        self.assertLessEqual(task["parent"]["rollouts"], task["teacher_start_rollouts"])
                        self.assertLessEqual(task["parent"]["clock"], task["start_clock"])
                    if event["event"] == "merge_start":
                        task = event["task"]
                        version = next(v for v in state.versions if v.key == task["version_key"])
                        self.assertLessEqual(version.available_at, task["start_clock"])
                self.assertTrue(all(v.used <= 1000 for v in state.versions))
                self.assertEqual(backend.student_training_rollouts, 16000)

    def test_scaling_panels_match_original_heroes(self):
        from tools.prepare_async_branching import plan, validate, BASE
        original = json.loads((BASE / "run_plan.json").read_text())["runs"]
        for size in (32, 64):
            with self.subTest(size=size):
                prepared = plan(size, Path("/scratch/fixture"), "slurm", 16)
                validate(prepared)
                for kind in ("mopd", "rl"):
                    hero = next(r for r in original if r["label"] == f"n{size}-{kind}")
                    self.assertEqual(prepared["environments"], hero["environments"])
                    self.assertEqual(prepared["id_eval"], hero["eval"])
                    self.assertEqual(prepared["test_eval"], hero["test_eval"])
                self.assertEqual(prepared["student_rollouts"], 16000)
                self.assertEqual(prepared["teacher_branch_rollouts"], 4000)
                self.assertEqual(prepared["physical_gpus"], 16)
                self.assertEqual(prepared["teacher_project"], "rl-data-scaling-async-teachers-v1")


if __name__ == "__main__":
    unittest.main()
