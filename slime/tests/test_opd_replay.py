import copy
import ast
import argparse
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from slime.utils.opd_replay import OPDReplayBuffer, replay_count
from async_branching.backend import SlimeBackend
from async_branching.schedule import Simulation, Snapshot, State, initial_state
from async_branching.segment import configure_args, worker_arguments
from tools.prepare_async_branching import plan, validate


def trace(environment="A", index=0):
    return {"tokens": [10, 11, 12, index], "response_length": 2,
            "rollout_log_probs": [-1., -1.], "teacher_log_probs": [-.5, -2.],
            "loss_mask": [0, 1], "metadata": {"environment": environment},
            "index": index, "reward": {"reward": 1., "accuracy": 1.},
            "status": "completed", "opd_replay": False}


class OPDReplayTest(unittest.TestCase):
    def test_exact_fraction_over_small_batches_and_segment_boundaries(self):
        for fraction in (.1, .3, .5):
            counts = [replay_count(step, 8, fraction) for step in range(2000)]
            self.assertEqual(sum(counts), round(16000*fraction))
            self.assertTrue(all(0 <= n <= 4 for n in counts))
            for start in (0, 25, 50, 125):
                self.assertEqual(sum(replay_count(s, 8, fraction) for s in range(start, start+50)),
                                 round(400*fraction))

    def test_newest_bounded_fresh_traces_and_masks_survive(self):
        buffer = OPDReplayBuffer(2)
        buffer.add([trace(index=i) for i in range(4)])
        self.assertEqual([r["index"] for r in buffer.entries["A"]], [2, 3])
        replayed = buffer.sample(2, seed=3)
        buffer.add(replayed)
        self.assertEqual(buffer.size, 2)
        self.assertTrue(all(t["opd_replay"] and t["loss_mask"] == [0, 1] for t in replayed))
        replayed[0]["tokens"][0] = 900
        self.assertTrue(all(t["tokens"][0] == 10 for t in buffer.entries["A"]))

    def test_other_environments_are_balanced_independent_of_buffer_size(self):
        buffer = OPDReplayBuffer(400)
        buffer.add([trace("A", i) for i in range(30)] + [trace("B")] + [trace("C")])
        values = buffer.sample(19, excluded_environments=["C"], seed=123)
        envs = [r["metadata"]["environment"] for r in values]
        self.assertNotIn("C", envs)
        self.assertLessEqual(abs(envs.count("A") - envs.count("B")), 1)
        more = buffer.sample(1, excluded_environments=["C"], seed=123)
        self.assertEqual(envs.count(more[0]["metadata"]["environment"]), 9)

    def test_warmup_is_fresh_when_only_current_environment_is_buffered(self):
        buffer = OPDReplayBuffer()
        self.assertEqual(buffer.sample(4, ["A"]), [])
        buffer.add([trace("A")])
        self.assertEqual(buffer.sample(4, ["A"]), [])
        self.assertEqual(buffer.cursor, 0)

    def test_snapshot_resume_is_exact_and_failed_work_does_not_change_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)/"parent.json.gz"
            buffer = OPDReplayBuffer(2)
            buffer.add([trace("A"), trace("B")])
            buffer.sample(1)
            buffer.save(parent)
            original = parent.read_bytes()
            resumed = OPDReplayBuffer.load(parent, 2)
            self.assertEqual(buffer.sample(4, seed=2), resumed.sample(4, seed=2))
            resumed.add([trace("C")])  # Simulate unfinished segment work.
            resumed.save(Path(directory)/"orphan.json.gz")
            retry = OPDReplayBuffer.load(parent, 2)
            self.assertNotIn("C", retry.entries)
            self.assertEqual(parent.read_bytes(), original)
            self.assertFalse(parent.with_name(parent.name+".tmp").exists())

    def test_malformed_or_wrong_capacity_snapshot_fails(self):
        for key, value in [("teacher_log_probs", [0.]), ("rollout_log_probs", [float("nan"), 0.]),
                           ("loss_mask", [1])]:
            bad = trace(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                OPDReplayBuffer().add([bad])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"buffer.gz"
            OPDReplayBuffer(2).save(path)
            with self.assertRaises(ValueError):
                OPDReplayBuffer.load(path, 3)

    def test_all_prepared_panels_have_replay_presets(self):
        for size in (16, 32, 64):
            for percent in (10, 30, 50):
                value = plan(size, Path("/scratch/test"), "slurm", 16, percent)
                validate(value)
                self.assertEqual(value["opd_replay"]["fraction"], percent/100)
        self.assertNotIn("opd_replay", plan(16, Path("/scratch/test")))

    def test_replay_worker_jobs_are_distinguishable_from_the_baseline(self):
        from async_branching.slurm import SlurmBackend
        with tempfile.TemporaryDirectory() as directory:
            config = plan(16, Path("/scratch/test"), "slurm", 16, 30)
            state = initial_state(config["environments"], "base", "ref")
            backend = SlurmBackend(config, directory, state)
            self.assertIn("--job-name=async-n16-replay30-student", backend.command("student-service"))
            self.assertIn("--job-name=async-n16-replay30-teacher-00-g0001",
                          backend.command("teacher-00-g0001", "/task.json"))

    def test_task_continuation_carries_only_committed_replay_snapshot(self):
        simulation = Simulation(initial_state(["A"], "base", "base-ref"))
        branch = simulation.branches()[0]
        simulation.materialize(branch["id"], {"rollouts": 4000, "accuracy": .5, "hf": "teacher"})
        simulation.state.clock = 4000
        simulation.publish()
        task = simulation.student_task()
        simulation.merge(task, {"rollouts": 400, "accuracy": .1, "hf": "student",
                                "resume": "opt", "replay": "committed-buffer.gz"})
        restored = State.from_dict(json.loads(json.dumps(simulation.state.to_dict())))
        self.assertEqual(restored.snapshots[-1].replay, "committed-buffer.gz")
        task = simulation.student_task()
        task.update(train_seed=123, checkpoint="next", opd_replay={"fraction": .3,
                    "capacity_per_env": 400, "clip_epsilon": .2})
        args, _, _ = configure_args(SimpleNamespace(), task)
        self.assertEqual(args.opd_replay_load, "committed-buffer.gz")
        self.assertIn("committed-buffer.gz", worker_arguments(task))
        teacher = copy.deepcopy(task); teacher["kind"] = "teacher"
        args, _, _ = configure_args(SimpleNamespace(), teacher)
        self.assertEqual(args.opd_replay_fraction, 0)
        self.assertIsNone(args.opd_replay_load)

    def test_collection_keeps_latest_replay_and_discards_old_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("old", "new"):
                d = root/"tasks"/name
                (d/"hf-final").mkdir(parents=True)
                (d/"ckpt").mkdir()
                (d/"opd-replay.json.gz").write_bytes(b"buffer")
                (d/"result.json").write_text(json.dumps({"hf": str(d/"hf-final"), "rollouts": 400}))
            latest = root/"tasks/new"
            state = initial_state(["A"], "base", "ref")
            state.snapshots.append(Snapshot(400, 4400, str(latest/"hf-final"),
                                            str(latest/"ckpt"), str(latest/"opd-replay.json.gz")))
            backend = SlimeBackend.__new__(SlimeBackend)
            backend.directory = root; backend.plan = {"base_hf": "base"}
            backend.collect(state)
            self.assertTrue((latest/"opd-replay.json.gz").exists())
            self.assertFalse((root/"tasks/old/opd-replay.json.gz").exists())

    def test_real_generation_loop_requests_only_the_fresh_slots(self):
        source = Path(__file__).parents[1]/"rollout/sglang_rollout.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                        and n.name == "generate_rollout_async")
        fetched = []
        class Progress:
            def __init__(self, **kwargs): pass
            def update(self, n): pass
            def close(self): pass
        class Generation:
            def reset(self):
                self.remaining_batch_size = 0
                self.pendings = set()
            def submit_generate_tasks(self, groups):
                async def complete(group): return group
                self.pendings.update(asyncio.create_task(complete(g)) for g in groups)
                self.remaining_batch_size += len(groups)
        state = Generation(); state.reset()
        async def abort(*args): return []
        namespace = {"asyncio": asyncio, "Sample": SimpleNamespace, "GenerateState": lambda args: state,
                     "tqdm": Progress, "abort": abort}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        def data_source(count):
            fetched.append(count)
            return [[SimpleNamespace(index=i, prompt="p", response="r", label="l", reward=1.)]
                    for i in range(count)]
        args = SimpleNamespace(rollout_global_dataset=False, rlve=True, dynamic_sampling_filter_path=None,
                               rollout_batch_size=8, opd_fresh_batch_size=4, over_sampling_batch_size=8,
                               n_samples_per_prompt=1, rm_type="rlve", use_wandb=False)
        manager = SimpleNamespace(update=lambda samples: {})
        with redirect_stdout(io.StringIO()):
            data, aborted = asyncio.run(namespace[function.name](
                args, 0, data_source, data_buffer=SimpleNamespace(rlve_manager=manager)))
        self.assertEqual(fetched, [4])
        self.assertEqual(len(data), 4)
        self.assertEqual(aborted, [])

    def test_real_dp_partition_keeps_replay_flags_aligned_with_cached_scores(self):
        source = Path(__file__).parents[1]/"backends/utils/data.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "process_rollout_data")
        data = {"tokens": [[i, 99] for i in range(8)], "raw_reward": [0]*8,
                "opd_replay_mask": [[float(i % 2)] for i in range(8)],
                "teacher_log_probs": [[-float(i)] for i in range(8)],
                "rollout_log_probs": [[-1.]]*8, "response_lengths": [1]*8}
        namespace = {"dist": SimpleNamespace(get_rank=lambda: 0, broadcast_object_list=lambda *a, **k: None),
                     "ray": SimpleNamespace(get=lambda value: value), "Timer": SimpleNamespace}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        result = namespace[function.name](SimpleNamespace(balance_data=False),
                                          SimpleNamespace(inner=data), dp_rank=1, dp_size=2)
        self.assertEqual(result["opd_replay_mask"], [[1.]]*4)
        self.assertEqual(result["teacher_log_probs"], [[-1.], [-3.], [-5.], [-7.]])
        self.assertEqual(result["tokens"], [[i, 99] for i in (1, 3, 5, 7)])

    def test_worker_replay_flags_are_accepted_by_the_real_argument_provider(self):
        source = Path(__file__).parents[1]/"utils/arguments.py"
        tree = ast.parse(source.read_text())
        nodes = [next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
                 for name in ("reset_arg", "add_cluster_arguments")]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        parser = argparse.ArgumentParser()
        namespace["add_cluster_arguments"](parser)
        task = {"kind": "student", "start_rollouts": 400, "cap": 400, "checkpoint": "next",
                "target": .45, "train_seed": 123, "parent": {"resume": "old", "replay": "buffer.gz"},
                "opd_replay": {"fraction": .3, "capacity_per_env": 400, "clip_epsilon": .2}}
        parsed, other_flags = parser.parse_known_args(worker_arguments(task))
        self.assertEqual(parsed.opd_replay_fraction, .3)
        self.assertEqual(parsed.opd_replay_capacity_per_env, 400)
        self.assertEqual(parsed.opd_replay_load, "buffer.gz")
        self.assertFalse(any("opd-replay" in flag for flag in other_flags))


if __name__ == "__main__":
    unittest.main()
