"""Parallel execution must retain checkpoint identity and claim each cell once."""

import asyncio
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from lora_clustering.balanced_pipeline import confirmation_assignments
from lora_clustering.parallel import (
    assign_teachers,
    checkpoint_inventory,
    claim,
    confirmation_tasks,
    evaluate_shared,
)
from lora_teachers.plan import sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ParallelTest(unittest.TestCase):
    def test_node_exclusions_change_only_pending_owned_jobs(self):
        import monitor_parallel_mopd_clustering as monitor

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write(directory / "node-exclusions.json", ["bad-node"])
            with (
                patch.object(
                    monitor.subprocess,
                    "check_output",
                    return_value="JobState=PENDING ExcNodeList=old-node",
                ) as query,
                patch.object(
                    monitor.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ) as update,
            ):
                monitor.apply_node_exclusions(directory, "123", {"state": "RUNNING"})
                query.assert_not_called()
                update.assert_not_called()
                monitor.apply_node_exclusions(directory, "123", {"state": "PENDING"})
                self.assertEqual(
                    update.call_args.args[0],
                    [
                        "scontrol",
                        "update",
                        "JobId=123",
                        "ExcNodeList=bad-node,old-node",
                    ],
                )

    def test_node_exclusion_race_keeps_the_running_allocation(self):
        import monitor_parallel_mopd_clustering as monitor

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write(directory / "node-exclusions.json", ["bad-node"])
            with (
                patch.object(
                    monitor.subprocess,
                    "check_output",
                    return_value="JobState=PENDING ExcNodeList=(null)",
                ),
                patch.object(
                    monitor.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=1, stderr="Job is no longer pending execution"
                    ),
                ) as update,
            ):
                monitor.apply_node_exclusions(directory, "123", {"state": "PENDING"})
                update.assert_called_once()

    def test_full_tmp_guard_defers_slurm_mutations_to_watchdog(self):
        import scale_mopd_clustering as scale

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write(directory / "backend.json", {"revision": str(directory)})
            with (
                patch.object(scale, "verify_inputs"),
                patch.object(
                    scale.shutil, "disk_usage", return_value=SimpleNamespace(free=0)
                ),
                patch.object(
                    scale.os, "uname", return_value=SimpleNamespace(nodename="bad-node")
                ),
                patch.object(scale.subprocess, "run") as mutation,
            ):
                with self.assertRaisesRegex(RuntimeError, "No space left.* /tmp guard"):
                    scale.worker(SimpleNamespace(experiment=directory))
                mutation.assert_not_called()
            self.assertIn(
                "bad-node", json.loads((directory / "full-tmp-nodes.json").read_text())
            )

    def test_small_gpu_admission_is_explicit_and_default_stays_strict(self):
        from submit_data_scaling_lora import gpu_environment

        rows = [f"{i}, GPU-example-{i}, 24576" for i in range(4)]
        with self.assertRaises(RuntimeError):
            gpu_environment(rows)
        result = gpu_environment(rows, minimum_gpu_mib=23000)
        self.assertEqual(result["APPTAINERENV_CUDA_VISIBLE_DEVICES"], "0,1,2,3")

    def test_hardware_replacement_never_cancels_a_running_shard(self):
        import replace_parallel_mopd_backend as replacement

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(
                root / "job.json",
                {"mode": "parallel", "parallel_manifest": str(root / "execution.json")},
            )
            write(
                root / "execution.json", {"backends": [{"index": 1, "job_id": "123"}]}
            )
            with (
                patch.object(
                    replacement.subprocess, "check_output", return_value="RUNNING\n"
                ),
                patch.object(replacement.subprocess, "run") as mutation,
            ):
                with self.assertRaisesRegex(ValueError, "Only a queued shard"):
                    replacement.replace(SimpleNamespace(experiment=root, index=1))
                mutation.assert_not_called()

    def test_claim_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "claim.lock"
            with claim(path) as first:
                self.assertTrue(first)
                with claim(path) as second:
                    self.assertFalse(second)
            with claim(path) as third:
                self.assertTrue(third)

    def test_assignments_cover_only_pending_teachers_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {
                "teachers": [{"source_label": f"teacher-{i:02d}"} for i in range(32)],
                "teacher_rollouts": 8000,
            }
            for i in range(32):
                directory = root / "teacher" / f"teacher-{i:02d}"
                if i < 3:
                    write(directory / "completed.json", {})
                else:
                    write(
                        directory / "latest.json",
                        {"hf": str(directory / "checkpoints/u0150/adapter")},
                    )
            groups = assign_teachers(root, plan, 6)
            self.assertEqual(sorted(i for g in groups for i in g), list(range(3, 32)))
            self.assertEqual(sorted(map(len, groups)), [4, 5, 5, 5, 5, 5])

    def test_confirmation_queue_deduplicates_cells(self):
        plan = {"environments": list("abcdefgh")}
        spec = confirmation_assignments([[0, 1], [2, 3], [4, 5], [6, 7]], 1234)
        tasks = confirmation_tasks(plan, spec)
        keys = [(t["label"], t["environment"]) for t in tasks]
        self.assertEqual(len(keys), len(set(keys)))
        for control in spec["controls"].values():
            for target, members in enumerate(control):
                self.assertTrue(
                    any(
                        t["environment"] == plan["environments"][target]
                        and t["members"] == members
                        for t in tasks
                    )
                )

    def test_recovery_inventory_requires_optimizer_and_client_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "teacher/teacher-00"
            bundle = directory / "checkpoints/u0150"
            for file in (
                "adapter/adapter_model.safetensors",
                "adapter/META.json",
                "state/metadata.json",
                "client-state.pkl",
                "reference.json",
            ):
                write(bundle / file, {})
            write(
                directory / "latest.json",
                {"hf": str(bundle / "adapter"), "resume": str(bundle / "state")},
            )
            plan = {"teachers": [{"source_label": "teacher-00"}]}
            self.assertEqual(
                checkpoint_inventory(root, plan)[0]["checkpoint_rollouts"], 1200
            )
            (bundle / "client-state.pkl").unlink()
            with self.assertRaises(FileNotFoundError):
                checkpoint_inventory(root, plan)

    def test_preparation_preserves_logical_plan_and_tenant(self):
        import scale_mopd_clustering as scale

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logical-run"
            (root / "source/rlve/tools/lora_clustering").mkdir(parents=True)
            plan = {
                "teachers": [{"source_label": f"teacher-{i:02d}"} for i in range(4)],
                "teacher_rollouts": 8000,
                "adapter_slots": 33,
            }
            write(root / "plan.json", plan)
            before = (root / "plan.json").read_bytes()
            write(root / "job.json", {"job_id": "123"})
            with (
                patch.object(scale, "verify_inputs"),
                patch.object(scale, "checkpoint_inventory", return_value=[]),
            ):
                revision, manifest = scale.prepare(root, 2)
            self.assertEqual((root / "plan.json").read_bytes(), before)
            for backend in manifest["backends"]:
                directory = Path(backend["backend"])
                self.assertEqual(directory.name, root.name)
                self.assertEqual((directory / "source").resolve(), revision / "source")
                self.assertEqual(
                    json.loads((directory / "plan.json").read_text())["adapter_slots"],
                    3,
                )

    def test_monitor_recovers_only_the_failed_owned_backend(self):
        import monitor_parallel_mopd_clustering as monitor

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "total_gpus": 8,
                "backends": [
                    {
                        "index": i,
                        "job_id": str(i),
                        "backend": str(root / str(i)),
                        "indices": [i],
                    }
                    for i in (1, 2)
                ],
            }
            health = {"active_teachers": 2, "errors": [], "completed_teachers": 0}
            with (
                patch.object(monitor, "inspect", return_value=health),
                patch.object(
                    monitor,
                    "slurm",
                    side_effect=[
                        {"state": "NODE_FAIL", "node": "a"},
                        {"state": "RUNNING", "node": "b"},
                    ],
                ),
                patch.object(monitor, "recover", return_value=True) as recover,
                patch.object(monitor, "validate_completion", return_value=False),
            ):
                actual = monitor.check(root, manifest, repair=True)
            recover.assert_called_once()
            self.assertEqual(recover.call_args.args[0], root / "1")
            self.assertEqual(actual["allocated_gpus"], 4)
            self.assertEqual(actual["status"], "healthy")

    def test_monitor_recognizes_frozen_worker_tmp_guard_failure(self):
        import monitor_parallel_mopd_clustering as monitor

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = root / "backend"
            write(backend / "full-tmp-nodes.json", ["full-node"])
            (backend / "slurm-123.out").write_text(
                "Job is no longer pending execution; ExcNodeList=full-node"
            )
            manifest = {
                "total_gpus": 4,
                "backends": [
                    {
                        "index": 0,
                        "job_id": "123",
                        "backend": str(backend),
                        "indices": [0],
                    }
                ],
            }
            with (
                patch.object(
                    monitor,
                    "inspect",
                    return_value={"active_teachers": 1, "errors": []},
                ),
                patch.object(
                    monitor,
                    "slurm",
                    return_value={"state": "FAILED", "node": "full-node"},
                ),
                patch.object(monitor, "recover", return_value=True) as recover,
                patch.object(monitor, "validate_completion", return_value=False),
            ):
                monitor.check(root, manifest, repair=True)
            local = recover.call_args.args[1]
            self.assertIn("No space left on device", local["errors"][0]["traceback"])

    def test_two_evaluation_workers_finish_without_duplicate_measurements(self):
        try:
            import torch  # noqa: F401
            from lora_clustering.evaluation import Evaluator  # noqa: F401
            from lora_clustering.plan import build_clustering_plan
        except ImportError as error:
            raise unittest.SkipTest(f"Runtime smoke check needs pinned image: {error}")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "plots"
            plan = build_clustering_plan(size=64)
            plan.update(
                size=4,
                environments=plan["environments"][:4],
                teachers=plan["teachers"][:4],
                plot_output=str(output),
            )
            plan["clustering"]["balanced_restarts"] = 2
            rows = {}
            for split in ("selection", "confirmation"):
                original = json.loads((ROOT / plan["clustering"][split]).read_text())
                chosen = [
                    r
                    for r in original
                    if json.loads(r["metadata"])["environment"] in plan["environments"]
                ]
                rows[split] = {
                    e: [
                        r
                        for r in chosen
                        if json.loads(r["metadata"])["environment"] == e
                    ]
                    for e in plan["environments"]
                }
                path = root / "source/rlve" / plan["clustering"][split]
                write(path, chosen)
                plan["clustering"][f"{split}_sha256"] = sha256(path)
            write(root / "plan.json", plan)
            for teacher in plan["teachers"]:
                write(
                    root / "teacher" / teacher["source_label"] / "completed.json",
                    {
                        "hf": "dummy",
                        "environment": teacher["environment"],
                        "accuracy": 0.5,
                        "training_rollouts": 8000,
                        "stop_reason": "budget",
                        "wandb_url": "test",
                    },
                )
            calls = Counter()
            sleep = asyncio.sleep

            class FakeEvaluator:
                def __init__(self, *args):
                    pass

                def cached(self, split, label, environment):
                    path = (
                        root
                        / "clustering/evaluations"
                        / split
                        / label
                        / f"{environment}.json"
                    )
                    return json.loads(path.read_text()) if path.exists() else None

                async def sampler(self, label, **kwargs):
                    return label

                async def cell(self, split, label, environment, sampler):
                    cached = self.cached(split, label, environment)
                    if cached is not None:
                        return cached
                    await sleep(0.001)
                    calls[(split, label, environment)] += 1
                    target = plan["environments"].index(environment)
                    source = (
                        int(label.removeprefix("individual-"))
                        if label.startswith("individual-")
                        else -10
                    )
                    correct = int(source // 2 == target // 2)
                    total = plan["clustering"][f"{split}_samples_per_prompt"]
                    cell = {
                        "eval_sha256": plan["clustering"][f"{split}_sha256"],
                        "accuracy": correct / total,
                        "observations": [
                            {
                                "prompt_id": hashlib.sha256(
                                    r["user_prompt"].encode()
                                ).hexdigest(),
                                "correct": correct,
                                "total": total,
                            }
                            for r in rows[split][environment]
                        ],
                    }
                    write(
                        root
                        / "clustering/evaluations"
                        / split
                        / label
                        / f"{environment}.json",
                        cell,
                    )
                    return cell

                async def vector(self, split, label, **kwargs):
                    return [
                        (await self.cell(split, label, e, label))["accuracy"]
                        for e in plan["environments"]
                    ]

            async def fast_sleep(_):
                await sleep(0.001)

            async def work():
                backends = [root / f"backend-{i}" for i in range(2)]
                for backend in backends:
                    backend.mkdir()
                await asyncio.gather(
                    *(
                        evaluate_shared(root, backend, plan, "base", "fake")
                        for backend in backends
                    )
                )

            with (
                patch("lora_clustering.evaluation.Evaluator", FakeEvaluator),
                patch(
                    "lora_clustering.pipeline.cluster_adapter",
                    return_value=root / "dummy",
                ),
                patch("lora_clustering.parallel.asyncio.sleep", side_effect=fast_sleep),
                patch.dict("os.environ", {"LORA_CHECKPOINT_ROOT": str(root / "local")}),
            ):
                asyncio.run(work())
            self.assertTrue((root / "completed.json").is_file())
            self.assertEqual(set(calls.values()), {1})
            self.assertTrue((output / "confirmation.json").is_file())


if __name__ == "__main__":
    unittest.main()
