"""Check signed transfer, real delta composition, panel identity and eval isolation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from lora_clustering.algorithm import (
    partition_metrics,
    random_partition,
    signed_hierarchy,
)
from lora_clustering.plan import build_clustering_plan, validate_evaluation


class GraphTest(unittest.TestCase):
    def test_recovers_positive_blocks_and_separates_negative_edges(self):
        transfer = np.full((6, 6), -0.2)
        transfer[:3, :3] = 0.3
        transfer[3:, 3:] = 0.3
        result = signed_hierarchy(transfer)
        self.assertEqual(result["automatic"], [[0, 1, 2], [3, 4, 5]])

    def test_self_improvement_does_not_create_cross_transfer(self):
        result = signed_hierarchy(np.eye(4))
        self.assertEqual(len(result["automatic"]), 4)

    def test_all_positive_can_legitimately_produce_one_cluster(self):
        self.assertEqual(signed_hierarchy(np.ones((4, 4)))["automatic"], [[0, 1, 2, 3]])

    def test_invalid_transfer_is_rejected(self):
        for matrix in [np.ones((2, 3)), [[0, float("nan")], [0, 0]]]:
            with self.assertRaises(ValueError):
                signed_hierarchy(matrix)

    def test_metrics_use_within_and_between_signs(self):
        result = partition_metrics(
            [[0, 1], [2]], [[0.7, 0.8, 0.2], [0.1, 0.2, 0.9]], [0.4, 0.4, 0.4]
        )
        self.assertAlmostEqual(result["assigned_accuracy"], 0.8)
        self.assertGreater(result["within_transfer"], 0)
        self.assertLess(result["between_transfer"], 0)

    def test_random_control_preserves_cluster_sizes(self):
        groups = [[0, 1, 2], [3, 4], [5]]
        random = random_partition(groups, 1234)
        self.assertEqual(sorted(map(len, random)), [1, 2, 3])
        self.assertEqual(sorted(i for g in random for i in g), list(range(6)))
        self.assertEqual(random, random_partition(groups, 1234))


class RecipeTest(unittest.TestCase):
    def test_sdk_installed_into_a_new_pythonpath_directory_is_discovered(self):
        import importlib.util

        import submit_mopd_clustering

        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            runtime = experiment / "runtime"
            sys.path.insert(0, str(runtime))
            self.assertIsNone(importlib.util.find_spec("clustering_bootstrap_sentinel"))

            def install(*args, **kwargs):
                runtime.mkdir()
                (runtime / "clustering_bootstrap_sentinel.py").write_text(
                    "ready = True\n"
                )

            def run(*args, **kwargs):
                self.assertIsNotNone(
                    importlib.util.find_spec("clustering_bootstrap_sentinel")
                )

            try:
                with (
                    patch.object(
                        submit_mopd_clustering.subprocess, "run", side_effect=install
                    ),
                    patch.dict(
                        sys.modules,
                        {
                            "lora_teachers.runner": SimpleNamespace(run=run),
                            "lora_clustering.pipeline": SimpleNamespace(
                                run_pipeline=object()
                            ),
                        },
                    ),
                ):
                    submit_mopd_clustering.inside(
                        SimpleNamespace(experiment=experiment, model_path="base")
                    )
            finally:
                sys.path.remove(str(runtime))
                sys.path_importer_cache.pop(str(runtime), None)

    def test_original_panel_and_only_requested_recipe_changes(self):
        import yaml

        plan = build_clustering_plan()
        runs = json.loads((ROOT / "data/DATA-SCALING-V1/run_plan.json").read_text())[
            "runs"
        ]
        panel = next(r for r in runs if r["label"] == "n32-mopd")
        self.assertEqual(plan["environments"], panel["environments"])
        original = yaml.safe_load(
            (ROOT / "configs/1gpu-data-scaling-teacher-v1.yaml").read_text()
        )
        expected = dict(original, lr=2e-5, **{"num-rollout": 1000})
        self.assertEqual(plan["teacher_config"], expected)
        self.assertEqual(plan["project"], "rl-data-scaling-lora-teachers-v3")
        self.assertEqual(
            plan["teacher_rollouts"], 1000 * original["n-samples-per-prompt"]
        )
        self.assertEqual(
            plan["teacher_validation_threshold"], original["early-stop-val-accuracy"]
        )
        self.assertEqual(len(plan["teachers"]), 32)

    def test_fresh_eval_sets_are_disjoint_from_gates_and_each_other(self):
        plan = validate_evaluation(build_clustering_plan())
        self.assertEqual(len(plan["clustering"]["selection_sha256"]), 64)
        self.assertEqual(len(plan["clustering"]["confirmation_sha256"]), 64)


class MergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Run delta-composition checks in the pinned image")
        torch.set_num_threads(1)
        cls.torch = torch
        from lora_clustering.merge import mean_factors

        cls.merge = staticmethod(mean_factors)

    def test_exact_delta_mean_without_rank_loss(self):
        t = self.torch
        t.manual_seed(12)
        factors = [(t.randn(2, 7), t.randn(6, 2)) for _ in range(3)]
        a, b, energy = self.merge(factors, [2, 3, 4], rank=6, output_scale=2)
        expected = sum(s * b @ a for (a, b), s in zip(factors, [2, 3, 4])) / 3
        t.testing.assert_close(2 * b @ a, expected, atol=2e-5, rtol=2e-5)
        self.assertAlmostEqual(energy, 1, places=5)

    def test_invariant_to_factor_reparameterization(self):
        t = self.torch
        t.manual_seed(23)
        a, b = t.randn(2, 9), t.randn(8, 2)
        a1, b1, _ = self.merge(
            [(a, b), (a * 10, b / 10)], [2, 2], rank=2, output_scale=2
        )
        t.testing.assert_close(b1 @ a1, b @ a, atol=1e-5, rtol=1e-5)

    def test_rank_truncation_matches_best_svd(self):
        t = self.torch
        t.manual_seed(34)
        factors = [(t.randn(3, 8), t.randn(7, 3)) for _ in range(2)]
        a, b, energy = self.merge(factors, [2, 2], rank=2, output_scale=2)
        target = sum(b @ a for a, b in factors)
        u, s, vh = t.linalg.svd(target, full_matrices=False)
        t.testing.assert_close(
            2 * b @ a, (u[:, :2] * s[:2]) @ vh[:2], atol=2e-5, rtol=2e-5
        )
        self.assertGreater(energy, 0)
        self.assertLess(energy, 1)

    def test_averaging_factors_would_give_wrong_result(self):
        t = self.torch
        a, b = t.ones(1, 2), t.ones(3, 1)
        merged_a, merged_b, _ = self.merge(
            [(a, b), (-a, -b)], [1, 1], rank=1, output_scale=1
        )
        t.testing.assert_close(merged_b @ merged_a, b @ a)
        self.assertGreater(float((merged_b @ merged_a).abs().sum()), 0)


class TeacherLoopTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        try:
            import torch
            from lora_clustering.teacher import Teacher
        except ImportError as error:
            raise unittest.SkipTest(
                f"Teacher runtime checks require the pinned image and Tinker SDK: {error}"
            )
        self.torch, self.Teacher = torch, Teacher
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def worker(self, accuracies, *, cap=16):
        worker = object.__new__(self.Teacher)
        worker.index, worker.experiment, worker.directory = (
            0,
            self.root,
            self.root / "workers",
        )
        worker.directory.mkdir(exist_ok=True)
        worker.root = self.root
        worker.training, worker.service = object(), object()
        worker.plan = {
            "teachers": [{"environment": "Example", "source_label": "teacher-00"}],
            "project": "test",
            "seed": 1234,
            "teacher_config": {"eval-interval": 1},
            "teacher_rollouts": cap,
            "teacher_validation_threshold": 0.98,
        }
        worker.open_run = Mock(
            return_value=SimpleNamespace(url="example", finish=Mock())
        )
        worker.source = Mock(
            return_value=SimpleNamespace(
                set_state=Mock(), get_state=Mock(return_value={})
            )
        )
        worker.update, worker.load = AsyncMock(return_value={}), AsyncMock()
        worker.gate, worker.checkpoint = AsyncMock(side_effect=accuracies), AsyncMock()
        return worker

    async def exercise(self, worker, resume=None):
        from async_branching.io import write_json

        if resume:
            directory = self.root / "teacher/teacher-00"
            directory.mkdir(parents=True)
            bundle = directory / "checkpoints/u0001"
            (bundle / "adapter").mkdir(parents=True)
            (bundle / "state").mkdir()
            write_json(
                directory / "latest.json",
                {"hf": str(bundle / "adapter"), "resume": str(bundle / "state")},
            )
        with (
            patch(
                "lora_clustering.teacher.export",
                new=AsyncMock(
                    return_value=(object(), "tinker://test/sampler_weights/1")
                ),
            ),
            patch(
                "lora_clustering.teacher.save_bundle",
                new=AsyncMock(return_value={"hf": "final", "resume": None}),
            ),
            patch("lora_clustering.teacher.log_metrics"),
            patch("lora_clustering.teacher.client_state", return_value=resume),
        ):
            await worker.train()
        return json.loads((self.root / "teacher/teacher-00/completed.json").read_text())

    async def test_initial_high_accuracy_still_trains_like_original_teachers(self):
        worker = self.worker([0.99, 0.99])
        result = await self.exercise(worker)
        self.assertEqual(result["training_rollouts"], 8)
        self.assertEqual(result["stop_reason"], "validation_accuracy")
        self.assertEqual(worker.update.await_count, 1)

    async def test_stops_at_rollout_cap(self):
        worker = self.worker([0.1, 0.2, 0.3])
        result = await self.exercise(worker)
        self.assertEqual(result["training_rollouts"], 16)
        self.assertEqual(result["stop_reason"], "budget")
        self.assertEqual(worker.update.await_count, 2)

    async def test_resume_restores_optimizer_and_source_before_continuing(self):
        import random

        worker = self.worker([0.3])
        saved = {
            "used": 8,
            "accuracy": 0.2,
            "source": {"prompt_cursor": 1},
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": self.torch.get_rng_state(),
        }
        result = await self.exercise(worker, saved)
        self.assertEqual(worker.update.await_count, 1)
        self.assertEqual(result["training_rollouts"], 16)
        self.assertTrue(worker.load.await_args.kwargs["optimizer"])
        worker.source.return_value.set_state.assert_called_once_with(saved["source"])

    async def test_base_sampling_client_is_given_an_explicit_model(self):
        from lora_clustering.evaluation import Evaluator

        evaluator = object.__new__(Evaluator)
        evaluator.plan = {"base_model": "example-base"}
        evaluator.service = SimpleNamespace(create_sampling_client_async=AsyncMock())
        await evaluator.sampler("base")
        evaluator.service.create_sampling_client_async.assert_awaited_once_with(
            model_path=None, base_model="example-base"
        )


if __name__ == "__main__":
    unittest.main()
