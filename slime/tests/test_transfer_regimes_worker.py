"""CPU tests inside the pinned Miles image; no GPU allocation is needed."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from transfer_regimes.worker import SyncWorker


class ObjectiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_joint_policy_uses_grpo_without_changing_student_identity(self):
        worker = object.__new__(SyncWorker)
        worker.role = "student"
        worker.plan = {"algorithm": "joint-rl"}
        worker.sampler = object()
        seen = []

        async def grpo(trainer, task, source, settings, teacher_sampler, *, used):
            seen.append((trainer.role, teacher_sampler, used))
            return {"grad_norm": 1.0}

        with patch("transfer_regimes.worker.Worker.update", new=grpo):
            result = await worker.update({}, object(), {}, used=24)
        self.assertEqual(seen, [("teacher", None, 24)])
        self.assertEqual(worker.role, "student")
        self.assertEqual(result["train/generated_completions"], 8)

    async def test_mixed_opd_batch_is_scored_by_four_correct_teachers(self):
        worker = object.__new__(SyncWorker)
        worker.role = "student"
        worker.plan = {
            "algorithm": "sync-mopd",
            "optimizer_eps": 1e-8,
            "grad_clip_norm": 1.0,
        }
        worker.sampler, worker.tokenizer = object(), object()
        worker.teacher_samplers = {e: e for e in "abcd"}
        rows = [([i], e, {}, 1) for i, e in enumerate("bdacbdac")]
        worker.generate = lambda source: next(source)
        completions = [SimpleNamespace(accuracy=1, reward=1.0) for _ in rows]
        seen = []

        async def sample(sampler, tokenizer, prompt, environment, config, settings):
            return [completions[prompt[0]]]

        async def score(batch, values, teacher):
            self.assertTrue(all(row[1] == teacher for row in batch))
            seen.append(teacher)
            return [row[0][0] for row in batch]

        async def optimizer(adam):
            async def result():
                return SimpleNamespace(metrics={"grad_norm": 1.0})

            return result()

        worker.training = SimpleNamespace(optim_step_async=optimizer)
        settings = {
            "rollout-batch-size": 8,
            "lr": 2e-5,
            "adam-beta1": 0.9,
            "adam-beta2": 0.98,
            "weight-decay": 0.01,
        }
        backward = AsyncMock(return_value={"loss": 0.01})
        with (
            patch("transfer_regimes.worker.sample_group", new=sample),
            patch("transfer_regimes.worker.score_trajectories", new=score),
            patch("transfer_regimes.worker.backward_batch", new=backward),
            patch("transfer_regimes.worker.trajectory_metrics", return_value={}),
        ):
            await worker.update({}, iter(rows), settings)
        self.assertEqual(sorted(seen), list("abcd"))
        self.assertEqual(backward.call_args.args[1], list(range(8)))
        self.assertEqual(backward.call_args.args[2:], ([], 8))


if __name__ == "__main__":
    unittest.main()
