"""Keep faster inference submission separate from training and causal changes."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from async_lora.plan import build_async_plan, experiment_stem
from async_lora.worker import Worker
from lora_teachers.client import evaluate
from lora_teachers.plan import gateway_argv


class AsyncLoRASpeedTest(unittest.TestCase):
    def test_v2_uses_one_trainer_three_samplers_and_only_reduces_teacher_eval(self):
        for single_question, replay in ((False, 0), (True, 30)):
            original = build_async_plan(single_question=single_question, replay_percent=replay)
            plan = build_async_plan(single_question=single_question, replay_percent=replay, speed_profile="v2")
            self.assertEqual(plan["physical_gpus"], 4)
            self.assertEqual((plan["train_gpus"], plan["sampling_gpus"]), (1, 3))
            self.assertEqual(plan["evaluation_concurrency"], 8)
            self.assertEqual(plan["student_config"], original["student_config"])
            self.assertEqual(plan["eval_interval_updates"], 50)
            expected = original["teacher_config"] | {"eval-interval": 100, "n-samples-per-eval-prompt": 4}
            self.assertEqual(plan["teacher_config"], expected)
            for key in ("student_rollouts", "student_quantum_rollouts", "teacher_branch_rollouts", "teacher_eval"):
                self.assertEqual(plan[key], original[key])
            self.assertEqual(plan.get("opd_replay"), original.get("opd_replay"))
            self.assertEqual(plan.get("single_question"), original.get("single_question"))
            self.assertTrue(experiment_stem(plan).endswith("-speed-v2"))
            argv = gateway_argv(plan, model_path="model", checkpoint_root="root", port=6000, router_port=7000)
            for flag, value in {
                "actor-num-gpus-per-node": 1,
                "rollout-num-gpus": 3,
                "num-gpus-per-node": 4,
                "multi-lora-n-adapters": 17,
                "global-batch-size": 8,
                "sglang-max-running-requests": 64,
                "sglang-cuda-graph-max-bs-decode": 64,
            }.items():
                self.assertEqual(argv[argv.index(f"--{flag}") + 1], str(value))

    def test_more_inflight_eval_prompts_preserve_samples_and_metric_weighting(self):
        async def measure(concurrency):
            active, peak, sampled = 0, 0, []

            async def sample(_sampler, _tokenizer, tokens, environment, config, settings, *, evaluation):
                nonlocal active, peak
                self.assertTrue(evaluation)
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0)
                active -= 1
                sampled.append((tokens[0], environment, config))
                return [
                    SimpleNamespace(accuracy=(tokens[0] + i) % 2, forced=False, truncated=False)
                    for i in range(settings["n-samples-per-eval-prompt"])
                ]

            rows = [{"user_prompt": str(i), "metadata": {"environment": "A", "config": {"i": i}}} for i in range(32)]
            with (
                patch("lora_teachers.client.prompt_tokens", side_effect=lambda _, prompt: [int(prompt)]),
                patch("lora_teachers.client.sample_group", side_effect=sample),
            ):
                metrics = await evaluate(None, None, rows, {"n-samples-per-eval-prompt": 4}, concurrency=concurrency)
            return metrics, peak, sorted(sampled)

        old, old_peak, old_samples = asyncio.run(measure(2))
        new, new_peak, new_samples = asyncio.run(measure(8))
        self.assertEqual((old_peak, new_peak), (2, 8))
        self.assertEqual(old_samples, new_samples)
        self.assertEqual(old, new)
        self.assertEqual(new["david/eval/val_envs/pass@1"], 0.5)

    def test_worker_submits_configured_concurrency_for_both_roles(self):
        for role in ("teacher", "student"):
            worker = Worker.__new__(Worker)
            worker.role, worker.sampler, worker.tokenizer = role, object(), object()
            worker.plan = {"evaluation_concurrency": 8}
            rows, settings = [object()], {"n-samples-per-eval-prompt": 4 if role == "teacher" else 8}
            with patch("async_lora.worker.evaluate", new=AsyncMock(return_value={})) as evaluator:
                asyncio.run(worker.evaluate_rows(rows, settings))
            evaluator.assert_awaited_once_with(worker.sampler, worker.tokenizer, rows, settings, concurrency=8)


if __name__ == "__main__":
    unittest.main()
