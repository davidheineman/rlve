"""Fixed-question gates, signed replay, batch normalization and durable continuation."""

import ast
import asyncio
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from async_lora.plan import build_async_plan, experiment_stem
from async_lora.questions import (
    SingleQuestionSource,
    evaluation_rows,
    load_questions,
    question_id,
)
from async_lora.replay import select_replay, trace_datums
from async_lora.storage import save_bundle, uri_directory
from async_lora.worker import Worker
from lora_teachers.plan import ROOT, sha256
from lora_teachers.sampling import Completion

from async_branching.io import read_json
from slime.utils.opd import sampled_reverse_kl_loss
from slime.utils.opd_replay import OPDReplayBuffer


def row(environment="A", value=7):
    result = {
        "user_prompt": f"Compute {value}",
        "metadata": {
            "environment": environment,
            "difficulty": 1,
            "config": {"seed": value, "parameter": {"value": value}},
        },
    }
    result["metadata"]["question_id"] = question_id(result)
    return result


def trace(environment="A", value=7):
    return {
        "tokens": [1, 2, 3, 4, 5],
        "response_length": 2,
        "rollout_log_probs": [-1.0, -1.0],
        "teacher_log_probs": [-0.5, -2.0],
        "loss_mask": [1, 0],
        "metadata": {"environment": environment, "question_id": row(environment, value)["metadata"]["question_id"]},
        "opd_replay": False,
        "reward": {"reward": 1.0, "accuracy": 1.0},
    }


class SingleQuestionTest(unittest.TestCase):
    def test_plan_preserves_randomized_panels_and_configures_replay30(self):
        before, plan = build_async_plan(16), build_async_plan(16, single_question=True, replay_percent=30)
        self.assertEqual(plan["id_eval"], before["id_eval"])
        self.assertEqual(plan["test_eval"], before["test_eval"])
        self.assertEqual(plan["physical_gpus"], 4)
        self.assertEqual(plan["adapter_slots"], 17)
        self.assertEqual(plan["opd_replay"], {"fraction": 0.3, "capacity_per_env": 400, "clip_epsilon": 0.2})
        self.assertEqual(plan["student_config"]["opd-replay-fraction"], 0.3)
        self.assertIn("fresh plus replay", plan["student_budget_unit"])
        self.assertEqual(experiment_stem(plan), "async-lora-n16-q1-replay30")
        manifest = read_json(ROOT / "data/DATA-SCALING-V1/manifest.json")
        for key in ("id_eval", "test_eval"):
            self.assertEqual(sha256(ROOT / plan[key]), manifest["sha256"][plan[key]])
        self.assertNotIn("single_question", before)
        self.assertNotIn("opd_replay", before)

    def test_fixed_questions_are_scoreable_and_disjoint_from_randomized_validation(self):
        from make_rlve_eval_set import _survives_json

        plan = build_async_plan(16, single_question=True)
        path = ROOT / plan["single_question"]["data"]
        questions = load_questions(path, plan["environments"])
        self.assertEqual(len(questions), 16)
        heldout = {r["user_prompt"] for key in ("id_eval", "test_eval") for r in read_json(ROOT / plan[key])}
        for environment, fixed in questions.items():
            original = read_json(ROOT / plan["teacher_eval"][environment])
            self.assertNotIn(fixed["user_prompt"], heldout | {r["user_prompt"] for r in original})
            self.assertTrue(_survives_json(environment, fixed["metadata"]["config"]))
        overall = evaluation_rows(questions)
        self.assertEqual(len(overall), 32)
        self.assertTrue(all(sum(r["metadata"]["environment"] == e for r in overall) == 2 for e in questions))
        self.assertEqual(len(evaluation_rows(questions, plan["environments"][0])), 32)

    def test_same_question_survives_generations_repeated_sampling_and_resume(self):
        fixed = row()
        for generation in (1, 2, 9):
            source = SingleQuestionSource(fixed)
            for _ in range(8):
                environment, difficulty, problem = source.generate_problem()
                self.assertEqual((environment, difficulty), ("A", 1))
                self.assertEqual(problem.prompt_generator(), fixed["user_prompt"])
                config = problem.get_config()
                self.assertEqual(config, fixed["metadata"]["config"])
                config["parameter"]["value"] = -100
            resumed = SingleQuestionSource(fixed)
            resumed.set_state(source.get_state())
            self.assertEqual(resumed.get_state()["draws"], 8)
            self.assertEqual(resumed.generate_problem()[2].get_config()["parameter"]["value"], 7)
            with self.assertRaises(ValueError):
                SingleQuestionSource(row(value=9)).set_state(source.get_state())

    def test_teacher_gate_and_student_recovery_use_same_question_without_relabeling_randomized_accuracy(self):
        for role in ("teacher", "student"):
            worker = Worker.__new__(Worker)
            worker.role, worker.sampler, worker.tokenizer = role, object(), object()
            worker.questions = {"A": row()}
            worker.plan = {"teacher_eval": {"A": "original.json"}, "eval_prompts": 32}
            worker.directory, worker.run = Path("/tmp/student"), object()
            worker.rows = Mock(return_value=[row(value=99)])
            task = {
                "environment": "A",
                "directory": "/tmp/task",
                "start_clock": 400,
                "generation": 1,
                "start_rollouts": 0,
                "target": 0.9,
            }
            with (
                patch(
                    "async_lora.worker.evaluate",
                    new=AsyncMock(
                        side_effect=[{"david/eval/val_envs/pass@1": 0.25}, {"david/eval/val_envs/pass@1": 0.99}]
                    ),
                ),
                patch("async_lora.worker.log_metrics") as logged,
            ):
                accuracy = asyncio.run(worker.gate(task, 0, {}))
            self.assertEqual(accuracy, 0.99)
            metrics = logged.call_args.args[-1]
            key = "david/eval/gate_envs/pass@1" if role == "student" else "david/eval/val_envs/pass@1"
            self.assertEqual(metrics[key], 0.25)
            if role == "student":
                self.assertEqual(metrics["david/eval/gate_envs/same_question_val_acc"], 0.99)
                self.assertNotIn("same_question_val_acc", metrics)
            else:
                self.assertEqual(metrics["same_question_val_acc"], 0.99)

    def test_student_overall_same_question_metric_keeps_original_val_and_test(self):
        worker = Worker.__new__(Worker)
        worker.questions = {"A": row()}
        worker.plan = {"student_config": {}, "id_eval": "id.json", "test_eval": "test.json", "eval_prompts": 32}
        worker.sampler, worker.tokenizer, worker.run, worker.directory = (
            object(),
            object(),
            object(),
            Path("/tmp/student"),
        )
        worker.rows = Mock(side_effect=[[row(value=100)], [row(value=200)]])
        with (
            patch(
                "async_lora.worker.evaluate",
                new=AsyncMock(
                    side_effect=[
                        {"david/eval/val_envs/pass@1": 0.3},
                        {"david/eval/val_envs/pass@1": 0.4},
                        {"david/eval/val_envs/pass@1": 0.9},
                    ]
                ),
            ),
            patch("async_lora.worker.log_metrics") as logged,
        ):
            asyncio.run(worker.overall_evaluation(50, 800))
        metrics = logged.call_args.args[-1]
        self.assertEqual(metrics["david/eval/val_envs/pass@1"], 0.3)
        self.assertEqual(metrics["david/eval/test_envs/pass@1"], 0.4)
        self.assertEqual(metrics["same_question_val_acc"], 0.9)


class LoRAReplayTest(unittest.TestCase):
    def test_exact_replay_fraction_excludes_current_environment_and_warms_up_fresh(self):
        buffer = OPDReplayBuffer(400)
        self.assertEqual(select_replay(buffer, "A", 0, 123, 8, 0.3), [])
        buffer.add([trace("A")])
        self.assertEqual(select_replay(buffer, "A", 1, 123, 8, 0.3), [])
        buffer.add([trace("B")])
        counts = []
        for step in range(125, 175):
            replayed = select_replay(buffer, "B", step, 123, 8, 0.3)
            counts.append(len(replayed))
            self.assertTrue(all(t["metadata"]["environment"] == "A" and t["opd_replay"] for t in replayed))
        self.assertEqual(sum(counts), 120)
        self.assertEqual(set(counts), {2, 3})

    def test_cached_replay_skips_resampling_and_teacher_scoring_and_uses_one_optimizer_step(self):
        worker = Worker.__new__(Worker)
        worker.role = "student"
        worker.plan = {
            "opd_replay": {"fraction": 0.3, "capacity_per_env": 400, "clip_epsilon": 0.2},
            "optimizer_eps": 1e-8,
            "grad_clip_norm": 1.0,
        }
        worker.replay = OPDReplayBuffer(400)
        worker.replay.add([trace("A")])
        worker.questions = {"B": row("B")}
        worker.tokenizer, worker.sampler = object(), object()
        worker.generate = Mock(return_value=([1, 2, 3], "B", {}, 1))

        async def result():
            return SimpleNamespace(metrics={"grad_norm": 0.2})

        worker.training = SimpleNamespace(
            forward_backward_async=AsyncMock(side_effect=lambda *a, **k: result()),
            optim_step_async=AsyncMock(side_effect=lambda *a, **k: result()),
        )
        sampler = SimpleNamespace(compute_logprobs_async=AsyncMock(return_value=[None, -9.0, -8.0, -0.5, -1.5]))
        completion = Completion([4, 5], [-1.0, -1.0], [1, 0], 1.0, 1, False, False)
        settings = {"rollout-batch-size": 8, "lr": 2e-5, "adam-beta1": 0.9, "adam-beta2": 0.98, "weight-decay": 0.01}
        with patch("async_lora.worker.sample_group", new=AsyncMock(return_value=[completion])) as sampled:
            metrics = asyncio.run(
                worker.update({"environment": "B", "start_rollouts": 0, "train_seed": 3}, object(), settings, sampler)
            )
        self.assertEqual(sampled.await_count, 6)
        self.assertEqual(sampler.compute_logprobs_async.await_count, 6)
        self.assertEqual(worker.training.optim_step_async.await_count, 1)
        calls = worker.training.forward_backward_async.call_args_list
        self.assertEqual([(len(c.args[0]), c.args[1]) for c in calls], [(6, "importance_sampling"), (2, "ppo")])
        self.assertEqual(calls[1].args[2], {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2})
        self.assertEqual(calls[0].args[0][0].loss_fn_inputs["advantages"].data, [0.0, 0.0, 0.0625, 0.0])
        self.assertEqual(metrics["david/opd/replay_fraction"], 0.25)
        self.assertEqual(worker.replay.fresh_count, 7)
        self.assertEqual(worker.replay.cursor, 2)
        self.assertEqual(worker.replay.size, 7)

    def test_mixed_native_losses_match_existing_signed_clipped_replay_loss_and_gradients(self):
        # Exercise the actual native loss bodies without importing the GPU runtime.
        source = ROOT.parent / "miles/miles/backends/training_utils/loss/hub/tinker_losses.py"
        tree = ast.parse(source.read_text())
        names = {"importance_sampling_loss_function", "ppo_loss_function"}
        functions = ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]
        )
        generator = torch.Generator().manual_seed(127)
        for _ in range(32):
            lengths = torch.randint(2, 20, (8,), generator=generator).tolist()
            sampled = [torch.randn(n, generator=generator) - 4 for n in lengths]
            teacher = [s + torch.randn(n, generator=generator) for s, n in zip(sampled, lengths, strict=True)]
            current = [
                (s + torch.randn(n, generator=generator)).requires_grad_()
                for s, n in zip(sampled, lengths, strict=True)
            ]
            masks = [torch.randint(0, 2, (n,), generator=generator).float() for n in lengths]
            traces = [
                {
                    "tokens": [1] * (n + 3),
                    "response_length": n,
                    "rollout_log_probs": s.tolist(),
                    "teacher_log_probs": t.tolist(),
                    "loss_mask": m.tolist(),
                    "metadata": {"environment": "A"},
                }
                for n, s, t, m in zip(lengths, sampled, teacher, masks, strict=True)
            ]

            def reduce(values, lengths=lengths, masks=masks):
                return (
                    sum((v * m).sum() / m.sum().clamp_min(1) for v, m in zip(values.split(lengths), masks, strict=True))
                    / 8
                )

            replay_mask = torch.cat([torch.full((n,), float(i >= 5)) for i, n in enumerate(lengths)])
            expected, _ = sampled_reverse_kl_loss(
                torch.cat(current),
                torch.cat(sampled),
                torch.cat(teacher),
                reduce,
                replay_mask=replay_mask,
                replay_clip_epsilon=0.2,
            )
            losses = []
            for indices, name in ((range(5), "importance_sampling_loss_function"), (range(5, 8), "ppo_loss_function")):
                lp = [current[i] for i in indices]
                datums = trace_datums([traces[i] for i in indices], 8)
                advantages = [
                    torch.tensor(d.loss_fn_inputs["advantages"].data[-lengths[i] :])
                    for d, i in zip(datums, indices, strict=True)
                ]
                namespace = {
                    "torch": torch,
                    "Namespace": SimpleNamespace,
                    "RolloutBatch": dict,
                    "Callable": Callable,
                    "PPO_DEFAULTS": {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
                    "_target_logprobs": lambda *args, lp=lp: lp,
                    "_as_tensor_like": lambda v, ref: torch.as_tensor(v, dtype=ref.dtype),
                    "_local_response_values": lambda args, batch, values: values,
                    "_local_response_masks": lambda *args, lp=lp: [torch.ones_like(x) for x in lp],
                    "_sum_loss_and_outputs": lambda args, batch, logits, logprobs, per_datum: (sum(per_datum), {}),
                }
                # Exercise the actual local Miles loss bodies without its GPU imports.
                exec(compile(functions, str(source), "exec"), namespace)  # noqa: S102
                loss, _ = namespace[name](
                    None, {"rollout_log_probs": [sampled[i] for i in indices], "advantages": advantages}, None, None
                )
                losses.append(loss)
            actual = sum(losses)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            a, b = torch.autograd.grad(actual, current, retain_graph=True), torch.autograd.grad(expected, current)
            for x, y in zip(a, b, strict=True):
                torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)

    def test_checkpoint_commits_replay_and_continuation_rejects_different_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, bundle = root / "local", root / "durable/final"
            sampler_uri = "tinker://model/sampler_weights/current"
            source = uri_directory(local, sampler_uri)
            source.mkdir(parents=True)
            (source / "META.json").write_text("{}")
            (source / "adapter_model.safetensors").write_bytes(b"adapter")
            buffer = OPDReplayBuffer(400)
            buffer.add([trace("A"), trace("B")])
            buffer.sample(3, seed=7)

            def copy_adapter(source, destination, plan):
                import shutil

                shutil.copytree(source, destination)

            with patch("async_lora.storage.persist_final_adapter", side_effect=copy_adapter):
                result = asyncio.run(save_bundle(None, sampler_uri, local, bundle, {}, optimizer=False, replay=buffer))
            self.assertTrue(Path(result["replay"]).is_file())
            worker = Worker.__new__(Worker)
            worker.role, worker.plan = "student", {"opd_replay": {"capacity_per_env": 400}}
            worker.questions = {"A": row("A"), "B": row("B")}
            worker.restore_replay(result)
            self.assertEqual(worker.replay.fresh_count, buffer.fresh_count)
            self.assertEqual(worker.replay.sample(4, seed=11), buffer.sample(4, seed=11))
            worker.questions["A"] = row("A", value=12)
            with self.assertRaises(ValueError):
                worker.restore_replay(result)
            with self.assertRaises(RuntimeError):
                worker.restore_replay({})


if __name__ == "__main__":
    unittest.main()
