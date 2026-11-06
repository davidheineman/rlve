"""Numerical parity, frozen inputs and masking for concurrent LoRA teachers."""

import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
from safetensors.torch import save_file
from tinker import types
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from lora_teachers.client import persist_final_adapter, persist_progress_adapter
from lora_teachers.objective import grpo_advantages, grpo_loss
from lora_teachers.plan import build_plan, gateway_argv
from lora_teachers.runner import preserved_results
from lora_teachers.sampling import (
    Completion,
    forcing_prefix,
    prompt_tokens,
    sample_group,
    training_datums,
)
from recover_data_scaling_lora import recovery_plan
from submit_data_scaling_lora import gpu_environment, submit


class LoRATeacherTest(unittest.TestCase):
    def test_chat_template_produces_sdk_token_ids_for_sampling_and_training(self):
        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "example": 1}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}",
        )
        prompt = prompt_tokens(tokenizer, "example")
        self.assertIsInstance(prompt, list)
        self.assertTrue(prompt)
        self.assertTrue(all(isinstance(token, int) for token in prompt))
        model_input = types.ModelInput.from_ints(prompt)
        self.assertEqual(model_input.to_ints(), prompt)
        completion = Completion([1], [-0.5], [1], 1.0, 1, False, False)
        datums, masks, logprobs = training_datums(prompt, [completion])
        self.assertEqual(datums[0].model_input.to_ints(), prompt)
        self.assertEqual(len(masks[0]), len(prompt))
        self.assertEqual(len(logprobs[0]), len(prompt))

    def test_noncontiguous_allocation_uses_numeric_ids_and_retains_uuid_checks(self):
        environment = gpu_environment([f"{index}, GPU-example-{index}, 49140" for index in (1, 3, 5, 7)])
        self.assertEqual(environment["APPTAINERENV_CUDA_VISIBLE_DEVICES"], "1,3,5,7")
        self.assertEqual(environment["APPTAINERENV_CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(
            environment["APPTAINERENV_LORA_EXPECTED_GPU_UUIDS"],
            "GPU-example-1,GPU-example-3,GPU-example-5,GPU-example-7",
        )
        remapped = gpu_environment([f"{index}, GPU-example-{index}, 49140" for index in (1, 3, 5, 7)], "0,1,2,3")
        self.assertEqual(remapped["APPTAINERENV_CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(
            remapped["APPTAINERENV_LORA_EXPECTED_GPU_UUIDS"], environment["APPTAINERENV_LORA_EXPECTED_GPU_UUIDS"]
        )
        pair = gpu_environment([f"{index}, GPU-example-{index}, 49140" for index in (2, 6)], expected_count=2)
        self.assertEqual(pair["APPTAINERENV_CUDA_VISIBLE_DEVICES"], "2,6")
        self.assertEqual(pair["APPTAINERENV_LORA_EXPECTED_GPU_UUIDS"], "GPU-example-2,GPU-example-6")
        with self.assertRaises(RuntimeError):
            gpu_environment([f"{index}, GPU-example-{index}, 49140" for index in (2, 6)])

    def test_final_adapter_survives_temporary_snapshot_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "local-snapshot"
            source.mkdir()
            (source / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64}))
            save_file(
                {"layer.lora_A.weight": torch.ones(32, 2), "layer.lora_B.weight": torch.ones(2, 32)},
                source / "adapter_model.safetensors",
            )
            destination = root / "adapter-final"
            persist_final_adapter(source, destination, build_plan())
            shutil.rmtree(source)
            self.assertTrue((destination / "adapter_model.safetensors").is_file())
            self.assertFalse(destination.is_symlink())

    def test_submission_uses_one_job_with_the_planned_gpu_count_and_a_frozen_worker(self):
        for train_gpus, sampling_gpus in ((2, 2), (1, 1)):
            with self.subTest(train_gpus=train_gpus), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                image = root / "image.sif"
                image.touch()
                model = root / "base"
                model.mkdir()
                (model / "config.json").write_text("{}")
                args = SimpleNamespace(
                    image=image,
                    model_path=model,
                    experiment=root / "experiment",
                    scratch=root,
                    miles_repo=root / "miles",
                )
                plan = build_plan() | {"train_gpus": train_gpus, "sampling_gpus": sampling_gpus}
                with (
                    patch("submit_data_scaling_lora.snapshot", side_effect=lambda path, *_: path.mkdir()),
                    patch("submit_data_scaling_lora.subprocess.check_output", return_value="12345\n") as sbatch,
                    patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}),
                ):
                    submit(args, plan, afterok=42)
                sbatch.assert_called_once()
                argv = sbatch.call_args.args[0]
                self.assertEqual(argv[0], "sbatch")
                self.assertIn(f"--gres=gpu:{train_gpus + sampling_gpus}", argv)
                self.assertIn("--no-requeue", argv)
                self.assertIn("--dependency=afterok:42", argv)
                self.assertFalse(any(arg.startswith("--array") for arg in argv))
                self.assertIn(str(args.experiment / "source/rlve/tools/submit_data_scaling_lora.py"), argv[-1])
                self.assertEqual(json.loads((args.experiment / "job.json").read_text())["job_id"], "12345")

    def test_progress_adapter_is_durable_and_keeps_prior_copy_if_validation_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "local"
            source.mkdir()
            (source / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64}))
            weights = {"layer.lora_A.weight": torch.ones(32, 2), "layer.lora_B.weight": torch.ones(2, 32)}
            save_file(weights, source / "adapter_model.safetensors")
            directory = root / "teacher"
            directory.mkdir()
            plan = build_plan()
            persist_progress_adapter(source, directory, plan, 50)
            old = Path(json.loads((directory / "latest-adapter.json").read_text())["adapter_path"])
            weights["layer.lora_B.weight"].fill_(float("nan"))
            save_file(weights, source / "adapter_model.safetensors")
            with self.assertRaises(ValueError):
                persist_progress_adapter(source, directory, plan, 100)
            self.assertTrue(old.is_dir())
            self.assertEqual(json.loads((directory / "latest-adapter.json").read_text())["updates"], 50)
            weights["layer.lora_B.weight"].fill_(2)
            save_file(weights, source / "adapter_model.safetensors")
            persist_progress_adapter(source, directory, plan, 100)
            latest = json.loads((directory / "latest-adapter.json").read_text())
            self.assertFalse(old.exists())
            self.assertEqual(latest["training_rollouts"], 800)
            self.assertFalse(latest["optimizer_state_saved"])
            shutil.rmtree(source)
            self.assertTrue((Path(latest["adapter_path"]) / "adapter_model.safetensors").exists())

    def test_recovery_retains_completed_teachers_and_restarts_only_incomplete_teachers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = build_plan(project="rl-data-scaling-lora-teachers-v2", learning_rate=2e-5)
            (root / "plan.json").write_text(json.dumps(original))
            (root / "job.json").write_text(json.dumps({"job_id": "42"}))
            teacher = original["teachers"][0]
            directory = root / "teacher" / teacher["source_label"]
            adapter = directory / "adapter-final"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"preserved")
            marker = directory / "completed.json"
            marker.write_text(json.dumps({"environment": teacher["environment"], "adapter_path": str(adapter)}))
            plan = recovery_plan(root)
            self.assertEqual(plan["teachers"], original["teachers"][1:])
            self.assertEqual(plan["teacher_config"], original["teacher_config"])
            self.assertEqual(plan["time_limit"], "1-00:00:00")
            self.assertEqual(plan["progress_adapter_interval"], 50)
            self.assertEqual(plan["recovery"]["bank_size"], 16)
            self.assertEqual(len(preserved_results(plan)), 1)
            (adapter / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                preserved_results(plan)

    def test_original_panel_and_budgets(self):
        plan = build_plan()
        original = json.loads((ROOT / "data/DATA-SCALING-V1/run_plan.json").read_text())
        panel = next(run for run in original["runs"] if run["label"] == "n16-mopd")
        self.assertEqual([t["environment"] for t in plan["teachers"]], panel["environments"])
        config = plan["teacher_config"]
        self.assertEqual(config["num-rollout"] * config["n-samples-per-prompt"], 4000)
        self.assertEqual(config["eval-interval"], 50)
        self.assertEqual(config["early-stop-val-accuracy"], 0.98)
        self.assertEqual(plan["project"], "rl-data-scaling-lora-teachers-v1")
        argv = gateway_argv(plan, model_path="/base", checkpoint_root="/adapters", port=6000, router_port=7000)
        self.assertEqual(argv[argv.index("--multi-lora-n-adapters") + 1], "16")
        self.assertEqual(argv[argv.index("--load") + 1], "/base")
        self.assertNotIn("--sglang-router-ip", argv)
        self.assertEqual(argv[argv.index("--sglang-router-port") + 1], "7000")
        self.assertEqual(argv[argv.index("--attention-backend") + 1], "flash")
        self.assertEqual(plan["flash_attention_version"], 2)
        self.assertEqual(
            argv[argv.index("--custom-megatron-init-path") + 1],
            "lora_teachers.attention.configure_flash_attention_2",
        )
        self.assertNotIn("--colocate", argv)
        self.assertNotIn("--optimizer-cpu-offload", argv)

    def test_v2_learning_rate_reaches_gateway_and_preserves_original_protocol(self):
        original = build_plan()
        tuned = build_plan(project="rl-data-scaling-lora-teachers-v2", learning_rate=2e-5)
        self.assertEqual(tuned["project"], "rl-data-scaling-lora-teachers-v2")
        self.assertEqual(tuned["teacher_config_overrides"], {"lr": {"source": 2e-6, "value": 2e-5}})
        self.assertEqual(tuned["teacher_config"], original["teacher_config"] | {"lr": 2e-5})
        self.assertEqual(tuned["teachers"], original["teachers"])
        self.assertEqual(tuned["teacher_config_sha256"], original["teacher_config_sha256"])
        self.assertEqual(build_plan()["teacher_config"]["lr"], 2e-6)
        argv = gateway_argv(tuned, model_path="/base", checkpoint_root="/adapters", port=6000, router_port=7000)
        self.assertEqual(float(argv[argv.index("--lr") + 1]), 2e-5)
        for invalid in (0, -1e-5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_plan(learning_rate=invalid)

    def test_gradient_matches_original_grpo_with_tis_and_response_means(self):
        rewards = [-1.0, 0.0, 1.0, 1.0, -1.0, 0.5, 0.0, 1.0]
        masks = [[0, 0, 1, 1, 0, 1][:n] for n in (3, 4, 5, 6, 3, 4, 5, 6)]
        lp = [torch.full((len(mask),), -2.0, requires_grad=True) for mask in masks]
        sampled = [[-4.0 if i % 2 else -1.0 for i in range(len(mask))] for mask in masks]
        loss, _ = grpo_loss(
            lp,
            rollout_logprobs=sampled,
            masks=masks,
            rewards=rewards,
            eps_low=0.2,
            eps_high=0.28,
            tis_low=0.0,
            tis_high=2.0,
        )
        loss.backward()
        advantage = grpo_advantages(rewards)
        for current, rollout, mask, adv in zip(lp, sampled, masks, advantage, strict=True):
            active = torch.tensor(mask)
            correction = (current.detach() - torch.tensor(rollout)).exp().clamp(0, 2)
            expected = -adv * correction * active / active.sum() / 8
            torch.testing.assert_close(current.grad, expected)
            self.assertTrue(torch.equal(current.grad[active == 0], torch.zeros_like(current.grad[active == 0])))

    def test_no_reward_variance_has_no_policy_gradient(self):
        lp = [torch.tensor([-2.0, -3.0], requires_grad=True) for _ in range(8)]
        loss, _ = grpo_loss(
            lp,
            rollout_logprobs=[[-1.0, -1.0]] * 8,
            masks=[[0, 1]] * 8,
            rewards=[1.0] * 8,
            eps_low=0.2,
            eps_high=0.28,
            tis_low=0,
            tis_high=2,
        )
        loss.backward()
        for item in lp:
            torch.testing.assert_close(item.grad, torch.zeros_like(item))

    def test_prompt_and_injected_tokens_are_aligned_to_shifted_targets(self):
        completion = Completion([7, 8, 9, 10], [-1.0, 0.0, 0.0, -2.0], [1, 0, 0, 1], 1, 1, True, False)
        datums, masks, logprobs = training_datums([1, 2, 3], [completion])
        self.assertEqual(datums[0].loss_fn_inputs["target_tokens"].data, [2, 3, 7, 8, 9, 10])
        self.assertEqual(masks, [[0, 0, 1, 0, 0, 1]])
        self.assertEqual(logprobs, [[0.0, 0.0, -1.0, 0.0, 0.0, -2.0]])

    def test_answer_forcing_continuation_uses_same_sampling_policy(self):
        plan = build_plan()
        tokenizer = SimpleNamespace(
            decode=lambda tokens, **kwargs: "reasoning" if tokens == [11, 12] else "answer",
            encode=lambda text, **kwargs: [20, 21],
        )
        first = SimpleNamespace(tokens=[11, 12], logprobs=[-0.2, -0.3], stop_reason="length")
        last = SimpleNamespace(tokens=[30], logprobs=[-0.1], stop_reason="stop")
        sampler = SimpleNamespace(
            sample_async=AsyncMock(side_effect=[SimpleNamespace(sequences=[first]), SimpleNamespace(sequences=[last])])
        )
        verifier = SimpleNamespace(
            set_config=lambda config: None, verifier=lambda response: {"reward": 1.0, "accuracy": 1}
        )
        with patch("lora_teachers.sampling.identifier2environment", {"example": lambda **kwargs: verifier}):
            result = asyncio.run(sample_group(sampler, tokenizer, [1], "example", {}, plan["teacher_config"]))
        self.assertEqual(result[0].mask, [1, 1, 0, 0, 1])
        self.assertFalse(result[0].truncated)
        second = sampler.sample_async.call_args_list[1].kwargs
        self.assertEqual(second["sampling_params"].max_tokens, 128)
        self.assertEqual(second["sampling_params"].temperature, 1.0)
        self.assertEqual(second["sampling_params"].top_p, 1.0)
        self.assertEqual(forcing_prefix("reasoning"), "\n</think>\n\n\\boxed{")
        self.assertEqual(forcing_prefix("</think>answer"), "\n\n\\boxed{")


if __name__ == "__main__":
    unittest.main()
