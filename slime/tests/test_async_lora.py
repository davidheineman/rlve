"""Loss equivalence, immutable checkpoints and causal panels for shared-base async MOPD."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from async_lora.objective import opd_advantages
from async_lora.plan import build_async_plan
from async_lora.storage import restore_reference, save_bundle, uri_directory
from lora_teachers.plan import gateway_argv

from async_branching.io import read_json, write_json
from slime.utils.opd import sampled_reverse_kl_loss


class AsyncLoRATest(unittest.TestCase):
    def test_student_scores_shifted_targets_and_masks_injected_tokens(self):
        from unittest.mock import AsyncMock

        from async_lora.worker import Worker
        from lora_teachers.sampling import Completion

        worker = Worker.__new__(Worker)
        sampler = SimpleNamespace(
            compute_logprobs_async=AsyncMock(
                side_effect=[[None, -2.0, -3.0, -1.5, -20.0, -2.5], [None, -9.0, -8.0, -7.0, -6.0, -1.25]]
            )
        )
        rows = [([1, 2, 3], "A", {}, 0), ([4, 5, 6, 7], "A", {}, 1)]
        completions = [
            Completion([8, 9, 10], [-1.0, 0.0, -2.0], [1, 0, 1], 0, 0, True, False),
            Completion([11, 12], [-3.0, -1.0], [1, 1], 1, 1, False, False),
        ]
        datums, _ = asyncio.run(worker.distillation_datums(rows, completions, sampler))
        self.assertEqual(datums[0].loss_fn_inputs["target_tokens"].data, [2, 3, 8, 9, 10])
        self.assertEqual(datums[1].loss_fn_inputs["target_tokens"].data, [5, 6, 7, 11, 12])
        self.assertEqual(datums[0].loss_fn_inputs["advantages"].data, [0.0, 0.0, -0.125, 0.0, -0.125])
        self.assertEqual(datums[1].loss_fn_inputs["advantages"].data, [0.0, 0.0, 0.0, -0.75, -0.0625])
        self.assertEqual(datums[0].loss_fn_inputs["logprobs"].data, [0.0, 0.0, -1.0, 0.0, -2.0])

    def test_replay_preserves_gate_decisions_before_published_rollout_watermarks(self):
        import random
        from unittest.mock import AsyncMock, Mock

        import numpy as np
        from async_lora.worker import Worker

        for used, interval, cap, expected in (
            (0, 50, 800, 400),
            (400, 50, 800, 800),
            (0, 100, 1600, 800),
            (800, 100, 1600, 1600),
            (800, 100, 1200, 1200),
        ):
            with (
                self.subTest(checkpoint_rollouts=used, interval=interval, cap=cap),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                task_directory = root / "tasks/teacher-00-g0001"
                task_directory.mkdir(parents=True)
                worker = Worker.__new__(Worker)
                worker.role = "teacher"
                worker.root = root / "local"
                worker.directory = root / "workers/teacher-00"
                worker.training, worker.service = object(), object()
                worker.plan = {
                    "teacher_config": {"eval-interval": interval, "rollout-batch-size": 1, "n-samples-per-prompt": 8},
                    "teacher_project": "teachers",
                    "teacher_validation_threshold": 0.98,
                    "eval_interval_updates": 50,
                }
                worker.open_run = Mock(return_value=Mock())
                worker.source = Mock(return_value=Mock())
                worker.load = AsyncMock()
                worker.gate = AsyncMock(return_value=0.99)
                worker.update = AsyncMock(return_value={})
                worker.rng_state = Mock(return_value={})
                write_json(task_directory / "initial-validation.json", {"accuracy": 0.2})
                checkpoint = task_directory / "checkpoints/u0050"
                checkpoint.mkdir(parents=True)
                if used:
                    write_json(
                        task_directory / "latest.json",
                        {"hf": str(checkpoint / "adapter"), "resume": str(checkpoint / "state")},
                    )
                task = {
                    "id": "teacher-00-g0001",
                    "directory": str(task_directory),
                    "cap": cap,
                    "parent": {"hf": "base-adapter", "resume": "base-state"},
                    "start_clock": 0,
                    "generation": 1,
                    "environment": "A",
                    "teacher_start_rollouts": 0,
                }
                saved = {
                    "used": used,
                    "accuracy": 0.2,
                    "source": {},
                    "random": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                }
                with (
                    patch(
                        "async_lora.worker.export", new=AsyncMock(return_value=(None, "tinker://m/sampler_weights/v"))
                    ),
                    patch("async_lora.worker.adapter_identity", return_value={}),
                    patch("async_lora.worker.client_state", return_value=saved),
                    patch("async_lora.worker.log_metrics"),
                    patch("async_lora.worker.save_bundle", new=AsyncMock(return_value={"hf": "final", "resume": None})),
                ):
                    result = asyncio.run(worker.execute(task))
                # Re-evaluating the cached negative gate would immediately return
                # .99 and stop before the next promised evaluation watermark.
                self.assertEqual(read_json(task_directory / "progress.json")["next_validation_rollouts"], expected)
                self.assertEqual(result["rollouts"], expected)
                worker.gate.assert_awaited_once()
                self.assertEqual(worker.update.await_count, (expected - used) // 8)

    def test_native_opd_matches_original_loss_and_gradient_with_forced_token_masks(self):
        generator = torch.Generator().manual_seed(41)
        for batch in (1, 8):
            for _ in range(16):
                lengths = torch.randint(2, 25, (batch,), generator=generator).tolist()
                sampled = [torch.randn(n, generator=generator) - 5 for n in lengths]
                teacher = [torch.randn(n, generator=generator) - 5 for n in lengths]
                masks = [torch.randint(0, 2, (n,), generator=generator).float() for n in lengths]
                current = [
                    (p + 0.2 * torch.randn(n, generator=generator)).requires_grad_()
                    for n, p in zip(lengths, sampled, strict=True)
                ]

                def reduce(values, lengths=lengths, masks=masks, batch=batch):
                    return (
                        sum(
                            (v * m).sum() / m.sum().clamp_min(1)
                            for v, m in zip(values.split(lengths), masks, strict=True)
                        )
                        / batch
                    )

                original, _ = sampled_reverse_kl_loss(
                    torch.cat(current), torch.cat(sampled), torch.cat(teacher), reduce
                )
                native = sum(
                    -(torch.exp(p - s) * torch.tensor(opd_advantages(s.tolist(), t.tolist(), m.tolist(), batch))).sum()
                    for p, s, t, m in zip(current, sampled, teacher, masks, strict=True)
                )
                torch.testing.assert_close(original, native)
                a = torch.autograd.grad(original, current, retain_graph=True)
                b = torch.autograd.grad(native, current)
                for x, y in zip(a, b, strict=True):
                    torch.testing.assert_close(x, y)

    def test_exact_panels_projects_and_four_gpu_layout(self):
        for size in (16, 32):
            plan = build_async_plan(size)
            self.assertEqual(len(plan["environments"]), size)
            self.assertEqual(plan["adapter_slots"], size + 1)
            self.assertEqual(plan["physical_gpus"], 4)
            self.assertEqual(plan["student_project"], "rl-data-scaling-students-v1")
            self.assertEqual(plan["teacher_project"], "rl-data-scaling-async-lora-teachers-v1")
            self.assertEqual(plan["teacher_branch_rollouts"], 4000)
            self.assertEqual(plan["student_rollouts"], 16000)
            self.assertIsNone(plan["teacher_device_rollout_horizon"])
            argv = gateway_argv(plan, model_path="model", checkpoint_root="root", port=6000, router_port=7000)
            self.assertEqual(argv[argv.index("--multi-lora-n-adapters") + 1], str(size + 1))
            self.assertEqual(argv[argv.index("--sglang-cuda-graph-max-bs-decode") + 1], "64")

    def test_durable_checkpoint_rehydrates_after_node_loss_and_keeps_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, bundle = base / "local", base / "durable/final"
            sampler_uri = "tinker://model-source/sampler_weights/first"
            source = uri_directory(root, sampler_uri)
            source.mkdir(parents=True)
            (source / "META.json").write_text("{}")
            (source / "adapter_model.safetensors").write_bytes(b"adapter")

            class Training:
                async def save_state_async(self, name):
                    uri = f"tinker://model-source/weights/{name}"
                    target = uri_directory(root, uri)
                    target.mkdir(parents=True)
                    (target / "adam").write_bytes(b"optimizer-and-weights")

                    async def saved():
                        return SimpleNamespace(path=uri)

                    return saved()

            def copy_adapter(source, destination, plan):
                import shutil

                shutil.copytree(source, destination)

            with patch("async_lora.storage.persist_final_adapter", side_effect=copy_adapter):
                result = asyncio.run(save_bundle(Training(), sampler_uri, root, bundle, {}, {"used": 400}))
            reference = read_json(bundle / "reference.json")
            self.assertFalse(uri_directory(root, reference["state_uri"]).exists())
            new_root = base / "new-node"
            uri = restore_reference(new_root, result["resume"])
            self.assertEqual((uri_directory(new_root, uri) / "adam").read_bytes(), b"optimizer-and-weights")
            sampled = restore_reference(new_root, result["hf"], sampling=True)
            self.assertEqual((uri_directory(new_root, sampled) / "adapter_model.safetensors").read_bytes(), b"adapter")
            # Rehydration is idempotent and does not replace immutable contents.
            self.assertEqual(restore_reference(new_root, result["resume"]), uri)

    def test_bad_checkpoint_paths_and_nonfinite_advantages_fail(self):
        for uri in ("/tmp/a", "tinker://../weights/a", "tinker://model/weights/../a", "tinker://model/foo/a"):
            with self.assertRaises(ValueError):
                uri_directory("/tmp/local", uri)
        with self.assertRaises(ValueError):
            opd_advantages([0], [float("nan")], [1], 8)


if __name__ == "__main__":
    unittest.main()
