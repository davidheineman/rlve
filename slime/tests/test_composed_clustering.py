"""The composed search must evaluate the exact multi-adapter update."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from composed_clustering.composition import compose, exact_factors
from composed_clustering.runner import Evaluator
from composed_clustering.search import canonical, partition_key, refine


class ExactCompositionTests(unittest.TestCase):
    def test_search_acceptance_uses_measured_composition_not_the_pairwise_surrogate(
        self,
    ):
        initial = [[[0, 1], [2, 3]]]
        preferred = canonical([[0, 2], [1, 3]])

        def accuracy(groups):
            return 0.9 if canonical(groups) == preferred else 0.2

        # The proposal surrogate favors the initial grouping; actual measured
        # composition accuracy must nevertheless determine acceptance.
        import numpy as np

        surrogate = np.zeros((4, 4))
        surrogate[0, 1] = surrogate[1, 0] = 100
        high = refine(initial, accuracy, surrogate, maximize=True, rounds=3, batch=4)
        low = refine(
            [preferred, *initial],
            accuracy,
            surrogate,
            maximize=False,
            rounds=3,
            batch=4,
        )
        self.assertEqual(high["groups"], preferred)
        self.assertEqual(high["selection_accuracy"], 0.9)
        self.assertEqual(low["selection_accuracy"], 0.2)
        self.assertTrue(
            all(
                a["score"] <= b["score"]
                for a, b in zip(high["trace"], high["trace"][1:])
            )
        )
        self.assertEqual(partition_key([[3, 1], [2, 0]]), partition_key(preferred))

    def test_parallel_target_chunks_keep_the_entire_composed_adapter_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks").mkdir()
            (root / "scores").mkdir()
            evaluator = Evaluator(root, {}, [SimpleNamespace(poll=lambda: None)])

            def complete(_):
                for path in (root / "tasks").glob("*.json"):
                    task = json.loads(path.read_text())
                    self.assertEqual(task["members"], list(range(32)))
                    self.assertLessEqual(len(task["targets"]), 8)
                    (root / "scores" / path.name).write_text(
                        json.dumps(
                            {
                                "cells": [
                                    {"environment": str(i), "accuracy": 1}
                                    for i in task["targets"]
                                ]
                            }
                        )
                    )

            with patch("composed_clustering.runner.time.sleep", side_effect=complete):
                result = evaluator.groups([list(range(32))], "selection")
            self.assertEqual(len(result[0]["cells"]), 32)
            self.assertEqual(len(list((root / "tasks").glob("*.json"))), 4)

    def test_concatenation_equals_mean_dense_update(self):
        generator = torch.Generator().manual_seed(8)
        factors = [
            (
                torch.randn(3, 5, generator=generator),
                torch.randn(7, 3, generator=generator),
            )
            for _ in range(4)
        ]
        scales = [0.5, 2.0, 3.0, 1.0]
        a, b = exact_factors(factors, scales)
        expected = (
            sum(scale * (b0 @ a0) for (a0, b0), scale in zip(factors, scales)) / 4
        )
        torch.testing.assert_close(b @ a, expected)
        self.assertEqual(a.shape, (12, 5))
        self.assertEqual(b.shape, (7, 12))

    def test_repeated_adapters_leave_the_effective_update_unchanged(self):
        a = torch.randn(4, 5)
        b = torch.randn(7, 4)
        aa, bb = exact_factors([(a, b)] * 32, [2.0] * 32)
        torch.testing.assert_close(bb @ aa, 2 * b @ a)

    def test_saved_adapter_has_uncompressed_rank_and_unit_output_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            expected = []
            for i in range(2):
                source = root / str(i)
                source.mkdir()
                sources.append(source)
                a = torch.randn(2, 5)
                b = torch.randn(7, 2)
                expected.append(2 * b @ a)
                save_file(
                    {"layer.lora_A.weight": a, "layer.lora_B.weight": b},
                    str(source / "adapter_model.safetensors"),
                )
                (source / "adapter_config.json").write_text(
                    json.dumps(
                        {
                            "r": 2,
                            "lora_alpha": 4,
                            "target_modules": ["layer"],
                            "base_model_name_or_path": "same",
                        }
                    )
                )
            output = root / "composed"
            compose(sources, output)
            tensors = load_file(str(output / "adapter_model.safetensors"))
            config = json.loads((output / "adapter_config.json").read_text())
            self.assertEqual((config["r"], config["lora_alpha"]), (4, 4))
            torch.testing.assert_close(
                tensors["layer.lora_B.weight"] @ tensors["layer.lora_A.weight"],
                sum(expected) / 2,
            )
            compact = root / "inference-storage"
            compose(sources, compact, inference_storage=True)
            inference = load_file(str(compact / "adapter_model.safetensors"))
            for key, value in tensors.items():
                self.assertEqual(inference[key].dtype, torch.bfloat16)
                self.assertTrue(torch.equal(inference[key], value.to(torch.bfloat16)))


if __name__ == "__main__":
    unittest.main()
