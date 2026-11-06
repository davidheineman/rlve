"""Check own-task controls, paired uncertainty, and deduplicated LOO probes."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "plot/mopd-clustering-v2/.plotting"))

from submit_mixing_transfer_loo import collect, task_specification, tier_for, write
from transfer_panels import assigned_gains, pairwise_panels, summarize


class TransferPanelTests(unittest.TestCase):
    def test_scores_apply_source_adapters_to_target_columns(self):
        matrix = np.arange(16).reshape(4, 4) / 16
        groups = [[0, 1], [2, 3]]
        np.testing.assert_allclose(assigned_gains(matrix, groups, include_own=False),
                                   [4 / 16, 1 / 16, 14 / 16, 11 / 16])
        np.testing.assert_allclose(assigned_gains(matrix, groups, include_own=True),
                                   [2 / 16, 3 / 16, 12 / 16, 13 / 16])
        self.assertIsNone(assigned_gains(matrix, [[i] for i in range(4)], include_own=False))

    def test_bootstrap_uses_environment_units_and_preserves_constants(self):
        constant = summarize([.25] * 64)
        self.assertEqual(constant["environment_bootstrap_95ci"], [.25, .25])
        first = summarize([0, 1, 2, 3])
        shifted = summarize([10, 11, 12, 13])
        np.testing.assert_allclose(np.asarray(first["environment_bootstrap_95ci"]) + 10,
                                   shifted["environment_bootstrap_95ci"])

    def test_equal_partitions_have_the_expected_own_adapter_weight(self):
        rng = np.random.default_rng(14)
        cube = rng.normal(size=(64, 64, 8))
        candidates = []
        for k in (2, 4, 8, 16, 32):
            groups = [list(range(i, i + 64 // k)) for i in range(0, 64, 64 // k)]
            candidates.append({"k": k, "folds": [
                {"groups": groups, "held_prompt_indices": [i, i + 4]} for i in range(4)]})
        source = {"environments": list(range(64)), "candidates": candidates}
        result = pairwise_panels(source, source, cube)
        diagonal = np.diag(cube.mean(axis=-1))
        for included, excluded in zip(result["included"]["points"][:-1],
                                      result["excluded"]["points"][:-1]):
            for label in ("best", "worst", "random"):
                size = included["size"]
                np.testing.assert_allclose(included[label]["per_environment"],
                    ((size - 1) * np.asarray(excluded[label]["per_environment"]) + diagonal) / size)
        self.assertIsNone(result["excluded"]["points"][-1]["best"])
        self.assertEqual(result["included"]["points"][0]["best"],
                         result["included"]["points"][0]["random"])

    def test_loo_tasks_exclude_each_target_and_deduplicate_repeated_partitions(self):
        part = {"groups": [[0, 1], [2, 3]]}
        completed = {"candidates": [{"k": 2, "best": part, "worst": part,
                                       "random": [part, part, part]}]}
        tasks, records = task_specification(completed, 4)
        self.assertEqual(len(tasks), 8)  # Four all-task LOO probes, four pair LOO probes.
        self.assertEqual(records[1]["best"], records[1]["worst"])
        for task in tasks.values():
            self.assertNotIn(task["targets"][0], task["members"])
        self.assertEqual(tier_for({"members": list(range(63))}), "large")
        self.assertEqual(tier_for({"members": list(range(31))}), "medium")
        self.assertEqual(tier_for({"members": list(range(15))}), "small")

    def test_collect_checks_prompt_pairing_before_publishing_curve_values(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = [{"user_prompt": name, "metadata": {"environment": name}} for name in ("A", "B")]
            write(root / "prompts.json", prompts)
            write(root / "plan.json", {"environments": ["A", "B"], "origin": "origin",
                                        "evaluation": {"confirmation": {"path": str(root / "prompts.json")}}})
            tasks, records = task_specification({"candidates": []}, 2)
            write(root / "specification.json", {"tasks": tasks, "records": records})
            for key, task in tasks.items():
                target = task["targets"][0]
                name = ["A", "B"][target]
                write(root / "scores" / f"{key}.json", {**task, "cells": [{
                    "environment": name, "accuracy": .5,
                    "observations": [{"prompt_id": hashlib.sha256(name.encode()).hexdigest(),
                                      "correct": 1, "total": 2}],
                }]})
            result = collect(root)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["records"][0]["random"], [.5, .5])
            path = next((root / "scores").glob("*.json"))
            corrupted = json.loads(path.read_text())
            corrupted["cells"][0]["observations"][0]["prompt_id"] = "wrong prompt"
            write(path, corrupted)
            with self.assertRaisesRegex(ValueError, "Mismatched LOO"):
                collect(root)


if __name__ == "__main__":
    unittest.main()
