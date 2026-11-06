"""Check the reverse objective, exact pair matching, and held-prompt isolation."""

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_low_transfer_lora_clustering import add_cached_confirmation
from lora_clustering.balanced import balanced_metrics
from lora_clustering.low_transfer import evaluate_fold, fit_low_transfer, summarize


def all_pairings(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for j, second in enumerate(rest):
        for others in all_pairings(rest[:j] + rest[j + 1 :]):
            yield [[first, second], *others]


class LowTransferTests(unittest.TestCase):
    def confirmation_fixture(self, root):
        files = {
            "plan.json": {
                "environments": ["A", "B"],
                "clustering": {"confirmation_sha256": "frozen"},
            },
            "clustering/evaluations/confirmation/individual-00/B.json": {
                "eval_sha256": "frozen",
                "accuracy": 0.125,
                "observations": [
                    {"prompt_id": "held-1", "correct": 0, "total": 4},
                    {"prompt_id": "held-2", "correct": 1, "total": 4},
                ],
            },
            "clustering/evaluations/confirmation/base/B.json": {
                "eval_sha256": "frozen",
                "accuracy": 0.75,
                "observations": [
                    {"prompt_id": "held-1", "correct": 4, "total": 4},
                    {"prompt_id": "held-2", "correct": 2, "total": 4},
                ],
            },
            "clustering/evaluations/selection/base/B.json": {
                "observations": [{"prompt_id": "selection-1"}]
            },
        }
        for relative, value in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        return {
            "environments": ["A", "B"],
            "full_fit_pairs": [{"task_a": "A", "task_b": "B"}],
            "seed": 1234,
        }

    def test_cached_confirmation_is_paired_and_does_not_impute_missing_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.confirmation_fixture(root)
            add_cached_confirmation(result, root)
            measured = result["cached_confirmation"]
            self.assertEqual(measured["available_directed_cells"], 1)
            self.assertEqual(measured["possible_directed_cells"], 2)
            self.assertFalse(measured["used_for_assignment_selection"])
            self.assertEqual(measured["records"][0]["gain"], -0.625)
            self.assertEqual(
                measured["records"][0]["prompt_bootstrap_95ci"], [-1, -0.25]
            )

    def test_confirmation_rejects_prompts_seen_during_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.confirmation_fixture(root)
            (root / "clustering/evaluations/selection/base/B.json").write_text(
                json.dumps({"observations": [{"prompt_id": "held-1"}]})
            )
            with self.assertRaisesRegex(ValueError, "disjoint"):
                add_cached_confirmation(result, root)

    def test_pair_matching_agrees_with_exhaustive_search(self):
        matrix = np.random.default_rng(19).integers(-10, 11, (8, 8)) / 16
        result = fit_low_transfer(matrix, 4)
        optimal = min(
            balanced_metrics(matrix, groups)["within"]
            for groups in all_pairings(list(range(8)))
        )
        self.assertAlmostEqual(result["within"], optimal)
        self.assertEqual(
            sorted(i for group in result["groups"] for i in group), list(range(8))
        )

    def test_balanced_swaps_minimize_mean_not_the_worst_group(self):
        matrix = np.random.default_rng(31).integers(-10, 11, (8, 8)) / 16
        alternatives = []
        for others in itertools.combinations(range(1, 8), 3):
            first = [0, *others]
            alternatives.append([first, [i for i in range(8) if i not in first]])
        optimal = min(
            balanced_metrics(matrix, groups)["within"] for groups in alternatives
        )
        result = fit_low_transfer(matrix, 2, restarts=64)
        self.assertAlmostEqual(result["within"], optimal)

    def test_recovers_negative_blocks_with_equal_sizes(self):
        matrix = np.full((12, 12), 0.2)
        expected = [list(range(i, i + 3)) for i in range(0, 12, 3)]
        for group in expected:
            matrix[np.ix_(group, group)] = -0.3
        result = fit_low_transfer(matrix, 4, restarts=48)
        self.assertEqual(result["groups"], expected)
        self.assertAlmostEqual(result["within"], -0.3)
        self.assertAlmostEqual(result["between"], 0.2)

    def test_self_transfer_is_ignored_and_search_is_reproducible(self):
        matrix = np.random.default_rng(15).normal(size=(12, 12))
        first = fit_low_transfer(matrix, 4, restarts=16, seed=9)
        np.fill_diagonal(matrix, 1000000)
        self.assertEqual(first, fit_low_transfer(matrix, 4, restarts=16, seed=9))

    def test_negative_mutual_mean_does_not_imply_both_directions_are_negative(self):
        matrix = np.zeros((4, 4))
        matrix[0, 1], matrix[1, 0] = -0.8, 0.4
        matrix[2, 3], matrix[3, 2] = -0.4, -0.2
        score = summarize(matrix, [[0, 1], [2, 3]])
        self.assertAlmostEqual(score["within"], -0.25)
        self.assertEqual(score["negative_clusters"], 2)
        self.assertEqual(score["both_directions_negative_pair_fraction"], 0.5)

    def test_held_prompt_changes_cannot_change_assignments(self):
        cube = np.random.default_rng(7).normal(size=(8, 8, 4))
        first = evaluate_fold(cube, 4, [0])
        cube[:, :, 0] = np.random.default_rng(40).normal(size=(8, 8)) * 100
        second = evaluate_fold(cube, 4, [0])
        self.assertEqual(first["train"], second["train"])
        self.assertEqual(first["groups"], second["groups"])
        self.assertNotEqual(first["held_out"]["within"], second["held_out"]["within"])

    def test_invalid_partitions_and_fold_indices_are_rejected(self):
        for k in [1, 3, 8]:
            with self.assertRaises(ValueError):
                fit_low_transfer(np.ones((8, 8)), k)
        for held in [[], [0, 0], [4], [0, 1, 2, 3]]:
            with self.assertRaises(ValueError):
                evaluate_fold(np.zeros((8, 8, 4)), 4, held)


if __name__ == "__main__":
    unittest.main()
