"""Synthetic recovery, exact balance, diagonal invariance, and CV isolation."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_pairwise_groupings import FOLDS, choose_nested
from lora_clustering.grouping_methods import (
    METHODS, PARAMETERS, affinity, balanced_kmeans, fit_methods,
    hierarchy, random_task_scores, swap_polish, task_scores, validate,
)


class GroupingMethodTests(unittest.TestCase):
    def test_capacity_assignment_covers_duplicate_points(self):
        groups = balanced_kmeans(np.zeros((8, 3)), 4, 1, restarts=2)
        validate(groups, 8, 4)

    def test_all_methods_recover_strong_blocks_and_ignore_diagonal(self):
        cube = np.full((8, 8, 4), -.2)
        for start in (0, 4):
            cube[start:start+4, start:start+4] = .5
        fast = {"mean_swap_restarts": 8, "risk_swap_restarts": 8, "kmeans_restarts": 4,
                "annealing_restarts": 2, "annealing_moves_per_restart": 100}
        with patch.dict(PARAMETERS, fast):
            first = fit_methods(cube, 2, 12)
            changed = cube.copy()
            for i in range(8):
                changed[i, i] = [1000, -1000, 2, 4]
            second = fit_methods(changed, 2, 12)
        self.assertEqual(set(first), set(METHODS))
        for method, groups in first.items():
            validate(groups, 8, 2)
            self.assertAlmostEqual(task_scores(cube.mean(axis=-1), groups).mean(), .5, msg=method)
            self.assertEqual(groups, second[method], method)

    def test_random_expectation_includes_self_with_correct_weight(self):
        matrix = np.arange(16, dtype=float).reshape(4, 4) / 16
        partitions = [[[0, other], [i for i in range(1, 4) if i != other]] for other in range(1, 4)]
        empirical = np.mean([task_scores(matrix, groups) for groups in partitions], axis=0)
        np.testing.assert_allclose(random_task_scores(matrix, 2), empirical)
        np.testing.assert_allclose(random_task_scores(matrix, 1), matrix.mean(axis=0))
        np.testing.assert_allclose(random_task_scores(matrix, 4), np.diag(matrix))

    def test_polish_preserves_balance_and_never_reduces_mean(self):
        matrix = affinity(np.random.default_rng(8).normal(size=(8, 8)))
        start = [[0, 1, 2, 3], [4, 5, 6, 7]]
        groups = swap_polish(matrix, start)
        validate(groups, 8, 2)
        self.assertGreaterEqual(task_scores(matrix, groups).mean(), task_scores(matrix, start).mean())

    def test_pair_hierarchy_is_exact_maximum_matching(self):
        matrix = affinity(np.random.default_rng(4).normal(size=(6, 6)))
        def pairings(items):
            if not items:
                yield []
                return
            for j in range(1, len(items)):
                for rest in pairings(items[1:j] + items[j+1:]):
                    yield [[items[0], items[j]], *rest]
        groups = hierarchy(matrix, 3, "average")
        best = max(task_scores(matrix, g).mean() for g in pairings(list(range(6))))
        self.assertAlmostEqual(task_scores(matrix, groups).mean(), best)

    def test_outer_test_outcomes_cannot_change_nested_method_selection(self):
        cube = np.random.default_rng(33).normal(size=(8, 8, 8))
        alternatives = [[[0,1,2,3],[4,5,6,7]], [[0,2,4,6],[1,3,5,7]]]
        fits = {(mask, 2): {method: alternatives[j % 2] for j, method in enumerate(METHODS)}
                for mask in range(1, 16) if mask.bit_count() == 2}
        before = choose_nested(cube, fits, 2, outer=0)
        cube[:, :, FOLDS[0]] += np.random.default_rng(10).normal(size=(8, 8, 2)) * 100
        after = choose_nested(cube, fits, 2, outer=0)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
