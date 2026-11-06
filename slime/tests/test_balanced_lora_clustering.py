"""The balanced objective must not reward self-transfer or a sacrificed group."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from lora_clustering.balanced import (
    balanced_metrics,
    candidate_ks,
    fit_balanced,
    random_balanced_baseline,
    select_k,
)


class BalancedClusteringTest(unittest.TestCase):
    def test_random_baseline_matches_partition_metrics_on_each_fold(self):
        matrices = np.random.default_rng(18).normal(size=(2, 8, 8))
        rng = np.random.default_rng(29)
        measured = np.zeros((7, 3))
        for matrix in matrices:
            for draw in range(7):
                order = rng.permutation(8)
                groups = [np.flatnonzero(order // 2 == g).tolist() for g in range(4)]
                score = balanced_metrics(matrix, groups)
                measured[draw] += [score["within"], score["between"], score["mean_gap"]]
        measured /= len(matrices)
        result = random_balanced_baseline(matrices, 4, draws=7, seed=29)
        for i, key in enumerate(("within", "between", "gap")):
            self.assertAlmostEqual(result[key]["sampled_mean"], measured[:, i].mean())
            np.testing.assert_allclose(
                result[key]["assignment_interval_95"],
                np.quantile(measured[:, i], [0.025, 0.975]),
            )
        expected = matrices[:, ~np.eye(8, dtype=bool)].mean()
        self.assertAlmostEqual(result["within"]["expected_mean"], expected)
        self.assertAlmostEqual(result["between"]["expected_mean"], expected)
        self.assertEqual(result["gap"]["expected_mean"], 0)

    def test_random_baseline_excludes_self_transfer(self):
        matrix = np.random.default_rng(50).normal(size=(8, 8))
        first = random_balanced_baseline([matrix], 2, draws=50, seed=13)
        np.fill_diagonal(matrix, 1000000)
        self.assertEqual(
            first, random_balanced_baseline([matrix], 2, draws=50, seed=13)
        )

    def test_singletons_have_no_random_within_or_gap(self):
        result = random_balanced_baseline([np.full((8, 8), 0.2)], 8, draws=30)
        self.assertIsNone(result["within"])
        self.assertIsNone(result["gap"])
        np.testing.assert_allclose(
            result["between"]["assignment_interval_95"], [0.2, 0.2]
        )

    def test_admissible_k_excludes_trivial_and_singleton_solutions(self):
        self.assertEqual(candidate_ks(32), [2, 4, 8, 16])

    def test_recovers_blocks_even_when_between_transfer_is_positive(self):
        transfer = np.full((12, 12), 0.02)
        groups = [list(range(i, i + 3)) for i in range(0, 12, 3)]
        for group in groups:
            transfer[np.ix_(group, group)] = 0.2
        result = fit_balanced(transfer, 4, restarts=48, seed=1234)
        self.assertEqual(result["groups"], groups)
        self.assertAlmostEqual(result["within"], 0.2)
        self.assertAlmostEqual(result["between"], 0.02)
        self.assertGreater(result["weakest_quality"], 0)

    def test_self_transfer_cannot_change_assignments_or_score(self):
        rng = np.random.default_rng(50)
        matrix = rng.normal(size=(12, 12))
        first = fit_balanced(matrix, 3, restarts=12, seed=13)
        np.fill_diagonal(matrix, np.arange(12) * 1000000)
        second = fit_balanced(matrix, 3, restarts=12, seed=13)
        self.assertEqual(first, second)

    def test_worst_cluster_counts_in_both_transfer_directions(self):
        matrix = np.full((8, 8), 0.2)
        matrix[:4, 4:] = 0.25
        matrix[4:, :4] = 0.01
        result = balanced_metrics(matrix, [list(range(4)), list(range(4, 8))])
        self.assertAlmostEqual(result["weakest_quality"], -0.05)

    def test_uniform_general_helpfulness_is_not_cluster_structure(self):
        result = fit_balanced(np.full((8, 8), 0.3), 2, restarts=8)
        self.assertAlmostEqual(result["weakest_quality"], 0)
        self.assertAlmostEqual(result["mean_gap"], 0)

    def test_partition_is_exactly_balanced_and_reproducible(self):
        matrix = np.random.default_rng(7).normal(size=(16, 16))
        result = fit_balanced(matrix, 4, restarts=16, seed=19)
        self.assertEqual([len(g) for g in result["groups"]], [4] * 4)
        self.assertEqual(
            sorted(i for g in result["groups"] for i in g), list(range(16))
        )
        self.assertEqual(result, fit_balanced(matrix, 4, restarts=16, seed=19))

    def test_invalid_partitions_and_matrices_are_rejected(self):
        for groups in [[[0], [1, 2, 3]], [[0, 1], [1, 2]]]:
            with self.assertRaises(ValueError):
                balanced_metrics(np.ones((4, 4)), groups)
        with self.assertRaises(ValueError):
            fit_balanced(np.ones((8, 8)), 8)
        with self.assertRaises(ValueError):
            fit_balanced(np.full((8, 8), float("nan")), 2)

    def test_k_selection_prefers_simpler_when_scores_are_indistinguishable(self):
        records = [
            {
                "k": k,
                "cv_quality_mean": quality,
                "cv_quality_se": 0.02,
                "cv_gap_mean": quality,
                "cv_gap_se": 0.02,
                "fit": {"weakest_quality": 0.1},
            }
            for k, quality in [(2, 0.065), (4, 0.08), (8, 0.07)]
        ]
        result = select_k(records)
        self.assertEqual(result["best_cv_k"], 4)
        self.assertEqual(result["k"], 2)


if __name__ == "__main__":
    unittest.main()
