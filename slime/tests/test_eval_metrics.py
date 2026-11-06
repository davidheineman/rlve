import json
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from math import comb
from pathlib import Path

metrics_spec = spec_from_file_location("eval_metrics", Path(__file__).parents[1] / "utils" / "eval_metrics.py")
eval_metrics = module_from_spec(metrics_spec)
metrics_spec.loader.exec_module(eval_metrics)
group_eval_results = eval_metrics.group_eval_results
pass_at_k = eval_metrics.pass_at_k
early_stop_accuracy = eval_metrics.early_stop_accuracy


class EvalMetricsTest(unittest.TestCase):
    def test_pass_at_k_matches_combinatorial_estimator(self):
        for group_size in (2, 8, 16):
            for correct_count in range(group_size + 1):
                estimates = pass_at_k([correct_count], group_size)
                for sample_count, estimate in estimates.items():
                    expected = 1 - comb(group_size - correct_count, sample_count) / comb(group_size, sample_count)
                    self.assertAlmostEqual(estimate, expected)

    def test_pairwise_groups_preserve_prompts_and_aggregate(self):
        rewards, truncated, metadata = [], [], []
        for prompt_index in range(12):
            for environment_index in range(16):
                environment = f"environment_{environment_index}"
                prompt_rewards = [1] * (prompt_index % 9) + [0] * (8 - prompt_index % 9)
                rewards.extend(prompt_rewards)
                truncated.extend([False] * 7 + [True])
                sample_metadata = {"environment": environment}
                if prompt_index % 2:
                    sample_metadata = json.dumps(sample_metadata)
                metadata.extend([sample_metadata] * 8)

        results = group_eval_results("test_envs", rewards, truncated, metadata)
        self.assertEqual(len(rewards), 1536)
        self.assertEqual(len(results), 17)
        self.assertEqual(results["test_envs"]["rewards"], rewards)
        expected_counts = [prompt_index % 9 for prompt_index in range(12)]
        expected_estimates = pass_at_k(expected_counts, 8)
        self.assertEqual(set(expected_estimates), {1, 2, 4, 8})
        for environment_index in range(16):
            result = results[f"test_envs/environment_{environment_index}"]
            self.assertEqual(len(result["rewards"]), 96)
            self.assertEqual(sum(result["truncated"]), 12)
            counts = [sum(result["rewards"][start:start + 8]) for start in range(0, 96, 8)]
            self.assertEqual(counts, expected_counts)
            self.assertEqual(pass_at_k(counts, 8), expected_estimates)

    def test_metadata_without_environment_preserves_existing_metrics(self):
        metadata = [None, {}, "plain text", "{}", "[]"]
        results = group_eval_results("other_eval", [0] * 5, [False] * 5, metadata)
        self.assertEqual(set(results), {"other_eval"})

    def test_misaligned_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            group_eval_results("test_envs", [1], [False], [])

    def test_early_stop_uses_only_selected_id_sets_or_function_slice(self):
        results = {
            "binary_id": {"rewards": [1, 0]},
            "path_id": {"rewards": [1, 1]},
            "primitives_id/compress_repeats": {"rewards": [1, 1, 1, 0]},
        }
        self.assertEqual(early_stop_accuracy(results, ["path_id"]), 1)
        self.assertEqual(early_stop_accuracy(results, ["binary_id", "path_id"]), .75)
        self.assertEqual(early_stop_accuracy(results, ["primitives_id/compress_repeats"]), .75)
        with self.assertRaises(ValueError):
            early_stop_accuracy(results, ["primitives_id/remove_vowels"])


if __name__ == "__main__":
    unittest.main()
