import json
import random
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

module_spec = spec_from_file_location(
    "eval_rollouts", Path(__file__).parents[1] / "utils/eval_rollouts.py"
)
eval_rollouts = module_from_spec(module_spec)
module_spec.loader.exec_module(eval_rollouts)


def make_samples(environment_count=32, prompt_count=12):
    samples = []
    for prompt_index in range(prompt_count):
        for environment_index in range(environment_count):
            metadata = {
                "environment": f"environment_{environment_index}",
                "config": {
                    "seed": 2000000 + prompt_index,
                    "parameter": {"reference_answer": 42},
                },
            }
            for rollout_index in range(8):
                samples.append(
                    SimpleNamespace(
                        index=len(samples),
                        metadata=json.dumps(metadata)
                        if environment_index % 2
                        else metadata,
                        prompt=f"fixed prompt for environment {environment_index}",
                        response=f"response {prompt_index}/{rollout_index}",
                        reward={"accuracy": float(rollout_index == 0), "reward": -0.5},
                        label=None,
                        status=SimpleNamespace(
                            value="truncated" if rollout_index == 7 else "completed"
                        ),
                        response_length=24,
                    )
                )
    return samples


class FakeTable:
    def __init__(self, columns, log_mode):
        self.columns = columns
        self.log_mode = log_mode
        self.data = []

    def add_data(self, *row):
        self.data.append(list(row))


class EvalRolloutsTest(unittest.TestCase):
    def test_selects_all_eight_rollouts_from_one_prompt_per_environment(self):
        samples = make_samples()
        random.Random(42).shuffle(samples)
        selected = eval_rollouts.select_eval_rollouts(samples, 8, "accuracy")
        self.assertEqual(len(samples), 3072)
        self.assertEqual(len(selected), 256)
        self.assertEqual(len({row["prompt_id"] for row in selected}), 32)
        for environment_index in range(32):
            rows = [
                row
                for row in selected
                if row["environment"] == f"environment_{environment_index}"
            ]
            self.assertEqual([row["rollout_index"] for row in rows], list(range(8)))
            self.assertEqual({row["problem_seed"] for row in rows}, {2000000})
            self.assertEqual(rows[0]["score"], 1.0)
            self.assertEqual(rows[0]["reference_answer"], "42")
            self.assertEqual(rows[7]["status"], "truncated")
            self.assertTrue(
                all(row["response"].startswith("response 0/") for row in rows)
            )

    def test_val_logs_only_one_prompt_and_identity_is_stable(self):
        samples = make_samples(environment_count=1, prompt_count=128)
        first = eval_rollouts.select_eval_rollouts(samples, 8, "accuracy")
        for sample in samples:
            sample.response = "changed response after training"
        second = eval_rollouts.select_eval_rollouts(
            list(reversed(samples)), 8, "accuracy"
        )
        self.assertEqual(len(first), 8)
        self.assertEqual(
            [row["prompt_id"] for row in first], [row["prompt_id"] for row in second]
        )
        self.assertNotEqual(first[0]["response"], second[0]["response"])

    def test_incomplete_group_is_rejected(self):
        samples = make_samples(environment_count=1, prompt_count=1)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            eval_rollouts.select_eval_rollouts(samples[:-1], 8, "accuracy")

    def test_tables_accumulate_across_evals_without_duplicating_per_env_groups(self):
        examples = eval_rollouts.select_eval_rollouts(make_samples(), 8, "accuracy")
        data = {
            "test_envs": {"rollout_examples": examples},
            "test_envs/environment_0": {"rewards": [1]},
        }
        tables = {}
        first = eval_rollouts.update_eval_rollout_tables(data, 0, tables, FakeTable)
        second = eval_rollouts.update_eval_rollout_tables(data, 9, tables, FakeTable)
        self.assertEqual(set(second), {"eval_rollouts/test_envs"})
        self.assertIs(
            first["eval_rollouts/test_envs"], second["eval_rollouts/test_envs"]
        )
        table = tables["test_envs"]
        self.assertEqual(table.log_mode, "INCREMENTAL")
        self.assertEqual(table.columns[0], "david/step")
        self.assertEqual(len(table.data), 512)
        self.assertEqual({row[0] for row in table.data[:256]}, {0})
        self.assertEqual({row[0] for row in table.data[256:]}, {9})


if __name__ == "__main__":
    unittest.main()
