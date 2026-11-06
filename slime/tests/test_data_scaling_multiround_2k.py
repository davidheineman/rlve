"""Check the length-only round-2 comparison and its isolated submission graph."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import submit_data_scaling_multiround_2k as variant
from slime.tests import test_data_scaling_multiround as original_tests


class MultiRound2kTest(unittest.TestCase):
    def test_only_training_generation_length_changes(self):
        variant.validate_configs()
        for baseline, longer in (
            ("configs/1gpu-data-scaling-teacher-v1.yaml", "configs/1gpu-data-scaling-multiround-teacher-2k-v1.yaml"),
            ("configs/data-scaling-multiround-mopd-v1.yaml", "configs/data-scaling-multiround-mopd-2k-v1.yaml"),
        ):
            before = yaml.safe_load((ROOT / baseline).read_text())
            after = yaml.safe_load((ROOT / longer).read_text())
            self.assertEqual({key for key in before if before[key] != after[key]}, {"rollout-max-response-len"})
            self.assertEqual(after["rollout-max-response-len"], 2048)
            self.assertEqual(after["eval-max-response-len"], 1024)
            self.assertEqual(after["answer-forcing-extra-tokens"], 128)
            self.assertEqual(after["early-stop-val-accuracy"], .98)

    def test_same_selective_teacher_and_student_pipeline_with_2k_configs(self):
        # Reuse the established model/export fixtures and fake Slurm assertions
        # for source initialization, 14 specialists, two retained teachers,
        # GPU allocations, dependencies, budgets, and project routing.
        captured = []
        build = variant.build_plan

        def capture(parent):
            plan = build(parent)
            captured.append(copy.deepcopy(plan))
            return plan

        with patch.object(original_tests, "submit", variant), \
             patch.object(variant, "build_plan", side_effect=capture):
            original_tests.MultiRoundTest.test_only_failed_teachers_retrain_and_student_has_another_4k_budget(self)
        plan = captured[0]
        self.assertEqual(plan["variant"], "gen-2k")
        self.assertEqual(plan["rollout_max_response_len"], 2048)
        self.assertEqual(plan["eval_max_response_len"], 1024)
        train = [run for run in plan["runs"] if run["action"] == "train"]
        self.assertEqual(len(train), 15)
        self.assertTrue(all("2k-v1.yaml" in run["config"] for run in train))
        self.assertTrue(all(run["initialization"] == "final-round1-mopd" for run in train))
        # SumLCM must be trained again even though it passed in the first
        # round-2 experiment; selection remains based on round-1 teachers.
        self.assertEqual(next(run for run in plan["runs"] if run["label"].startswith("02-"))["action"], "train")


if __name__ == "__main__":
    unittest.main()
