"""Explicit tiny-domain exceptions never admit student validation/test prompts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_async_lora_questions import prepare


class QuestionPreparationTest(unittest.TestCase):
    def test_teacher_only_overlap_exception_preserves_student_holdout_exclusion(self):
        for allow in (False, True):
            with self.subTest(allow=allow), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "data/DATA-SCALING-ASYNC-V1").mkdir(parents=True)
                protocol = {
                    "environments": ["Tiny"],
                    "teacher_eval": {"Tiny": "teacher.json"},
                    "id_eval": "id.json",
                    "test_eval": "test.json",
                }
                (root / "data/DATA-SCALING-ASYNC-V1/n32.json").write_text(json.dumps(protocol))
                for name, prompt in (("teacher", "teacher-only"), ("id", "student-val"), ("test", "student-test")):
                    (root / f"{name}.json").write_text(json.dumps([{"user_prompt": prompt}]))

                def generate(_environment, _difficulty, _count, _seed, _param_seed, *, seen_prompts, allow=allow):
                    self.assertIn("student-val", seen_prompts)
                    self.assertIn("student-test", seen_prompts)
                    self.assertEqual("teacher-only" in seen_prompts, not allow)
                    return [
                        {
                            "user_prompt": "teacher-only" if allow else "unseen",
                            "metadata": json.dumps({"environment": "Tiny", "config": {"seed": 1}}),
                        }
                    ]

                with (
                    patch("prepare_async_lora_questions.ROOT", root),
                    patch("prepare_async_lora_questions._records_for_environment", side_effect=generate),
                ):
                    result = prepare(32, allow_teacher_eval_overlap_for=["Tiny"] if allow else [])
                self.assertEqual(bool(result.get("teacher_eval_overlap")), allow)
                if allow:
                    self.assertFalse(result["teacher_eval_overlap"]["Tiny"]["student_validation_and_test_overlap"])

    def test_unknown_overlap_exception_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare(32, allow_teacher_eval_overlap_for=["not-in-the-panel"])


if __name__ == "__main__":
    unittest.main()
