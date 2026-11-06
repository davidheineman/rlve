"""Synchronous barriers, teacher lineage, work accounting, and OPD routing."""

import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path

from transfer_regimes.controller import execute, validate_budget
from transfer_regimes.routing import score_with_teachers
from transfer_regimes.selection import select_groups

from async_branching.io import read_json, write_json


def plan(algorithm="sync-mopd"):
    return {
        "algorithm": algorithm,
        "group": "example",
        "rounds": 2,
        "environments": ["a", "b", "c", "d"],
        "teacher_branch_rollouts": 4000,
        "student_round_rollouts": 4000,
        "total_training_rollout_cap": 40000,
        "teacher_validation_threshold": 0.98,
        "seed": 1234,
    }


class Backend:
    def __init__(self, *, success=False, fail_student_once=False):
        self.tasks, self.pending = [], []
        self.success, self.fail_student_once = success, fail_student_once

    def teacher(self, task):
        self.tasks.append(task)
        future = Future()
        future.set_result(
            {
                "hf": task["id"] + "/adapter",
                "resume": None,
                "rollouts": 4000,
                "accuracy": 0.99
                if self.success and task["environment_index"] == 0
                else 0.6,
            }
        )
        self.pending.append(future)
        return future

    def student(self, task):
        self.tasks.append(task)
        if task.get("evaluation_only"):
            return {**task["parent"], "rollouts": 0}
        if task["teacher_bank"]:
            assert all(f.done() for f in self.pending)
            assert set(task["teacher_bank"]) == {"a", "b", "c", "d"}
        if self.fail_student_once:
            self.fail_student_once = False
            raise RuntimeError("interrupted before student phase")
        return {
            "hf": task["id"] + "/adapter",
            "resume": task["id"] + "/state",
            "rollouts": task["cap"],
            "accuracy": 0,
        }


class ControllerTests(unittest.TestCase):
    def run_controller(self, settings=None, backend=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        write_json(
            directory / "base.json", {"hf": "base/adapter", "resume": "base/state"}
        )
        backend = backend or Backend()
        result = execute(settings or plan(), directory, backend)
        return directory, backend, result

    def test_two_rounds_barrier_lineage_and_complete_teacher_cost(self):
        directory, backend, result = self.run_controller()
        teachers = [t for t in backend.tasks if t["kind"] == "teacher"]
        students = [
            t
            for t in backend.tasks
            if t["kind"] == "student" and not t.get("evaluation_only")
        ]
        self.assertEqual(len(teachers), 8)
        self.assertEqual([t["generation"] for t in teachers], [1] * 4 + [2] * 4)
        self.assertTrue(all(t["parent"]["hf"] == "base/adapter" for t in teachers[:4]))
        self.assertTrue(
            all(t["parent"]["hf"] == "round-01-student/adapter" for t in teachers[4:])
        )
        self.assertEqual(students[1]["parent"]["resume"], "round-01-student/state")
        self.assertEqual(result["teacher_rollouts"], 32000)
        self.assertEqual(result["student_rollouts"], 8000)
        self.assertEqual(result["total_training_rollouts"], 40000)
        self.assertEqual(students[0]["start_work"], 16000)
        self.assertEqual(students[1]["start_work"], 36000)
        self.assertEqual(backend.tasks[-1]["id"], "final-evaluation")
        self.assertTrue((directory / "completed.json").exists())

    def test_joint_rl_is_one_continuous_four_task_policy_at_same_cap(self):
        _, backend, result = self.run_controller(plan("joint-rl"))
        self.assertFalse(any(t["kind"] == "teacher" for t in backend.tasks))
        training = [t for t in backend.tasks if not t.get("evaluation_only")]
        self.assertEqual(len(training), 1)
        self.assertEqual(training[0]["environments"], ["a", "b", "c", "d"])
        self.assertEqual(training[0]["cap"], 40000)
        self.assertEqual(result["total_training_rollouts"], 40000)

    def test_successful_specialist_is_retained_in_next_round(self):
        _, backend, result = self.run_controller(backend=Backend(success=True))
        self.assertEqual(sum(t["kind"] == "teacher" for t in backend.tasks), 7)
        self.assertEqual(result["total_training_rollouts"], 36000)
        second = next(t for t in backend.tasks if t["id"] == "round-02-student")
        self.assertEqual(
            second["teacher_bank"]["a"]["hf"], "round-01-teacher-00/adapter"
        )

    def test_resume_after_barrier_does_not_retrain_or_double_charge(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(
                directory / "base.json", {"hf": "base/adapter", "resume": "base/state"}
            )
            backend = Backend(fail_student_once=True)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                execute(plan(), directory, backend)
            state = read_json(directory / "sync-state.json")
            self.assertEqual(state["phase"], "student")
            self.assertEqual(state["total_training_rollouts"], 16000)
            result = execute(plan(), directory, backend)
            self.assertEqual(sum(t["kind"] == "teacher" for t in backend.tasks), 8)
            self.assertEqual(result["total_training_rollouts"], 40000)
            count = len(backend.tasks)
            execute(plan(), directory, backend)
            self.assertEqual(count, len(backend.tasks))

    def test_budget_rejects_student_only_accounting(self):
        settings = plan()
        settings["total_training_rollout_cap"] = 8000
        with self.assertRaises(ValueError):
            validate_budget(settings)


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_batch_uses_correct_teacher_and_preserves_order(self):
        rows = [(i, e) for i, e in enumerate(["b", "a", "d", "b", "c", "a", "c", "d"])]
        calls = []

        async def score(batch, completions, teacher):
            self.assertTrue(all(row[1] == teacher for row in batch))
            calls.append(teacher)
            return [
                (row[0], teacher, answer) for row, answer in zip(batch, completions)
            ]

        traces = await score_with_teachers(
            rows, list(range(8)), {e: e for e in "abcd"}, score
        )
        self.assertEqual([t[0] for t in traces], list(range(8)))
        self.assertEqual(sorted(calls), list("abcd"))

    async def test_missing_teacher_fails_before_any_scoring(self):
        async def forbidden(*args):
            self.fail("Must reject a missing teacher before scoring")

        with self.assertRaisesRegex(ValueError, "No teacher"):
            await score_with_teachers(
                [(0, "a"), (1, "b")], [0, 1], {"a": "teacher-a"}, forbidden
            )


class SelectionTests(unittest.TestCase):
    def test_earlier_quartets_and_unfiltered_random_controls_exclude_self(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            groups = [list(range(i, i + 4)) for i in range(0, 64, 4)]
            names = [f"e{i}" for i in range(64)]
            matrix = [[1.0 for _ in names] for _ in names]
            for i in range(64):
                matrix[i][i] = 99999
            for group in groups:
                for i in group:
                    for j in group:
                        if i != j:
                            matrix[i][j] = i // 4 - 8
            study = {
                "environments": names,
                "candidates": [{"size": 4, "fit": {"groups": groups}}],
            }
            write_json(root / "positive.json", study)
            write_json(root / "negative.json", study)
            write_json(
                root / "transfer.json", {"environments": names, "transfer": matrix}
            )
            selected = select_groups(
                root / "positive.json",
                root / "negative.json",
                root / "transfer.json",
                seed=1234,
            )
            self.assertEqual(selected["groups"][0]["indices"], groups[-1])
            self.assertEqual(selected["groups"][2]["indices"], groups[0])
            self.assertEqual(selected["groups"][0]["mean_transfer"], 7)
            self.assertEqual(selected["groups"][2]["mean_transfer"], -8)
            near_zero = select_groups(
                root / "positive.json",
                root / "negative.json",
                root / "transfer.json",
                controls="near-zero",
            )
            neutral = [
                r for r in near_zero["groups"] if r["label"].startswith("neutral")
            ]
            self.assertEqual(len(neutral), 2)
            self.assertEqual(neutral[0]["mean_transfer"], 0)
            self.assertEqual(abs(neutral[1]["mean_transfer"]), 1)
            self.assertEqual(near_zero["groups"][:4], selected["groups"][:4])
            before = [
                r["indices"]
                for r in selected["groups"]
                if r["label"].startswith("random")
            ]
            matrix = [[-value for value in row] for row in matrix]
            write_json(
                root / "transfer.json", {"environments": names, "transfer": matrix}
            )
            after = select_groups(
                root / "positive.json",
                root / "negative.json",
                root / "transfer.json",
                seed=1234,
            )
            self.assertEqual(
                before,
                [
                    r["indices"]
                    for r in after["groups"]
                    if r["label"].startswith("random")
                ],
            )


if __name__ == "__main__":
    unittest.main()
