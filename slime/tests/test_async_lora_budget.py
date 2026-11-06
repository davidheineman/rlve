"""A budget extension keeps the causal student/teacher continuation past the old cap."""

import copy
import unittest

from async_branching.schedule import Simulation, Snapshot, State, Version, initial_state
from tools.extend_async_lora_budget import extended_state


class AsyncLoRABudgetTest(unittest.TestCase):
    def state_at_old_cap(self):
        state = initial_state(["A"], "base", "base-optimizer")
        state.student_rollouts, state.clock = 16000, 20000
        state.snapshots.append(Snapshot(16000, 20000, "student-16000", "optimizer-16000"))
        state.devices[0].rollouts = 16000
        state.versions.append(Version("A", 1, 20000, 4000, 0.8, "teacher"))
        return state

    def test_student_keeps_training_past_16000_with_original_checkpoint(self):
        state = self.state_at_old_cap()
        self.assertIsNone(Simulation(state).student_task())
        self.assertEqual(state.stopped, "student_budget")
        resumed = Simulation(State.from_dict(extended_state(state.to_dict(), 32000)))
        task = resumed.student_task()
        self.assertEqual(task["start_rollouts"], 16000)
        self.assertEqual(task["cap"], 400)
        self.assertEqual(task["parent"]["resume"], "optimizer-16000")
        resumed.merge(task, {"rollouts": 400, "accuracy": 0.8, "hf": "student-16400", "resume": "optimizer-16400"})
        self.assertEqual(resumed.state.student_rollouts, 16400)
        self.assertEqual(resumed.state.budget, 32000)

    def test_teachers_branch_after_the_original_student_cap(self):
        state = self.state_at_old_cap()
        self.assertEqual(Simulation(state).branches(), [])
        resumed = Simulation(State.from_dict(extended_state(state.to_dict(), 32000)))
        task = resumed.branches()[0]
        self.assertEqual(task["parent"]["hf"], "student-16000")
        self.assertEqual(task["teacher_start_rollouts"], 16000)
        self.assertEqual(task["cap"], 4000)

    def test_extension_changes_only_budget_and_preserves_original_state(self):
        before = self.state_at_old_cap().to_dict()
        original = copy.deepcopy(before)
        after = extended_state(before, 32000)
        self.assertEqual(before, original)
        self.assertEqual(
            {k: v for k, v in after.items() if k != "budget"}, {k: v for k, v in before.items() if k != "budget"}
        )
        self.assertIsNot(after["snapshots"], before["snapshots"])

    def test_invalid_budget_or_unexpected_stop_reason_is_rejected(self):
        before = self.state_at_old_cap().to_dict()
        for budget in (16000, 8000, 32001, -8, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                extended_state(before, budget)
        before["stopped"] = "unknown_failure"
        with self.assertRaises(ValueError):
            extended_state(before, 32000)


if __name__ == "__main__":
    unittest.main()
