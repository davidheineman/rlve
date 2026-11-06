import importlib.util
import random
import unittest
from pathlib import Path


module_path = Path(__file__).resolve().parents[1] / "utils/rlve_difficulty.py"
spec = importlib.util.spec_from_file_location("rlve_difficulty", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BalancedDifficultySchedule = module.BalancedDifficultySchedule


class BalancedDifficultyScheduleTest(unittest.TestCase):
    def test_covers_every_pair_before_repeating(self):
        random.seed(17)
        schedule = BalancedDifficultySchedule(["CantorExpansion", "GoldWashing"], 2)
        first_cycle = []
        for _ in range(6):
            pair = schedule.peek()
            self.assertEqual(schedule.peek(), pair)
            first_cycle.append(pair)
            schedule.accept()
        self.assertEqual(set(first_cycle), {
            (environment, difficulty)
            for environment in ("CantorExpansion", "GoldWashing")
            for difficulty in range(3)
        })
        self.assertEqual(set(schedule.counts.values()), {1})
        for _ in range(6):
            schedule.peek()
            schedule.accept()
        self.assertEqual(set(schedule.counts.values()), {2})

    def test_restores_partial_cycle(self):
        random.seed(19)
        schedule = BalancedDifficultySchedule(["CantorExpansion", "GoldWashing"], 1)
        schedule.peek()
        schedule.accept()
        pending_pair = schedule.peek()
        restored = BalancedDifficultySchedule(
            ["CantorExpansion", "GoldWashing"], 1, schedule.get_state()
        )
        self.assertEqual(restored.peek(), pending_pair)
        restored.accept()
        self.assertEqual(restored.counts[pending_pair], 1)
        self.assertEqual(schedule.counts[pending_pair], 0)


if __name__ == "__main__":
    unittest.main()
