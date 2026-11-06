import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "early_stopping", Path(__file__).parents[1] / "utils/early_stopping.py"
)
early_stopping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(early_stopping)
update_plateau = early_stopping.update_plateau


class PlateauTest(unittest.TestCase):
    def test_improvement_resets_patience_and_small_noise_does_not(self):
        self.assertEqual(update_plateau(None, 0, 0.5, 0.01), (0.5, 0, True))
        self.assertEqual(update_plateau(0.5, 0, 0.505, 0.01), (0.5, 1, False))
        self.assertEqual(update_plateau(0.5, 1, 0.52, 0.01), (0.52, 0, True))


if __name__ == "__main__":
    unittest.main()
