import importlib.util
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("finalize_data_scaling", Path(__file__).parents[2] / "tools/finalize_data_scaling.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FinalizationRecoveryTest(unittest.TestCase):
    def checkpoint(self, root, iteration, log):
        model = root / "final-ckpt" / f"iter_{iteration:07d}"
        model.mkdir(parents=True)
        for name in ("common.pt", ".metadata", "__0_0.distcp"):
            (model / name).write_bytes(b"nonempty fixture; tensor validation belongs to the exporter")
        (root / "final-ckpt/latest_checkpointed_iteration.txt").write_text(str(iteration))
        (root / "train.log").write_text(log)

    def test_recovers_cap_and_validated_accuracy_stop(self):
        for iteration, log in ((499, "david 499: training metrics\n"),
                               (49, "david 49: training metrics\nEarly stop at update 50: reason=accuracy, validation=0.9961\n")):
            with self.subTest(iteration=iteration), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.checkpoint(root, iteration, log)
                self.assertEqual(module.validate_terminal_checkpoint(root, 500), iteration)

    def test_refuses_partial_or_incomplete_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkpoint(root, 49, "david 49: metrics\n")
            with self.assertRaises(RuntimeError):
                module.validate_terminal_checkpoint(root, 500)
            (root / "train.log").write_text("david 49: metrics\nEarly stop at update 50: reason=accuracy, validation=0.99\n")
            (root / "final-ckpt/iter_0000049/__0_0.distcp").unlink()
            with self.assertRaises(RuntimeError):
                module.validate_terminal_checkpoint(root, 500)
