import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


spec = spec_from_file_location("opd_teacher_marker", Path(__file__).parents[1] / "utils" / "opd_teacher_marker.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)
validate_teacher_marker = module.validate_teacher_marker


class TeacherMarkerTest(unittest.TestCase):
    def test_v3_final_checkpoint_is_valid(self):
        marker = {
            "checkpoint_step": 499,
            "format": "hf-safetensors",
            "path": "hf-final",
            "weight_bytes": 3554214720,
        }
        validate_teacher_marker(marker, "Circuit", "/teacher/hf-final")

    def test_incomplete_export_is_rejected(self):
        marker = {
            "checkpoint_step": -1,
            "format": "hf-safetensors",
            "path": "hf-final",
            "weight_bytes": 3554214720,
        }
        with self.assertRaises(RuntimeError):
            validate_teacher_marker(marker, "Circuit", "/teacher/hf-final")


if __name__ == "__main__":
    unittest.main()
