"""Check experiment inputs and exercise submission using an isolated fake sbatch."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from data_scaling_plan import build_plan, validate


class DataScalingPlanTest(unittest.TestCase):
    def test_frozen_inputs_and_run_budgets(self):
        validate(build_plan())

    def test_standalone_submission_and_only_required_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake = temporary / "sbatch"
            fake.write_text("#!/usr/bin/python3\n"
                            "import json, os, pathlib, sys\n"
                            "path = pathlib.Path(os.environ['SCALING_FAKE_SBATCH_LOG'])\n"
                            "calls = json.loads(path.read_text()) if path.exists() else []\n"
                            "calls.append(sys.argv[1:])\n"
                            "path.write_text(json.dumps(calls))\n"
                            "print(9000 + len(calls))\n")
            fake.chmod(0o755)
            log = temporary / "calls.json"
            runs = temporary / "runs"
            runs.mkdir()
            result = subprocess.run(
                ["bash", "scripts/slurm/submit-data-scaling-v1.sh", "--submit"],
                cwd=ROOT, capture_output=True, text=True,
                env={**os.environ, "PATH": str(temporary) + os.pathsep + os.environ["PATH"],
                     "RLVE_RUNS_DIR": str(runs), "SCALING_FAKE_SBATCH_LOG": str(log)},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = json.loads(log.read_text())
            self.assertEqual(len(calls), 70)
            for command in calls:
                self.assertFalse(any(arg.startswith("--array") for arg in command))
            for command in calls[:3]:
                self.assertIn("--gres=gpu:4", command)
                self.assertFalse(any(arg.startswith("--dependency") for arg in command))
            for command in calls[3:67]:
                self.assertIn("--gres=gpu:1", command)
                self.assertFalse(any(arg.startswith("--dependency") for arg in command))
            teacher_ids = [str(9000 + index + 1) for index in range(3, 67)]
            for size, command in zip((16, 32, 64), calls[67:], strict=True):
                self.assertIn("--gres=gpu:4", command)
                self.assertIn("--dependency=afterok:" + ":".join(teacher_ids[:size]), command)


if __name__ == "__main__":
    unittest.main()
