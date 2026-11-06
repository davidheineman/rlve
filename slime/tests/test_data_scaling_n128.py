"""Check N128 nesting and exercise submission without contacting Slurm."""

import contextlib
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import submit_data_scaling_n128 as submit
from prepare_data_scaling_n128 import OUTPUT, build_plan


class DataScalingN128Test(unittest.TestCase):
    def test_nested_panel_and_original_runs_are_preserved(self):
        manifest = json.loads((OUTPUT / "manifest.json").read_text())
        plan = build_plan(manifest["environments"])
        original = json.loads((ROOT / "data/DATA-SCALING-V1/run_plan.json").read_text())
        self.assertEqual(plan["runs"][:64], original["runs"][:64])
        self.assertEqual(len(plan["runs"]), 130)
        self.assertEqual(len(set(manifest["environments"])), 128)
        self.assertFalse(set(manifest["environments"]) & set(manifest["test_environments"]))
        for run in plan["runs"][64:128]:
            self.assertEqual((run["gpus"], run["updates"], run["training_rollouts"]), (1, 500, 4000))
        rl, mopd = plan["runs"][128:]
        self.assertEqual((rl["gpus"], rl["updates"], rl["training_rollouts"]), (4, 500, 64000))
        self.assertEqual((mopd["gpus"], mopd["updates"], mopd["training_rollouts"]), (4, 2000, 16000))
        self.assertEqual(rl["test_eval"], original["runs"][65]["test_eval"])

    def test_66_standalone_jobs_and_only_teacher_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            parent = scratch / "runs/parent"
            parent.mkdir(parents=True)
            model = scratch / "models/DeepSeek-R1-Distill-Qwen-1.5B"
            model.mkdir(parents=True)
            for name in ("config.json", "tokenizer.json"):
                (model / name).write_text("{}")
            reference = model.with_name(model.name + "_torch_dist")
            reference.mkdir()
            (reference / "latest_checkpointed_iteration.txt").write_text("0\n")
            original = ROOT / "data/DATA-SCALING-V1/run_plan.json"
            (parent / "run_plan.json").write_bytes(original.read_bytes())
            (parent / "input_sha256.txt").write_text(hashlib.sha256(original.read_bytes()).hexdigest() + "  " + str(original) + "\n")
            teachers = json.loads(original.read_text())["runs"][:64]
            with (parent / "jobs.tsv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=["kind", "n", "label", "job", "gpus", "dependency"], delimiter="\t")
                writer.writeheader()
                for index, spec in enumerate(teachers):
                    writer.writerow(dict(kind="teacher", n=1, label=spec["label"], job=1000 + index, gpus=1, dependency="none"))
            commands = []

            def output(command, **kwargs):
                if command[0] == "squeue":
                    return "".join(f"{1000 + i}|RUNNING\n" for i in range(64))
                self.assertEqual(command[0], "sbatch")
                commands.append(command)
                return str(9000 + len(commands)) + "\n"

            experiment = scratch / "runs/extension"
            with patch.object(sys, "argv", ["submit", "--parent-experiment", str(parent), "--experiment", str(experiment), "--submit"]), \
                 patch.dict(os.environ, {"WANDB_API_KEY": "test-only-key"}), \
                 patch.object(submit.shutil, "disk_usage", return_value=SimpleNamespace(free=1000 * 1024**3)), \
                 patch.object(submit.subprocess, "check_output", side_effect=output), \
                 contextlib.redirect_stdout(io.StringIO()):
                submit.main()
            self.assertEqual(len(commands), 66)
            for command in commands:
                self.assertFalse(any(arg.startswith("--array") for arg in command))
                self.assertIn("--partition=" + submit.PARTITIONS, command)
            self.assertIn("--gres=gpu:4", commands[0])
            for command in commands[:65]:
                self.assertFalse(any(arg.startswith("--dependency") for arg in command))
            for command in commands[1:65]:
                self.assertIn("--gres=gpu:1", command)
            expected = [str(1000 + i) for i in range(64)] + [str(9002 + i) for i in range(64)]
            self.assertIn("--dependency=afterok:" + ":".join(expected), commands[-1])
            self.assertIn("--gres=gpu:4", commands[-1])
            self.assertIn("--mem=512G", commands[-1])
            with (experiment / "jobs.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 130)
            self.assertEqual(sum(row["phase"] == "shared" for row in rows), 64)
            self.assertTrue(all((experiment / "teacher" / spec["label"]).is_symlink() for spec in teachers))


if __name__ == "__main__":
    unittest.main()
