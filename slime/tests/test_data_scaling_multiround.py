"""Verify selective teachers, learned initialization, and the round-2 job graph."""

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
import data_scaling_multiround as method
import submit_data_scaling_multiround as submit


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class MultiRoundTest(unittest.TestCase):
    def test_teacher_selection_uses_validation_of_the_exported_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            put_json(run / "weights_only_complete.json", {"format": "hf-safetensors", "checkpoint_step": 49})
            (run / "train.log").write_text(
                "eval 49: {'eval/val_envs': 0.99, 'eval/test_envs': 1.0}\n"
                "eval 49: {'eval/val_envs': 0.97, 'eval/test_envs': 1.0}\n")
            self.assertTrue(method.teacher_result(run)["retrain"])
            with (run / "train.log").open("a") as log:
                log.write("eval 49: {'eval/val_envs': 0.98, 'eval/test_envs': 0.0}\n")
            self.assertFalse(method.teacher_result(run)["retrain"])
            put_json(run / "weights_only_complete.json", {"format": "hf-safetensors", "checkpoint_step": 99})
            with self.assertRaises(ValueError):
                method.teacher_result(run)

    def test_warmstart_identity_and_cleanup_preserve_round1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "r2"
            hf = root / "r1/hf-final"
            put_json(hf / "model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
            put_json(hf.parent / "weights_only_complete.json", {"format": "hf-safetensors", "checkpoint_step": 1999})
            (hf.parent / "wandb_run_id.txt").write_text("round1-id\n")
            reference = experiment / "warmstart/round1-torch-dist"
            release = reference / "release"
            release.mkdir(parents=True)
            for name in ("common.pt", ".metadata", "__0_0.distcp"):
                (release / name).write_bytes(b"model-only test fixture")
            (reference / "latest_checkpointed_iteration.txt").write_text("release")
            runs = [{"kind": "teacher", "label": "00-env"}, {"kind": "mopd", "label": "n16-mopd"}]
            put_json(experiment / "run_plan.json", {"initial_hf": str(hf), "initial_ref": "warmstart/round1-torch-dist", "runs": runs})
            method.record_warmstart(experiment)
            method.check_warmstart(experiment)
            (reference / "latest_checkpointed_iteration.txt").write_text("1999")
            with self.assertRaises(RuntimeError):
                method.check_warmstart(experiment)
            (reference / "latest_checkpointed_iteration.txt").write_text("release")
            with self.assertRaises(RuntimeError):
                method.cleanup_warmstart(experiment)
            for run in runs:
                put_json(experiment / run["kind"] / run["label"] / "weights_only_complete.json", {})
            method.cleanup_warmstart(experiment)
            self.assertFalse(reference.exists())
            self.assertTrue((hf / "model.safetensors.index.json").exists())
            self.assertTrue((experiment / "initialization.json").exists())

    def test_only_failed_teachers_retrain_and_student_has_another_4k_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            parent = scratch / "runs/parent"
            original = ROOT / "data/DATA-SCALING-V1/run_plan.json"
            source = json.loads(original.read_text())
            put_json(parent / "run_plan.json", source)
            (parent / "input_sha256.txt").write_text(hashlib.sha256(original.read_bytes()).hexdigest() + "  " + str(original) + "\n")
            base = scratch / "models/DeepSeek-R1-Distill-Qwen-1.5B"
            for name in ("config.json", "tokenizer.json"):
                put_json(base / name, {})
            rows = []
            for index, run in enumerate(source["runs"][:16]):
                folder = parent / "teacher" / run["label"]
                put_json(folder / "weights_only_complete.json", {"format": "hf-safetensors", "checkpoint_step": 499,
                                                                 "tensor_count": 1, "weight_bytes": 1})
                accuracy = .98 if index in (6, 13) else .9
                (folder / "train.log").write_text(f"eval 499: {{'eval/val_envs': {accuracy}}}\n")
                for name in ("config.json", "tokenizer.json"):
                    put_json(folder / "hf-final" / name, {})
                put_json(folder / "hf-final/model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
                (folder / "hf-final/model.safetensors").write_bytes(b"x")
                rows.append(dict(kind="teacher", n=1, label=run["label"], job=2000 + index, gpus=1, dependency="none"))
            rows.append(dict(kind="mopd", n=16, label="n16-mopd", job=4321, gpus=4, dependency="none"))
            with (parent / "jobs.tsv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            calls = []

            def output(command, **kwargs):
                if command[0] == "squeue":
                    return "4321|RUNNING\n"
                self.assertEqual(command[0], "sbatch")
                calls.append(command)
                return str(9000 + len(calls)) + "\n"

            experiment = scratch / "runs/r2"
            with patch.object(sys, "argv", ["submit", "--parent-experiment", str(parent), "--experiment", str(experiment), "--submit"]), \
                 patch.dict(os.environ, {"WANDB_API_KEY": "test-only-key"}), \
                 patch.object(submit.shutil, "disk_usage", return_value=SimpleNamespace(free=500 * 1024**3)), \
                 patch.object(submit.subprocess, "check_output", side_effect=output), \
                 contextlib.redirect_stdout(io.StringIO()):
                submit.main()
            self.assertEqual(len(calls), 16)
            self.assertIn("--dependency=afterok:4321", calls[0])
            for call in calls:
                self.assertFalse(any(arg.startswith("--array") for arg in call))
                self.assertIn("--partition=" + submit.PARTITIONS, call)
                export = next(arg for arg in call if arg.startswith("--export="))
                self.assertIn(f"RLVE_HF_CKPT={parent}/mopd/n16-mopd/hf-final", export)
                self.assertIn(f"RLVE_REF_LOAD={experiment}/warmstart/round1-torch-dist", export)
            for call in calls[1:15]:
                self.assertIn("--gres=gpu:1", call)
                self.assertIn("--dependency=afterok:9001", call)
            self.assertIn("--gres=gpu:4", calls[-1])
            self.assertIn("--dependency=afterok:" + ":".join(str(i) for i in range(9002, 9016)), calls[-1])
            plan = json.loads((experiment / "run_plan.json").read_text())
            self.assertEqual(plan["method"], "multi-round")
            self.assertEqual(plan["runs"][-1]["updates"], 500)
            self.assertEqual(plan["runs"][-1]["training_rollouts"], 4000)
            self.assertEqual(plan["runs"][-1]["project"], "rl-data-scaling-students-v1")
            for run in plan["runs"][:16]:
                self.assertEqual(run["project"], "rl-data-scaling-multiround-teachers-v1")
                self.assertEqual(run["training_rollouts"], 4000)
            retained = [run for run in plan["runs"] if run["action"] == "retain"]
            self.assertEqual(len(retained), 2)
            self.assertTrue(all((experiment / "teacher" / run["label"]).is_symlink() for run in retained))


if __name__ == "__main__":
    unittest.main()
