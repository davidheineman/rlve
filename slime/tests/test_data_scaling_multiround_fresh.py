"""Exercise the complete fresh 4k + 4k job graph with isolated fake Slurm."""

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
import submit_data_scaling_multiround as round2
import submit_data_scaling_multiround_fresh as fresh


def put_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class FreshMultiRoundTest(unittest.TestCase):
    def test_new_round1_is_fresh_4k_and_is_the_only_round2_model_source(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            parent = scratch / "runs/original"
            original = ROOT / "data/DATA-SCALING-V1/run_plan.json"
            source = json.loads(original.read_text())
            put_json(parent / "run_plan.json", source)
            (parent / "input_sha256.txt").write_text(hashlib.sha256(original.read_bytes()).hexdigest() + "  " + str(original) + "\n")
            base = scratch / "models/DeepSeek-R1-Distill-Qwen-1.5B"
            for name in ("config.json", "tokenizer.json"):
                put_json(base / name, {})
            reference = base.with_name(base.name + "_torch_dist")
            reference.mkdir()
            (reference / "latest_checkpointed_iteration.txt").write_text("release")
            rows = []
            for index, spec in enumerate(source["runs"][:16]):
                folder = parent / "teacher" / spec["label"]
                put_json(folder / "weights_only_complete.json", {"format": "hf-safetensors", "checkpoint_step": 499,
                                                                 "tensor_count": 1, "weight_bytes": 1})
                accuracy = .99 if index in (6, 13) else .9
                (folder / "train.log").write_text(f"eval 499: {{'eval/val_envs': {accuracy}}}\n")
                for name in ("config.json", "tokenizer.json"):
                    put_json(folder / "hf-final" / name, {})
                put_json(folder / "hf-final/model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
                (folder / "hf-final/model.safetensors").write_bytes(b"x")
                rows.append(dict(kind="teacher", n=1, label=spec["label"], job=2000 + index, gpus=1, dependency="none"))
            rows.append(dict(kind="mopd", n=16, label="n16-mopd", job=4321, gpus=4, dependency="none"))
            with (parent / "jobs.tsv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            calls = []
            real_run = fresh.subprocess.run

            def output(command, **kwargs):
                if command[0] == "squeue":
                    self.assertEqual(command[command.index('-j') + 1], "9001")
                    return "9001|PENDING\n"
                self.assertEqual(command[0], "sbatch")
                calls.append(command)
                return str(9000 + len(calls)) + "\n"

            def run(command, **kwargs):
                if len(command) > 1 and command[1].endswith("submit_data_scaling_multiround.py"):
                    with patch.object(sys, "argv", command[1:]):
                        round2.main()
                    return fresh.subprocess.CompletedProcess(command, 0)
                return real_run(command, **kwargs)

            r1, r2 = scratch / "runs/new-r1", scratch / "runs/new-r2"
            with patch.object(sys, "argv", ["submit", "--parent-experiment", str(parent), "--experiment", str(r1),
                                             "--round2-experiment", str(r2), "--submit"]), \
                 patch.dict(os.environ, {"WANDB_API_KEY": "test-only-key"}), \
                 patch.object(fresh.shutil, "disk_usage", return_value=SimpleNamespace(free=500 * 1024**3)), \
                 patch.object(fresh.subprocess, "check_output", side_effect=output), \
                 patch.object(fresh.subprocess, "run", side_effect=run), \
                 contextlib.redirect_stdout(io.StringIO()):
                fresh.main()
            self.assertEqual(len(calls), 17)
            self.assertIn("--gres=gpu:4", calls[0])
            self.assertFalse(any(arg.startswith("--dependency") for arg in calls[0]))
            first_export = next(arg for arg in calls[0] if arg.startswith("--export="))
            self.assertIn(f"RLVE_HF_CKPT={base}", first_export)
            self.assertIn(f"RLVE_REF_LOAD={reference}", first_export)
            self.assertIn("--dependency=afterok:9001", calls[1])
            for call in calls[2:16]:
                self.assertIn("--gres=gpu:1", call)
                self.assertIn("--dependency=afterok:9002", call)
            self.assertIn("--gres=gpu:4", calls[-1])
            self.assertIn("--dependency=afterok:" + ":".join(str(i) for i in range(9003, 9017)), calls[-1])
            for call in calls[1:]:
                export = next(arg for arg in call if arg.startswith("--export="))
                self.assertIn(f"RLVE_HF_CKPT={r1}/mopd/n16-mopd/hf-final", export)
                self.assertNotIn(f"RLVE_HF_CKPT={parent}/mopd/n16-mopd/hf-final", export)
                self.assertFalse(any("afterok:4321" in arg for arg in call))
            for path in (r1, r2):
                plan = json.loads((path / "run_plan.json").read_text())
                self.assertEqual(plan["runs"][-1]["updates"], 500)
                self.assertEqual(plan["runs"][-1]["training_rollouts"], 4000)
                self.assertEqual(plan["runs"][-1]["project"], "rl-data-scaling-students-v1")
                self.assertEqual(plan["runs"][-1]["environments"], source["runs"][64]["environments"])
            self.assertEqual(json.loads((r2 / "run_plan.json").read_text())["parent_experiment"], str(r1))


if __name__ == "__main__":
    unittest.main()
