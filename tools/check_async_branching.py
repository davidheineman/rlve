"""CPU-only validation of actual Slime CLI arguments for all three worker modes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import argparse
import ast
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_scheduler_cli(task):
    """Check scheduler flags against installed Megatron without importing CUDA."""
    source = Path("/root/Megatron-LM/megatron/training/arguments.py")
    try:
        source_text = source.read_text()
    except (FileNotFoundError, PermissionError):
        return False
    from async_branching.segment import worker_arguments
    tree = ast.parse(source_text)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_add_learning_rate_args")
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    parser = argparse.ArgumentParser()
    namespace[function.name](parser)
    flags = [flag for flag in worker_arguments(task) if flag.startswith("--use-checkpoint-opt")]
    parsed = parser.parse_args(flags)
    assert parsed.use_checkpoint_opt_param_scheduler == (task["kind"] == "student" and task["start_rollouts"] > 0)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="Also import Slime/SGLang; requires a CUDA-driver-equipped node even without training")
    parser.add_argument("--plan", type=Path, default=ROOT / "data/DATA-SCALING-ASYNC-V1/n16.json")
    options = parser.parse_args()
    from async_branching.segment import configure_args, worker_arguments
    os.chdir(ROOT)
    plan = json.loads(options.plan.read_text())
    result = subprocess.check_output(["bash", "-c", 'source scripts/models/deepseek-r1-distill-qwen-1.5B.sh; printf "%s\\0" "${MODEL_ARGS[@]}"'])
    model_args = result.decode().split("\0")[:-1]
    with tempfile.TemporaryDirectory(prefix="async-slime-parse-") as directory:
        d = Path(directory)
        cases = (("teacher", 0, 4000, .45), ("student", 0, 400, .45),
                 ("student", 400, 400, .45), ("student", 1000, 200, .45),
                 ("student", 1200, 400, 0.0))
        for kind, start, cap, target in cases:
            config = yaml.safe_load((ROOT / plan[f"{kind}_config"]).read_text())
            flags = []
            for key, value in config.items():
                if value is False or value is None:
                    continue
                flags.append("--" + key)
                if value is not True:
                    flags.extend(map(str, value if isinstance(value, list) else [value]))
            task = {"kind": kind, "start_rollouts": start, "cap": cap,
                    "checkpoint": str(d / "save"), "target": target, "train_seed": 1234,
                    "parent": {"resume": str(d / "resume")}}
            if kind == "student" and plan.get("opd_replay"):
                task["opd_replay"] = plan["opd_replay"]
            if validate_scheduler_cli(task):
                print(f"Installed Megatron scheduler CLI passed: {kind}, start={start}")
            if not options.runtime:
                fields = {k.replace("-", "_"): v for k, v in config.items()}
                fields["transactional_checkpoints"] = True
                fields.setdefault("opd_teacher_gpus", 0)
                if kind == "student":
                    fields.update(actor_num_gpus_per_node=1, num_gpus_per_node=1, opd_teacher_gpus=1)
                checked, update, count = configure_args(SimpleNamespace(**fields), task)
                assert checked.actor_num_gpus_per_node == 1
                assert checked.opd_teacher_gpus == (1 if kind == "student" else 0)
                assert checked.no_load_optim == (not (kind == "student" and start > 0))
                assert checked.no_save_optim == (kind != "student")
                assert checked.transactional_checkpoints
                assert checked.opd_replay_fraction == (plan.get("opd_replay", {}).get("fraction", 0)
                                                       if kind == "student" else 0)
                arguments = worker_arguments(task)
                assert 0 < float(arguments[arguments.index("--early-stop-val-accuracy") + 1]) <= 1
                assert checked.early_stop_val_accuracy == (target if kind == "student" else .98)
                print(f"Static configuration passed: {kind}, start={start}, updates={count}, "
                      f"optimizer_resume={not checked.no_load_optim}, target={checked.early_stop_val_accuracy}")
                continue
            (d / "resume").mkdir(exist_ok=True)
            (d / "resume/latest_checkpointed_iteration.txt").write_text(f"{max(0, start // 8 - 1)}\n")
            path = d / "task.json"
            path.write_text(json.dumps(task))
            bank = d / "bank"
            bank.mkdir(exist_ok=True)
            evaluation = ["val_envs", plan["id_eval"] if kind == "student" else plan["teacher_eval"][plan["environments"][0]]]
            extra = []
            if kind == "student":
                evaluation.extend(["test_envs", plan["test_eval"], "gate_envs", plan["teacher_eval"][plan["environments"][0]]])
                extra = ["--actor-num-gpus-per-node", "1", "--num-gpus-per-node", "1", "--opd-teacher-gpus", "1",
                         "--opd-teacher-dir", str(bank), "--opd-teacher-cache-size", "1"]
                if start:
                    extra.extend(["--load", str(d / "resume")])
            command = [sys.executable, "-m", "async_branching.segment", *model_args, *flags,
                       "--hf-checkpoint", plan["base_hf"], "--ref-load", plan["base_ref"],
                       *worker_arguments(task),
                       "--environment-list", plan["environments"][0], "--eval-prompt-data", *evaluation, *extra]
            proc = subprocess.run(command, capture_output=True, text=True,
                                  env={**os.environ, "RLVE_ASYNC_TASK": str(path), "RLVE_ASYNC_PARSE_ONLY": "1"})
            if proc.returncode:
                raise RuntimeError(proc.stdout[-3000:] + proc.stderr[-6000:])
            print(proc.stdout.splitlines()[-1])


if __name__ == "__main__":
    main()
