"""Prepare N16/N32/N64 rollout-clock simulations; does not submit GPU jobs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from async_branching.io import write_json

OUTPUT = ROOT / "data/DATA-SCALING-ASYNC-V1"
BASE = ROOT / "data/DATA-SCALING-V1"


def with_replay(value, percent):
    if percent not in (0, 10, 30, 50):
        raise ValueError("Prepared OPD replay percentages are 0, 10, 30 and 50")
    if percent:
        value["method"] = "async-branching-opd-replay-v1"
        value["opd_replay"] = {"fraction": percent / 100, "capacity_per_env": 400, "clip_epsilon": 0.2}
        value["student_config"] = f"configs/data-scaling-mopd-replay{percent}-v1.yaml"
        value["student_budget_unit"] = "Training trajectories: fresh plus replay; eight per update"
    return value


def plan(size, scratch, backend="local", physical_gpus=4, replay_percent=0):
    if size not in (16, 32, 64):
        raise ValueError("Prepared panels are N16, N32 and N64")
    source = json.loads((BASE / "manifest.json").read_text())
    for name, digest in source["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source dataset changed: {name}")
    names = source["environments"][:size]
    model = Path(scratch) / "models/DeepSeek-R1-Distill-Qwen-1.5B"
    value = {
        "method": "async-branching-v1", "size": size,
        "physical_gpus": 4, "simulated_teacher_devices": size, "simulated_student_devices": 1,
        "environments": names, "base_hf": str(model), "base_ref": str(model) + "_torch_dist",
        "parent_policy": "latest-available", "student_rollouts": 16000,
        "teacher_branch_rollouts": 4000, "teacher_validation_threshold": .98,
        "student_rollouts_per_teacher_version": 1000, "teacher_recovery_fraction": .9,
        "student_quantum_rollouts": 400, "teacher_device_rollout_horizon": None,
        "teacher_config": "configs/1gpu-data-scaling-teacher-v1.yaml",
        "student_config": "configs/data-scaling-mopd-v1.yaml",
        "teacher_eval": {name: f"data/DATA-SCALING-V1/{name}_d01.json" for name in names},
        "id_eval": f"data/DATA-SCALING-V1/id_n{size}_d01.json",
        "test_eval": "data/DATA-SCALING-V1/test_32env_d01.json",
        "eval_prompts": 32, "eval_samples_per_prompt": 8, "eval_interval_updates": 50,
        "teacher_project": "rl-data-scaling-async-teachers-v1",
        "student_project": "rl-data-scaling-students-v1",
        "teacher_batch": {"prompts": 1, "samples_per_prompt": 8},
        "student_batch": {"prompts": 8, "samples_per_prompt": 1},
        "maximum_initial_response_tokens": 1024, "answer_forcing_extra_tokens": 128,
        "physical_layout": "student segments: 1 colocated actor/rollout GPU + 1 scorer GPU; remaining GPUs train specialists; all four train specialists while the student waits",
        "clock_definition": "Training rollout counts exclude validation. Availability clocks also include idle time. Teacher rebranch parent is the highest currently available student checkpoint count <= that teacher device's cumulative training rollouts.",
    }
    if backend == "slurm":
        if not 4 <= physical_gpus <= 16:
            raise ValueError("Elastic GPU cap must be between 4 and 16")
        value.update(backend="slurm", physical_gpus=physical_gpus,
                     physical_layout="CPU coordinator; independent one-GPU teacher jobs; persistent two-GPU student service; teacher submission window capped at max_gpus - 2, including pending jobs")
    elif backend != "local" or physical_gpus != 4:
        raise ValueError("Local backend expects four GPUs")
    return with_replay(value, replay_percent)


def validate(value):
    assert (4 <= value["physical_gpus"] <= 16 if value.get("backend") == "slurm"
            else value["physical_gpus"] == 4)
    assert value["simulated_teacher_devices"] == len(value["environments"]) == value["size"]
    assert value["simulated_student_devices"] == 1
    assert value["parent_policy"] == "latest-available"
    assert value["student_rollouts"] == 16000
    assert value["teacher_branch_rollouts"] == 4000
    assert value["student_rollouts_per_teacher_version"] == 1000
    assert value["teacher_recovery_fraction"] == .9
    for role, prompts, samples in (("teacher", "1", "8"), ("student", "8", "1")):
        scalars = {}
        for line in (ROOT / value[f"{role}_config"]).read_text().splitlines():
            key, separator, scalar = line.partition(":")
            if separator:
                scalars[key] = scalar.strip()
        for key, expected in {"rollout-batch-size": prompts, "n-samples-per-prompt": samples,
                              "rollout-max-response-len": "1024", "answer-forcing": "true",
                              "answer-forcing-extra-tokens": "128", "eval-interval": "50",
                              "n-samples-per-eval-prompt": "8", "rlve-balanced-difficulty": "true"}.items():
            assert scalars[key] == expected, (role, key)
        if role == "student":
            assert scalars["custom-loss-function-path"] == "slime.backends.megatron_utils.opd_loss.opd_loss_function"
            if value.get("opd_replay"):
                replay = value["opd_replay"]
                assert replay["fraction"] in (.1, .3, .5)
                assert replay["capacity_per_env"] == int(scalars["opd-replay-capacity-per-env"]) > 0
                assert replay["fraction"] == float(scalars["opd-replay-fraction"])
                assert replay["clip_epsilon"] == float(scalars["opd-replay-clip-epsilon"])
        else:
            assert scalars["advantage-estimator"] == "grpo"
    assert len(json.loads((ROOT / value["id_eval"]).read_text())) == 32
    assert len(json.loads((ROOT / value["test_eval"]).read_text())) == 32
    for path in value["teacher_eval"].values():
        assert len(json.loads((ROOT / path).read_text())) == 32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("/nlp/scr") / __import__("getpass").getuser())
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--replay-percent", type=int, choices=(0, 10, 30, 50), default=0)
    args = parser.parse_args()
    for size in (16, 32, 64):
        for backend, cap in (("local", 4), ("slurm", 16)):
            value = plan(size, args.scratch.resolve(), backend, cap, args.replay_percent)
            validate(value)
            suffix = f"-replay{args.replay_percent}" if args.replay_percent else ""
            path = OUTPUT / f"n{size}{'-elastic' if backend == 'slurm' else ''}{suffix}.json"
            if args.check_only:
                assert json.loads(path.read_text()) == value, path
            else:
                write_json(path, value)
            print(f"Validated N{size}: {backend} backend; GPU cap={cap}; 16k student rollout cap")


if __name__ == "__main__":
    main()
