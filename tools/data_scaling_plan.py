"""Build and validate the reviewable 64-teacher / six-hero scaling plan."""

import json

import yaml

from prepare_data_scaling import OUTPUT, ROOT, SIZES, check, freeze


def build_plan():
    names = json.loads((OUTPUT / "manifest.json").read_text())["environments"]
    runs = [{
        "label": f"{index:02d}-{name}", "kind": "teacher", "size": 1,
        "environments": [name], "config": "configs/1gpu-data-scaling-teacher-v1.yaml",
        "eval": f"data/DATA-SCALING-V1/{name}_d01.json", "test_eval": "", "gpus": 1,
        "memory_gib": 96,
        "teacher_gpus": 0, "updates": 500, "training_rollouts": 4000,
        "project": "rl-data-scaling-teachers-v1", "constraint": "ada|48G|80G|141G",
    } for index, name in enumerate(names)]
    for size in SIZES:
        for kind in ("mopd", "rl"):
            teacher_gpus = 2 if kind == "mopd" else 0
            runs.append({
                "label": f"n{size}-{kind}", "kind": kind, "size": size,
                "environments": names[:size], "config": f"configs/data-scaling-{kind}-v1.yaml",
                "eval": f"data/DATA-SCALING-V1/id_n{size}_d01.json",
                "test_eval": "data/DATA-SCALING-V1/test_32env_d01.json",
                "gpus": 4,
                "memory_gib": ({16: 192, 32: 256, 64: 384}[size] if kind == "mopd" else 160),
                "teacher_gpus": teacher_gpus, "updates": 2000 if kind == "mopd" else 500,
                "training_rollouts": 16000 if kind == "mopd" else 64000,
                "project": "rl-data-scaling-students-v1",
                "constraint": "ada|48G|80G|141G",
            })
    return {"runs": runs,
            "scheduling": "70 standalone jobs, no arrays or concurrency caps; joint RL starts independently; each MOPD depends only on its matching N teachers",
            "initialization": "All actors freshly initialized from base DeepSeek-R1-Distill-Qwen-1.5B; no previously trained checkpoint reuse"}


def validate(plan):
    check()
    assert plan == build_plan()
    assert len(plan["runs"]) == 70
    for run in plan["runs"]:
        config = yaml.safe_load((ROOT / run["config"]).read_text())
        assert config["num-rollout"] == run["updates"]
        assert config["rollout-batch-size"] * config["n-samples-per-prompt"] * run["updates"] == run["training_rollouts"]
        assert config["initial-difficulty"] == 1 and config["difficulty-sliding-window-size"] == 2
        assert config["rlve-balanced-difficulty"] and config["min-metric-to-increase-difficulty"] > 1
        assert config["answer-forcing"] and config["answer-forcing-extra-tokens"] == 128
        assert config["rollout-max-response-len"] == config["eval-max-response-len"] == 1024
        assert config["early-stop-val-accuracy"] == .98 and config["eval-on-last-rollout"]
        assert config["early-stop-eval-names"] == ["val_envs"]
        assert config["eval-interval"] == 50 and config["n-samples-per-eval-prompt"] == 8
        assert not config["log-eval-rollouts"] and config["no-save-optim"] and config["no-save-rng"]
        for key in ("save-debug-rollout-data", "save-debug-train-data", "dump-details", "save-rollouts-jsonl", "rlve-final-ood-eval-path"):
            assert not config.get(key), key
        assert len(json.loads((ROOT / run["eval"]).read_text())) == 32
        if run["kind"] != "teacher":
            assert len(json.loads((ROOT / run["test_eval"]).read_text())) == 32
        if run["kind"] == "mopd":
            assert config["opd-teacher-cache-size"] == 8 and run["teacher_gpus"] == 2
    heroes = plan["runs"][64:]
    assert {run["gpus"] for run in heroes} == {4}
    print("Verified 64 fresh 1-GPU teachers, six 4-GPU heroes, 32+32 hero evals, 16k/64k budgets")


if __name__ == "__main__":
    plan = build_plan()
    validate(plan)
    freeze(OUTPUT / "run_plan.json", plan)
