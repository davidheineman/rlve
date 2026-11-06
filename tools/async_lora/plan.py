"""Derive the LoRA experiment from the existing async protocol and hero panels."""

import json

import yaml
from lora_teachers.plan import ROOT, build_plan, sha256
from prepare_async_branching import with_replay

from async_lora.questions import load_questions


def apply_speed_profile(plan, profile):
    if profile not in ("original", "v2", "v3"):
        raise ValueError(f"Unknown speed profile: {profile}")
    plan["speed_profile"] = profile
    plan["evaluation_concurrency"] = 2 if profile == "original" else 8
    if profile in ("v2", "v3"):
        plan.update(train_gpus=1 if profile == "v2" else 2, sampling_gpus=3 if profile == "v2" else 2)
        for key, value in {"eval-interval": 100, "n-samples-per-eval-prompt": 4}.items():
            plan["teacher_config_overrides"][key] = {"source": plan["teacher_config"][key], "value": value}
            plan["teacher_config"][key] = value
    placement = (
        f"{plan['train_gpus']} training DP rank{'s' if plan['train_gpus'] != 1 else ''} "
        f"and {plan['sampling_gpus']} TP1 sampling engines; "
        "one student and N teacher slots"
    )
    plan["physical_layout"] = placement
    plan["execution_changes"]["placement"] = placement


def build_async_plan(
    size=16,
    *,
    learning_rate=2e-5,
    single_question=False,
    replay_percent=0,
    question_data=None,
    speed_profile="original",
    flex_gpus=0,
):
    if size not in (16, 32):
        raise ValueError("Supported original panels are N16 and N32")
    plan = build_plan(project="rl-data-scaling-async-lora-teachers-v1", learning_rate=learning_rate)
    protocol_path = ROOT / f"data/DATA-SCALING-ASYNC-V1/n{size}.json"
    protocol = with_replay(json.loads(protocol_path.read_text()), replay_percent)
    student_path = ROOT / protocol["student_config"]
    student = yaml.safe_load(student_path.read_text())
    student["lr"] = learning_rate
    manifest = json.loads((ROOT / "data/DATA-SCALING-V1/manifest.json").read_text())
    original = json.loads((ROOT / "data/DATA-SCALING-V1/run_plan.json").read_text())["runs"]
    hero = next(r for r in original if r["label"] == f"n{size}-mopd")
    if protocol["environments"] != hero["environments"] or protocol["id_eval"] != hero["eval"]:
        raise ValueError("Async panel differs from the original MOPD hero")
    teachers = []
    for index, environment in enumerate(protocol["environments"]):
        path = protocol["teacher_eval"][environment]
        if sha256(ROOT / path) != manifest["sha256"][path]:
            raise ValueError(f"Original teacher gate changed: {environment}")
        teachers.append(
            {
                "environment": environment,
                "source_label": f"teacher-{index:02d}",
                "eval": path,
                "eval_sha256": manifest["sha256"][path],
            }
        )
    settings = {k: v for k, v in protocol.items() if k not in {"teacher_config", "student_config"}}
    plan.update(settings)
    plan.update(
        method="async-branching-lora-v1",
        source_panel=f"n{size}-mopd",
        teachers=teachers,
        size=size,
        teacher_project="rl-data-scaling-async-lora-teachers-v1",
        student_project="rl-data-scaling-students-v1",
        student_config=student,
        source_student_config=protocol["student_config"],
        student_config_sha256=sha256(student_path),
        source_async_protocol_sha256=sha256(protocol_path),
        physical_gpus=4,
        adapter_slots=size + 1,
        decode_graph_max_batch_size=64,
        branching="clone the selected student's complete LoRA weights; fresh teacher Adam; frozen base",
        checkpointing="durable adapter and optimizer checkpoints; immutable sampling versions; causal scheduler",
        student_learning_rate_override={
            "dense": yaml.safe_load(student_path.read_text())["lr"],
            "lora": learning_rate,
        },
    )
    if question_data is not None and not single_question:
        raise ValueError("--single-question-data requires --single-question")
    if single_question:
        path = (
            question_data or ROOT / f"data/DATA-SCALING-ASYNC-LORA-V1/single_questions_n{size}_d1_s1234.json"
        ).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Fixed-question data must live inside the source RLVE repository")
        questions = load_questions(path, plan["environments"])
        data = json.loads(path.read_text())
        plan["single_question"] = {
            "data": str(path.relative_to(ROOT.resolve())),
            "sha256": sha256(path),
            "difficulty": data["difficulty"],
            "seed": data["seed"],
            "question_ids": {e: r["metadata"]["question_id"] for e, r in questions.items()},
            "teacher_gate_metric": "same_question_val_acc",
            "student_recovery_metric": "same_question_val_acc",
            "student_overall_metric": "david/eval/val_envs/pass@1 (original randomized inputs)",
        }
        if data.get("teacher_eval_overlap"):
            plan["single_question"]["teacher_eval_overlap"] = data["teacher_eval_overlap"]
        plan["method"] = "async-branching-lora-single-question-v1"
    apply_speed_profile(plan, speed_profile)
    if type(flex_gpus) is not int or not 0 <= flex_gpus <= 8:
        raise ValueError("flex_gpus must be an integer from zero to eight")
    if flex_gpus:
        if plan["train_gpus"] != 2 or plan["sampling_gpus"] != 2:
            raise ValueError("Flexible generators require the two-trainer/two-generator core layout")
        plan["elastic_generation"] = {
            "max_extra_gpus": flex_gpus,
            "poll_seconds": 5,
            "heartbeat_timeout_seconds": 30,
            "router_request_timeout_seconds": 600,
            "checkpoint_transport": "shared-filesystem immutable adapters",
        }
        plan["max_physical_gpus"] = plan["physical_gpus"] + flex_gpus
        plan["physical_layout"] += f"; up to {flex_gpus} additional one-GPU generation jobs on any eligible node"
        plan["execution_changes"]["placement"] = plan["physical_layout"]
        plan["execution_changes"]["storage"] = "shared immutable sampler exports for cross-node adapter loading"
    return plan


def experiment_stem(plan):
    stem = f"async-lora-n{plan['size']}"
    if plan.get("single_question"):
        stem += "-q1"
    if plan.get("opd_replay"):
        stem += f"-replay{round(100 * plan['opd_replay']['fraction'])}"
    if plan.get("speed_profile", "original") != "original":
        stem += f"-speed-{plan['speed_profile']}"
    if plan.get("elastic_generation"):
        stem += f"-flex{plan['elastic_generation']['max_extra_gpus']}"
    return stem
