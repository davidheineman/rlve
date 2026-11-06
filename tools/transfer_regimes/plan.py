"""Matched initialization, disjoint monitoring/final panels, and explicit budgets."""

import copy
import json
from pathlib import Path

from transfer_regimes.selection import read


def metadata(row):
    value = row["metadata"]
    return json.loads(value) if isinstance(value, str) else value


def build_plans(
    selection,
    reference,
    discovery,
    *,
    rounds=2,
    seed=1234,
    student_project="rl-transfer-regimes-students",
    teacher_project="rl-transfer-regimes-teachers",
):
    if rounds < 2:
        raise ValueError("Multi-round MOPD requires at least two rounds")
    base = read(reference / "plan.json")
    study = read(discovery / "plan.json")
    source = discovery / "source/rlve"
    confirmation = read(source / study["clustering"]["confirmation"])
    selection_prompts = {
        r["user_prompt"] for r in read(source / study["clustering"]["selection"])
    }
    by_env = {t["environment"]: t for t in study["teachers"]}
    grouped = {}
    for row in confirmation:
        grouped.setdefault(metadata(row)["environment"], []).append(row)
    plans, generated = [], {}
    initial = read(reference / "base.json")
    total_cap = rounds * (4 * 4000 + 4000)
    for group in selection["groups"]:
        environments = group["environments"]
        monitoring, held = [], []
        teachers = [copy.deepcopy(by_env[e]) for e in environments]
        for environment in environments:
            rows = grouped[environment]
            if len(rows) != 32:
                raise ValueError("Expected 32 disjoint confirmation prompts per task")
            monitor_indices = [*range(4), *range(16, 20)]
            monitoring.extend(rows[i] for i in monitor_indices)
            held.extend(row for i, row in enumerate(rows) if i not in monitor_indices)
        monitor_set = {r["user_prompt"] for r in monitoring}
        held_set = {r["user_prompt"] for r in held}
        gates = {r["user_prompt"] for t in teachers for r in read(source / t["eval"])}
        if monitor_set & held_set or (monitor_set | held_set) & (
            gates | selection_prompts
        ):
            raise ValueError(
                "Evaluation panels overlap stopping or assignment-selection data"
            )
        if len(monitor_set) != 32 or len(held_set) != 96:
            raise ValueError("Evaluation panels contain duplicate questions")
        prefix = f"data/TRANSFER-REGIMES-V1/{group['label']}"
        generated[prefix + "/monitor.json"] = monitoring
        generated[prefix + "/final.json"] = held
        for algorithm in ["joint-rl", "sync-mopd"]:
            plan = copy.deepcopy(base)
            for key in [
                "elastic_generation",
                "single_question",
                "opd_replay",
                "clusters",
                "simulated_devices",
                "simulated_clock_budget",
                "teacher_worker_count",
                "teacher_recovery_fraction",
            ]:
                plan.pop(key, None)
            plan.update(
                method="transfer-regimes-sync-v1",
                algorithm=algorithm,
                group=group["label"],
                transfer_condition="positive"
                if group["label"].startswith("best")
                else "negative"
                if group["label"].startswith("worst")
                else "near-zero"
                if group["label"].startswith("neutral")
                else "random",
                transfer_selection=group,
                size=4,
                environments=environments,
                teachers=teachers,
                teacher_eval={t["environment"]: t["eval"] for t in teachers},
                rounds=rounds,
                seed=seed,
                teacher_branch_rollouts=4000,
                student_round_rollouts=4000,
                teacher_validation_threshold=0.98,
                total_training_rollout_cap=total_cap,
                student_rollouts=total_cap
                if algorithm == "joint-rl"
                else 4000 * rounds,
                id_eval=prefix + "/monitor.json",
                final_id_eval=prefix + "/final.json",
                id_eval_prompts=32,
                final_id_eval_prompts=96,
                test_eval_prompts=32,
                joint_eval_rollouts=2000,
                checkpoint_rollouts=400,
                student_project=student_project,
                teacher_project=teacher_project,
                project=teacher_project,
                physical_gpus=4,
                train_gpus=2,
                sampling_gpus=2,
                adapter_slots=1 if algorithm == "joint-rl" else 5,
                minimum_gpu_mib=23000,
                evaluation_concurrency=2,
                decode_graph_max_batch_size=64,
                initialization="identical untrained LoRA factors and fresh Adam for both methods",
                initial_student_reference={
                    "bundle": "initial-base",
                    "source_experiment": str(reference),
                    "source_bundle": str(Path(initial["hf"]).parent),
                    "optimizer": "fresh; import the common untrained adapter only",
                },
                synchronous_protocol="all specialists finish, then freeze the bank for one mixed-task OPD student phase; next-round specialists clone that completed student",
                budget_definition="Generated training completions: GRPO 1 prompt x 8 responses; OPD 8 prompts x 1 response; all teacher work is charged",
                comparison="Common maximum training-completion budget; MOPD can use less through teacher early stopping. Also plot actual work and four-GPU elapsed time; no simulated devices.",
                final_endpoint="96 prompts held aside from discovery, teacher stopping and monitoring; 24 per task, 8 completions each",
            )
            for settings in [plan["teacher_config"], plan["student_config"]]:
                settings["lr"] = 2e-5
            plans.append(plan)
    return plans, generated
