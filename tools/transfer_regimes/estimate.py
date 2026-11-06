"""Measured reference rates plus conservative phase and validation allowances."""

import json
import statistics

from clustering_mopd.estimate import calibrate


def calibration():
    value = calibrate()
    # An isolated joint teacher measures GRPO without concurrent teacher slots.
    from pathlib import Path

    sweep = Path("/juice5b/scr5b/dhei/runs/clustering-mopd-n32-d8-20261101-174231")
    path = next(
        sweep.glob("experiments/*k01-joint-repeat1/tasks/teacher-*-g0001/metrics.jsonl")
    )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    value["isolated_grpo_seconds_per_update_median"] = statistics.median(
        r["train/seconds_per_update"] for r in rows if "train/seconds_per_update" in r
    )
    value["isolated_grpo_source"] = str(path)
    student_path = next(
        sweep.glob("experiments/*k01-joint-repeat1/workers/student/metrics.jsonl")
    )
    student = [
        json.loads(line) for line in student_path.read_text().splitlines() if line
    ]
    value["historical_32_adapter_student_seconds_per_update_median"] = value[
        "student_seconds_per_update_median"
    ]
    value["student_seconds_per_update_median"] = statistics.median(
        r["train/seconds_per_update"]
        for r in student
        if "train/seconds_per_update" in r
    )
    value["seconds_per_256_sample_gate_median"] = statistics.median(
        (r["clustering/val_envs_seconds"] + r["clustering/test_envs_seconds"]) / 2
        for r in student
        if "clustering/val_envs_seconds" in r
    )
    value["low_contention_student_and_eval_source"] = str(student_path)
    return value


def estimate(plan, rates):
    gate = rates["seconds_per_256_sample_gate_median"]
    # Both methods have 20 ordinary ID+unseen panels plus baseline at the
    # default budget. Final ID uses 96 prompts and final unseen uses 32.
    ordinary = (
        plan["total_training_rollout_cap"] / plan["joint_eval_rollouts"]
        if plan["algorithm"] == "joint-rl"
        else plan["rounds"] * plan["student_round_rollouts"] / 400
    )
    main_eval_hours = ((ordinary + 1) * 2 + 4) * gate / 3600
    if plan["algorithm"] == "joint-rl":
        training_hours = (
            plan["total_training_rollout_cap"]
            / 8
            * rates["isolated_grpo_seconds_per_update_median"]
            / 3600
        )
        specialist_hours = 0
    else:
        # This observed aggregate rate already includes 32-prompt specialist
        # gates. Four adapters may fill sampling batches less well than the
        # 32-adapter reference, accounted for by the planning range below.
        specialist_hours = (
            plan["rounds"]
            * 4
            * plan["teacher_branch_rollouts"]
            / rates["teacher_four_gpu_aggregate_rollouts_per_hour"]
        )
        training_hours = (
            plan["rounds"]
            * plan["student_round_rollouts"]
            / 8
            * rates["student_seconds_per_update_median"]
            / 3600
        )
    central = training_hours + specialist_hours + main_eval_hours + 0.2
    low, high = round(central * 0.7, 1), round(central * 1.7 + 0.5, 1)
    return {
        "group": plan["group"],
        "algorithm": plan["algorithm"],
        "physical_gpus": 4,
        "hours_range": [low, high],
        "central_hours": round(central, 2),
        "main_training_hours": round(training_hours, 2),
        "specialist_training_and_gates_hours": round(specialist_hours, 2),
        "main_evaluation_hours": round(main_eval_hours, 2),
        "maximum_training_completions": plan["total_training_rollout_cap"],
        "queue_and_preemption_time_included": False,
        "assumptions": "Full teacher caps; no trained discovery adapters reused. Rates calibrated on earlier four-GPU jobs; compatible 24GB+ GPUs are eligible, as in the existing eight-slot clustering run on RTX 3090s. Early teacher stopping can lower MOPD work. Task lengths, four-adapter batching and GPU type can change speed; these are planning ranges, not a benchmark of this new implementation.",
    }
