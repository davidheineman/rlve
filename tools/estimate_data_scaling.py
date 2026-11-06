"""Estimate capped compute and a conservative 24-GPU reference wall time."""

import json

from data_scaling_plan import build_plan, validate
from prepare_data_scaling import OUTPUT


def main():
    plan = build_plan()
    validate(plan)
    # Planning ranges, not measurements of the new d0/d1 + forcing condition.
    hours = {"n16-mopd": (5, 8), "n32-mopd": (6, 9), "n64-mopd": (7, 11),
             "n16-rl": (4, 6), "n32-rl": (4, 6), "n64-rl": (4, 6)}
    heroes = []
    for run in plan["runs"][64:]:
        low, high = hours[run["label"]]
        heroes.append({"run": run["label"], "gpus": run["gpus"],
                       "hours": [low, high], "gpu_hours": [run["gpus"] * low, run["gpus"] * high],
                       "eval_completions_at_cap": (1 + run["updates"] // 50) * 32 * 2 * 8})
    # Conservative 24-GPU reference: teacher phase, then all six 4-GPU heroes.
    # Actual submission has no cap and RL controls have no teacher dependencies.
    hero_phase = [max(values[i] for values in hours.values()) for i in (0, 1)]
    teacher_hours = 2
    teacher_phase = 3 * teacher_hours  # 24 + 24 + 16 teachers
    teacher_gpu_hours = 64 * teacher_hours
    report = {
        "assumptions": ["24 dedicated eligible GPUs; no queue/preemption delay",
                        "64 teachers take two hours each, including ID eval and export, as requested",
                        "No savings assumed from 98% early stops",
                        "32 total ID + 32 unseen-env test prompts per hero, eight samples each; 1024 initial tokens plus up to 128 forced-answer tokens",
                        "Fresh heroes: MOPD p8r1/2000 steps, RL p8r16/500 steps",
                        "Every hero uses four GPUs; MOPD caches up to eight teachers per scorer GPU",
                        "Actual launch uses standalone jobs without concurrency caps; independent RL can overlap teachers"],
        "reference_runs": [
            {"url": "https://wandb.ai/david-heineman/rl-data-opd-sweeps-teachers-v3/runs/5bd4c7fa",
             "description": "Circuit teacher, d0, no forcing: 500 steps, mean train step 8.84s, total 2.21h"},
            {"url": "https://wandb.ai/david-heineman/rl-data-opd-sweeps-students-v3/runs/6461d5a7",
             "description": "N16 direct MOPD: mean train step 9.29s, 2.58h training, 5.42h total"},
            {"url": "https://wandb.ai/david-heineman/rl-data-opd-sweeps-students-v3/runs/5c9120ea",
             "description": "N16 joint RL: mean train step 27.72s, 3.85h training, 6.98h total"}],
        "teacher_stage_hours": teacher_phase, "teacher_gpu_hours": teacher_gpu_hours,
        "heroes": heroes, "hero_stage_hours": hero_phase,
        "total_wall_hours": [teacher_phase + value for value in hero_phase],
        "total_gpu_hours": [teacher_gpu_hours + sum(run["gpu_hours"][i] for run in heroes) for i in (0, 1)],
        "sensitivity": "If teachers average three hours instead, add three hours to this conservative staged wall time and 64 GPU-hours. Teacher CPU/GPU cache transfers and newly selected environments are unbenchmarked.",
    }
    path = OUTPUT / "compute_estimate.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
