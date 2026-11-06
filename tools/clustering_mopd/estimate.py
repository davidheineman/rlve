"""Conditional four-GPU planning ranges from measured reference workloads."""

import json
import statistics
import subprocess
import tempfile
from pathlib import Path

from async_branching.replay import ReplayBackend
from async_branching.runner import Runner
from clustering_mopd.plan import CLUSTERING, REFERENCE
from clustering_mopd.schedule import ClusterSimulation, initial_state

COMPLETED_K8_SWEEP = Path(
    "/juice5b/scr5b/dhei/runs/clustering-mopd-n32-d8-20261101-174231"
)


def calibrate_completed(sweep=COMPLETED_K8_SWEEP):
    """Use allocated GPU time, excluding queue gaps, from completed k=8 runs."""
    sweep = Path(sweep)
    manifest = json.loads((sweep / "sweep.json").read_text())
    records, gates = [], []
    for item in manifest["experiments"]:
        if item["cluster_count"] != 8:
            continue
        directory = Path(item["experiment"])
        complete = json.loads((directory / "completed.json").read_text())
        if complete["simulated_clock"] != 16000:
            raise ValueError("Calibration requires completed 16k k=8 runs")
        old_plan = json.loads((directory / "plan.json").read_text())
        if old_plan["simulated_devices"] != 8 or old_plan["physical_gpus"] != 4:
            raise ValueError(
                "Calibration must match eight simulated and four physical devices"
            )
        rows = [
            json.loads(line)
            for line in (directory / "workers/student/metrics.jsonl")
            .read_text()
            .splitlines()
            if line
        ]
        seconds = [
            r["train/seconds_per_update"]
            for r in rows
            if "train/seconds_per_update" in r
        ]
        gates.extend(
            r["clustering/val_envs_seconds"] / (old_plan["id_eval_prompts"] / 32)
            for r in rows
            if "clustering/val_envs_seconds" in r
        )
        teacher_rollouts = 0
        for path in directory.glob("tasks/teacher-*/metrics.jsonl"):
            values = [
                json.loads(line) for line in path.read_text().splitlines() if line
            ]
            teacher_rollouts += max(
                (r.get("david/rollouts", 0) for r in values), default=0
            )
        job = json.loads((directory / "job.json").read_text())
        attempts = []
        for job_id in [*job.get("previous_job_ids", []), job["job_id"]]:
            output = subprocess.check_output(
                [
                    "sacct",
                    "-X",
                    "-D",
                    "-n",
                    "-P",
                    "-j",
                    str(job_id),
                    "--format=JobID,ElapsedRaw,State",
                ],
                text=True,
                timeout=30,
            )
            attempts.extend(
                {"job_id": str(job_id), "seconds": int(fields[1]), "state": fields[2]}
                for line in output.splitlines()
                if (fields := line.split("|"))[0] == str(job_id)
            )
        if not attempts or not seconds or not teacher_rollouts:
            raise ValueError("Incomplete measured calibration workload")
        records.append(
            {
                "label": item["label"],
                "allocated_hours": sum(a["seconds"] for a in attempts) / 3600,
                "allocation_attempts": attempts,
                "teacher_rollouts": teacher_rollouts,
                "student_rollouts": complete["student"]["rollouts"],
                "student_seconds_per_update_median": statistics.median(seconds),
                "id_eval_prompts": old_plan["id_eval_prompts"],
                "test_eval_prompts": old_plan.get("test_eval_prompts", 32),
                "student_evaluation_points": len(
                    {
                        r.get("async/simulated_clock", r.get("david/rollouts", 0))
                        for r in rows
                        if "david/eval/val_envs/pass@1" in r
                    }
                ),
            }
        )
    if len(records) < 2 or not gates:
        raise ValueError(
            "Need two completed k=8 comparison runs and measured evaluations"
        )
    return {
        "method": "completed-k8-allocated-time",
        "reference_sweep": str(sweep),
        "runs": records,
        "seconds_per_256_sample_gate_median": statistics.median(gates),
        "queue_gaps_excluded": True,
        "startup_and_failed_allocations_included": True,
    }


def completed_range(
    plan, calibration, teacher_rollouts, student_rollouts, student_segments
):
    """Scale measured k=8 work; allow extra student updates and the larger ID panel."""
    if plan["simulated_devices"] != 8:
        raise ValueError("Completed-run calibration requires eight simulated devices")
    projections = []
    for reference in calibration["runs"]:
        teacher_scale = teacher_rollouts / reference["teacher_rollouts"]
        extra_evals = (
            max(0, plan["id_eval_prompts"] - reference["id_eval_prompts"])
            / 32
            * (student_segments + 2)
            * calibration["seconds_per_256_sample_gate_median"]
            / 3600
        )
        if reference.get("student_evaluation_points"):
            # Teacher scaling includes the reference student evaluations; charge
            # additional merge evaluations after scaling their full ID/unseen panels.
            current_panels = (
                plan["id_eval_prompts"] + plan.get("test_eval_prompts", 32)
            ) / 32
            reference_panels = (
                reference["id_eval_prompts"] + reference.get("test_eval_prompts", 32)
            ) / 32
            extra_evals = (
                max(
                    0,
                    (student_segments + 2) * current_panels
                    - reference["student_evaluation_points"]
                    * reference_panels
                    * teacher_scale,
                )
                * calibration["seconds_per_256_sample_gate_median"]
                / 3600
            )
        extra_updates = (
            max(0, student_rollouts - reference["student_rollouts"])
            / 8
            * reference["student_seconds_per_update_median"]
            / 3600
        )
        base = reference["allocated_hours"] * teacher_scale + extra_evals
        projections.append(
            {
                "reference": reference["label"],
                "matched_work_hours": base,
                "extra_student_update_hours": extra_updates,
                "extra_id_evaluation_hours": extra_evals,
                "upper_planning_hours": 1.3 * (base + extra_updates),
            }
        )
    return {
        "estimate_basis": "Measured completed four-GPU k=8 runs; scaled teacher work, extra ID evaluations and upper-envelope student updates",
        "reference_projections": projections,
        "physical_hours_range": [
            round(min(p["matched_work_hours"] for p in projections), 1),
            round(max(p["upper_planning_hours"] for p in projections), 1),
        ],
    }


def calibrate(reference=REFERENCE, clustering=CLUSTERING):
    rows = [
        json.loads(line)
        for line in (reference / "workers/student/metrics.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    seconds = [
        r["train/seconds_per_update"] for r in rows if "train/seconds_per_update" in r
    ]
    gates = []
    for directory in (reference / "tasks").glob("student-*"):
        result = directory / "result.json"
        task = directory / "task.json"
        if (
            result.exists()
            and task.exists()
            and json.loads(result.read_text())["rollouts"] == 0
        ):
            gates.append(result.stat().st_mtime - task.stat().st_mtime)
    history = [
        json.loads(line)
        for line in (clustering / "watchdog/health.jsonl").read_text().splitlines()
        if line
    ]
    old = [
        r for r in history if r["job_id"] == "18221879" and r.get("stage") == "training"
    ]
    end = old[-1]
    start = next(r for r in old if r["time"] >= end["time"] - 3600)
    teacher_rate = (
        (end["sum_training_rollouts"] - start["sum_training_rollouts"])
        / (end["time"] - start["time"])
        * 3600
    )
    return {
        "reference_async": str(reference),
        "reference_clustering": str(clustering),
        "student_update_observations": len(seconds),
        "student_seconds_per_update_median": statistics.median(seconds),
        "student_seconds_per_update_mean": statistics.mean(seconds),
        "zero_update_student_gate_observations": len(gates),
        "seconds_per_256_sample_gate_median": statistics.median(gates),
        "teacher_four_gpu_aggregate_rollouts_per_hour": teacher_rate,
        "teacher_rate_includes_original_specialist_validation": True,
    }


def estimate(plan, calibration):
    with tempfile.TemporaryDirectory() as temporary:
        sim = ClusterSimulation(
            initial_state(plan, {"hf": "base", "resume": "base-state"}),
            teacher_cap=plan["teacher_branch_rollouts"],
            merge_cap=plan["student_rollouts_per_teacher_version"],
            recovery=plan["teacher_recovery_fraction"],
            quantum=plan["student_quantum_rollouts"],
        )
        backend = ReplayBackend(teacher_rollouts=plan["teacher_branch_rollouts"])
        Runner(sim, backend, Path(temporary)).run()
        teacher_tasks = [t for t in backend.tasks if t["kind"] == "teacher"]
        student_tasks = [t for t in backend.tasks if t["kind"] == "student"]
        teacher_rollouts = sum(t["cap"] for t in teacher_tasks)
        student_rollouts = backend.student_training_rollouts
        if calibration.get("method") == "completed-k8-allocated-time":
            projected = completed_range(
                plan,
                calibration,
                teacher_rollouts,
                student_rollouts,
                len(student_tasks),
            )
            return {
                "label": f"k{plan['cluster_count']:02d}-{plan['assignment']}",
                "cluster_count": plan["cluster_count"],
                "assignment": plan["assignment"],
                "physical_gpus": 4,
                "simulated_devices": plan["simulated_devices"],
                "simulated_teacher_lanes": plan["teacher_worker_count"],
                "simulated_clock_budget": plan["simulated_clock_budget"],
                "replay_scenario": "4000-rollout teacher caps and no early student recovery; no predicted accuracies",
                "replay_teacher_branches": len(teacher_tasks),
                "replay_teacher_rollouts": teacher_rollouts,
                "replay_student_rollouts": student_rollouts,
                "replay_student_segments": len(student_tasks),
                **projected,
                "allocation_gpu_hours_range": [
                    round(4 * x, 1) for x in projected["physical_hours_range"]
                ],
                "queue_time_included": False,
                "uncertainty": "Conditional on healthy allocations. New tasks, sampling lengths, recovery, storage, queue waits and preemption can change elapsed time; 30% upper planning margin is not a statistical interval or a bound.",
            }
        # The old teacher aggregate rate already includes ordinary 32-prompt
        # validation. Charge additional gates explicitly for the 64-task policy.
        gate_seconds = calibration["seconds_per_256_sample_gate_median"]
        teacher_hours = (
            teacher_rollouts
            / calibration["teacher_four_gpu_aggregate_rollouts_per_hour"]
        )
        cluster_prompts = plan["clusters"][0]["eval_prompts"]
        extra_teacher_gates = sum(
            1 + (t["cap"] + 399) // 400 for t in teacher_tasks
        ) * max(0, cluster_prompts / 32 - 1)
        teacher_hours += extra_teacher_gates * gate_seconds / 3600
        # Upper-envelope replay assumes every branch reaches its cap and the
        # student never recovers before its 1000-rollout/version allowance.
        # Every nonzero merge has an initial gate, terminal gate, and full ID
        # and unseen evaluation, using the actual panel prompt counts.
        overall_gates = (
            plan.get("id_eval_prompts", plan["size"])
            + plan.get("test_eval_prompts", 32)
        ) / 32
        student_eval_gate_equivalents = (
            len(student_tasks) * (2 * cluster_prompts / 32 + overall_gates)
            + 2 * overall_gates
        )
        eval_hours = student_eval_gate_equivalents * gate_seconds / 3600
        update_hours = (
            student_rollouts
            / 8
            * calibration["student_seconds_per_update_median"]
            / 3600
        )
        central = teacher_hours + update_hours + eval_hours
        return {
            "label": f"k{plan['cluster_count']:02d}-{plan['assignment']}",
            "cluster_count": plan["cluster_count"],
            "assignment": plan["assignment"],
            "physical_gpus": 4,
            "simulated_devices": plan["simulated_devices"],
            "simulated_teacher_lanes": plan["teacher_worker_count"],
            "simulated_clock_budget": plan["simulated_clock_budget"],
            "replay_scenario": "planning workload only: 4000-rollout teacher caps and no early student recovery; no predicted accuracies",
            "replay_teacher_branches": len(teacher_tasks),
            "replay_teacher_rollouts": teacher_rollouts,
            "replay_student_rollouts": student_rollouts,
            "replay_student_segments": len(student_tasks),
            "teacher_elapsed_component_hours": teacher_hours,
            "student_updates_component_hours": update_hours,
            "student_evaluation_component_hours": eval_hours,
            "physical_hours_range": [
                round(max(1, central * 0.65), 1),
                round(central * 1.5 + 1, 1),
            ],
            "allocation_gpu_hours_range": [
                round(4 * max(1, central * 0.65), 1),
                round(4 * (central * 1.5 + 1), 1),
            ],
            "queue_time_included": False,
            "uncertainty": "Early gates/recovery can lower work or introduce more short branches; sampling length, simultaneous jobs, storage and preemption affect elapsed time. Not a bound or a measured runtime for these new configurations.",
        }
