"""Read-only status for one shared-base async LoRA experiment."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def status(experiment):
    plan, state = read(experiment / "plan.json"), read(experiment / "state.json")
    devices = {d["environment"]: d for d in state.get("devices", [])}
    teachers = []
    for index, environment in enumerate(plan["environments"]):
        directory = experiment / "workers" / f"teacher-{index:02d}"
        heartbeat = read(directory / "heartbeat.json")
        task_id = heartbeat.get("task_id")
        task = read(experiment / "tasks" / task_id / "task.json") if task_id else {}
        device = devices.get(environment, {})
        active = device.get("active") or {}
        if active:
            current = experiment / "tasks" / active["id"]
            gate = read(current / "progress.json") or read(current / "initial-validation.json")
        else:
            gate = device.get("version") or {}
        teachers.append(
            {
                "environment": environment,
                "generation": device.get("generation", 0),
                "logical_completed_rollouts": device.get("rollouts", 0),
                "physical_task": task_id,
                "physical_updates": heartbeat.get("used", 0) // 8,
                "physical_generation": task.get("generation"),
                "accuracy": gate.get("accuracy"),
                "accuracy_metric": "same_question_val_acc" if plan.get("single_question") else "randomized_val_acc",
                "active_logical_task": active.get("id"),
            }
        )
    metrics = experiment / "workers/student/metrics.jsonl"
    overall = None
    if metrics.exists():
        for line in reversed(metrics.read_text().splitlines()):
            row = json.loads(line)
            if "david/eval/val_envs/pass@1" in row:
                overall = row
                break
    return {
        "job": read(experiment / "job.json"),
        "student": read(experiment / "workers/student/identity.json"),
        "ready_slots": len(list((experiment / "workers").glob("*/ready.json"))),
        "expected_slots": plan["adapter_slots"],
        "single_question": plan.get("single_question"),
        "opd_replay": plan.get("opd_replay"),
        "gpu_preflight_passed": (experiment / "preflight/passed.json").exists(),
        "student_rollouts": state.get("student_rollouts", 0),
        "simulated_clock": state.get("clock", 0),
        "teacher_versions": len(state.get("versions", [])),
        "student_latest_overall_eval": overall,
        "teachers": teachers,
        "errors": {p.parent.name: read(p) for p in (experiment / "workers").glob("*/error.json")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    print(json.dumps(status(args.experiment), indent=2))


if __name__ == "__main__":
    main()
