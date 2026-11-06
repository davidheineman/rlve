"""A persisted teacher barrier followed by a single mixed-task student phase."""

from pathlib import Path

from async_branching.io import read_json, write_json


def validate_budget(plan):
    expected = plan["rounds"] * (
        4 * plan["teacher_branch_rollouts"] + plan["student_round_rollouts"]
    )
    if len(plan["environments"]) != 4 or expected != plan["total_training_rollout_cap"]:
        raise ValueError(
            "Expected four tasks and an explicitly accounted training budget"
        )
    for value in [
        plan["teacher_branch_rollouts"],
        plan["student_round_rollouts"],
        expected,
    ]:
        if value <= 0 or value % 400:
            raise ValueError("Budgets must be positive multiples of 400 completions")


def task(
    plan,
    experiment,
    identifier,
    parent,
    *,
    role,
    round_index,
    start_student,
    start_work,
    environment_index=0,
    teacher_bank=None,
):
    specialist = role == "teacher"
    environments = (
        [plan["environments"][environment_index]]
        if specialist
        else plan["environments"]
    )
    return {
        "id": identifier,
        "kind": role,
        "directory": str(experiment / "tasks" / identifier),
        "environment_index": environment_index,
        "environment": environments[0] if specialist else "joint",
        "environments": environments,
        "generation": round_index,
        "parent": parent,
        "teacher_bank": teacher_bank or {},
        "cap": plan["teacher_branch_rollouts"]
        if specialist
        else (
            plan["total_training_rollout_cap"]
            if plan["algorithm"] == "joint-rl"
            else plan["student_round_rollouts"]
        ),
        "start_rollouts": 0 if specialist else start_student,
        "start_work": start_work,
        "train_seed": plan["seed"]
        + (
            100000 * round_index + environment_index
            if specialist
            else 2000000 + round_index
        ),
    }


def execute(plan, experiment, backend):
    """All teacher futures resolve before a student task can be dispatched."""
    validate_budget(plan)
    experiment = Path(experiment)
    final = experiment / "completed.json"
    if final.exists():
        return read_json(final)
    path = experiment / "sync-state.json"
    state = (
        read_json(path)
        if path.exists()
        else {
            "round": 1,
            "phase": "teachers",
            "student": read_json(experiment / "base.json"),
            "student_rollouts": 0,
            "teacher_rollouts": 0,
            "total_training_rollouts": 0,
            "teachers": {},
            "completed_rounds": [],
        }
    )

    def commit():
        write_json(path, state)
        write_json(
            experiment / "status.json",
            {
                k: v
                for k, v in state.items()
                if k not in {"teachers", "completed_rounds"}
            },
        )

    if plan["algorithm"] == "joint-rl":
        raw = task(
            plan,
            experiment,
            "joint-rl",
            state["student"],
            role="student",
            round_index=0,
            start_student=0,
            start_work=0,
        )
        result = backend.student(raw)
        state.update(
            student=result,
            student_rollouts=result["rollouts"],
            total_training_rollouts=result["rollouts"],
            phase="complete",
        )
        commit()
    else:
        while state["round"] <= plan["rounds"]:
            round_index = state["round"]
            if state["phase"] == "teachers":
                futures, retained = {}, {}
                for index, environment in enumerate(plan["environments"]):
                    previous = state["teachers"].get(environment)
                    if (
                        previous
                        and previous["accuracy"] >= plan["teacher_validation_threshold"]
                    ):
                        retained[environment] = previous
                        continue
                    raw = task(
                        plan,
                        experiment,
                        f"round-{round_index:02d}-teacher-{index:02d}",
                        state["student"],
                        role="teacher",
                        round_index=round_index,
                        start_student=state["student_rollouts"],
                        start_work=state["total_training_rollouts"],
                        environment_index=index,
                    )
                    futures[environment] = backend.teacher(raw)
                # Do not advance the phase or use partially arrived teachers.
                trained = {
                    environment: future.result()
                    for environment, future in futures.items()
                }
                spent = sum(r["rollouts"] for r in trained.values())
                state["teachers"] = {**retained, **trained}
                state["teacher_rollouts"] += spent
                state["total_training_rollouts"] += spent
                state["phase"] = "student"
                commit()
            raw = task(
                plan,
                experiment,
                f"round-{round_index:02d}-student",
                state["student"],
                role="student",
                round_index=round_index,
                start_student=state["student_rollouts"],
                start_work=state["total_training_rollouts"],
                teacher_bank=state["teachers"],
            )
            result = backend.student(raw)
            if result["rollouts"] != plan["student_round_rollouts"]:
                raise ValueError(
                    "Synchronous student phase did not finish its fixed budget"
                )
            state["student"] = result
            state["student_rollouts"] += result["rollouts"]
            state["total_training_rollouts"] += result["rollouts"]
            state["completed_rounds"].append(
                {
                    "round": round_index,
                    "student": result,
                    "teacher_rollouts": state["teacher_rollouts"],
                    "total_training_rollouts": state["total_training_rollouts"],
                }
            )
            state.update(round=round_index + 1, phase="teachers")
            commit()
        state["phase"] = "complete"
        commit()
    if state["total_training_rollouts"] > plan["total_training_rollout_cap"]:
        raise ValueError("Total teacher plus student work exceeded the common cap")
    evaluation = task(
        plan,
        experiment,
        "final-evaluation",
        state["student"],
        role="student",
        round_index=plan["rounds"],
        start_student=state["student_rollouts"],
        start_work=state["total_training_rollouts"],
    )
    evaluation.update(evaluation_only=True, cap=0)
    backend.student(evaluation)
    result = {
        **state,
        "algorithm": plan["algorithm"],
        "group": plan["group"],
        "stage": "complete",
    }
    write_json(final, result)
    return result
