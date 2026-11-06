"""Independent teacher shards and shared, restartable evaluation work queues."""

import asyncio
import fcntl
import hashlib
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from async_branching.io import read_json, write_json
from lora_clustering.algorithm import members_label


@contextmanager
def claim(path):
    """Cross-node POSIX locks release automatically on process termination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def assign_teachers(experiment, plan, shards):
    if shards < 1:
        raise ValueError("Expected at least one shard")
    assignments, loads = [[] for _ in range(shards)], [0] * shards
    pending = []
    for i, teacher in enumerate(plan["teachers"]):
        directory = experiment / "teacher" / teacher["source_label"]
        if (directory / "completed.json").exists():
            continue
        latest = directory / "latest.json"
        used = 0
        if latest.exists():
            bundle = read_json(latest)
            used = int(Path(bundle["hf"]).parent.name.removeprefix("u")) * 8
        pending.append((plan["teacher_rollouts"] - used, i))
    for remaining, i in sorted(pending, reverse=True):
        target = min(range(shards), key=lambda s: (loads[s], len(assignments[s]), s))
        assignments[target].append(i)
        loads[target] += remaining
    return [sorted(group) for group in assignments]


def completed_results(experiment, plan):
    results = []
    for teacher in plan["teachers"]:
        path = experiment / "teacher" / teacher["source_label"] / "completed.json"
        results.append(read_json(path) if path.exists() else None)
    return results


def confirmation_tasks(plan, specification):
    """Deduplicate actual adapter/task cells shared by multiple controls."""
    tasks = {}

    def add(label, target, members):
        environment = plan["environments"][target]
        tasks[(label, environment)] = {
            "label": label,
            "environment": environment,
            "members": members,
        }

    for i in range(len(plan["environments"])):
        add("base", i, None)
        add(f"individual-{i:02d}", i, [i])
    for memberships in specification["controls"].values():
        for i, members in enumerate(memberships):
            add(members_label(members), i, members)
    return list(tasks.values())


def checkpoint_inventory(experiment, plan):
    """Validate recovery bundles before relinquishing the old allocation."""
    rows = []
    for i, teacher in enumerate(plan["teachers"]):
        directory = experiment / "teacher" / teacher["source_label"]
        if (directory / "completed.json").exists():
            rows.append({"index": i, "complete": True})
            continue
        saved = read_json(directory / "latest.json")
        hf, state = Path(saved["hf"]), Path(saved["resume"])
        for required in (
            hf / "adapter_model.safetensors",
            hf / "META.json",
            state / "metadata.json",
            hf.parent / "client-state.pkl",
            hf.parent / "reference.json",
        ):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Missing durable recovery artifact: {required}"
                )
        rows.append(
            {
                "index": i,
                "complete": False,
                "hf": str(hf),
                "resume": str(state),
                "checkpoint_rollouts": int(hf.parent.name.removeprefix("u")) * 8,
            }
        )
    return rows


async def evaluate_shared(experiment, backend, plan, model_path, base_url):
    from analyze_balanced_lora_clustering import analyze

    from lora_clustering.balanced_pipeline import confirmation_assignments, run_balanced
    from lora_clustering.evaluation import Evaluator
    from lora_clustering.pipeline import cluster_adapter

    evaluator = Evaluator(
        experiment, plan, model_path, base_url, os.environ["LORA_CHECKPOINT_ROOT"]
    )
    directory = experiment / "clustering/parallel"
    directory.mkdir(parents=True, exist_ok=True)

    def status(stage, **extra):
        write_json(
            backend / "pipeline-status.json",
            {"stage": stage, "time": time.time(), **extra},
        )

    def marker(label):
        return directory / "selection-done" / f"{label}.json"

    labels = ["base"] + [f"individual-{i:02d}" for i in range(len(plan["teachers"]))]
    while not all(marker(label).exists() for label in labels):
        results = completed_results(experiment, plan)
        available = [("base", None)] + [
            (f"individual-{i:02d}", row)
            for i, row in enumerate(results)
            if row is not None
        ]
        advanced = False
        for label, bundle in available:
            if marker(label).exists():
                continue
            with claim(directory / "locks" / f"selection-{label}.lock") as owned:
                if not owned or marker(label).exists():
                    continue
                status("transfer_evaluation", adapter=label)
                await evaluator.vector("selection", label, bundle=bundle)
                write_json(
                    marker(label),
                    {
                        "label": label,
                        "eval_sha256": plan["clustering"]["selection_sha256"],
                        "job_id": os.environ.get("SLURM_JOB_ID"),
                    },
                )
                advanced = True
                break
        if not advanced:
            status("waiting_for_selection")
            await asyncio.sleep(10)
    results = completed_results(experiment, plan)
    if any(row is None for row in results):
        raise RuntimeError("Selection completed without all final teacher adapters")
    selected_file = experiment / "clustering/balanced-selected.json"
    specification_file = experiment / "clustering/confirmation-assignment-spec.json"
    while not specification_file.exists():
        with claim(directory / "locks/balanced-search.lock") as owned:
            if owned and not specification_file.exists():
                status("balanced_search")
                if selected_file.exists():
                    selected = read_json(selected_file)
                else:
                    selected = analyze(
                        experiment,
                        Path(plan["plot_output"]) / "balanced",
                        plan["clustering"]["balanced_restarts"],
                        plan["seed"],
                    )
                    write_json(selected_file, selected)
                if (
                    selected["evaluation_input_sha256"]
                    != plan["clustering"]["selection_sha256"]
                ):
                    raise ValueError("Balanced selection inputs changed")
                specification = confirmation_assignments(
                    selected["selected"]["fit"]["groups"], plan["seed"] + 20000
                )
                write_json(specification_file, specification)
        if not specification_file.exists():
            await asyncio.sleep(10)
    tasks = confirmation_tasks(plan, read_json(specification_file))
    while True:
        missing = [
            task
            for task in tasks
            if evaluator.cached("confirmation", task["label"], task["environment"])
            is None
        ]
        if not missing:
            break
        advanced = False
        for task in missing:
            key = hashlib.sha256(
                f"{task['label']}/{task['environment']}".encode()
            ).hexdigest()
            with claim(directory / "locks" / f"confirmation-{key}.lock") as owned:
                if (
                    not owned
                    or evaluator.cached(
                        "confirmation", task["label"], task["environment"]
                    )
                    is not None
                ):
                    continue
                status(
                    "merged_confirmation",
                    adapter=task["label"],
                    environment=task["environment"],
                )
                members = task["members"]
                if members is None:
                    sampler = await evaluator.sampler("base")
                elif len(members) == 1:
                    sampler = await evaluator.sampler(
                        task["label"], bundle=results[members[0]]
                    )
                else:
                    adapter = cluster_adapter(experiment, plan, results, members)
                    sampler = await evaluator.sampler(task["label"], adapter=adapter)
                await evaluator.cell(
                    "confirmation", task["label"], task["environment"], sampler
                )
                advanced = True
                break
        if not advanced:
            status("waiting_for_confirmation")
            await asyncio.sleep(10)
    while not (experiment / "completed.json").exists():
        with claim(directory / "locks/finalize.lock") as owned:
            if owned and not (experiment / "completed.json").exists():
                status("finalizing")
                write_json(experiment / "teachers-completed.json", results)
                final = await run_balanced(
                    experiment, plan, results, model_path, base_url
                )
                record = {
                    "stage": "complete",
                    "teachers": len(results),
                    "k": final["selection"]["selected"]["k"],
                    "plots": plan["plot_output"],
                    "execution": "six independent four-GPU backends",
                    "time": time.time(),
                }
                write_json(experiment / "completed.json", record)
                write_json(experiment / "pipeline-status.json", record)
        if not (experiment / "completed.json").exists():
            await asyncio.sleep(10)
    status("complete")


def run_shard(backend, backend_plan, *, base_url, model_path, gateway):
    import torch

    from lora_clustering.pipeline import train_teachers
    from lora_clustering.preflight import check_backend

    torch.set_num_threads(4)
    config = read_json(backend / "backend.json")
    experiment = Path(config["experiment"])
    plan = read_json(experiment / "plan.json")
    with claim(backend / "run.lock") as owned:
        if not owned:
            raise RuntimeError("This backend already has a running owner")
        if (experiment / "completed.json").exists():
            return
        write_json(
            backend / "pipeline-status.json",
            {"stage": "preflight", "time": time.time()},
        )
        asyncio.run(check_backend(backend, backend_plan, model_path, base_url))
        # Diagnostics are disposable; retain verification JSON, reclaim only
        # this backend's diagnostic tensor bundles before starting real work.
        for path in (backend / "preflight").iterdir():
            if path.is_dir():
                shutil.rmtree(path)
        os.environ.update(MOPD_VERIFY_RESUME="1", MOPD_SHARD_INDEX=str(config["index"]))
        train_teachers(
            experiment,
            plan,
            model_path,
            base_url,
            gateway,
            indices=config["indices"],
            status_path=backend / "pipeline-status.json",
        )
        asyncio.run(evaluate_shared(experiment, backend, plan, model_path, base_url))
