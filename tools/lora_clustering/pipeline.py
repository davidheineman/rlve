"""Train all 32 specialists once, then search and confirm adapter-based clusters."""

import asyncio
import fcntl
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lora_teachers.runner import stop_process

from async_branching.io import read_json, write_json
from lora_clustering.algorithm import (
    members_label,
    partition_metrics,
    random_partition,
    signed_hierarchy,
)
from lora_clustering.evaluation import Evaluator
from lora_clustering.merge import combine_adapters


def train_teachers(
    experiment, plan, model_path, base_url, gateway, *, indices=None, status_path=None
):
    selected = list(range(len(plan["teachers"]))) if indices is None else list(indices)
    status_path = status_path or experiment / "pipeline-status.json"
    processes, logs = {}, []
    try:
        for index in selected:
            teacher = plan["teachers"][index]
            completion = (
                experiment / "teacher" / teacher["source_label"] / "completed.json"
            )
            if completion.exists():
                if not Path(read_json(completion)["hf"]).is_dir():
                    raise FileNotFoundError(completion)
                continue
            directory = experiment / "workers" / teacher["source_label"]
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("error.json", "ready.json", "heartbeat.json"):
                (directory / name).unlink(missing_ok=True)
            log = (directory / "worker.log").open("a")
            logs.append(log)
            command = [
                sys.executable,
                "-m",
                "lora_clustering.teacher",
                "--experiment",
                str(experiment),
                "--index",
                str(index),
                "--model-path",
                str(model_path),
                "--base-url",
                base_url,
            ]
            processes[teacher["source_label"]] = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
                start_new_session=True,
            )
        while processes:
            if gateway.poll() is not None:
                raise RuntimeError("Miles gateway exited")
            for name, process in list(processes.items()):
                status = process.poll()
                if status is not None:
                    if status != 0:
                        raise RuntimeError(
                            f"Teacher {name} failed; inspect workers/{name}/worker.log"
                        )
                    del processes[name]
            write_json(
                status_path,
                {
                    "stage": "training",
                    "active_teachers": len(processes),
                    "completed_teachers": len(selected) - len(processes),
                    "assigned_indices": selected,
                    "time": time.time(),
                },
            )
            if processes:
                time.sleep(5)
        results = [
            read_json(
                experiment
                / "teacher"
                / plan["teachers"][i]["source_label"]
                / "completed.json"
            )
            for i in selected
        ]
        write_json(status_path.parent / "teachers-completed.json", results)
        return results
    finally:
        for process in processes.values():
            stop_process(process)
        for log in logs:
            log.close()


def cluster_adapter(experiment, plan, results, members):
    if len(members) == 1:
        return Path(results[members[0]]["hf"])
    key = hashlib.sha256(",".join(map(str, sorted(members))).encode()).hexdigest()[:16]
    destination = experiment / "clustering/adapters" / key
    lock = experiment / "clustering/adapter-locks" / f"{key}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not destination.exists():
            diagnostics = combine_adapters(
                [results[i]["hf"] for i in members],
                destination,
                rank=plan["lora_rank"],
                alpha=plan["lora_alpha"],
            )
            print(f"CLUSTER_MERGE {members} {diagnostics}", flush=True)
    return destination


async def evaluate_partition(
    evaluator, experiment, plan, results, split, label, groups, base
):
    vectors, adapters = [], []
    for members in groups:
        adapter = cluster_adapter(experiment, plan, results, members)
        # Deduplicate identical member sets across k candidates and controls.
        key = members_label(members)
        vectors.append(await evaluator.vector(split, key, adapter=adapter))
        adapters.append(str(adapter))
    return {
        "label": label,
        "groups": groups,
        "adapters": adapters,
        "matrix": vectors,
        **partition_metrics(groups, vectors, base),
    }


async def clustering(experiment, plan, results, model_path, base_url):
    torch.set_num_threads(8)
    evaluator = Evaluator(
        experiment, plan, model_path, base_url, os.environ["LORA_CHECKPOINT_ROOT"]
    )
    directory = experiment / "clustering"
    selection_file = directory / "selected.json"
    if selection_file.exists():
        selection = read_json(selection_file)
    else:
        base = await evaluator.vector("selection", "base")
        individual = []
        for i, result in enumerate(results):
            individual.append(
                await evaluator.vector(
                    "selection", f"individual-{i:02d}", bundle=result
                )
            )
        transfer = np.asarray(individual) - np.asarray(base)[None, :]
        hierarchy = signed_hierarchy(transfer)
        write_json(
            directory / "transfer.json",
            {
                "environments": plan["environments"],
                "base": base,
                "individual": individual,
                "transfer": transfer.tolist(),
                "hierarchy": hierarchy,
            },
        )
        candidates = {"automatic": hierarchy["automatic"]}
        for k in plan["clustering"]["candidate_k"]:
            if hierarchy["partitions"][k] not in candidates.values():
                candidates[f"k{k}"] = hierarchy["partitions"][k]
        measured = []
        for label, groups in candidates.items():
            measured.append(
                await evaluate_partition(
                    evaluator,
                    experiment,
                    plan,
                    results,
                    "selection",
                    label,
                    groups,
                    base,
                )
            )
            write_json(directory / "candidate-results.json", measured)
        # k=1 has no between-cluster comparison, so is a baseline, not a candidate.
        eligible = [r for r in measured if r["separation"] is not None]
        selected = (
            max(
                eligible,
                key=lambda r: (r["separation"], r["assigned_accuracy"], -r["k"]),
            )
            if eligible
            else measured[0]
        )
        selection = {
            "selected": selected,
            "candidates": measured,
            "selection_base": base,
            "selection_individual": individual,
            "criterion": plan["clustering"]["selection_objective"],
            "selection_inputs_sha256": plan["clustering"]["selection_sha256"],
        }
        write_json(selection_file, selection)
    groups = selection["selected"]["groups"]
    base = await evaluator.vector("confirmation", "base")
    individual = []
    for i, result in enumerate(results):
        # The specialist control needs only its own environment on fresh inputs.
        sampler = await evaluator.sampler(f"individual-{i:02d}", bundle=result)
        cell = await evaluator.cell(
            "confirmation", f"individual-{i:02d}", plan["environments"][i], sampler
        )
        individual.append(cell["accuracy"])
    confirm = await evaluate_partition(
        evaluator, experiment, plan, results, "confirmation", "selected", groups, base
    )
    controls = [confirm]
    random_groups = random_partition(groups, plan["seed"])
    if random_groups != groups:
        controls.append(
            await evaluate_partition(
                evaluator,
                experiment,
                plan,
                results,
                "confirmation",
                "random-matched-sizes",
                random_groups,
                base,
            )
        )
    controls.append(
        await evaluate_partition(
            evaluator,
            experiment,
            plan,
            results,
            "confirmation",
            "all-in-one",
            [list(range(plan["size"]))],
            base,
        )
    )
    final = {
        "environments": plan["environments"],
        "selection": selection,
        "confirmation": {
            "base": base,
            "specialist": individual,
            "partitions": controls,
        },
        "positive_within": confirm["within_transfer"] > 0,
        "negative_between": confirm["between_transfer"] is not None
        and confirm["between_transfer"] < 0,
        "interpretation": "Adapter composition is a proxy for joint training; these results do not establish joint-training transfer.",
    }
    write_json(directory / "results.json", final)
    write_json(
        directory / "mopd-clusters.json",
        {
            "selected_k": len(groups),
            "source_experiment": str(experiment),
            "clusters": [
                {
                    "environments": [plan["environments"][i] for i in group],
                    "adapter": adapter,
                    "specialist_teachers": {
                        plan["environments"][i]: results[i]["hf"] for i in group
                    },
                }
                for group, adapter in zip(groups, confirm["adapters"])
            ],
            "confirmation_positive_within": final["positive_within"],
            "confirmation_negative_between": final["negative_between"],
        },
    )
    return final


def run_pipeline(experiment, plan, *, base_url, model_path, gateway):
    from lora_clustering.plot import plot_results
    from lora_clustering.preflight import check_backend

    with (experiment / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(
            experiment / "pipeline-status.json",
            {"stage": "preflight", "time": time.time()},
        )
        asyncio.run(check_backend(experiment, plan, model_path, base_url))
        results = train_teachers(experiment, plan, model_path, base_url, gateway)
        write_json(
            experiment / "pipeline-status.json",
            {"stage": "clustering", "time": time.time()},
        )
        if plan["clustering"].get("mode") == "balanced":
            from lora_clustering.balanced_pipeline import run_balanced

            final = asyncio.run(
                run_balanced(experiment, plan, results, model_path, base_url)
            )
        else:
            final = asyncio.run(
                clustering(experiment, plan, results, model_path, base_url)
            )
            plot_results(experiment, Path(plan["plot_output"]))
        write_json(
            experiment / "completed.json",
            {
                "stage": "complete",
                "teachers": len(results),
                "k": final["selection"]["selected"]["k"],
                "plots": plan["plot_output"],
                "time": time.time(),
            },
        )
        write_json(
            experiment / "pipeline-status.json",
            read_json(experiment / "completed.json"),
        )
