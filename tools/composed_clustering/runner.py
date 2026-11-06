"""Freeze selection-only assignments, then independently score group ID gain."""

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from async_branching.io import read_json, write_json
from composed_clustering.search import (
    canonical,
    random_partition,
    refine,
)


class Evaluator:
    def __init__(self, directory, plan, workers):
        self.directory, self.plan, self.workers = directory, plan, workers

    def groups(self, groups, split, *, base=False):
        ids = []
        group_ids = []
        for group in groups:
            keys = []
            # Large groups are evaluated by several GPUs using the same full
            # composition, with disjoint target-task chunks.
            targets = sorted(group)
            for start in range(0, len(targets), 8):
                raw = {
                    "members": None if base else sorted(group),
                    "targets": targets[start : start + 8],
                    "split": split,
                }
                key = hashlib.sha256(
                    json.dumps(raw, sort_keys=True).encode()
                ).hexdigest()[:24]
                ids.append(key)
                keys.append(key)
                task = self.directory / "tasks" / f"{key}.json"
                if (
                    not task.exists()
                    and not (self.directory / "scores" / f"{key}.json").exists()
                ):
                    write_json(task, {"id": key, **raw})
            group_ids.append(keys)
        deadline = time.monotonic() + self.plan.get("evaluation_timeout_seconds", 7200)
        while not all(
            (self.directory / "scores" / f"{key}.json").exists() for key in ids
        ):
            if any(p.poll() is not None for p in self.workers):
                raise RuntimeError(
                    "Inference worker exited; inspect workers/*/worker.log"
                )
            if time.monotonic() > deadline:
                raise TimeoutError("Group evaluation exceeded two hours")
            time.sleep(0.5)
        return [
            {
                "cells": [
                    cell
                    for key in keys
                    for cell in read_json(self.directory / "scores" / f"{key}.json")[
                        "cells"
                    ]
                ]
            }
            for keys in group_ids
        ]

    def partitions(self, groups_list, split):
        self.groups([group for groups in groups_list for group in groups], split)
        return [self.partition(groups, split) for groups in groups_list]

    def partition(self, groups, split):
        groups = canonical(groups)
        scored = self.groups(groups, split)
        values = {
            cell["environment"]: cell["accuracy"]
            for group in scored
            for cell in group["cells"]
        }
        if set(values) != set(self.plan["environments"]):
            raise ValueError("Partition score omitted tasks")
        vector = [values[name] for name in self.plan["environments"]]
        return {
            "groups": groups,
            "accuracy": float(np.mean(vector)),
            "per_environment": vector,
            "split": split,
        }


def execute(directory):
    directory = Path(directory)
    plan = read_json(directory / "plan.json")
    if (directory / "completed.json").exists():
        return
    workers = []
    logs = []
    for name in [
        "tasks",
        "scores",
        "cells",
        "workers",
        "task-locks",
        "completed-tasks",
    ]:
        (directory / name).mkdir(exist_ok=True)
    (directory / "stop").unlink(missing_ok=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if len(visible) != plan["gpus"]:
        raise RuntimeError("Expected exactly the requested GPU allocation")

    def shutdown(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    try:
        for i in range(plan["gpus"]):
            work = directory / "workers" / f"worker-{i}"
            work.mkdir(exist_ok=True)
            for name in ["ready.json", "error.json"]:
                (work / name).unlink(missing_ok=True)
            log = (work / "worker.log").open("a")
            logs.append(log)
            workers.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "composed_clustering.worker",
                        "--experiment",
                        str(directory),
                        "--index",
                        str(i),
                    ],
                    env=os.environ | {"CUDA_VISIBLE_DEVICES": visible[i]},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 1200
        while not all(
            (directory / "workers" / f"worker-{i}" / "ready.json").exists()
            for i in range(plan["gpus"])
        ):
            if any(p.poll() is not None for p in workers):
                raise RuntimeError("Inference worker failed during startup")
            if time.monotonic() > deadline:
                raise TimeoutError("Worker initialization exceeded 20 minutes")
            time.sleep(2)
        evaluator = Evaluator(directory, plan, workers)
        frozen = directory / "assignments.json"
        if not frozen.exists():
            positive = read_json(plan["positive_results"])
            low = read_json(plan["low_results"])
            transfer = np.asarray(read_json(plan["transfer"]))
            selected = []
            for k in plan["ks"]:
                checkpoint = directory / f"selected-k{k:02d}.json"
                if checkpoint.exists():
                    selected.append(read_json(checkpoint))
                    continue
                random = [
                    random_partition(len(plan["environments"]), k, seed)
                    for seed in plan["random_seeds"]
                ]
                high_seed = next(
                    r["fit"]["groups"] for r in positive["candidates"] if r["k"] == k
                )
                low_seed = next(
                    r["fit"]["groups"] for r in low["candidates"] if r["k"] == k
                )
                initial = [*random, high_seed, low_seed]

                def score(groups):
                    return evaluator.partition(groups, "selection")["accuracy"]

                def score_many(groups_list):
                    return [
                        r["accuracy"]
                        for r in evaluator.partitions(groups_list, "selection")
                    ]

                cases = {"k": k, "random_groups": random}
                for label, maximize in [("best", True), ("worst", False)]:
                    write_json(
                        directory / "status.json",
                        {
                            "stage": "selection_search",
                            "k": k,
                            "direction": label,
                            "time": time.time(),
                        },
                    )
                    cases[label] = refine(
                        initial,
                        score,
                        transfer,
                        maximize=maximize,
                        rounds=plan["rounds"],
                        batch=plan["proposals_per_round"],
                        seed=plan["seed"] + 100 * k + int(maximize),
                        evaluate_many=score_many,
                        on_round=lambda r: write_json(
                            directory / "search-progress.json",
                            {
                                "k": k,
                                "direction": label,
                                "record": r,
                                "time": time.time(),
                            },
                        ),
                    )
                write_json(checkpoint, cases)
                selected.append(cases)
            # Never read confirmation outcomes before all assignments are fixed.
            write_json(
                frozen,
                {
                    "candidates": selected,
                    "frozen_at": time.time(),
                    "used_confirmation": False,
                },
            )
        selection = read_json(frozen)
        write_json(
            directory / "status.json",
            {"stage": "held_out_evaluation", "time": time.time()},
        )
        n = len(plan["environments"])
        chunks = [list(range(i, n, plan["gpus"])) for i in range(plan["gpus"])]
        base = evaluator.groups(chunks, "confirmation", base=True)
        base_map = {
            cell["environment"]: cell["accuracy"]
            for block in base
            for cell in block["cells"]
        }
        records = []
        for row in selection["candidates"]:
            checkpoint = directory / f"held-k{row['k']:02d}.json"
            if checkpoint.exists():
                records.append(read_json(checkpoint))
                continue
            record = {
                "k": row["k"],
                "best": evaluator.partition(row["best"]["groups"], "confirmation"),
                "worst": evaluator.partition(row["worst"]["groups"], "confirmation"),
                "random": [
                    evaluator.partition(g, "confirmation") for g in row["random_groups"]
                ],
            }
            write_json(checkpoint, record)
            records.append(record)
            from composed_clustering.plot import render

            render(directory, records, base_map)
        singleton = evaluator.partition([[i] for i in range(n)], "confirmation")
        result = {
            "candidates": records,
            "base": base_map,
            "singleton": singleton,
            "complete": True,
            "finished_at": time.time(),
        }
        from composed_clustering.plot import render

        render(directory, records, base_map, singleton=singleton)
        write_json(directory / "completed.json", result)
        write_json(
            directory / "status.json", {"stage": "complete", "time": time.time()}
        )
    finally:
        (directory / "stop").touch()
        for p in workers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    with (args.experiment / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args.experiment)
