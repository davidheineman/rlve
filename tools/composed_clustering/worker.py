"""One inference GPU consumes restartable group-scoring tasks."""

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import signal
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from async_branching.io import read_json, write_json
from composed_clustering.composition import compose
from composed_clustering.sampling import score_cell
from composed_clustering.server import (
    load_adapter,
    start_server,
    stop_server,
    unload_adapter,
)


def run(directory, index):
    directory = Path(directory)
    plan = read_json(directory / "plan.json")
    torch.set_num_threads(2)
    work = directory / "workers" / f"worker-{index}"
    work.mkdir(parents=True, exist_ok=True)
    local = (
        Path("/tmp")
        / f"composed-search-{os.environ.get('SLURM_JOB_ID', os.getpid())}-{index}"
    )
    local.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(local).free < 6 * 2**30:
        raise RuntimeError("Need 6 GiB free node-local /tmp per composition worker")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(local / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(local / "triton")
    paths = plan["adapters"]
    names = plan["environments"]
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model_path"], trust_remote_code=True
    )
    rows = {}
    for split in ["selection", "confirmation"]:
        data = read_json(plan["evaluation"][split]["path"])
        rows[split] = {name: [] for name in names}
        for row in data:
            meta = row["metadata"]
            meta = json.loads(meta) if isinstance(meta, str) else meta
            rows[split][meta["environment"]].append(row)
    port = 40000 + (int(os.environ.get("SLURM_JOB_ID", "0")) % 3000) * 4 + index
    process, log, url = start_server(
        plan["model_path"],
        paths[0],
        work,
        port=port,
        max_rank=plan.get("max_lora_rank", 1024),
    )
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        unload_adapter(url, "native")
        write_json(
            work / "ready.json",
            {
                "time": time.time(),
                "url": url,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory
                / 2**30,
            },
        )
        while not stopping and not (directory / "stop").exists():
            advanced = False
            for task_path in sorted((directory / "tasks").glob("*.json")):
                try:
                    task = read_json(task_path)
                except FileNotFoundError:
                    # Another worker may have completed and moved this task
                    # after the directory snapshot was taken.
                    continue
                result = directory / "scores" / f"{task['id']}.json"
                if result.exists():
                    continue
                with (directory / "task-locks" / f"{task['id']}.lock").open(
                    "a"
                ) as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if result.exists():
                        continue
                    started_at = time.time()
                    write_json(
                        work / "heartbeat.json",
                        {
                            "time": time.time(),
                            "task": task["id"],
                            "split": task["split"],
                            "members": task["members"],
                        },
                    )
                    adapter = None
                    destination = local / "adapter"
                    try:
                        if task["members"] is not None:
                            shutil.rmtree(destination, ignore_errors=True)
                            compose(
                                [paths[i] for i in task["members"]],
                                destination,
                                inference_storage=True,
                            )
                            adapter = "group-" + task["id"]
                            load_adapter(url, adapter, destination)
                        inference_started_at = time.time()
                        cells = []

                        async def evaluate():
                            semaphore = asyncio.Semaphore(plan["concurrency"])

                            async def cell(i):
                                path = (
                                    directory / "cells" / task["id"] / f"{i:02d}.json"
                                )
                                if path.exists():
                                    return read_json(path)
                                measured = await score_cell(
                                    url,
                                    adapter,
                                    names[i],
                                    rows[task["split"]][names[i]],
                                    tokenizer,
                                    plan["settings"],
                                    plan["evaluation"][task["split"]]["samples"],
                                    semaphore,
                                )
                                write_json(path, measured)
                                write_json(
                                    work / "heartbeat.json",
                                    {
                                        "time": time.time(),
                                        "task": task["id"],
                                        "completed_environment": names[i],
                                    },
                                )
                                return measured

                            return await asyncio.gather(
                                *(cell(i) for i in task["targets"])
                            )

                        cells = asyncio.run(evaluate())
                        write_json(
                            result,
                            {
                                "id": task["id"],
                                "members": task["members"],
                                "split": task["split"],
                                "targets": task["targets"],
                                "cells": cells,
                                "accuracy": sum(c["accuracy"] for c in cells)
                                / len(cells),
                                "finished_at": time.time(),
                                "started_at": started_at,
                                "composition_and_load_seconds": inference_started_at
                                - started_at,
                                "inference_seconds": time.time() - inference_started_at,
                            },
                        )
                        task_path.replace(
                            directory / "completed-tasks" / task_path.name
                        )
                    finally:
                        if adapter:
                            unload_adapter(url, adapter)
                        shutil.rmtree(destination, ignore_errors=True)
                    advanced = True
                    break
            if not advanced:
                time.sleep(0.5)
    finally:
        stop_server(process, log)
        shutil.rmtree(local, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    a = p.parse_args()
    try:
        run(a.experiment, a.index)
    except Exception as error:
        import traceback

        write_json(
            a.experiment / "workers" / f"worker-{a.index}" / "error.json",
            {
                "time": time.time(),
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
