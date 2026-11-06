"""Check rank-1024 composition, actual RLVE scoring, and memory on one GPU."""

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from composed_clustering.composition import compose
from composed_clustering.sampling import score_cell
from composed_clustering.server import (
    load_adapter,
    start_server,
    stop_server,
    unload_adapter,
)


def probe(experiment, output):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((experiment / "plan.json").read_text())
    data = json.loads(Path(plan["evaluation"]["selection"]["path"]).read_text())
    local = Path("/tmp") / f"composed-hardware-{os.environ['SLURM_JOB_ID']}"
    local.mkdir(exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(local / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(local / "triton")
    process, log, url = start_server(
        plan["model_path"], plan["adapters"][0], output, port=17601, max_rank=1024
    )
    try:
        unload_adapter(url, "native")
        compose(plan["adapters"][:32], local / "adapter", inference_storage=True)
        load_adapter(url, "distinct", local / "adapter")
        tokenizer = AutoTokenizer.from_pretrained(plan["model_path"])

        async def evaluate():
            semaphore = asyncio.Semaphore(32)
            cells = []
            for name in plan["environments"][:8]:
                rows = []
                for row in data:
                    meta = row["metadata"]
                    meta = json.loads(meta) if isinstance(meta, str) else meta
                    if meta["environment"] == name:
                        rows.append(row)
                cells.append(
                    score_cell(
                        url,
                        "distinct",
                        name,
                        rows[:2],
                        tokenizer,
                        plan["settings"],
                        2,
                        semaphore,
                    )
                )
            return await asyncio.gather(*cells)

        started = time.time()
        cells = asyncio.run(evaluate())
        record = {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "rank": 1024,
            "concurrent_completions": 32,
            "inference_seconds": time.time() - started,
            "cells": cells,
        }
        (output / "passed.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        unload_adapter(url, "distinct")
    finally:
        stop_server(process, log)
        shutil.rmtree(local, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.experiment, args.output)
