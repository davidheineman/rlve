"""One original-style GRPO specialist with durable optimizer and RNG recovery."""

import argparse
import asyncio
import fcntl
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from async_lora.storage import adapter_identity, client_state, export, save_bundle
from async_lora.worker import Worker
from lora_teachers.client import log_metrics, snapshot_directory

from async_branching.io import read_json, write_json


class Teacher(Worker):
    def source(self, task):
        source = super().source(task)
        # Match the original independent-teacher prompt stream (seed 0 onward),
        # rather than the async branch runner's generation-specific prompt seed.
        source.problem_generation_seed = 0
        return source

    async def checkpoint(self, directory, source, used, accuracy):
        latest = directory / "latest.json"
        # The previous checkpoint remains authoritative until this one is durable.
        destination = directory / "checkpoints" / f"u{used // 8:04d}"
        if destination.exists():
            shutil.rmtree(destination)
        saved = await save_bundle(
            self.training,
            self.sampler_uri,
            self.root,
            destination,
            self.plan,
            self.rng_state(source, used, accuracy),
        )
        previous = read_json(latest) if latest.exists() else None
        write_json(latest, saved)
        if previous and Path(previous["hf"]).parent != destination:
            shutil.rmtree(Path(previous["hf"]).parent)

    async def train(self):
        teacher = self.plan["teachers"][self.index]
        directory = self.experiment / "teacher" / teacher["source_label"]
        directory.mkdir(parents=True, exist_ok=True)
        task = dict(
            id=teacher["source_label"],
            kind="teacher",
            environment=teacher["environment"],
            directory=str(directory),
            generation=1,
            start_clock=0,
            train_seed=self.plan["seed"],
        )
        self.run = self.open_run(
            directory,
            self.plan["project"],
            teacher["environment"],
            dict(self.plan, environment=teacher["environment"], teacher=teacher),
        )
        source, used = self.source(task), 0
        settings = self.plan["teacher_config"]
        latest = directory / "latest.json"
        if latest.exists():
            saved = read_json(latest)
            await self.load(saved["resume"], optimizer=True)
            state = client_state(Path(saved["resume"]).parent)
            used, accuracy = state["used"], state["accuracy"]
            source.set_state(state["source"])
            random.setstate(state["random"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
        self.sampler, self.sampler_uri = await export(
            self.training, self.service, self.root, f"start-{used}"
        )
        if latest.exists() and os.environ.get("MOPD_VERIFY_RESUME") == "1":
            verification = adapter_identity(
                saved["hf"], snapshot_directory(self.root, self.sampler_uri)
            )
            write_json(
                directory / f"resume-verification-{os.environ['SLURM_JOB_ID']}.json",
                {
                    **verification,
                    "used": used,
                    "optimizer_loaded": True,
                    "source_checkpoint": saved["resume"],
                    "job_id": os.environ["SLURM_JOB_ID"],
                    "shard": os.environ.get("MOPD_SHARD_INDEX"),
                },
            )
        if not latest.exists():
            initial = directory / "initial-validation.json"
            if initial.exists():
                accuracy = read_json(initial)["accuracy"]
            else:
                accuracy = await self.gate(task, 0, settings)
                write_json(initial, {"accuracy": accuracy})
        cap = self.plan["teacher_rollouts"]
        threshold = self.plan["teacher_validation_threshold"]
        while used < cap and (used == 0 or accuracy < threshold):
            # Original independent teachers always perform at least one evaluation
            # interval, even if the untrained policy passes the initial gate.
            while used < cap:
                began = time.monotonic()
                metrics = await self.update(task, source, settings, None, used=used)
                used += 8
                self.sampler, self.sampler_uri = await export(
                    self.training,
                    self.service,
                    self.root,
                    f"u{used // 8:04d}",
                    self.sampler_uri,
                )
                log_metrics(
                    self.run,
                    directory,
                    dict(
                        metrics,
                        **{
                            "train/step": used // 8,
                            "david/step": used // 8,
                            "david/rollouts": used,
                            "train/seconds_per_update": time.monotonic() - began,
                        },
                    ),
                )
                write_json(
                    self.directory / "heartbeat.json",
                    {"used": used, "step": used // 8, "time": time.time()},
                )
                if (used // 8) % settings["eval-interval"] == 0 or used == cap:
                    break
            accuracy = await self.gate(task, used, settings)
            if used < cap and accuracy < threshold:
                await self.checkpoint(directory, source, used, accuracy)
        # A final directory without its atomic completion marker is uncommitted.
        shutil.rmtree(directory / "final", ignore_errors=True)
        result = await save_bundle(
            self.training,
            self.sampler_uri,
            self.root,
            directory / "final",
            self.plan,
            self.rng_state(source, used, accuracy),
            optimizer=False,
        )
        result.update(
            environment=teacher["environment"],
            training_rollouts=used,
            updates=used // 8,
            accuracy=accuracy,
            stop_reason="validation_accuracy" if accuracy >= threshold else "budget",
            wandb_url=self.run.url,
        )
        write_json(directory / "completed.json", result)
        if latest.exists():
            previous = read_json(latest)
            latest.unlink()
            shutil.rmtree(Path(previous["hf"]).parent)
        self.run.finish()
        self.run = None


async def run(args):
    torch.set_num_threads(1)
    worker = Teacher(
        args.experiment, "teacher", args.index, args.model_path, args.base_url
    )
    ownership = (
        args.experiment / "teacher" / f"teacher-{args.index:02d}" / "ownership.lock"
    )
    ownership.parent.mkdir(parents=True, exist_ok=True)
    lock = ownership.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await worker.initialize()
        await worker.train()
    except BaseException:
        write_json(
            worker.directory / "error.json", {"traceback": traceback.format_exc()}
        )
        if worker.run:
            worker.run.finish(exit_code=1)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
