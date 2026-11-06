"""Multi-environment policies; retain the existing GRPO and native OPD updates."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import random
import shutil
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from async_lora.storage import export, uri_directory
from async_lora.worker import Worker
from lora_teachers.client import log_metrics
from lora_teachers.sampling import prompt_tokens, sample_group

from async_branching.io import read_json, write_json
from slime.ray.rollout_data_source import RLVEManager


class ClusterWorker(Worker):
    @contextmanager
    def sampler_lease(self, uri, *, shared=False):
        directory = self.root / ".sampler-leases"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (hashlib.sha256(uri.encode()).hexdigest() + ".lock")
        with path.open("a") as lock:
            # Only one student consumes a teacher at once. Its shared lease
            # keeps the immutable export available for SGLang cache backfills.
            fcntl.flock(
                lock, fcntl.LOCK_SH if shared else fcntl.LOCK_EX | fcntl.LOCK_NB
            )
            yield

    def retire_sampler(self, uri):
        try:
            with self.sampler_lease(uri):
                shutil.rmtree(uri_directory(self.root, uri))
            return True
        except FileNotFoundError:
            return True
        except BlockingIOError:
            # The student reclaims this export after its current merge ends.
            return False

    async def load(self, path, *, optimizer):
        # Many lanes can import one parent concurrently. Hold a separate lease
        # through loading and reclamation so a sibling never loses its source.
        # The frozen durable bundle remains available for every future import.
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".cluster-load.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            await super().load(path, optimizer=optimizer)
            uri = read_json(Path(path).parent / "reference.json")["state_uri"]
            shutil.rmtree(uri_directory(self.root, uri))

    def source(self, task):
        settings = self.plan[
            "student_config" if self.role == "student" else "teacher_config"
        ]
        random.seed(task["train_seed"])
        np.random.seed(task["train_seed"] % 2**32)
        torch.manual_seed(task["train_seed"])
        source = RLVEManager(
            SimpleNamespace(
                environment_list=task["environments"],
                initial_difficulty=settings["initial-difficulty"],
                rlve_balanced_difficulty=settings["rlve-balanced-difficulty"],
                rlve_prompt_pool=None,
                difficulty_sliding_window_size=settings[
                    "difficulty-sliding-window-size"
                ],
            ),
            self.tokenizer,
        )
        source.problem_generation_seed = task["train_seed"]
        return source

    def open_run(self, directory, project, suffix, config):
        run = super().open_run(directory, project, suffix, config)
        run.define_metric("async/simulated_clock")
        run.define_metric("david/eval/*", step_metric="async/simulated_clock")
        run.define_metric("physical/elapsed_seconds")
        return run

    async def panel(self, rows, settings):
        semaphore = asyncio.Semaphore(self.plan.get("evaluation_concurrency", 2))

        async def score(row):
            async with semaphore:
                meta = row["metadata"]
                meta = json.loads(meta) if isinstance(meta, str) else meta
                values = await sample_group(
                    self.sampler,
                    self.tokenizer,
                    prompt_tokens(self.tokenizer, row["user_prompt"]),
                    meta["environment"],
                    meta["config"],
                    settings,
                    evaluation=True,
                )
                return meta["environment"], values

        groups = await asyncio.gather(*(score(row) for row in rows))
        by_env = {}
        for environment, completions in groups:
            by_env.setdefault(environment, []).extend(completions)
        accuracy = {
            e: sum(c.accuracy for c in values) / len(values)
            for e, values in by_env.items()
        }
        values = [c for _, group in groups for c in group]
        return {
            "accuracy": sum(accuracy.values()) / len(accuracy),
            "per_environment": accuracy,
            "answer_forcing_fraction": sum(c.forced for c in values) / len(values),
            "truncated_fraction": sum(c.truncated for c in values) / len(values),
            "samples": len(values),
        }

    def elapsed(self):
        return time.time() - read_json(self.experiment / "physical-start.json")["time"]

    async def gate(self, task, used, settings):
        cluster = next(
            c for c in self.plan["clusters"] if c["id"] == task["environment"]
        )
        began = time.monotonic()
        result = await self.panel(self.rows(cluster["eval"]), settings)
        step = (task.get("start_rollouts", 0) + used) // 8
        namespace = "gate_envs" if self.role == "student" else "val_envs"
        metrics = {
            "train/step": step,
            "david/step": step,
            "david/rollouts": step * 8,
            "async/simulated_clock": task["start_clock"] + used,
            "async/teacher_generation": task["generation"],
            f"david/eval/{namespace}/pass@1": result["accuracy"],
            "physical/elapsed_seconds": self.elapsed(),
            "clustering/gate_seconds": time.monotonic() - began,
            "clustering/gate_samples": result["samples"],
            **{
                f"david/eval/{namespace}/{e}/pass@1": value
                for e, value in result["per_environment"].items()
            },
        }
        if self.role == "student":
            metrics["async/recovery_target"] = task["target"]
        log_metrics(
            self.run,
            self.directory if self.role == "student" else Path(task["directory"]),
            metrics,
        )
        destination = (
            Path(task["directory"]) / "gate-details" / f"u{used // 8:04d}.json"
        )
        write_json(destination, result)
        return result["accuracy"]

    async def overall_evaluation(self, step, clock):
        metrics = {
            "train/step": step,
            "david/step": step,
            "david/rollouts": step * 8,
            "async/simulated_clock": clock,
            "physical/elapsed_seconds": self.elapsed(),
        }
        settings = self.plan["student_config"]
        for label, path in (
            ("val_envs", self.plan["id_eval"]),
            ("test_envs", self.plan["test_eval"]),
        ):
            began = time.monotonic()
            result = await self.panel(self.rows(path), settings)
            metrics.update(
                {
                    f"david/eval/{label}/pass@1": result["accuracy"],
                    f"eval/{label}/answer_forcing_fraction": result[
                        "answer_forcing_fraction"
                    ],
                    f"eval/{label}/truncated_fraction": result["truncated_fraction"],
                    f"clustering/{label}_seconds": time.monotonic() - began,
                    **{
                        f"david/eval/{label}/{e}/pass@1": v
                        for e, v in result["per_environment"].items()
                    },
                }
            )
            if label == "val_envs":
                legacy = [
                    result["per_environment"][e]
                    for e in self.plan["legacy_id_environments"]
                ]
                metrics["david/eval/reference_val_envs/pass@1"] = sum(legacy) / len(
                    legacy
                )
        log_metrics(self.run, self.directory, metrics)

    async def execute(self, task):
        if not task.get("evaluation_only"):
            previous = self.sampler_uri
            teacher_uri = None
            if self.role == "student":
                reference = read_json(
                    Path(task["teacher_hf"]).parent / "reference.json"
                )
                teacher_uri = reference["sampler_uri"]
            lease = (
                self.sampler_lease(teacher_uri, shared=True)
                if teacher_uri
                else nullcontext()
            )
            with lease:
                result = await super().execute(task)
            obsolete = (
                {previous} if previous and previous != self.sampler_uri else set()
            )
            if teacher_uri:
                obsolete.add(teacher_uri)
            for uri in obsolete:
                self.retire_sampler(uri)
            return result
        await self.load(task["parent"]["resume"], optimizer=True)
        self.sampler, self.sampler_uri = await export(
            self.training, self.service, self.root, task["id"], self.sampler_uri
        )
        await self.overall_evaluation(task["start_rollouts"] // 8, task["start_clock"])
        result = {
            "hf": task["parent"]["hf"],
            "resume": task["parent"]["resume"],
            "rollouts": 0,
            "evaluation_only": True,
        }
        write_json(Path(task["directory"]) / "result.json", result)
        return result


async def run(args):
    worker = ClusterWorker(
        args.experiment, args.role, args.index, args.model_path, args.base_url
    )
    try:
        await worker.loop()
    except BaseException:
        write_json(
            worker.directory / "error.json", {"traceback": traceback.format_exc()}
        )
        if worker.run:
            worker.run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--role", choices=("student", "teacher"), required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
