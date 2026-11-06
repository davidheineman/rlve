"""Persistent GRPO specialists and a joint GRPO or mixed-teacher OPD policy."""

import argparse
import asyncio
import copy
import random
import shutil
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from async_lora.replay import backward_batch, score_trajectories, trajectory_metrics
from async_lora.storage import (
    adapter_identity,
    client_state,
    export,
    restore_reference,
    save_bundle,
    uri_directory,
)
from async_lora.worker import Worker
from clustering_mopd.worker import ClusterWorker
from lora_teachers.client import log_metrics
from lora_teachers.sampling import sample_group
from tinker import types

from async_branching.io import read_json, write_json
from transfer_regimes.routing import score_with_teachers


class CountingSampler:
    def __init__(self, sampler):
        self.sampler, self.generated_tokens = sampler, 0

    async def sample_async(self, *args, **kwargs):
        result = await self.sampler.sample_async(*args, **kwargs)
        self.generated_tokens += sum(
            len(sequence.tokens) for sequence in result.sequences
        )
        return result


class SyncWorker(ClusterWorker):
    def open_run(self, directory, project, suffix, config):
        run = Worker.open_run(self, directory, project, suffix, config)
        if self.role == "student":
            run.define_metric("budget/total_training_rollouts")
            run.define_metric(
                "david/eval/*", step_metric="budget/total_training_rollouts"
            )
        return run

    async def overall_evaluation(self, step, clock, *, final=False):
        if (
            not final
            and self.plan["algorithm"] == "joint-rl"
            and step * 8 % self.plan["joint_eval_rollouts"]
        ):
            return
        metrics = {
            "train/step": step,
            "david/step": step,
            "david/rollouts": step * 8,
            "budget/total_training_rollouts": clock,
            "sync/round": getattr(self, "current_round", 0),
            "physical/elapsed_seconds": self.elapsed(),
        }
        panels = (
            [("final_id_envs", self.plan["final_id_eval"])]
            if final
            else [
                ("val_envs", self.plan["id_eval"]),
                ("test_envs", self.plan["test_eval"]),
            ]
        )
        if final:
            panels.append(("final_test_envs", self.plan["test_eval"]))
        for label, path in panels:
            began = time.monotonic()
            result = await self.panel(self.rows(path), self.plan["student_config"])
            metrics[f"david/eval/{label}/pass@1"] = result["accuracy"]
            metrics[f"eval/{label}/seconds"] = time.monotonic() - began
            metrics[f"eval/{label}/samples"] = result["samples"]
            metrics.update(
                {
                    f"david/eval/{label}/{e}/pass@1": v
                    for e, v in result["per_environment"].items()
                }
            )
        log_metrics(self.run, self.directory, metrics)
        if final:
            write_json(self.experiment / "final-evaluation.json", metrics)

    async def gate(self, task, used, settings):
        began = time.monotonic()
        result = await self.panel(
            self.rows(self.plan["teacher_eval"][task["environment"]]), settings
        )
        log_metrics(
            self.run,
            Path(task["directory"]),
            {
                "train/step": used // 8,
                "david/rollouts": used,
                "david/eval/val_envs/pass@1": result["accuracy"],
                "sync/round": task["generation"],
                "eval/gate_seconds": time.monotonic() - began,
                "physical/elapsed_seconds": self.elapsed(),
            },
        )
        return result["accuracy"]

    async def update(self, task, source, settings, teacher_sampler=None, *, used=0):
        sampler = CountingSampler(self.sampler)
        if self.role == "teacher" or self.plan["algorithm"] == "joint-rl":
            # Reuse the tested GRPO objective without changing this persistent
            # worker's student identity or allocating an unused teacher slot.
            trainer = copy.copy(self)
            trainer.role, trainer.sampler = "teacher", sampler
            metrics = await Worker.update(
                trainer, task, source, settings, None, used=used
            )
        else:
            rows = [
                self.generate(source) for _ in range(settings["rollout-batch-size"])
            ]
            groups = await asyncio.gather(
                *(
                    sample_group(sampler, self.tokenizer, p, e, c, settings)
                    for p, e, c, _ in rows
                )
            )
            completions = [c for group in groups for c in group]
            traces = await score_with_teachers(
                rows, completions, self.teacher_samplers, score_trajectories
            )
            metrics = await backward_batch(self.training, traces, [], len(rows))
            metrics.update(trajectory_metrics(traces))
            adam = types.AdamParams(
                learning_rate=settings["lr"],
                beta1=settings["adam-beta1"],
                beta2=settings["adam-beta2"],
                eps=self.plan["optimizer_eps"],
                weight_decay=settings["weight-decay"],
                grad_clip_norm=self.plan["grad_clip_norm"],
            )
            result = await (await self.training.optim_step_async(adam))
            if result.metrics.get("skipped_nonfinite", 0):
                raise RuntimeError("Optimizer skipped a nonfinite update")
            metrics.update(result.metrics)
            metrics.update(
                {
                    "train/accuracy": sum(c.accuracy for c in completions)
                    / len(completions),
                    "train/reward": sum(c.reward for c in completions)
                    / len(completions),
                    "train/difficulty": sum(row[3] for row in rows) / len(rows),
                    "train/lr": settings["lr"],
                }
            )
        return {
            **metrics,
            "train/generated_tokens": sampler.generated_tokens,
            "train/generated_completions": 8,
        }

    async def execute(self, task):
        directory = Path(task["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        self.current_round = task["generation"]
        if task.get("evaluation_only"):
            await self.load(task["parent"]["resume"], optimizer=True)
            self.sampler, self.sampler_uri = await export(
                self.training, self.service, self.root, task["id"], self.sampler_uri
            )
            await self.overall_evaluation(
                task["start_rollouts"] // 8, task["start_work"], final=True
            )
            result = {**task["parent"], "rollouts": 0}
            write_json(directory / "result.json", result)
            return result
        specialist = self.role == "teacher"
        settings = (
            self.plan["teacher_config"]
            if specialist or self.plan["algorithm"] == "joint-rl"
            else self.plan["student_config"]
        )
        if settings["rollout-batch-size"] * settings["n-samples-per-prompt"] != 8:
            raise ValueError(
                "An optimizer update must consume eight training completions"
            )
        shutil.rmtree(directory / "final", ignore_errors=True)
        if specialist:
            self.run = self.open_run(
                directory,
                self.plan["teacher_project"],
                task["id"],
                {**self.plan, **task},
            )
        source, used = self.source(task), 0
        latest = directory / "latest.json"
        if latest.exists():
            checkpoint = read_json(latest)
            await self.load(checkpoint["resume"], optimizer=True)
            saved = client_state(Path(checkpoint["resume"]).parent)
            used, accuracy = saved["used"], saved["accuracy"]
            source.set_state(saved["source"])
            random.setstate(saved["random"])
            np.random.set_state(saved["numpy"])
            torch.set_rng_state(saved["torch"])
        else:
            await self.load(task["parent"]["resume"], optimizer=not specialist)
            accuracy = 0
        self.sampler, self.sampler_uri = await export(
            self.training,
            self.service,
            self.root,
            f"{task['id']}-start-{used}",
            self.sampler_uri,
        )
        initial = directory / "initial-validation.json"
        if specialist and not latest.exists():
            identity = adapter_identity(
                task["parent"]["hf"], uri_directory(self.root, self.sampler_uri)
            )
            write_json(
                directory / "branch_identity.json",
                {**identity, "parent": task["parent"], "optimizer": "fresh"},
            )
            accuracy = (
                read_json(initial)["accuracy"]
                if initial.exists()
                else await self.gate(task, 0, settings)
            )
            write_json(initial, {"accuracy": accuracy})
        previous_exports = []
        with ExitStack() as leases:
            self.teacher_samplers = {}
            if not specialist and self.plan["algorithm"] == "sync-mopd":
                if set(task["teacher_bank"]) != set(task["environments"]):
                    raise ValueError("Student needs all four frozen teachers")
                for environment, teacher in task["teacher_bank"].items():
                    uri = restore_reference(self.root, teacher["hf"], sampling=True)
                    leases.enter_context(self.sampler_lease(uri, shared=True))
                    self.teacher_samplers[
                        environment
                    ] = await self.service.create_sampling_client_async(model_path=uri)
                    previous_exports.append(uri)
            while used < task["cap"] and (
                not specialist or accuracy < self.plan["teacher_validation_threshold"]
            ):
                began = time.monotonic()
                metrics = await self.update(task, source, settings, used=used)
                used += 8
                step = (task["start_rollouts"] + used) // 8
                self.sampler, self.sampler_uri = await export(
                    self.training,
                    self.service,
                    self.root,
                    f"{task['id']}-u{used // 8:05d}",
                    self.sampler_uri,
                )
                metrics.update(
                    {
                        "train/step": step,
                        "david/step": step,
                        "david/rollouts": step * 8,
                        "sync/round": task["generation"],
                        "budget/task_rollouts": used,
                        "train/seconds_per_update": time.monotonic() - began,
                        "sync/task_id": task["id"],
                    }
                )
                if not specialist:
                    metrics["budget/total_training_rollouts"] = (
                        task["start_work"] + used
                    )
                log_metrics(
                    self.run, directory if specialist else self.directory, metrics
                )
                write_json(
                    self.directory / "heartbeat.json",
                    {
                        "task_id": task["id"],
                        "used": used,
                        "time": time.time(),
                        "step": step,
                    },
                )
                if used % 400 and used != task["cap"]:
                    continue
                if specialist:
                    accuracy = await self.gate(task, used, settings)
                else:
                    await self.overall_evaluation(step, task["start_work"] + used)
                if used < task["cap"] and (
                    not specialist
                    or accuracy < self.plan["teacher_validation_threshold"]
                ):
                    bundle = directory / "checkpoints" / f"u{used // 8:05d}"
                    checkpoint = await save_bundle(
                        self.training,
                        self.sampler_uri,
                        self.root,
                        bundle,
                        self.plan,
                        self.rng_state(source, used, accuracy),
                    )
                    previous = read_json(latest) if latest.exists() else None
                    write_json(latest, checkpoint)
                    if previous:
                        shutil.rmtree(Path(previous["hf"]).parent)
            result = await save_bundle(
                self.training,
                self.sampler_uri,
                self.root,
                directory / "final",
                self.plan,
                self.rng_state(source, used, accuracy),
                optimizer=not specialist,
            )
            result.update(
                rollouts=used,
                accuracy=accuracy,
                reason="validation_threshold"
                if specialist and accuracy >= self.plan["teacher_validation_threshold"]
                else "rollout_cap",
            )
            write_json(directory / "result.json", result)
            if latest.exists():
                previous = read_json(latest)
                latest.unlink()
                shutil.rmtree(Path(previous["hf"]).parent)
        for uri in previous_exports:
            self.retire_sampler(uri)
        if specialist:
            self.run.finish()
            self.run = None
        return result


async def run(args):
    worker = SyncWorker(
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
    parser = argparse.ArgumentParser()
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
