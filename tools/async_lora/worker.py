"""Persistent CPU worker: one training slot, independent RNG and W&B lifecycle."""

import argparse
import asyncio
import os
import random
import shutil
import time
import traceback
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tinker
import torch
from lora_teachers.client import evaluate, log_metrics
from lora_teachers.objective import grpo_loss
from lora_teachers.sampling import prompt_tokens, sample_group, training_datums
from tinker import types
from transformers import AutoTokenizer

import wandb
from async_branching.io import read_json, write_json
from async_lora.preflight import check_student
from async_lora.questions import SingleQuestionSource, evaluation_rows, load_questions
from async_lora.replay import (
    backward_batch,
    score_trajectories,
    select_replay,
    trace_datums,
    trajectory_metrics,
)
from async_lora.storage import (
    adapter_identity,
    client_state,
    export,
    restore_reference,
    save_bundle,
    uri_directory,
)
from slime.ray.rollout_data_source import RLVEManager
from slime.utils.opd_replay import OPDReplayBuffer


class Worker:
    def __init__(self, experiment, role, index, model_path, base_url):
        self.experiment, self.role, self.index = experiment, role, index
        self.plan = read_json(experiment / "plan.json")
        self.root = Path(os.environ["LORA_CHECKPOINT_ROOT"])
        self.directory = experiment / "workers" / ("student" if role == "student" else f"teacher-{index:02d}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # A shared experiment tenant permits explicit student-to-teacher checkpoint clones.
        # Slots, optimizers, process RNG and runs remain separate.
        self.service = tinker.ServiceClient(base_url=base_url, api_key=f"tml-async-{experiment.name}")
        if self.plan.get("elastic_generation"):
            from async_lora.flex_sampling import ResilientSamplingService

            self.service = ResilientSamplingService(self.service)
        self.training = None
        self.run = None
        self.sampler_uri = None
        self.sampler = None
        self.questions = {}
        if self.plan.get("single_question"):
            data = self.plan["single_question"]
            path = experiment / "source/rlve" / data["data"]
            from lora_teachers.plan import sha256

            if sha256(path) != data["sha256"]:
                raise ValueError("Frozen single-question inputs changed")
            self.questions = load_questions(path, self.plan["environments"])
        self.replay = (
            OPDReplayBuffer(self.plan["opd_replay"]["capacity_per_env"])
            if role == "student" and self.plan.get("opd_replay")
            else None
        )

    async def initialize(self):
        self.training = await self.service.create_lora_training_client_async(
            base_model=self.plan["base_model"],
            rank=self.plan["lora_rank"],
            train_attn=True,
            train_mlp=True,
            train_unembed=False,
        )
        if self.role == "student":
            self.run = self.open_run(self.directory, self.plan["student_project"], "student", self.plan)
            if not (self.experiment / "base.json").exists():
                initial = self.plan.get("initial_student_reference")
                if initial:
                    bundle = self.experiment / initial["bundle"]
                    await self.load(str(bundle / "state"), optimizer=False)
                self.sampler, self.sampler_uri = await export(self.training, self.service, self.root, "base")
                if initial:
                    identity = adapter_identity(bundle / "adapter", uri_directory(self.root, self.sampler_uri))
                    write_json(self.experiment / "initial-student-identity.json", identity)
                result = await save_bundle(
                    self.training,
                    self.sampler_uri,
                    self.root,
                    self.experiment / "checkpoints/base",
                    self.plan,
                    replay=self.replay,
                )
                write_json(self.experiment / "base.json", result)
            if (
                not (self.experiment / "preflight/passed.json").exists()
                and not (self.experiment / "state.json").exists()
            ):
                await check_student(self, read_json(self.experiment / "base.json"))
            if not (self.experiment / "baseline.json").exists():
                if self.sampler is None:
                    await self.load(read_json(self.experiment / "base.json")["resume"], optimizer=True)
                    self.sampler, self.sampler_uri = await export(self.training, self.service, self.root, "baseline")
                await self.overall_evaluation(0, 0)
                write_json(self.experiment / "baseline.json", {"complete": True})
        write_json(self.directory / "ready.json", {"model_id": self.training.model_id, "role": self.role})

    def open_run(self, directory, project, suffix, config):
        identity = directory / "identity.json"
        previous = read_json(identity) if identity.exists() else {}
        run = wandb.init(
            entity=self.plan["entity"],
            project=project,
            group=self.experiment.name,
            name=f"{self.experiment.name}-{suffix}",
            dir=str(directory),
            config=config,
            id=previous.get("wandb_id"),
            resume="allow" if previous else None,
        )
        run.define_metric("train/step")
        run.define_metric("*", step_metric="train/step")
        write_json(identity, {"wandb_id": run.id, "wandb_url": run.url, "project": project})
        return run

    def rows(self, relative):
        return read_json(self.experiment / "source/rlve" / relative)

    async def evaluate_rows(self, rows, settings):
        return await evaluate(
            self.sampler, self.tokenizer, rows, settings, concurrency=self.plan.get("evaluation_concurrency", 2)
        )

    async def overall_evaluation(self, step, clock):
        metrics = {"train/step": step, "david/step": step, "david/rollouts": step * 8, "async/simulated_clock": clock}
        settings = self.plan["student_config"]
        for key, path in (("val_envs", self.plan["id_eval"]), ("test_envs", self.plan["test_eval"])):
            result = await self.evaluate_rows(self.rows(path), settings)
            metrics.update({name.replace("val_envs", key): value for name, value in result.items()})
        if self.questions:
            result = await self.evaluate_rows(
                evaluation_rows(self.questions, count=self.plan["eval_prompts"]), settings
            )
            accuracy = result["david/eval/val_envs/pass@1"]
            metrics.update({"same_question_val_acc": accuracy, "david/eval/same_question_val_acc": accuracy})
        log_metrics(self.run, self.directory, metrics)

    def source(self, task):
        settings = self.plan["student_config" if self.role == "student" else "teacher_config"]
        random.seed(task["train_seed"])
        np.random.seed(task["train_seed"] % 2**32)
        torch.manual_seed(task["train_seed"])
        if self.questions:
            return SingleQuestionSource(self.questions[task["environment"]])
        source = RLVEManager(
            SimpleNamespace(
                environment_list=[task["environment"]],
                initial_difficulty=settings["initial-difficulty"],
                rlve_balanced_difficulty=settings["rlve-balanced-difficulty"],
                rlve_prompt_pool=None,
                difficulty_sliding_window_size=settings["difficulty-sliding-window-size"],
            ),
            self.tokenizer,
        )
        source.problem_generation_seed = task["train_seed"]
        return source

    def generate(self, source):
        for _ in range(1000):
            environment, difficulty, problem = source.generate_problem()
            if problem is not None:
                return (
                    prompt_tokens(self.tokenizer, problem.prompt_generator()),
                    environment,
                    problem.get_config(),
                    difficulty,
                )
        raise RuntimeError("Failed to generate a scoreable RLVE prompt")

    async def load(self, path, *, optimizer):
        uri = restore_reference(self.root, path)
        future = (
            await self.training.load_state_with_optimizer_async(uri)
            if optimizer
            else await self.training.load_state_async(uri)
        )
        await future

    async def gate(self, task, used, settings):
        result = await self.evaluate_rows(self.rows(self.plan["teacher_eval"][task["environment"]]), settings)
        accuracy = result["david/eval/val_envs/pass@1"]
        step = (task.get("start_rollouts", 0) + used) // 8
        if self.role == "student":
            result = {k.replace("val_envs", "gate_envs"): v for k, v in result.items()}
            result["async/recovery_target"] = task["target"]
        if self.questions:
            same = await self.evaluate_rows(
                evaluation_rows(self.questions, task["environment"], self.plan["eval_prompts"]), settings
            )
            accuracy = same["david/eval/val_envs/pass@1"]
            if self.role == "student":
                result["david/eval/gate_envs/same_question_val_acc"] = accuracy
            else:
                result.update({"same_question_val_acc": accuracy, "david/eval/same_question_val_acc": accuracy})
        metrics = {
            "train/step": step,
            "david/step": step,
            "david/rollouts": step * 8,
            "async/simulated_clock": task["start_clock"] + used,
            "async/teacher_generation": task["generation"],
            **result,
        }
        log_metrics(self.run, self.directory if self.role == "student" else Path(task["directory"]), metrics)
        return accuracy

    async def update(self, task, source, settings, teacher_sampler, *, used=0):
        batch_size = settings["rollout-batch-size"]
        replayed = []
        if self.role == "student":
            replayed = select_replay(
                self.replay,
                task["environment"],
                (task["start_rollouts"] + used) // 8,
                task["train_seed"],
                batch_size,
                self.plan.get("opd_replay", {}).get("fraction", 0),
            )
        rows = [self.generate(source) for _ in range(batch_size - len(replayed))]
        groups = await asyncio.gather(
            *(sample_group(self.sampler, self.tokenizer, p, e, c, settings) for p, e, c, _ in rows)
        )
        completions = [c for group in groups for c in group]
        if self.role == "teacher":
            datums, masks, logprobs = training_datums(rows[0][0], completions)
            objective = partial(
                grpo_loss,
                rollout_logprobs=logprobs,
                masks=masks,
                rewards=[c.reward for c in completions],
                eps_low=settings["eps-clip"],
                eps_high=settings["eps-clip-high"],
                tis_low=self.plan["tis_clip_low"],
                tis_high=self.plan["tis_clip_high"],
            )
            result = await (await self.training.forward_backward_custom_async(datums, lambda _data, lp: objective(lp)))
            metrics = result.metrics
        else:
            fresh = await score_trajectories(rows, completions, teacher_sampler)
            if self.questions:
                for trace in fresh:
                    trace["metadata"]["question_id"] = self.questions[trace["metadata"]["environment"]]["metadata"][
                        "question_id"
                    ]
            metrics = await backward_batch(
                self.training, fresh, replayed, batch_size, self.plan.get("opd_replay", {}).get("clip_epsilon", 0.2)
            )
            metrics.update(trajectory_metrics(fresh + replayed))
        adam = types.AdamParams(
            learning_rate=settings["lr"],
            beta1=settings["adam-beta1"],
            beta2=settings["adam-beta2"],
            eps=self.plan["optimizer_eps"],
            weight_decay=settings["weight-decay"],
            grad_clip_norm=self.plan["grad_clip_norm"],
        )
        optimizer = await (await self.training.optim_step_async(adam))
        if optimizer.metrics.get("skipped_nonfinite", 0):
            raise RuntimeError("Optimizer skipped a nonfinite update")
        if self.role == "student" and self.replay is not None:
            self.replay.add(fresh)
            metrics.update(
                {
                    "david/opd/replay_fraction": len(replayed) / batch_size,
                    "david/opd/replay_trajectories": len(replayed),
                    "david/opd/fresh_trajectories": len(fresh),
                    "david/opd/fresh_trajectories_total": self.replay.fresh_count,
                    "david/opd/replay_trajectories_total": self.replay.cursor,
                    "david/opd/buffer_trajectories": self.replay.size,
                    "david/opd/buffer_environments": len(self.replay.entries),
                }
            )
        return {
            "train/reward": sum(c.reward for c in completions) / len(completions),
            "train/accuracy": sum(c.accuracy for c in completions) / len(completions),
            "train/difficulty": sum(row[3] for row in rows) / len(rows),
            "train/lr": settings["lr"],
            **metrics,
            **optimizer.metrics,
        }

    async def distillation_datums(self, rows, completions, teacher_sampler):
        traces = await score_trajectories(rows, completions, teacher_sampler)
        return trace_datums(traces, len(traces)), trajectory_metrics(traces)

    def restore_replay(self, checkpoint):
        if self.role != "student" or not self.plan.get("opd_replay"):
            return
        path = checkpoint.get("replay")
        if not path or not Path(path).is_file():
            raise RuntimeError("Student continuation is missing its committed replay buffer")
        self.replay = OPDReplayBuffer.load(path, self.plan["opd_replay"]["capacity_per_env"])
        if self.questions:
            for environment, traces in self.replay.entries.items():
                expected = self.questions[environment]["metadata"]["question_id"]
                if any(trace["metadata"].get("question_id") != expected for trace in traces):
                    raise ValueError("Replay buffer contains a different fixed training question")

    def rng_state(self, source, used, accuracy):
        return {
            "used": used,
            "accuracy": accuracy,
            "source": source.get_state(),
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

    async def execute(self, task):
        directory = Path(task["directory"])
        # A final bundle without result.json was never committed to the scheduler.
        shutil.rmtree(directory / "final", ignore_errors=True)
        settings = self.plan["student_config" if self.role == "student" else "teacher_config"]
        evaluation_interval = settings["eval-interval"]
        evaluation_rollouts = evaluation_interval * settings["rollout-batch-size"] * settings["n-samples-per-prompt"]
        if self.role == "teacher":
            self.run = self.open_run(directory, self.plan["teacher_project"], task["id"], {**self.plan, **task})
        source, used = self.source(task), 0
        latest = directory / "latest.json"
        if latest.exists():
            checkpoint = read_json(latest)
            await self.load(checkpoint["resume"], optimizer=True)
            self.restore_replay(checkpoint)
            saved = client_state(Path(checkpoint["resume"]).parent)
            used = saved["used"]
            source.set_state(saved["source"])
            random.setstate(saved["random"])
            np.random.set_state(saved["numpy"])
            torch.set_rng_state(saved["torch"])
        else:
            await self.load(task["parent"]["resume"], optimizer=self.role == "student")
            self.restore_replay(task["parent"])
        self.sampler, self.sampler_uri = await export(
            self.training, self.service, self.root, f"{task['id']}-start-{used}"
        )
        if self.role == "teacher" and not latest.exists():
            identity = adapter_identity(task["parent"]["hf"], uri_directory(self.root, self.sampler_uri))
            write_json(directory / "branch_identity.json", {**identity, "parent": task["parent"], "optimizer": "fresh"})
        teacher_sampler = None
        if self.role == "student":
            teacher_uri = restore_reference(self.root, task["teacher_hf"], sampling=True)
            teacher_sampler = await self.service.create_sampling_client_async(model_path=teacher_uri)
        # Reuse a measured gate on replay. Re-sampling an earlier gate could
        # stop before a rollout lower bound already published to the scheduler.
        initial_gate = directory / "initial-validation.json"
        if latest.exists():
            accuracy = saved["accuracy"]
        elif initial_gate.exists():
            accuracy = read_json(initial_gate)["accuracy"]
        else:
            accuracy = await self.gate(task, used, settings)
            write_json(
                initial_gate,
                {
                    "accuracy": accuracy,
                    "accuracy_metric": "same_question_val_acc"
                    if self.plan.get("single_question")
                    else "randomized_val_acc",
                },
            )
        threshold = task["target"] if self.role == "student" else self.plan["teacher_validation_threshold"]
        while used < task["cap"] and accuracy + 1e-12 < threshold:
            if self.role == "teacher":
                write_json(
                    directory / "progress.json",
                    {
                        "rollouts": used,
                        "accuracy": accuracy,
                        "accuracy_metric": "same_question_val_acc"
                        if self.plan.get("single_question")
                        else "randomized_val_acc",
                        "next_validation_rollouts": min(
                            task["cap"], ((used // evaluation_rollouts) + 1) * evaluation_rollouts
                        ),
                    },
                )
            while used < task["cap"]:
                began = time.monotonic()
                metrics = await self.update(task, source, settings, teacher_sampler, used=used)
                used += 8
                step = (task.get("start_rollouts", 0) + used) // 8
                self.sampler, self.sampler_uri = await export(
                    self.training, self.service, self.root, f"{task['id']}-u{used // 8:04d}", self.sampler_uri
                )
                metrics.update(
                    {
                        "train/step": step,
                        "david/step": step,
                        "david/rollouts": step * 8,
                        "async/simulated_clock": task["start_clock"] + used,
                        "async/teacher_device_rollouts": task.get("teacher_start_rollouts", 0) + used,
                        "train/seconds_per_update": time.monotonic() - began,
                        "async/teacher_generation": task["generation"],
                    }
                )
                log_metrics(self.run, self.directory if self.role == "student" else directory, metrics)
                write_json(
                    self.directory / "heartbeat.json",
                    {"task_id": task["id"], "used": used, "time": time.time(), "step": step},
                )
                if step % evaluation_interval == 0 or used == task["cap"]:
                    break
            accuracy = await self.gate(task, used, settings)
            if self.role == "student":
                await self.overall_evaluation(step, task["start_clock"] + used)
            if used < task["cap"] and accuracy + 1e-12 < threshold:
                bundle = directory / "checkpoints" / f"u{used // 8:04d}"
                checkpoint = await save_bundle(
                    self.training,
                    self.sampler_uri,
                    self.root,
                    bundle,
                    self.plan,
                    self.rng_state(source, used, accuracy),
                    replay=getattr(self, "replay", None),
                )
                previous = read_json(latest) if latest.exists() else None
                write_json(latest, checkpoint)
                if previous:
                    shutil.rmtree(Path(previous["hf"]).parent)
        if self.role == "student" and used == 0:
            result = {"hf": task["parent"]["hf"], "resume": task["parent"]["resume"]}
            if self.plan.get("opd_replay"):
                result["replay"] = task["parent"]["replay"]
        else:
            result = await save_bundle(
                self.training,
                self.sampler_uri,
                self.root,
                directory / "final",
                self.plan,
                self.rng_state(source, used, accuracy),
                optimizer=self.role == "student",
                replay=getattr(self, "replay", None),
            )
        result.update(
            rollouts=used,
            accuracy=accuracy,
            reason="recovered" if accuracy + 1e-12 >= threshold else "rollout_cap",
            accuracy_metric="same_question_val_acc" if self.plan.get("single_question") else "randomized_val_acc",
        )
        write_json(directory / "result.json", result)
        if latest.exists():
            previous = read_json(latest)
            latest.unlink()
            shutil.rmtree(Path(previous["hf"]).parent)
        if self.role == "teacher":
            self.run.finish()
            self.run = None
        return result

    async def loop(self):
        await self.initialize()
        inbox = self.directory / "inbox.json"
        while not (self.experiment / "workers.stop").exists():
            if not inbox.exists():
                await asyncio.sleep(0.2)
                continue
            task = read_json(inbox)
            inbox.unlink()
            await self.execute(task)
        if self.run:
            self.run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--role", choices=("student", "teacher"), required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    worker = Worker(args.experiment, args.role, args.index, args.model_path, args.base_url)
    try:
        asyncio.run(worker.loop())
    except BaseException:
        write_json(worker.directory / "error.json", {"traceback": traceback.format_exc()})
        if worker.run:
            worker.run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
