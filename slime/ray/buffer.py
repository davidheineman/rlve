import logging
import os
import random
import statistics
from math import sqrt
from pathlib import Path
from time import time
from typing import Union

import numpy as np
import ray
import torch
import wandb

from slime.ray.rollout_data_source import RolloutDataSourceWithBuffer
from slime.utils.eval_metrics import early_stop_accuracy, pass_at_k
from slime.utils.eval_rollouts import update_eval_rollout_tables
from slime.utils.misc import load_function
from slime.utils.opd_replay import OPDReplayBuffer, replay_count
from slime.utils.rollout_export import export_training_rollouts
from slime.utils.ray_utils import Box
from slime.utils.types import Sample
from slime.utils.wandb_utils import init_wandb_secondary

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@ray.remote
class RolloutController:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, wandb_run_id):
        self.args = args
        init_wandb_secondary(args, wandb_run_id)
        self.eval_rollout_tables = {}

        self.data_source = RolloutDataSourceWithBuffer(args)
        self.opd_replay = None
        if getattr(args, "opd_replay_fraction", 0):
            self.opd_replay = (OPDReplayBuffer.load(args.opd_replay_load, args.opd_replay_capacity_per_env)
                               if args.opd_replay_load else OPDReplayBuffer(args.opd_replay_capacity_per_env))

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        self.rollout_id = rollout_id

        start_time = time()
        replay_samples = []
        if self.opd_replay is not None:
            count = replay_count(rollout_id, self.args.rollout_batch_size, self.args.opd_replay_fraction)
            replay_samples = [Sample.from_dict(t) for t in self.opd_replay.sample(
                count, excluded_environments=self.args.environment_list,
                seed=self.args.seed + rollout_id,
            )]
        if self.args.load_debug_rollout_data:
            if self.opd_replay is not None:
                raise ValueError("Debug rollout loading cannot be combined with OPD replay")
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
        else:
            self.args.opd_fresh_batch_size = (self.args.rollout_batch_size - len(replay_samples)
                                               if self.opd_replay is not None else None)
            try:
                data = self.generate_rollout(self.args, rollout_id, self.data_source, evaluation=False)
            finally:
                self.args.opd_fresh_batch_size = None
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = sum(data, [])
            if self.opd_replay is not None:
                expected = self.args.rollout_batch_size - len(replay_samples)
                if len(data) != expected:
                    raise ValueError(f"OPD replay requested {expected} fresh trajectories, got {len(data)}")
                data.extend(replay_samples)

            if len(data) % self.args.global_batch_size != 0:
                trim_len = (len(data) // self.args.global_batch_size) * self.args.global_batch_size
                origin_data_length = len(data)
                data = data[:trim_len]
                print(f"trim number of samples from {origin_data_length} to {trim_len}")

        if self.args.loss_type == "dpo_loss":
            if len(data) != self.args.rollout_batch_size * 2:
                raise ValueError("DPO replay must contain exactly one pair per prompt")
            for pair_index in range(0, len(data), 2):
                chosen, rejected = data[pair_index:pair_index + 2]
                if (chosen.prompt != rejected.prompt or
                        chosen.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or
                        rejected.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or
                        chosen.response_length < 1 or rejected.response_length < 1 or
                        (not self.args.dpo_online and
                         chosen.reward["accuracy"] <= rejected.reward["accuracy"])):
                    raise ValueError(f"Invalid matched DPO pair {pair_index // 2}")

        # TODO to be refactored (originally Buffer._set_data)
        # TODO extract to a function during refactor
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=self.rollout_id))
            print(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    rollout_id=self.rollout_id,
                    samples=[sample.to_dict() for sample in data],
                ),
                path,
            )
        if (path_template := self.args.save_rollouts_jsonl) is not None:
            path = export_training_rollouts(
                path_template, rollout_id, data, self.args.wandb_run_name, self.args.wandb_group
            )
            print(f"Saved {len(data)} scored training rollouts to {path}")
        
        log_rollout_data(rollout_id, self.args, data, time() - start_time)
        if self.args.opd_teacher_dir is not None:
            teacher_start_time = time()
            if not getattr(self, "teacher_scorers", None):
                raise RuntimeError("OPD teacher scorers have not been initialized")
            groups = [[] for _ in self.teacher_scorers]
            environment_indices = {name: index for index, name in enumerate(self.args.environment_list)}
            for index, sample in enumerate(data):
                if sample.opd_replay:
                    # The original teacher may have been evicted. Its cached
                    # scores and behavior probabilities define the replay advantage.
                    continue
                environment = sample.metadata["environment"]
                if sample.rollout_log_probs is None or len(sample.rollout_log_probs) != sample.response_length:
                    raise RuntimeError(f"Missing student sampling log probabilities for {environment}")
                scorer_index = environment_indices[environment] % len(self.teacher_scorers)
                groups[scorer_index].append((index, environment, sample.tokens, sample.response_length))
            scored = ray.get([
                scorer.score.remote(trajectories)
                for scorer, trajectories in zip(self.teacher_scorers, groups)
            ])
            for result in scored:
                for index, teacher_log_probs in result:
                    data[index].teacher_log_probs = teacher_log_probs
            if any(sample.teacher_log_probs is None for sample in data):
                raise RuntimeError("Some student trajectories were not scored by their teacher")
            teacher_time = time() - teacher_start_time
            print(f"opd {rollout_id}: scored {sum(map(len, groups))} fresh trajectories with {len(self.teacher_scorers)} teacher GPUs in {teacher_time:.2f}s")
            if self.args.use_wandb:
                wandb.log({
                    "david/step": wandb_step(rollout_id, self.args),
                    "david/opd/teacher_score_time": teacher_time,
                })
        if self.opd_replay is not None:
            self.opd_replay.add(data)
            metrics = {"david/step": wandb_step(rollout_id, self.args),
                       "david/opd/replay_fraction": len(replay_samples) / len(data),
                       "david/opd/replay_trajectories": len(replay_samples),
                       "david/opd/fresh_trajectories": len(data) - len(replay_samples),
                       "david/opd/fresh_trajectories_total": self.opd_replay.fresh_count,
                       "david/opd/replay_trajectories_total": self.opd_replay.cursor,
                       "david/opd/buffer_trajectories": self.opd_replay.size,
                       "david/opd/buffer_environments": len(self.opd_replay.entries)}
            print(f"opd_replay {rollout_id}: {metrics}")
            if self.args.use_wandb:
                wandb.log(metrics)
        # Kept so the driver can turn its own step wall-clock into a tokens/second figure.
        # Prompt + response, i.e. everything the step actually pushed through the model.
        self.last_rollout_tokens = sum(
            len(sample.tokens) if sample.tokens is not None else (sample.response_length or 0)
            for sample in data
        )
        data = self._convert_samples_to_train_data(data)
        return Box(ray.put(data))

    def get_last_rollout_tokens(self):
        return getattr(self, "last_rollout_tokens", 0)

    def save_opd_replay(self, path):
        return self.opd_replay.save(path) if self.opd_replay is not None else None

    def set_rollouts_per_prompt(self, count):
        assert count > 0
        old_count = self.args.n_samples_per_prompt
        if count == old_count:
            return
        # Partial responses in the buffer still have the old group size.
        discarded = len(self.data_source.buffer)
        self.data_source.buffer.clear()
        self.args.n_samples_per_prompt = count
        self.args.global_batch_size = self.args.rollout_batch_size * count
        print(f"Rollout annealing: {old_count} -> {count} per prompt; "
              f"discarded {discarded} buffered groups", flush=True)
        if self.args.use_wandb:
            wandb.log({"david/rollout_anneal/rollouts_per_prompt": count})

    def set_teacher_scorers(self, scorers):
        self.teacher_scorers = scorers

    def set_dpo_base_generator(self, generator):
        self.data_source.dpo_base_generator = generator

    def finish_wandb(self):
        """Flush this actor's wandb writer.

        Eval is logged from here, not from the driver, and in wandb's shared mode each
        process buffers its own points. When train.py returns it tears the actors down, so
        without an explicit flush the final eval - the one after the last step, i.e. the
        pass@k you actually care about - never reaches the server.
        """
        if self.args.use_wandb:
            wandb.finish()

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return

        data = self.eval_generate_rollout(self.args, rollout_id, self.data_source, evaluation=True)
        log_eval_data(rollout_id, self.args, data, rollout_tables=self.eval_rollout_tables)
        names = getattr(self.args, "early_stop_eval_names", ["val_envs"])
        if self.args.early_stop_val_accuracy is not None:
            return early_stop_accuracy(data, names)
        if all(name in data for name in names):
            return early_stop_accuracy(data, names)

    def post_process_rewards(self, samples: Union[list[Sample], list[list[Sample]]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            from slime.utils.reward_normalization import normalize_group_rewards

            rewards = normalize_group_rewards(rewards,
                self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
        """
        Convert inference generated samples to training data.
        """
        raw_rewards, rewards = self.post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length
            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]
        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]
        if self.opd_replay is not None:
            train_data["opd_replay_mask"] = [
                [float(sample.opd_replay)] * sample.response_length for sample in samples
            ]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        return train_data

    def save(self, rollout_id):
        self.data_source.save(rollout_id)
        if getattr(self.args, "transactional_checkpoints", False):
            self.save_runtime(rollout_id)

    def save_runtime(self, rollout_id):
        path = Path(self.args.save) / "rollout" / f"runtime_state_{rollout_id}.pt"
        temporary = path.with_suffix(".pt.tmp")
        state = {
            "buffer": [[sample.to_dict() for sample in group] for group in self.data_source.buffer],
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
            "eval_tables": {
                name: {"columns": table.columns, "data": table.data}
                for name, table in self.eval_rollout_tables.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)
        if getattr(self.args, "transactional_checkpoints", False) and rollout_id is not None and rollout_id >= 0:
            path = Path(self.args.load) / "rollout" / f"runtime_state_{rollout_id}.pt"
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.data_source.buffer = [[Sample.from_dict(sample) for sample in group] for group in state["buffer"]]
            random.setstate(state["random_state"])
            np.random.set_state(state["numpy_state"])
            torch.set_rng_state(state["torch_state"])
            self.eval_rollout_tables = {
                name: wandb.Table(columns=table["columns"], data=table["data"], log_mode="INCREMENTAL")
                for name, table in state["eval_tables"].items()
            }


def wandb_step(rollout_id, args):
    return (
        rollout_id
        if not args.wandb_always_use_train_step
        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )


def log_eval_data(rollout_id, args, data, rollout_tables=None):
    log_dict = {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        full_rewards = [float(reward >= 1.0 - 1e-6) for reward in rewards]
        log_dict[f"eval/{key}-solve_rate"] = sum(full_rewards) / len(full_rewards)
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

        # pass@k. eval_rollout_single_dataset emits samples ordered by index, so every
        # consecutive run of n_samples_per_eval_prompt rewards belongs to one prompt.
        group_size = args.n_samples_per_eval_prompt
        if len(rewards) % group_size == 0:
            # Every prompt is a fixed hidden-state seed. Treat its sampled mean
            # as one independent unit when estimating uncertainty across seeds.
            prompt_means = [
                sum(rewards[i : i + group_size]) / group_size
                for i in range(0, len(rewards), group_size)
            ]
            if len(prompt_means) > 1:
                log_dict[f"eval/{key}-stderr"] = statistics.stdev(prompt_means) / sqrt(len(prompt_means))
                full_prompt_means = [
                    sum(full_rewards[i : i + group_size]) / group_size
                    for i in range(0, len(full_rewards), group_size)
                ]
                log_dict[f"eval/{key}-solve_rate-stderr"] = (
                    statistics.stdev(full_prompt_means) / sqrt(len(full_prompt_means))
                )
        if group_size > 1 and len(rewards) % group_size == 0:
            eps = 1e-6
            correct_per_prompt = [
                sum(1 for r in rewards[i : i + group_size] if r >= 1.0 - eps)
                for i in range(0, len(rewards), group_size)
            ]
            for k, estimate in pass_at_k(correct_per_prompt, group_size).items():
                log_dict[f"david/eval/{key}/pass@{k}"] = estimate

    print(f"eval {rollout_id}: {log_dict}")
    if args.use_wandb:
        log_dict["eval/step"] = wandb_step(rollout_id, args)
        log_dict["david/step"] = log_dict["eval/step"]
        if getattr(args, "log_eval_rollouts", False):
            if rollout_tables is None:
                rollout_tables = {}
            log_dict.update(update_eval_rollout_tables(data, log_dict["david/step"], rollout_tables, wandb.Table))
        wandb.log(log_dict)


def log_rollout_data(rollout_id, args, samples, rollout_time):
    if args.load_debug_rollout_data:
        return

    log_dict = {}
    response_lengths = [
        sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length for sample in samples
    ]
    log_dict["perf/rollout_time"] = rollout_time
    if args.rollout_num_gpus is not None:
        log_dict["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
    log_dict["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time
    print(f"perf {rollout_id}: {log_dict}")
    if args.use_wandb:
        log_dict["rollout/step"] = (
            rollout_id
            if not args.wandb_always_use_train_step
            else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        )
        wandb.log(log_dict)
