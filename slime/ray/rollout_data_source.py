import copy
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.types import Sample

import random
from Gym.environment import VerifiableEnvironment
from Gym.environments import identifier2environment
from Gym.parameter_controller import ParameterController
from Gym.parameter_controllers import identifier2controller
from typing import List, Optional, Tuple, Dict, Any

from slime.utils.data import custom_prompt_preprocessor
from slime.utils.rlve_mixing import FrozenPromptMix
from slime.utils.rlve_difficulty import BalancedDifficultySchedule


class RLVEManager :
    def __init__(self, args, tokenizer) :
        self.args = args
        self.tokenizer = tokenizer

        assert args.environment_list, "Environment list is not set."

        self.environment2difficulty = {environment : args.initial_difficulty for environment in args.environment_list}

        self.environment2accuracy = {environment : dict(accuracy = 0, num_samples = 0) for environment in args.environment_list}
        self.problem_generation_seed = 0
        self.balanced_difficulty = (
            BalancedDifficultySchedule(args.environment_list, args.initial_difficulty)
            if getattr(args, "rlve_balanced_difficulty", False) else None
        )
        self.frozen_mix = None
        if args.rlve_prompt_pool is not None:
            self.frozen_mix = FrozenPromptMix(
                args.rlve_prompt_pool,
                args.environment_list,
                args.rlve_mix_weights,
                args.rlve_mix_seed,
                args.rlve_pool_sha256,
            )


    def generate_problem(self) -> Tuple[str, Optional[VerifiableEnvironment]] :
        pair = self.balanced_difficulty.peek() if self.balanced_difficulty is not None else None
        environment : str = pair[0] if pair is not None else random.choice(self.args.environment_list)

        parameter_controller : ParameterController = identifier2controller[environment]()
        maximum_difficulty : int = self.environment2difficulty[environment]
        parameter_lists : List[List[Dict]] = []
        for problem_difficulty in range(maximum_difficulty + 1) :
            if problem_difficulty > maximum_difficulty - self.args.difficulty_sliding_window_size :
                parameter_lists.append((problem_difficulty, copy.deepcopy(parameter_controller.get_parameter_list())))
            parameter_controller.update()

        if pair is None:
            problem_difficulty, parameter_list = random.choice(parameter_lists)
        else:
            problem_difficulty = pair[1]
            parameter_list = next(parameters for level, parameters in parameter_lists
                                  if level == problem_difficulty)
        parameter : Dict = random.choice(parameter_list)
        problem : VerifiableEnvironment = identifier2environment[environment]()
        if problem.generator(seed = self.problem_generation_seed, parameter = parameter) :
            generated_problem = problem
            if self.balanced_difficulty is not None:
                self.balanced_difficulty.accept()
        else :
            generated_problem = None
            print("Generating problem for environment {} failed\nparameter: {}\n\n\n".format(environment, parameter), flush=True)
        self.problem_generation_seed += 1

        return environment, problem_difficulty, generated_problem

    def get_sample(self) -> Optional[Sample] :
        if self.frozen_mix is None:
            environment, problem_difficulty, problem = self.generate_problem()
            if problem is None:
                return None
            user_prompt = problem.prompt_generator()
            config = problem.get_config()
        else:
            environment, user_prompt, config = self.frozen_mix.draw()
            problem_difficulty = 0
            self.problem_generation_seed += 1
        prompt = custom_prompt_preprocessor(args = self.args, user_prompt = user_prompt, apply_chat_template = self.args.apply_chat_template)

        apply_chat_template = self.args.apply_chat_template
        tokenizer = self.tokenizer
        max_length = self.args.rollout_max_prompt_len
        tool_key = self.args.tool_key
        if apply_chat_template:
            if tool_key is not None:
                assert False, "Tool key is not supported for RLVE yet."
            else:
                tools = None
            prompt = tokenizer.apply_chat_template(prompt, tools, tokenize=False, add_generation_prompt=True)

        # TODO: this is slow.
        if max_length is not None:
            assert False, "For now, we don't discard overlong prompts"
            if len(tokenizer(prompt)["input_ids"]) > max_length:
                return None
        
        return Sample(
            prompt = prompt,
            label = None,
            metadata = dict(environment = environment, problem_difficulty = problem_difficulty, config = config),
        )

    def get_state(self) -> Dict[str, Any] :
        return dict(
            environment2difficulty = self.environment2difficulty,
            environment2accuracy = self.environment2accuracy,
            problem_generation_seed = self.problem_generation_seed,
            frozen_mix_state = self.frozen_mix.get_state() if self.frozen_mix is not None else None,
            balanced_difficulty_state = self.balanced_difficulty.get_state() if self.balanced_difficulty is not None else None,
        )

    def set_state(self, state : Dict[str, Any]) -> None :
        self.environment2difficulty = state["environment2difficulty"]
        self.environment2accuracy = state["environment2accuracy"]
        self.problem_generation_seed = state["problem_generation_seed"]
        if self.frozen_mix is not None:
            self.frozen_mix.set_state(state["frozen_mix_state"])
        if self.balanced_difficulty is not None:
            self.balanced_difficulty = BalancedDifficultySchedule(
                self.args.environment_list, self.args.initial_difficulty,
                state["balanced_difficulty_state"],
            )

    def update(self, samples : List[Sample]) -> Dict[str, Any] :
        """
        Update accuracy statistics based on completed samples.
        Also update the difficulty when necessary.
        This should be called after rewards have been computed.
        """
        log_dict = {}

        for sample in samples :
            environment = sample.metadata["environment"]

            problem_difficulty, maximum_difficulty = sample.metadata["problem_difficulty"], self.environment2difficulty[environment]
            assert problem_difficulty <= maximum_difficulty, "The difficulty of the sample is higher than the current difficulty of the problem, which should not happen."
            if problem_difficulty < maximum_difficulty :
                continue
            self.environment2accuracy[environment]["num_samples"] += 1
            self.environment2accuracy[environment]["accuracy"] += sample.reward["accuracy"]

        log_dict["rollout/problem_generation_seed"] = self.problem_generation_seed
        if self.balanced_difficulty is not None:
            for difficulty in range(self.args.initial_difficulty + 1):
                log_dict[f"RLVE/difficulty_prompts/{difficulty}"] = sum(
                    count for (_, level), count in self.balanced_difficulty.counts.items()
                    if level == difficulty
                )

        for environment in self.args.environment_list :
            num_samples, accuracy = self.environment2accuracy[environment]["num_samples"], self.environment2accuracy[environment]["accuracy"]
            if num_samples >= self.args.min_prompts_before_difficulty_check * self.args.n_samples_per_prompt :
                accuracy = accuracy / num_samples
                log_dict["RLVE/{}/accuracy".format(environment)] = accuracy

                if accuracy >= self.args.min_metric_to_increase_difficulty :
                    self.environment2difficulty[environment] += 1
                    log_dict["RLVE/{}/difficulty".format(environment)] = self.environment2difficulty[environment]

                self.environment2accuracy[environment] = dict(accuracy = 0, num_samples = 0)

        return log_dict


# TODO may further refactor data-loading part later
class RolloutDataSource:
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        assert (args.rollout_global_dataset and not args.rlve) or (not args.rollout_global_dataset and args.rlve), "Incompatible arguments: args.rollout_global_dataset and args.rlve must be mutually exclusive, and one of them must be set."
        self.rlve_manager = None
        self.tree_sampler = None

        if args.rollout_global_dataset:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")

            if args.program_tree_sampler_spec is not None:
                from program_trees.next_experiments import ProgramTreeSampler

                self.tree_sampler = ProgramTreeSampler.from_file(args.program_tree_sampler_spec)
                self.tokenizer = tokenizer
                self.dataset = None
            else:
                self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                seed=args.rollout_seed,
                args=args,
                )
            if self.args.rollout_shuffle and self.dataset is not None:
                self.dataset.shuffle(self.epoch_id)
        elif args.rlve:
            self.dataset = None
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            self.rlve_manager = RLVEManager(args, tokenizer)
        else:
            assert False, "None of args.rollout_global_dataset or args.rlve is set - there is no data source."
            self.dataset = None

    def get_samples(self, num_samples):
        samples = []

        # TODO unify the two branches
        if self.tree_sampler is not None:
            for _ in range(num_samples):
                row = self.tree_sampler.draw()
                prompt = custom_prompt_preprocessor(
                    args=self.args, user_prompt=row["user_prompt"],
                    apply_chat_template=self.args.apply_chat_template,
                )
                if self.args.apply_chat_template:
                    if self.args.tool_key is not None:
                        raise ValueError("Program-tree sampler does not use tools")
                    prompt = self.tokenizer.apply_chat_template(
                        prompt, None, tokenize=False, add_generation_prompt=True,
                    )
                if (self.args.rollout_max_prompt_len is not None and
                        len(self.tokenizer(prompt)["input_ids"]) > self.args.rollout_max_prompt_len):
                    raise ValueError("Sampled program-tree prompt exceeds rollout-max-prompt-len")
                prompt_sample = Sample(prompt=prompt, label=row["label"], metadata=row["metadata"])
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = copy.deepcopy(prompt_sample)
                    sample.index = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                samples.append(group)
        elif self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
            for prompt_sample in prompt_samples:
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = copy.deepcopy(prompt_sample)
                    sample.index = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                samples.append(group)
        elif self.rlve_manager is not None:
            while len(samples) < num_samples :
                prompt_sample = self.rlve_manager.get_sample()
                if prompt_sample is None :
                    continue
                '''[Zhiyuan TODO]
                The following part (converting one prompt_sample to args.n_samples_per_prompt samples in one group) is fully copied from `https://github.com/THUDM/slime/tree/main/ray/rollout_data_source.py#L63-#L69`, which may not be clean.
                '''
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = copy.deepcopy(prompt_sample)
                    sample.index = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                samples.append(group)
            assert len(samples) == num_samples
        else:
            assert False, "There is no valid data source (both self.dataset and self.rlve_manager are None)"
            for _ in range(num_samples):
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = Sample(
                        index=self.sample_index,
                    )
                    self.sample_index += 1
                    group.append(sample)
                samples.append(group)

        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            if self.args.rlve :
                state_dict = self.rlve_manager.get_state()

                path = os.path.join(self.args.save, f"rollout/rlve_manager_state_dict_{rollout_id}.pt")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(state_dict, path)
            else :
                assert False, "None of args.rollout_global_dataset or args.rlve is set - there is no data source."
            # return
        else :
            assert not self.args.rlve, "If args.rollout_global_dataset is set, args.rlve must not be set."

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        if self.tree_sampler is not None:
            state_dict["program_tree_sampler"] = self.tree_sampler.state_dict()
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if self.args.load is None:
            return
        if rollout_id == -1 :
            return
        load_options = {"weights_only": False} if getattr(self.args, "transactional_checkpoints", False) else {}

        if not self.args.rollout_global_dataset:
            if self.args.rlve :
                path = os.path.join(self.args.load, f"rollout/rlve_manager_state_dict_{rollout_id}.pt")
                if not os.path.exists(path):
                    assert False, f"Checkpoint {path} does not exist."
                state_dict = torch.load(path, **load_options)
                self.rlve_manager.set_state(state_dict)
            else :
                assert False, "None of args.rollout_global_dataset or args.rlve is set - there is no data source."
            # return
        else :
            assert not self.args.rlve, "If args.rollout_global_dataset is set, args.rlve must not be set."

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            assert False, f"Checkpoint {path} does not exist."
            print(f"Checkpoint {path} does not exist.")
            return

        print(f"load metadata from {path}")
        print(f"load metadata: {self.metadata}")
        state_dict = torch.load(path, **load_options)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        if self.tree_sampler is not None:
            if "program_tree_sampler" not in state_dict:
                raise ValueError("Checkpoint lacks program-tree sampler state")
            self.tree_sampler.load_state_dict(state_dict["program_tree_sampler"])

        if self.dataset is not None and self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
