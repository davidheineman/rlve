import asyncio
import copy
import json
import wandb

from tqdm import tqdm
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_metrics import group_eval_results
from slime.utils.eval_rollouts import select_eval_rollouts
from slime.utils.final_ood_eval import eval_path, should_run_dataset
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

# Successful R1 Distill depth-1 tree traces usually close reasoning with
# </think> and put the answer in a JSON object. Only the framing is supplied;
# the model must still generate the value and the closing syntax.
ANSWER_FORCING_PREFIX = '\n</think>\n\n{"output": "'
ANSWER_FORCING_PREFIX_AFTER_THINK = '\n\n{"output": "'
ANSWER_FORCING_KEY = "_answer_forcing"


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args):
        # persistant state for the generation process
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        self.reset()

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]):
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args, sample: Sample, sampling_params) -> Sample:
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"
    initial_cap = sampling_params["max_new_tokens"]
    forcing_state = sample.metadata.get(ANSWER_FORCING_KEY) if isinstance(sample.metadata, dict) else None
    if forcing_state is not None:
        # An aborted continuation can be resumed without inserting the prefix twice.
        generated_after_prefix = sample.response_length - forcing_state["initial_tokens"] - forcing_state["prefix_tokens"]
        remaining = args.answer_forcing_extra_tokens - generated_after_prefix
    else:
        remaining = initial_cap - sample.response_length
    assert remaining >= 0, f"remaining generation tokens {remaining} should not be negative"
    if remaining == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    if not sample.tokens:
        sample.tokens = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]

    async def generate_chunk(max_new_tokens):
        params = sampling_params.copy()
        params["max_new_tokens"] = max_new_tokens
        output = await post(
            url,
            {"input_ids": sample.tokens.copy(), "sampling_params": params, "return_logprob": True},
            use_http2=args.use_http2,
        )
        token_logprobs = output["meta_info"].get("output_token_logprobs", [])
        new_tokens = [item[1] for item in token_logprobs]
        sample.tokens.extend(new_tokens)
        sample.response_length += len(new_tokens)
        sample.response += output["text"]
        if sample.loss_mask is not None:
            sample.loss_mask.extend([1] * len(new_tokens))
        if "weight_version" in output["meta_info"]:
            sample.weight_versions.append(output["meta_info"]["weight_version"])
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs.extend(item[0] for item in token_logprobs)
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED
        return sample

    if forcing_state is not None:
        return await generate_chunk(remaining)

    await generate_chunk(remaining)
    if sample.status != Sample.Status.TRUNCATED or not getattr(args, "answer_forcing", False):
        return sample

    # Frozen RLVE eval files encode metadata as JSON; live training uses dicts.
    if isinstance(sample.metadata, str):
        sample.metadata = json.loads(sample.metadata)
    if getattr(args, "rlve", False):
        marker = getattr(args, "answer_marker_type", r"\boxed{}")
        if marker == r"\boxed{}":
            answer_prefix = r"\boxed{"
        elif marker == "<answer></answer>":
            answer_prefix = "<answer>"
        else:
            raise ValueError(f"Unsupported RLVE answer marker: {marker}")
        prefix = ("\n\n" if "</think>" in sample.response else "\n</think>\n\n") + answer_prefix
    else:
        prefix = ANSWER_FORCING_PREFIX_AFTER_THINK if "</think>" in sample.response else ANSWER_FORCING_PREFIX
    if not isinstance(sample.metadata, dict):
        raise TypeError("answer forcing requires sample.metadata to be a dict")
    prefix_tokens = state.tokenizer(prefix, add_special_tokens=False)["input_ids"]
    initial_tokens = sample.response_length
    if sample.loss_mask is None:
        sample.loss_mask = [1] * initial_tokens
    sample.tokens.extend(prefix_tokens)
    sample.response_length += len(prefix_tokens)
    sample.response += prefix
    sample.loss_mask.extend([0] * len(prefix_tokens))
    sample.rollout_log_probs.extend([0.0] * len(prefix_tokens))
    sample.metadata[ANSWER_FORCING_KEY] = {"initial_tokens": initial_tokens, "prefix_tokens": len(prefix_tokens)}
    await generate_chunk(args.answer_forcing_extra_tokens)
    return sample


async def generate_and_rm(args, sample: Sample, sampling_params: dict, evaluation=False) -> Sample:
    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        if args.custom_generate_function_path is not None:
            custom_generate_func = load_function(args.custom_generate_function_path)
            sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample

        sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(args, group: list[Sample], sampling_params: dict, evaluation=False) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    group = await asyncio.gather(
        *[generate_and_rm(args, sample, sampling_params.copy(), evaluation=evaluation) for sample in group]
    )

    # for the rm that need the whole group, we will not do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards):
            sample.reward = reward

    return group


async def abort(args, rollout_id: int):
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True
    response = await get(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers", use_http2=args.use_http2
    )

    # abort all the requests
    for url in response["urls"]:
        print(f"Abort request for {url}", flush=True)
        await post(f"{url}/abort_request", {"abort_all": True}, use_http2=False)

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED, timeout=30.0)
        
        # Check if no tasks completed within timeout
        if not done:
            print(f"Abort timeout after 30s, forcing completion of {len(state.pendings)} pending tasks", flush=True)
            # Forcefully cancel all pending tasks
            for task in state.pendings:
                if not task.done():
                    task.cancel()
            state.pendings.clear()
            break

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        print(f"Collected {count} partial samples into the data buffer", flush=True)

    return aborted_samples


async def generate_rollout_async(args, rollout_id: int, data_source, data_buffer=None) -> list[list[Sample]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        list[list[Sample]]: a list of samples generated by the rollout, the length of the list is exactly the same as the `rollout_batch_size`
    """
    assert (args.rollout_global_dataset and not args.rlve) or (not args.rollout_global_dataset and args.rlve), "Incompatible arguments: args.rollout_global_dataset and args.rlve must be mutually exclusive, and one of them must be set."

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    # target_data_size is the total number of valid samples to get
    target_data_size = getattr(args, "opd_fresh_batch_size", None) or args.rollout_batch_size

    data = []
    all_data = []
    rejected_groups = 0
    fallback_groups = 0
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(min(args.over_sampling_batch_size, target_data_size)
                                  if getattr(args, "opd_fresh_batch_size", None)
                                  else args.over_sampling_batch_size)
            print("Fetched {} samples from the data source".format(len(samples)), flush=True)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                print(
                    f"First rollout sample: {[sample.prompt + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                    flush=True,
                )
                do_print = False

            all_data.append(group)

            assert len(group) == args.n_samples_per_prompt
            if dynamic_filter is not None and not dynamic_filter(args, group):
                max_rejections = args.dynamic_sampling_max_rejections
                if max_rejections is None or rejected_groups < max_rejections:
                    rejected_groups += 1
                    state.remaining_batch_size -= 1
                    continue
                fallback_groups += 1

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    if dynamic_filter is not None:
        print(
            f"Dynamic sampling rollout {rollout_id}: rejected={rejected_groups}, "
            f"fallback_accepted={fallback_groups}", flush=True
        )
        if args.use_wandb:
            wandb.log({
                "rollout/step": rollout_id if not args.wandb_always_use_train_step else
                    rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size,
                "david/dynamic_sampling/rejected_groups": rejected_groups,
                "david/dynamic_sampling/fallback_groups": fallback_groups,
            })

    if args.rlve :
        assert args.rm_type == "rlve", f"Expected rm_type to be 'rlve', got {args.rm_type}"
        assert data_buffer is not None, "data_buffer must be provided when using RLVE"
        assert hasattr(data_buffer, "rlve_manager"), "data_buffer must have rlve_manager attribute when using RLVE"
        assert data_buffer.rlve_manager is not None, "data_buffer.rlve_manager must be initialized when using RLVE"

        assert len(all_data) >= len(data), "all_data should be larger than data"
        while isinstance(all_data[0], list):
            all_data = sum(all_data, [])
        log_dict = data_buffer.rlve_manager.update(samples = all_data)

        if args.use_wandb :
            # Ensure the rlve metrics use rollout/step as x-axis by defining them explicitly
            try:
                for k in list(log_dict.keys()):
                    if k.startswith("RLVE/"):
                        wandb.define_metric(k, step_metric="rollout/step")
            except Exception:
                # Metric may already be defined; ignore
                pass
            log_dict["rollout/step"] = (
                rollout_id
                if not args.wandb_always_use_train_step
                else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
            )
            wandb.log(log_dict)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    print(
        f"Finish rollout: {[sample.prompt + sample.response]}, label: {sample.label}, reward: {sample.reward}",
        flush=True,
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == target_data_size, f"Got {len(data)} samples, expected {target_data_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    return data, aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args, rollout_id):
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    results = {}
    for i in range(0, len(args.eval_prompt_data), 2):
        name, path = args.eval_prompt_data[i : i + 2]
        if not should_run_dataset(name, rollout_id, args.num_rollout, args.eval_interval,
                                  getattr(args, "rlve_final_ood_eval_path", None),
                                  getattr(args, "rlve_final_ood_only", False)):
            continue
        path = eval_path(name, path, rollout_id, args.num_rollout,
                         getattr(args, "rlve_final_ood_eval_path", None))
        results.update(await eval_rollout_single_dataset(args, rollout_id, name, path))
    return results, []


async def eval_rollout_single_dataset(args, rollout_id, name, path):
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        name: str, the name of the dataset
        path: str, the path of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = (name, path)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key if args.eval_input_key is None else args.eval_input_key,
            label_key=args.label_key if args.eval_label_key is None else args.eval_label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key if args.eval_tool_key is None else args.eval_tool_key,
            apply_chat_template=args.apply_chat_template,
            args=args,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    sampling_params = dict(
        temperature=args.rollout_temperature if args.eval_temperature is None else args.eval_temperature,
        top_p=args.rollout_top_p if args.eval_top_p is None else args.eval_top_p,
        top_k=args.rollout_top_k if args.eval_top_k is None else args.eval_top_k,
        max_new_tokens=(
            args.rollout_max_response_len if args.eval_max_response_len is None else args.eval_max_response_len
        ),
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(args.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            tasks.append(
                generate_and_rm(
                    args,
                    sample,
                    sampling_params=sampling_params,
                    evaluation=True,
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc="Rollout generation", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            print([sample.prompt + sample.response], sample.reward, flush=True)
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    results = group_eval_results(
        name,
        rewards=[sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
        truncated=[sample.status == Sample.Status.TRUNCATED for sample in data],
        metadata=[sample.metadata for sample in data],
    )
    if getattr(args, "log_eval_rollouts", False):
        results[name]["rollout_examples"] = select_eval_rollouts(data, args.n_samples_per_eval_prompt, reward_key)
    return results


# TODO remove this temp function
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    completed_samples, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation,
        data_buffer=data_buffer,
    )
    data_buffer.add_samples(aborted_samples)
    return completed_samples


def generate_abortable_samples(args, rollout_id, data_source, evaluation=False, data_buffer=None):
    # assert args.rollout_global_dataset
    assert (args.rollout_global_dataset and not args.rlve) or (not args.rollout_global_dataset and args.rlve), "Incompatible arguments: args.rollout_global_dataset and args.rlve must be mutually exclusive, and one of them must be set."
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_source, data_buffer=data_buffer))
