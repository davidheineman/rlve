"""Task-specific answer forcing inside the configured sampled-token budget."""

import copy
import json
import os
import time
from pathlib import Path

from modular_rl import rollout as v1

FORCING_KEY = "_modular_v2_forcing"
_COUNTS = []
_SAMPLES = []


def format_prompt(tokenizer, messages):
    mode = os.environ.get('MODULAR_RL_ENABLE_THINKING', '1')
    if mode not in ('0', '1'):
        raise ValueError('MODULAR_RL_ENABLE_THINKING must be 0 or 1')
    return tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=mode == '1')


def forcing_prefix(task, response, *, base_model=False):
    close_thinking = "</think>" not in response and (not base_model or "<think>" in response)
    closing = "\n</think>\n\n" if close_thinking else "\n\n"
    if task == "math":
        return closing + r"\boxed{"
    if task == "code":
        return closing + "```python\n"
    if task == "chat":
        return closing + "Final answer:\n"
    raise ValueError(f"Unknown task: {task}")


def sampled_tokens(sample):
    return sample.response_length - sample.metadata.get(FORCING_KEY, {}).get(
        "prefix_tokens", 0
    )


async def generate(args, sample, sampling_params):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.rollout.sglang_rollout import generate as base_generate
    from slime.utils.types import Sample

    total = sampling_params["max_new_tokens"]
    reserve = args.answer_forcing_extra_tokens
    if not 0 < reserve < total:
        raise ValueError("Answer reserve must fit inside the response token budget")
    native_args = copy.copy(args)
    native_args.answer_forcing = (
        False  # v2 supplies the task prefix and shares the cap.
    )
    if not sample.tokens:
        sample.prompt = format_prompt(GenerateState(args).tokenizer, sample.metadata['messages'])
    state = sample.metadata.get(FORCING_KEY)
    if state is None:
        params = {**sampling_params, "max_new_tokens": total - reserve}
        sample = await base_generate(native_args, sample, params)
        if sample.status == Sample.Status.ABORTED:
            return sample
        if sample.status == Sample.Status.TRUNCATED:
            prefix = forcing_prefix(sample.metadata["task"], sample.response,
                                    base_model=(os.environ.get('MODULAR_RL_BASE_TOKENIZER') == '1'
                                                or str(getattr(args, "hf_checkpoint", "")).rstrip("/").endswith("-Base")))
            ids = GenerateState(args).tokenizer(prefix, add_special_tokens=False)[
                "input_ids"
            ]
            initial_tokens, initial_chars = sample.response_length, len(sample.response)
            sample.metadata[FORCING_KEY] = {
                "initial_tokens": initial_tokens,
                "prefix_tokens": len(ids),
                "answer_char_start": initial_chars,
                "sampled_token_budget": total,
            }
            if sample.loss_mask is None:
                sample.loss_mask = [1] * initial_tokens
            sample.tokens.extend(ids)
            sample.response += prefix
            sample.response_length += len(ids)
            sample.loss_mask.extend([0] * len(ids))
            sample.rollout_log_probs.extend([0.0] * len(ids))
            state = sample.metadata[FORCING_KEY]
    if state is not None:
        remaining = total - sampled_tokens(sample)
        if remaining > 0:
            sample.status = Sample.Status.ABORTED
            params = {
                **sampling_params,
                "max_new_tokens": sample.response_length + remaining,
            }
            sample = await base_generate(native_args, sample, params)
        if sample.status == Sample.Status.ABORTED:
            return sample
    generated = sampled_tokens(sample)
    if generated > total or generated < 0:
        raise RuntimeError(
            f"Generation exceeded its sampled-token cap: {generated}/{total}"
        )
    _COUNTS.append(
        {
            "sampled_tokens": generated,
            "forced": int(state is not None),
            "truncated": int(sample.status == Sample.Status.TRUNCATED),
        }
    )
    # Keep the object: the backend attaches its reward after generation returns.
    _SAMPLES.append(sample)
    return sample


async def reward(args, sample, **kwargs):
    from .math_reward import reward as math_reward

    task, target = sample.metadata["task"], sample.metadata["target"]
    if task == "math":
        return await math_reward(sample.response, target["answer"])
    state = sample.metadata.get(FORCING_KEY)
    if task == "code" and state:
        # A forced implementation is independent of any incomplete earlier code fence.
        graded = copy.copy(sample)
        graded.response = code_submission(sample.response, state,
            fallback=os.environ.get('MODULAR_RL_CODE_FORCING_FALLBACK') == '1')
        return await v1.reward(args, graded, **kwargs)
    return await v1.reward(args, sample, **kwargs)


def code_submission(response, forcing_state, *, fallback=False):
    """Prefer the forced solution; optionally retain an earlier complete implementation.

    Selection uses Python syntax only, without executing tests or observing rewards.
    The default preserves historical scores; enable the fallback for new experiments.
    """
    final=response[forcing_state['answer_char_start']:]
    if not fallback:return final
    import ast
    from modular_rl.graders import extract_code

    def complete_implementation(text):
        try:tree=ast.parse(extract_code(text))
        except (SyntaxError,ValueError):return False
        return any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for node in tree.body)

    if complete_implementation(final):return final
    earlier=response[:forcing_state['answer_char_start']]
    return earlier if complete_implementation(earlier) else final


def validation_prompt_count(task="math"):
    """Keep full validation at32; allow only explicitly marked Base GPU preflights."""
    if declared := os.environ.get('MODULAR_RL_VALIDATION_PROMPTS'):
        count = int(declared)
        path = Path(os.environ['MODULAR_RL_SELECTION']) / task / 'selection.json'
        info = json.loads(path.read_text())
        if count not in (32, 64) and not (info.get('preflight_only') and 1 <= count <= 32):
            raise ValueError('Declare 32/64 prompts, or an explicitly marked smaller preflight')
        if info['validation_prompts'] != count:
            raise ValueError('Validation count differs from the frozen selection')
        return count
    if os.environ.get("MODULAR_RL_REQUIRE_PRETRAINED_BASE") == "1":
        selection = os.environ.get("MODULAR_RL_SELECTION")
        if selection:
            path = Path(selection) / task / "selection.json"
            if path.exists():
                info = json.loads(path.read_text())
                if info.get("preflight_only"):
                    count = info["validation_prompts"]
                    if not isinstance(count, int) or not 1 <= count <= 32:
                        raise ValueError("Invalid declared preflight validation size")
                    return count
    return 32


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    from slime.ray.buffer import wandb_step

    _COUNTS.clear()
    _SAMPLES.clear()
    started = time.monotonic()
    data = v1.generate_rollout(args, rollout_id, data_buffer, evaluation=evaluation)
    elapsed = time.monotonic() - started
    phase = "eval" if evaluation else "train"
    task = _SAMPLES[0].metadata["task"] if _SAMPLES else "math"
    expected = (
        validation_prompt_count(task) * args.n_samples_per_eval_prompt
        if evaluation
        else args.rollout_batch_size * args.n_samples_per_prompt
    )
    if evaluation and len(data["val_envs"]["rewards"]) != expected:
        raise RuntimeError(
            f"Validation must retain all {validation_prompt_count(task)} prompts and eight samples each"
        )
    if len(_COUNTS) != expected:
        raise RuntimeError(
            f"Unaccounted {phase} generations: {len(_COUNTS)}/{expected}"
        )
    record = {
        "phase": phase,
        "rollout_id": rollout_id,
        "responses": expected,
        "sampled_token_budget": args.eval_max_response_len
        if evaluation
        else args.rollout_max_response_len,
        "sampled_tokens": sum(row["sampled_tokens"] for row in _COUNTS),
        "max_sampled_tokens": max(row["sampled_tokens"] for row in _COUNTS),
        "forced_responses": sum(row["forced"] for row in _COUNTS),
        "truncated_responses": sum(row["truncated"] for row in _COUNTS),
        "wall_seconds": elapsed,
    }
    directory = Path(os.environ["MODULAR_RL_RUN_DIR"])
    if evaluation:
        export_evaluation_samples(
            directory, rollout_id, _SAMPLES, args.wandb_run_name, args.wandb_group
        )
    with (directory / "generation.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if args.use_wandb:
        import wandb

        wandb.log(
            {
                "david/step": wandb_step(rollout_id, args),
                **{
                    f"david/{phase}/generation/{key}": value
                    for key, value in record.items()
                    if key not in ("phase", "rollout_id")
                },
            }
        )
    return data


def export_evaluation_samples(directory, rollout_id, samples, run_name, run_group):
    """Retain every validation response in prompt order, including its grader target."""
    from slime.utils.rollout_export import export_training_rollouts

    ordered = sorted(samples, key=lambda sample: sample.index)
    if [sample.index for sample in ordered] != list(range(len(ordered))):
        raise RuntimeError("Validation sample indices must be complete and unique")
    return export_training_rollouts(
        str(directory / "rollouts/eval/{rollout_id:04d}.jsonl"),
        rollout_id,
        ordered,
        run_name,
        run_group,
    )
