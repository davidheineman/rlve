"""Multi-turn SPADE gameplay for Slime's custom generation and reward hooks."""

import asyncio
import os
from pathlib import Path

from slime.utils.http_utils import post
from slime.utils.types import Sample
from spade_games.protocol import observation_suffix_ids, parse_action
from spade_games.synthetic_game_env import make_synthetic_env

_ENV_CALL_LOCK = asyncio.Lock()


def _setting(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _append_observation(sample: Sample, tokenizer, observation: str) -> None:
    if sample.metadata.get("no_think"):
        observation += "\n/no_think"
    obs_ids = observation_suffix_ids(tokenizer, observation)
    sample.tokens.extend(obs_ids)
    sample.loss_mask.extend([0] * len(obs_ids))
    sample.rollout_log_probs.extend([0.0] * len(obs_ids))
    sample.response_length += len(obs_ids)


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """Play one frozen game episode, preserving model and observation token masks."""
    # Imported here so tests can exercise the adapter without constructing a
    # live SGLang GenerateState.
    from slime.rollout.sglang_rollout import GenerateState

    state = GenerateState(args)
    metadata = sample.metadata
    game_file = Path(metadata["game_file"]).resolve()
    env = None
    try:
        # Generated games sometimes seed Python's module-global RNG. Serialize
        # load/reset and step calls so concurrent GRPO samples keep their seed.
        async with _ENV_CALL_LOCK:
            env = await asyncio.wait_for(
                asyncio.to_thread(
                    make_synthetic_env,
                    str(game_file),
                    max_turns=_setting("SPADE_MAX_TURNS", 12),
                    respect_game_max_turns=True,
                ),
                timeout=20,
            )
            observation, _ = await asyncio.wait_for(
                asyncio.to_thread(env.reset, seed=int(metadata["seed"])), timeout=10
            )
        if observation != metadata.get("initial_observation", observation):
            raise ValueError(f"Reset observation drift for {game_file}")
        tokenizer = state.tokenizer
        sample.tokens = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.response_length = 0
        sample.response = ""
        sample.metadata["spade_turns"] = 0
        sample.metadata["spade_terminated"] = False
        sample.metadata["spade_terminal_reward"] = 0.0
        max_turns = _setting("SPADE_MAX_TURNS", 12)
        turn_tokens = _setting("SPADE_TURN_TOKENS", 256)
        context_budget = _setting("SPADE_MAX_CONTEXT_TOKENS", 4096)
        total_budget = sampling_params["max_new_tokens"]
        generated_tokens = 0
        for turn in range(max_turns):
            if state.aborted:
                sample.status = Sample.Status.ABORTED
                break
            remaining = min(turn_tokens, total_budget - generated_tokens,
                            context_budget - len(sample.tokens))
            if remaining < 1:
                sample.status = Sample.Status.TRUNCATED
                break
            params = {**sampling_params, "max_new_tokens": remaining}
            output = await post(
                f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
                {"input_ids": sample.tokens, "sampling_params": params, "return_logprob": True},
                use_http2=args.use_http2,
            )
            meta = output["meta_info"]
            logprobs = meta.get("output_token_logprobs")
            if not logprobs:
                raise RuntimeError("SGLang returned no token log probabilities")
            ids = [entry[1] for entry in logprobs]
            probs = [entry[0] for entry in logprobs]
            sample.tokens.extend(ids)
            sample.loss_mask.extend([1] * len(ids))
            sample.rollout_log_probs.extend(probs)
            sample.response_length += len(ids)
            generated_tokens += len(ids)
            action = output["text"]
            sample.response += action
            sample.metadata["spade_turns"] = turn + 1
            if "weight_version" in meta:
                sample.weight_versions.append(meta["weight_version"])
            finish_reason = meta["finish_reason"]["type"]
            if finish_reason == "abort":
                raise RuntimeError("SGLang aborted the game turn")
            if finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
                break
            if finish_reason != "stop":
                raise ValueError(f"Unexpected finish reason: {finish_reason}")
            # Some servers omit EOS from returned token IDs. Add it as masked
            # context so the next user message starts at a valid chat boundary.
            eos = tokenizer.eos_token_id
            if eos is not None and ids[-1] != eos:
                sample.tokens.append(eos)
                sample.loss_mask.append(0)
                sample.rollout_log_probs.append(0.0)
                sample.response_length += 1
            async with _ENV_CALL_LOCK:
                observation, reward, terminated, truncated, _ = await asyncio.wait_for(
                    asyncio.to_thread(env.step, parse_action(action)), timeout=10
                )
            if terminated or truncated:
                sample.metadata["spade_terminated"] = bool(terminated)
                sample.metadata["spade_terminal_reward"] = float(reward)
                sample.status = Sample.Status.COMPLETED if terminated else Sample.Status.TRUNCATED
                break
            _append_observation(sample, tokenizer, observation)
        else:
            sample.status = Sample.Status.TRUNCATED
        # SPADE's published episode reward uses the final reward only when the
        # game terminates, clipped to [-1, 1]. Intermediate shaping is ignored.
        final_reward = max(-1.0, min(1.0, sample.metadata["spade_terminal_reward"]))
        if not sample.metadata["spade_terminated"]:
            final_reward = 0.0
        sample.metadata["spade_reward"] = {
            "reward": final_reward,
            "accuracy": float(final_reward >= 1.0),
            "positive_reward": float(final_reward > 0),
        }
        assert sample.response_length == len(sample.loss_mask) == len(sample.rollout_log_probs)
        assert len(sample.tokens) >= sample.response_length
        return sample
    except Exception as error:
        sample.status = Sample.Status.ABORTED
        sample.metadata["spade_error"] = repr(error)
        raise RuntimeError(f"SPADE rollout failed for {game_file.name} seed {metadata['seed']}") from error
    finally:
        if env is not None:
            env.close()


async def reward(args, sample: Sample) -> dict[str, float]:
    return sample.metadata["spade_reward"]
