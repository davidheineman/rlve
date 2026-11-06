"""Measure episode rewards of a local model on the frozen SPADE pool."""

import argparse
import json
import signal
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spade_games.protocol import observation_suffix_ids, parse_action
from spade_games.synthetic_game_env import make_synthetic_env


@contextmanager
def game_call_timeout(seconds: float):
    """Bound generated game code without interrupting GPU model generation."""
    def raise_timeout(_signum, _frame):
        raise TimeoutError(f"game call exceeded {seconds:g}s")

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def probe(model_path: str, prompts_path: Path, output_path: Path, seeds: int,
          repeats: int, max_turns: int, tokens_per_turn: int, total_tokens: int,
          max_context_tokens: int, temperature: float, top_p: float = 0.95,
          dtype: str = "fp16", game_timeout: float = 10.0) -> None:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
        device_map="cuda", trust_remote_code=True
    ).eval()
    rows = [json.loads(line) for line in prompts_path.open()]
    rows = [row for row in rows if row["metadata"]["seed"] < seeds]
    rows = [(row, repeat) for row in rows for repeat in range(repeats)]
    if not rows:
        raise ValueError("No prompts selected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output_path.open("w") as stream, torch.inference_mode():
        for index, (row, repeat) in enumerate(rows):
            metadata = row["metadata"]
            env = None
            rewards = []
            turns = []
            stop_reasons = []
            output_tokens = 0
            terminated = truncated = False
            error = None
            try:
                with game_call_timeout(game_timeout):
                    env = make_synthetic_env(
                        metadata["game_file"], max_turns=max_turns,
                        respect_game_max_turns=True,
                    )
                    observation, _ = env.reset(seed=metadata["seed"])
                if observation != metadata["initial_observation"]:
                    raise ValueError("Reset observation differs from frozen prompt")
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["user_prompt"]}],
                    tokenize=True, add_generation_prompt=True,
                )
                for _ in range(max_turns):
                    inputs = torch.tensor([input_ids], device=model.device)
                    remaining = min(tokens_per_turn, total_tokens - output_tokens,
                                    max_context_tokens - inputs.shape[-1])
                    if remaining < 1:
                        truncated = True
                        break
                    generated = model.generate(
                        inputs,
                        attention_mask=torch.ones_like(inputs),
                        max_new_tokens=remaining,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        top_p=top_p if temperature > 0 else None,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0, inputs.shape[-1]:]
                    output_tokens += len(generated)
                    input_ids.extend(generated.tolist())
                    action = tokenizer.decode(generated, skip_special_tokens=True)
                    turns.append(action)
                    hit_eos = bool(len(generated) and generated[-1].item() == tokenizer.eos_token_id)
                    stop_reasons.append("eos" if hit_eos else "length")
                    if not hit_eos:
                        truncated = True
                        break
                    with game_call_timeout(game_timeout):
                        observation, reward, terminated, truncated, _ = env.step(parse_action(action))
                    rewards.append(float(reward))
                    if observation.startswith("Error:") or observation == "Invalid game response":
                        raise RuntimeError(f"Game wrapper failed: {observation}")
                    if terminated or truncated:
                        break
                    if metadata.get("no_think"):
                        observation += "\n/no_think"
                    input_ids.extend(observation_suffix_ids(tokenizer, observation))
            except Exception as exc:  # noqa: BLE001 - isolate broken games during a scan
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if env is not None:
                    try:
                        with game_call_timeout(game_timeout):
                            env.close()
                    except Exception as exc:  # noqa: BLE001 - keep scanning other games
                        error = error or f"{type(exc).__name__}: {exc}"
            outcome = max(-1.0, min(1.0, rewards[-1])) if rewards and terminated and not error else 0.0
            result = {
                "environment": metadata["environment"], "seed": metadata["seed"],
                "repeat": repeat, "reward": outcome, "terminated": terminated,
                "truncated": truncated, "turns": len(turns), "step_rewards": rewards,
                "actions": turns, "stop_reasons": stop_reasons,
                "output_tokens": output_tokens, "error": error,
            }
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index + 1}/{len(rows)} {metadata['environment']} "
                  f"reward={outcome:.3f} turns={len(turns)}"
                  f"{f' error={error}' if error else ''}", flush=True)
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["environment"], result["seed"])].append(result["reward"])
    informative = sum(min(values) != max(values) for values in grouped.values())
    positive = sum(result["reward"] > 0 for result in results)
    first_turn_lengths = sum(
        bool(result["stop_reasons"]) and result["stop_reasons"][0] == "length"
        for result in results
    )
    print(f"Summary: {positive}/{len(results)} positive rewards; "
          f"{first_turn_lengths}/{len(results)} first turns hit token cap; "
          f"{informative}/{len(grouped)} prompt groups have reward variance", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", type=Path, default=Path("data/SPADE-concise/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/SPADE-concise/probe.jsonl"))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--tokens-per-turn", type=int, default=256)
    parser.add_argument("--total-tokens", type=int, default=2048)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--game-timeout", type=float, default=10.0)
    args = parser.parse_args()
    probe(args.model_path, args.prompts, args.output, args.seeds, args.repeats,
          args.max_turns, args.tokens_per_turn, args.total_tokens,
          args.max_context_tokens, args.temperature, args.top_p, args.dtype,
          args.game_timeout)


if __name__ == "__main__":
    main()
