"""Build matched, correctness-ordered pairs from N=8 teacher periods.

Each environment is processed independently on one GPU. Existing high-period
correct traces are paired with a fresh low-snapshot response, and existing
low-period incorrect traces are paired with a fresh final-teacher response.
Fresh training problems fill any shortfall. No evaluation prompts are used.
"""

import argparse
import hashlib
import json
import os
import random
import re
from bisect import bisect_right
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Gym.environments import identifier2environment
from slime.ray.rollout_data_source import RLVEManager
from slime.rollout.rm_hub.rlve_rm import rlve_rm


VAL_PATTERN = re.compile(r"eval (\d+): \{'eval/val_envs': ([0-9.]+)")


def validation_periods(log_path):
    values = []
    with log_path.open(errors="replace") as stream:
        for line in stream:
            match = VAL_PATTERN.search(line)
            if match:
                values.append((int(match.group(1)), float(match.group(2))))
    if not values or values[0][0] != 0:
        raise RuntimeError(f"Missing initial validation history: {log_path}")
    return values


def trace_candidates(run_dir, log_path, environment):
    periods = validation_periods(log_path)
    steps = [step for step, _ in periods]
    low, high = [], []
    for shard in sorted((run_dir / "rollouts" / "train").glob("*.jsonl")):
        step = int(shard.stem)
        period = bisect_right(steps, step) - 1
        if period < 0:
            continue
        accuracy = periods[period][1]
        with shard.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["source_environment"] != environment or row["status"] not in ("completed", "truncated") or not row["response"]:
                    continue
                if accuracy < 0.25 and not row["correct"]:
                    low.append(row)
                elif accuracy > 0.25 and row["correct"]:
                    high.append(row)
    return low, high, periods


def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval().to("cuda")


@torch.inference_mode()
def generate(model, tokenizer, prompts, max_response_tokens):
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    output = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        do_sample=True, temperature=1.0, top_p=1.0,
        max_new_tokens=max_response_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    prompt_width = input_ids.size(1)
    eos = tokenizer.eos_token_id
    results = []
    for prompt, row in zip(prompts, output):
        generated = row[prompt_width:].tolist()
        if eos in generated:
            generated = generated[:generated.index(eos) + 1]
            completed = True
        else:
            completed = False
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        results.append((generated, len(prompt_ids) + len(generated) <= 2048))
    return results


def tokens_for_trace(tokenizer, row):
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
    if row["status"] == "completed" and tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != tokenizer.eos_token_id):
        response_ids.append(tokenizer.eos_token_id)
    if not response_ids or len(prompt_ids) + len(response_ids) > 2048:
        return None
    return prompt_ids, response_ids


def sample_dict(prompt, response_ids, tokenizer, environment, config, accuracy, source, source_step):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return {
        "prompt": prompt,
        "tokens": prompt_ids + response_ids,
        "response": tokenizer.decode(response_ids, skip_special_tokens=False),
        "response_length": len(response_ids),
        "reward": {"accuracy": float(accuracy), "reward": 1.0 if accuracy else -1.0},
        "status": "completed" if response_ids[-1] == tokenizer.eos_token_id else "truncated",
        "metadata": {
            "environment": environment, "config": config,
            "source": source, "teacher_step": source_step,
        },
    }


def build(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_hf, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    environment = args.environment
    if environment not in identifier2environment:
        raise ValueError(f"Unknown environment: {environment}")
    frozen_prompts = {
        row["user_prompt"] for row in json.loads((args.eval_root / f"{environment}_d0.json").read_text())
    }
    frozen_prompts.update(
        row["user_prompt"] for row in json.loads((args.eval_root / "id_n8_d0_x4.json").read_text())
        if json.loads(row["metadata"])["environment"] == environment
    )

    def is_frozen(prompt):
        return any(user_prompt in prompt for user_prompt in frozen_prompts)
    run = args.teacher_root / f"{args.index:02d}-{environment}"
    if not (run / "weights_only_complete.json").exists():
        raise RuntimeError(f"Incomplete high teacher: {run}")
    low_run = args.low_root / f"{args.index:02d}-{environment}"
    if environment != "CRT" and not (low_run / "weights_only_complete.json").exists():
        raise RuntimeError(f"Incomplete low teacher: {low_run}")
    low_rows, high_rows, periods = trace_candidates(
        run, args.teacher_root.parent / f"{args.index:02d}-{environment}.log", environment,
    )
    if environment != "CRT" and not any(value < .25 for _, value in periods):
        raise RuntimeError(f"No low validation period for {environment}")
    if not any(value > .25 for _, value in periods):
        raise RuntimeError(f"No high validation period for {environment}")
    rng = random.Random(args.seed)
    rng.shuffle(low_rows)
    rng.shuffle(high_rows)
    high = load_model(run / "hf-final")
    low = load_model(args.base_hf if environment == "CRT" else low_run / "hf-final")
    verifier_args = SimpleNamespace(
        answer_marker_type=r"\boxed{}", custom_prompt_preprocessor="ChatTemplate_BoxedAnswer",
    )
    fresh_args = SimpleNamespace(
        environment_list=[environment], initial_difficulty=0, rlve_prompt_pool=None,
        difficulty_sliding_window_size=4,
        custom_prompt_preprocessor="ChatTemplate_BoxedAnswer", apply_chat_template=True,
        answer_marker_type=r"\boxed{}", rollout_max_prompt_len=None, tool_key=None,
    )
    manager = RLVEManager(fresh_args, tokenizer)
    manager.problem_generation_seed = 1_000_000 + args.index * 100_000
    pairs = []
    seen_prompts = set()
    counts = {"high_trace": 0, "low_trace": 0, "fresh": 0, "attempted": 0,
              "high_candidates": len(high_rows), "low_candidates": len(low_rows)}

    def append_pair(prompt, config, high_ids, low_ids, source, high_step, low_step):
        if prompt in seen_prompts or is_frozen(prompt) or not high_ids or not low_ids:
            return False
        high_text = tokenizer.decode(high_ids, skip_special_tokens=False)
        low_text = tokenizer.decode(low_ids, skip_special_tokens=False)
        high_score = rlve_rm(verifier_args, environment, config, high_text)["accuracy"]
        low_score = rlve_rm(verifier_args, environment, config, low_text)["accuracy"]
        if high_score != 1 or low_score != 0:
            return False
        chosen = sample_dict(prompt, high_ids, tokenizer, environment, config, 1, source + ":high", high_step)
        rejected = sample_dict(prompt, low_ids, tokenizer, environment, config, 0, source + ":low", low_step)
        chosen["index"] = len(pairs) * 2
        rejected["index"] = len(pairs) * 2 + 1
        pairs.append({"prompt_id": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                      "chosen": chosen, "rejected": rejected})
        seen_prompts.add(prompt)
        counts[source] += 1
        return True

    for source, rows, model in (("high_trace", high_rows, low), ("low_trace", low_rows, high)):
        for start in range(0, len(rows), args.batch_size):
            if len(pairs) >= args.target_pairs:
                break
            batch = rows[start:start + args.batch_size]
            generated = generate(model, tokenizer, [row["prompt"] for row in batch], args.max_response_tokens)
            for row, (new_ids, completed) in zip(batch, generated):
                if not completed or row["prompt"] in seen_prompts:
                    continue
                original = tokens_for_trace(tokenizer, row)
                if original is None:
                    continue
                _, original_ids = original
                config = json.loads(row["metadata_json"])["config"]
                counts["attempted"] += 1
                if source == "high_trace":
                    append_pair(row["prompt"], config, original_ids, new_ids,
                                source, row["rollout_id"], 0 if environment != "CRT" else -1)
                else:
                    append_pair(row["prompt"], config, new_ids, original_ids,
                                source, int(json.loads((run / "weights_only_complete.json").read_text())["checkpoint_step"]),
                                row["rollout_id"])
        if len(pairs) >= args.target_pairs:
            break

    while len(pairs) < args.target_pairs and counts["attempted"] < args.max_attempts:
        samples = []
        while len(samples) < args.batch_size:
            sample = manager.get_sample()
            if sample is not None and sample.prompt not in seen_prompts and not is_frozen(sample.prompt):
                samples.append(sample)
        prompts = [sample.prompt for sample in samples]
        high_generated = generate(high, tokenizer, prompts, args.max_response_tokens)
        low_generated = generate(low, tokenizer, prompts, args.max_response_tokens)
        for sample, (high_ids, high_done), (low_ids, low_done) in zip(samples, high_generated, low_generated):
            counts["attempted"] += 1
            if high_done and low_done:
                append_pair(sample.prompt, sample.metadata["config"], high_ids, low_ids,
                            "fresh", periods[-1][0], 0 if environment != "CRT" else -1)
        if counts["attempted"] % 1000 < args.batch_size:
            print(f"{environment}: {len(pairs)}/{args.target_pairs} pairs after {counts['attempted']} candidates", flush=True)

    if len(pairs) < args.target_pairs:
        raise RuntimeError(f"Only {len(pairs)} pairs after {counts['attempted']} candidates for {environment}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"environment": environment, "pairs": len(pairs), **counts}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=Path("data/MOPD-SWEEPS-V2"))
    parser.add_argument("--environment", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-pairs", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
