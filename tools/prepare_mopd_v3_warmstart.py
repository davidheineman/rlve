"""Turn completed v3 teacher training traces into one epoch of SFT replay batches."""

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import deque
from pathlib import Path

import torch
from transformers import AutoTokenizer


def prepare(teacher_dir, environment_file, output_dir, tokenizer, batch_size=8, seed=17,
            max_total_tokens=2048, expected_num_environments=16):
    teacher_dir = Path(teacher_dir)
    output_dir = Path(output_dir)
    environments = Path(environment_file).read_text().split()
    if (expected_num_environments not in (4, 8, 16) or
            len(environments) != expected_num_environments or
            len(set(environments)) != expected_num_environments):
        raise ValueError(f"Expected {expected_num_environments} distinct v3 teacher environments")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if (output_dir / "manifest.json").exists():
        manifest = json.loads((output_dir / "manifest.json").read_text())
        if set(manifest.get("per_environment", {})) != set(environments):
            raise RuntimeError("Existing SFT replay uses different teacher environments")
        if len(list(output_dir.glob("*.pt"))) == manifest["num_batches"]:
            return manifest
        raise RuntimeError("Existing SFT replay directory is incomplete")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    rng = random.Random(seed)
    trace_hash = hashlib.sha256()
    per_environment = {}
    queues = {}
    for index, environment in enumerate(environments):
        run = teacher_dir / f"{index:02d}-{environment}"
        marker = json.loads((run / "weights_only_complete.json").read_text())
        terminal_step = int(marker["checkpoint_step"])
        if not 0 <= terminal_step <= 499 or not (run / "hf-final" / "model.safetensors.index.json").exists():
            raise RuntimeError(f"Teacher export is incomplete: {run}")
        selected = []
        counts = {"seen": 0, "correct": 0, "usable": 0, "overlong": 0}
        for rollout_id in range(terminal_step + 1):
            shard = run / "rollouts" / "train" / f"{rollout_id:04d}.jsonl"
            if not shard.is_file():
                raise FileNotFoundError(f"Missing teacher trace shard: {shard}")
            trace_hash.update(f"{environment}/{rollout_id}\n".encode())
            with shard.open("rb") as source:
                for line in source:
                    trace_hash.update(line)
                    row = json.loads(line)
                    counts["seen"] += 1
                    if row["source_environment"] != environment or row["rollout_id"] != rollout_id:
                        raise ValueError(f"Trace identity mismatch in {shard}")
                    if not row["correct"] or row["status"] != "completed" or not row["response"]:
                        continue
                    counts["correct"] += 1
                    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
                    response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
                    if tokenizer.eos_token_id is not None:
                        response_ids.append(tokenizer.eos_token_id)
                    if not response_ids or len(prompt_ids) + len(response_ids) > max_total_tokens:
                        counts["overlong"] += 1
                        continue
                    selected.append({
                        "prompt": row["prompt"],
                        "response": row["response"],
                        "tokens": prompt_ids + response_ids,
                        "response_length": len(response_ids),
                        "reward": {"accuracy": 1.0, "reward": 1.0},
                        "status": "completed",
                        "metadata": {
                            "environment": environment,
                            "source_rollout_id": rollout_id,
                            "prompt_id": row["prompt_id"],
                        },
                    })
                    counts["usable"] += 1
        if not selected:
            raise RuntimeError(f"No usable correct teacher traces for {environment}")
        rng.shuffle(selected)
        queues[environment] = deque(selected)
        per_environment[environment] = counts | {"terminal_step": terminal_step}

    batch = []
    num_batches = 0
    total_selected = sum(len(queue) for queue in queues.values())
    while any(queues.values()):
        for environment in environments:
            if not queues[environment]:
                continue
            sample = queues[environment].popleft()
            sample["index"] = num_batches * batch_size + len(batch)
            batch.append(sample)
            if len(batch) == batch_size:
                path = temporary / f"{num_batches:06d}.pt"
                torch.save({"rollout_id": num_batches, "samples": batch}, path)
                num_batches += 1
                batch = []
    if num_batches < 2:
        raise RuntimeError("At least two SFT batches are needed for checkpointing")
    manifest = {
        "source_environments": environments,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "selected_traces": total_selected,
        "discarded_remainder": len(batch),
        "selection": "completed correct training traces, one epoch, round-robin across environments",
        "max_total_tokens": max_total_tokens,
        "seed": seed,
        "source_sha256": trace_hash.hexdigest(),
        "per_environment": per_environment,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--expected-num-environments", type=int, default=16,
                        choices=(4, 8, 16))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    manifest = prepare(args.teacher_dir, args.environment_file, args.output_dir, tokenizer,
                       args.batch_size, max_total_tokens=args.max_total_tokens,
                       expected_num_environments=args.expected_num_environments)
    print(json.dumps({key: value for key, value in manifest.items() if key != "per_environment"}, sort_keys=True))


if __name__ == "__main__":
    main()
