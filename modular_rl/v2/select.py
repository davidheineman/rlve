"""Freeze full MATH/MBPP training pools and a proportional 32-prompt holdout."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from modular_rl.prepare import REWARD_MODEL, REWARD_REVISION, write_json

from .data import stratified_sample, stratum


def select(data, output, model, task, seed=42, chat_train_prompts=8192, val_prompts=32):
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    rm = (
        AutoTokenizer.from_pretrained(REWARD_MODEL, revision=REWARD_REVISION)
        if task == "chat"
        else None
    )
    selected, checksums, excluded, distributions = {}, {}, Counter(), {}
    for split in ("train", "val"):
        rows = []
        with (data / task / f"{split}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                meta = row["metadata"]
                meta["environment"] = task
                messages = meta["messages"]
                if (
                    len(
                        tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True, enable_thinking=True
                        )
                    )
                    > 2048
                ):
                    excluded[f"{split}/policy_prompt_exceeds_2048"] += 1
                    continue
                if rm:
                    reference = messages + [
                        {
                            "role": "assistant",
                            "content": meta["target"]["rejected_reference"],
                        }
                    ]
                    if len(rm.apply_chat_template(reference)) > 8192:
                        excluded[f"{split}/rm_reference_exceeds_8192"] += 1
                        continue
                    if (
                        len(
                            rm.apply_chat_template(messages, add_generation_prompt=True)
                        )
                        > 4096
                    ):
                        excluded[f"{split}/rm_prompt_exceeds_4096"] += 1
                        continue
                rows.append(row)
                if (
                    task == "chat"
                    and split == "train"
                    and len(rows) == chat_train_prompts
                ):
                    break
        if split == "val":
            rows = stratified_sample(rows, val_prompts, seed)
        path = output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        checksums[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected[split] = [row["metadata"]["key"] for row in rows]
        distributions[split] = dict(Counter("/".join(stratum(row)) for row in rows))
    if (
        not selected["train"]
        or len(selected["val"]) != val_prompts
        or set(selected["train"]) & set(selected["val"])
    ):
        raise ValueError("Invalid or overlapping selected pools")
    write_json(
        output / "selection.json",
        {
            "source": str(data / task),
            "keys": selected,
            "sha256": checksums,
            "excluded": dict(excluded),
            "strata": distributions,
            "enable_thinking": True,
            "seed": seed,
            "validation_prompts": val_prompts,
            "chat_train_limit": chat_train_prompts if task == "chat" else None,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", choices=("math", "code", "chat"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat-train-prompts", type=int, default=8192)
    parser.add_argument("--val-prompts", type=int, default=32)
    select(**vars(parser.parse_args()))
