"""Select a bounded pilot from each frozen split without truncating prompts."""

import argparse
import hashlib
import json
from pathlib import Path

from .prepare import REWARD_MODEL, REWARD_REVISION, write_json


def select(data, output, task, model, train_prompts=256, val_prompts=16):
    from transformers import AutoTokenizer

    if min(train_prompts, val_prompts) < 0:
        raise ValueError("Prompt limits must be nonnegative; zero selects the full split")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    rm_tokenizer = (
        AutoTokenizer.from_pretrained(REWARD_MODEL, revision=REWARD_REVISION)
        if task == "chat"
        else None
    )
    selected, excluded, checksums = {}, [], {}
    for split, count in (("train", train_prompts), ("val", val_prompts)):
        rows = []
        with (data / task / f"{split}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                metadata = row["metadata"]
                # Native RLVE validation tables group examples by this field.
                metadata["environment"] = task
                messages = metadata["messages"]
                tokens = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, enable_thinking=False
                )
                reason = "policy_prompt_exceeds_2048" if len(tokens) > 2048 else None
                if rm_tokenizer and reason is None:
                    reference = messages + [
                        {
                            "role": "assistant",
                            "content": metadata["target"]["rejected_reference"],
                        }
                    ]
                    if len(rm_tokenizer.apply_chat_template(reference)) > 8192:
                        reason = "reward_reference_exceeds_8192"
                    elif (
                        len(
                            rm_tokenizer.apply_chat_template(
                                messages, add_generation_prompt=True
                            )
                        )
                        > 4096
                    ):
                        reason = "reward_prompt_exceeds_4096"
                if reason:
                    excluded.append(
                        {"split": split, "id": metadata["id"], "reason": reason}
                    )
                    continue
                rows.append(row)
                if count and len(rows) == count:
                    break
        if not rows or (count and len(rows) != count):
            raise ValueError(f"Insufficient eligible {task}/{split} prompts")
        destination = output / f"{split}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        checksums[split] = hashlib.sha256(destination.read_bytes()).hexdigest()
        selected[split] = [row["metadata"]["key"] for row in rows]
    if set(selected["train"]) & set(selected["val"]):
        raise ValueError("Pilot train/validation overlap")
    write_json(
        output / "selection.json",
        {
            "source": str(data / task),
            "keys": selected,
            "excluded": excluded,
            "sha256": checksums,
            "policy_tokenizer": str(model),
            "enable_thinking": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["math", "code", "chat"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--train-prompts",
        type=int,
        default=256,
        help="0 selects the full eligible train split",
    )
    parser.add_argument(
        "--val-prompts",
        type=int,
        default=16,
        help="0 selects the full eligible val split",
    )
    select(**vars(parser.parse_args()))
