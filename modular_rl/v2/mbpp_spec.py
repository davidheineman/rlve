"""Prepare a separately versioned MBPP variant with one public specification test."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from modular_rl.prepare import digest, write_json

from .inspect_rollouts import read_events


def with_public_example(row):
    if row["metadata"]["task"] != "code":
        raise ValueError("The public-example variant is only for MBPP")
    if row["metadata"].get("public_example_tests"):
        raise ValueError("This prompt already has a public specification example")
    result = copy.deepcopy(row)
    metadata = result["metadata"]
    example = metadata["target"]["tests"][0]
    message = metadata["messages"][-1]
    suffix = "\nReturn a self-contained Python solution in a ```python code block."
    if not message["content"].endswith(suffix):
        raise ValueError("Unexpected original MBPP prompt format")
    message["content"] = (
        message["content"][: -len(suffix)]
        + "\nPublic example specifying the required behavior:\n```python\n"
        + example
        + "\n```"
        + suffix
    )
    metadata["original_key"] = metadata["key"]
    metadata["key"] = digest(metadata["messages"])
    metadata["public_example_tests"] = [example]
    result["user_prompt"] = copy.deepcopy(metadata["messages"])
    return result


def prepare(source, output, model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if output.exists():
        raise ValueError("Use a fresh prompt-variant directory")
    output.mkdir(parents=True)
    descriptor = json.loads((source / "selection.json").read_text())
    for split in ("train", "val"):
        original = read_events(source / f"{split}.jsonl")
        rows = [with_public_example(row) for row in original]
        for row in rows:
            tokens = tokenizer.apply_chat_template(
                row["metadata"]["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if len(tokens) > 2048:
                raise ValueError("Example exceeds the original prompt budget")
        path = output / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        descriptor["keys"][split] = [r["metadata"]["key"] for r in rows]
        descriptor["sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor.update(
        prompt_specification="one-public-test-v1",
        original_selection=str(source),
        grading_tests_unchanged=True,
        public_tests_per_prompt=1,
        source_implementation="https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/mbpp.py",
    )
    write_json(output / "selection.json", descriptor)
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": {k: len(v) for k, v in descriptor["keys"].items()},
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    prepare(**vars(parser.parse_args()))
