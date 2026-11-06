"""Materialize pinned, deduplicated task pools and reproducible train/val splits."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .graders import grade_code, numeric_answer

DATASETS = {
    "math": ("apple/GSM-Symbolic", "p1", "test"),
    "code": ("Muennighoff/mbpp", "full", "test"),
    "chat": ("allenai/llama-3.1-tulu-3-8b-preference-mixture", None, "train"),
}
REWARD_MODEL = "Skywork/Skywork-Reward-V2-Llama-3.2-1B"
REWARD_REVISION = "c7b03c1a5e065600a8d23815faff18204aebefad"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize(task, row):
    if task == "math":
        if numeric_answer(row["answer"], reference=True) is None:
            raise ValueError("Unparseable GSM reference")
        messages = [
            {
                "role": "user",
                "content": row["question"]
                + "\n\nSolve step by step and put your final numeric answer in \\boxed{}.",
            }
        ]
        identifier = f"{row['id']}:{row['instance']}"
        # All 50 instantiations of a GSM template stay in the same split.
        group = f"gsm-original:{row['original_id']}"
        target = {"answer": row["answer"]}
    elif task == "code":
        tree = ast.parse(row["code"])
        tests = row["test_list"] + row.get("challenge_test_list", [])
        signatures = []
        test_text = "\n".join(tests)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in test_text:
                signatures.append(f"def {node.name}({ast.unparse(node.args)}): ...")
        if not tests or not signatures:
            raise ValueError("MBPP requires nonempty tests and an API signature")
        prompt = (
            row["text"]
            + "\nRequired API (implementation omitted):\n"
            + "\n".join(signatures)
        )
        prompt += "\nReturn a self-contained Python solution in a ```python code block."
        messages = [{"role": "user", "content": prompt}]
        identifier, group = str(row["task_id"]), digest(row["text"].strip())
        target = {"setup": row.get("test_setup_code", ""), "tests": tests}
    else:
        chosen, rejected = row["chosen"], row["rejected"]
        if (
            chosen[:-1] != rejected[:-1]
            or chosen[-1]["role"] != "assistant"
            or rejected[-1]["role"] != "assistant"
        ):
            raise ValueError("Preference pair has inconsistent conversation prefixes")
        messages = chosen[:-1]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("Preference prompt must end with a user turn")
        if any(m["role"] == "system" for m in messages):
            return None  # Skywork's model card excludes system messages.
        identifier, group = str(row["id"]), digest(messages)
        target = {"rejected_reference": rejected[-1]["content"]}
    key = digest(messages)
    return {
        "user_prompt": messages,
        "metadata": {
            "task": task,
            "environment": task,
            "id": identifier,
            "key": key,
            "split_group": group,
            "messages": messages,
            "target": target,
        },
    }


def assign_splits(examples, seed=42, val_fraction=0.1):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    groups = sorted({e["metadata"]["split_group"] for e in examples})
    if len(groups) < 2:
        raise ValueError("Need at least two independent groups")
    random.Random(seed).shuffle(groups)
    val_groups = set(
        groups[: max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))]
    )
    result = {"train": [], "val": []}
    for example in examples:
        split = "val" if example["metadata"]["split_group"] in val_groups else "train"
        result[split].append(example)
    for rows in result.values():
        random.Random(seed).shuffle(rows)
    assert not (
        {e["metadata"]["key"] for e in result["train"]}
        & {e["metadata"]["key"] for e in result["val"]}
    )
    return result


def prepare(
    root, seed=42, val_fraction=0.1, chat_limit=0, val_prompts=16, model_dir=None
):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifests = {}
    for task, (repo, subset, split) in DATASETS.items():
        directory = root / task
        directory.mkdir(exist_ok=True)
        manifest_path = directory / "manifest.json"
        options = {
            "seed": seed,
            "val_fraction": val_fraction,
            "chat_limit": chat_limit,
            "val_prompts": val_prompts,
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["options"] != options:
                raise ValueError(
                    f"Use a new output directory for changed split options: {directory}"
                )
            for filename, checksum in manifest["sha256"].items():
                if (
                    hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                    != checksum
                ):
                    raise ValueError(f"Dataset file changed: {directory / filename}")
            manifests[task] = manifest
            continue
        revision = api.dataset_info(repo).sha
        if task == "code":
            # This repo's legacy loader is a dataset script, unsupported by datasets>=4.
            # Read the very same pinned JSONL directly; no remote Python execution.
            source = hf_hub_download(
                repo, "data/mbpp.jsonl", repo_type="dataset", revision=revision
            )
            rows = [json.loads(line) for line in Path(source).read_text().splitlines()]
        else:
            rows = load_dataset(
                repo,
                name=subset,
                revision=revision,
                split=split,
                streaming=task == "chat",
            )
        if task == "chat" and chat_limit:
            rows = rows.shuffle(seed=seed, buffer_size=8192).take(chat_limit)
        examples, excluded, seen = [], [], set()
        counts = Counter()

        def audit(row, task=task):
            try:
                example = normalize(task, row)
            except (ValueError, SyntaxError) as exc:
                if task != "code":
                    raise
                return None, {"id": row["task_id"], "reason": str(exc)}
            if example is not None and task == "code":
                result = grade_code(row["code"], example["metadata"]["target"])
                if not result["accuracy"]:
                    return None, {
                        "id": row["task_id"],
                        "reason": "reference_fails_tests",
                        "grade": result,
                    }
            return example, None

        if task == "code":
            pool = ThreadPoolExecutor(max_workers=4)
            audited = pool.map(audit, rows)
        else:
            audited = map(audit, rows)
        for example, exclusion in audited:
            if exclusion:
                excluded.append(exclusion)
                continue
            if example is None:
                counts["system_prompt_excluded"] += 1
                continue
            key = example["metadata"]["key"]
            if key in seen:
                counts["duplicate_prompt_excluded"] += 1
                continue
            seen.add(key)
            examples.append(example)
        if task == "code":
            pool.shutdown()
        splits = assign_splits(examples, seed, val_fraction)
        splits["val-smoke"] = splits["val"][:val_prompts]
        checksums = {}
        for name, values in splits.items():
            path = directory / f"{name}.jsonl"
            temporary = path.with_suffix(".jsonl.tmp")
            with temporary.open("w") as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            temporary.replace(path)
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            counts[name] = len(values)
        write_json(directory / "excluded.json", excluded)
        manifest = {
            "dataset": repo,
            "subset": subset,
            "source_split": split,
            "revision": revision,
            "options": options,
            "counts": dict(counts),
            "reference_failures_excluded": len(excluded),
            "sha256": checksums,
            "split_unit": "original_id/template"
            if task == "math"
            else "deduplicated prompt",
        }
        write_json(manifest_path, manifest)
        manifests[task] = manifest
        print(json.dumps({"task": task, **manifest}), flush=True)
    snapshot_download(
        REWARD_MODEL,
        revision=REWARD_REVISION,
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
    )
    for name in ("Qwen3-0.6B", "Qwen3-1.7B"):
        path = (model_dir or root.parent / "models") / name
        if not (path / "config.json").exists():
            snapshot_download(
                "Qwen/" + name,
                local_dir=path,
                allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt"],
            )
    write_json(
        root / "ready.json",
        {
            "datasets": manifests,
            "reward_model": REWARD_MODEL,
            "reward_revision": REWARD_REVISION,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--chat-limit", type=int, default=0, help="0 uses the full preference mixture"
    )
    parser.add_argument(
        "--val-prompts",
        type=int,
        default=16,
        help="Prompts in the fast smoke eval file",
    )
    parser.add_argument("--model-dir", type=Path)
    prepare(**vars(parser.parse_args()))
