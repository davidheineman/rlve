"""Upload validated R1 OPD teacher exports to Hugging Face."""

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi

from verify_opd_teachers import verify


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def remote_weight_matches(api, repo_id, filename, expected_sha256):
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception:
        return False
    for sibling in info.siblings:
        if sibling.rfilename == filename:
            return sibling.lfs is not None and sibling.lfs.sha256 == expected_sha256
    return False


def upload_teacher(index, environment, args, collection_slug):
    api = HfApi()
    run = args.teacher_dir / f"{index:02d}-{environment}"
    export = run / "hf-final"
    weight = next(export.glob("model-*.safetensors"))
    digest = sha256(weight)
    repo_id = f"{args.namespace}/opd-teacher-R1Distill-{environment}-step149"
    api.create_repo(repo_id, private=False, exist_ok=True)
    if not remote_weight_matches(api, repo_id, weight.name, digest):
        api.upload_folder(
            repo_id=repo_id,
            folder_path=export,
            commit_message=f"Upload R1 Distill OPD teacher for {environment}",
        )
    if not remote_weight_matches(api, repo_id, weight.name, digest):
        raise RuntimeError(f"Uploaded weight digest does not match {environment}")
    card = f"""---
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
tags:
- rlve
- grpo
- opd-teacher
---

# RLVE OPD teacher: {environment}

DeepSeek-R1-Distill-Qwen-1.5B trained for 150 steps on difficulty-0
`{environment}` prompts, with four prompts and 16 rollouts per step. Training
did not use DAPO prompt filtering. These are the final model weights at step
149 for environment-specific on-policy distillation.

Training project: `david-heineman/rl-data-opd-teachers-r1-distil`.
Training group: `{args.group}`.
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add teacher model card",
    )
    api.add_collection_item(collection_slug, repo_id, "model", exists_ok=True)
    return repo_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--base-hf-dir", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--namespace", default="davidheineman")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    environments = verify(args.teacher_dir, args.environment_file, args.base_hf_dir)
    print(f"Verified {len(environments)} complete R1 teachers", flush=True)
    api = HfApi()
    collection = api.create_collection(
        "RLVE OPD Teachers (R1 Distill)",
        namespace=args.namespace,
        description="32 difficulty-0 RLVE OPD teachers trained from DeepSeek-R1-Distill-Qwen-1.5B for 150 steps each.",
        private=False,
        exists_ok=True,
    )
    print(f"Collection: https://huggingface.co/collections/{collection.slug}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(upload_teacher, index, environment, args, collection.slug): environment
            for index, environment in enumerate(environments)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            environment = futures[future]
            for attempt in range(3):
                try:
                    repo_id = future.result() if attempt == 0 else upload_teacher(
                        environments.index(environment), environment, args, collection.slug
                    )
                    break
                except Exception as error:
                    if attempt == 2:
                        raise
                    print(f"Retrying {environment}: {type(error).__name__}: {error}", flush=True)
                    time.sleep(2 ** attempt)
            print(f"Uploaded {completed}/{len(environments)}: {repo_id}", flush=True)
    collection = HfApi().get_collection(collection.slug)
    expected = {f"{args.namespace}/opd-teacher-R1Distill-{environment}-step149" for environment in environments}
    found = {item.item_id for item in collection.items if item.item_type == "model"}
    if found != expected:
        raise RuntimeError(f"Collection mismatch: missing={expected - found}, extra={found - expected}")
    print(f"Validated all {len(expected)} teacher repositories in collection", flush=True)


if __name__ == "__main__":
    main()
