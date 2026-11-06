"""Export the completed program-tree base RL checkpoint and upload its model weights."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

from archive_completed_mixing_checkpoints import tensor_shapes, validate_export


WANDB_RUN = "https://wandb.ai/david-heineman/rl-trees-base-rl/runs/f75d51ca"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def model_card() -> str:
    return f"""---
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
library_name: transformers
tags:
- reinforcement-learning
- rlve
- grpo
- program-trees
---

# RL-Base-R1-Distill-Q1.5B

DeepSeek-R1-Distill-Qwen-1.5B continued with reinforcement learning on
random program trees built from string-operation primitives. The trees have
depth 1–2 and up to two children at each node. Training used 20,000 sampled
rollouts (2,500 updates), one prompt and eight rollouts per update, four GPUs,
and 1,024 initial generation tokens plus answer forcing with up to 128 more
tokens. This is the final model checkpoint at update 2,499.

The training run and its in-distribution evaluation history are on
[Weights & Biases]({WANDB_RUN}). The optimizer and sampler state are retained
in the original training checkpoint; this repository provides model weights
and tokenizer files for inference or further fine-tuning.
"""


def remote_weight_matches(api: HfApi, repo_id: str, name: str, digest: str) -> bool:
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception:
        return False
    for sibling in info.siblings:
        if sibling.rfilename == name:
            return sibling.lfs is not None and sibling.lfs.sha256 == digest
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-hf-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="davidheineman/RL-Base-R1-Distill-Q1.5B")
    args = parser.parse_args()

    run = args.run_dir.resolve()
    origin = args.base_hf_dir.resolve()
    checkpoint_root = run / "ckpt"
    with (run / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((checkpoint_root / "rlve_checkpoint_state.json").read_text())
        iteration = int((checkpoint_root / "latest_checkpointed_iteration.txt").read_text())
        if not (state["completed"] and state["rollout_id"] == 2499
                and state["evaluated_rollout_id"] == 2499 and iteration == 2499):
            raise RuntimeError("Base RL run has not completed all 20,000 rollouts and final evaluation")
        checkpoint = checkpoint_root / f"iter_{iteration:07d}"
        if not (checkpoint / ".metadata").is_file():
            raise FileNotFoundError(checkpoint / ".metadata")

        reference_shapes, _ = tensor_shapes(origin)
        output = run / "hf-final"
        temporary = run / "hf-final.tmp"
        if not output.exists():
            if temporary.exists():
                shutil.rmtree(temporary)
            vocab_size = json.loads((origin / "config.json").read_text())["vocab_size"]
            subprocess.run([
                sys.executable, "-u", "tools/convert_torch_dist_to_hf.py",
                "--input-dir", str(checkpoint), "--output-dir", str(temporary),
                "--origin-hf-dir", str(origin), "--vocab-size", str(vocab_size),
                "--chunk-size", str(512 * 1024**2),
            ], check=True)
            count, size = validate_export(temporary, origin, reference_shapes)
            os.replace(temporary, output)
            print(f"Converted {count} tensors ({size / 1024**3:.2f} GiB)", flush=True)
        else:
            count, size = validate_export(output, origin, reference_shapes)
            print(f"Validated existing export: {count} tensors ({size / 1024**3:.2f} GiB)", flush=True)

        card = model_card()
        (output / "README.md").write_text(card)
        weights = sorted(output.glob("model-*.safetensors"))
        if not weights:
            raise RuntimeError("Converted model has no weights")
        digests = {weight.name: sha256(weight) for weight in weights}
        api = HfApi()
        if api.whoami().get("name") != args.repo_id.split("/", 1)[0]:
            raise RuntimeError("Hugging Face credentials do not match the target namespace")
        api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
        if not all(remote_weight_matches(api, args.repo_id, name, digest)
                   for name, digest in digests.items()):
            commit = api.upload_folder(
                repo_id=args.repo_id, repo_type="model", folder_path=str(output),
                commit_message="Upload completed program-tree base RL model",
            )
            print(f"Uploaded commit: {commit.oid}", flush=True)
        info = api.model_info(args.repo_id, files_metadata=True)
        siblings = {sibling.rfilename: sibling for sibling in info.siblings}
        expected_files = {p.name for p in output.iterdir() if p.is_file()}
        if not expected_files <= siblings.keys():
            raise RuntimeError(f"Missing remote files: {expected_files - siblings.keys()}")
        for name, digest in digests.items():
            sibling = siblings[name]
            if sibling.lfs is None or sibling.lfs.sha256 != digest:
                raise RuntimeError(f"Remote weight checksum mismatch: {name}")
        result = {
            "repo_id": args.repo_id, "revision": info.sha,
            "checkpoint_iteration": iteration, "rollouts": 20000,
            "weight_sha256": digests, "weight_bytes": size,
        }
        marker = run / "uploaded_to_hf.json"
        pending = marker.with_suffix(".json.tmp")
        pending.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(pending, marker)
        print(f"Verified https://huggingface.co/{args.repo_id} at {info.sha}", flush=True)
        print(f"Local HF model: {output}", flush=True)
        print(f"Original optimizer checkpoint retained: {checkpoint_root}", flush=True)


if __name__ == "__main__":
    main()
