"""Freeze the original N16 teachers and their RL settings for a shared-base run."""

import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "rl-data-scaling-lora-teachers-v1"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
IMAGE_TAG = "dev-202610231907"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(root=ROOT, *, project=PROJECT, learning_rate=None):
    source = root / "data/DATA-SCALING-V1"
    manifest = json.loads((source / "manifest.json").read_text())
    original = json.loads((source / "run_plan.json").read_text())
    panel = next(run for run in original["runs"] if run["label"] == "n16-mopd")
    names = (source / "n16.txt").read_text().split()
    if names != panel["environments"] or names != manifest["environments"][:16]:
        raise ValueError("N16 panel disagrees with the original MOPD plan")
    config_path = root / "configs/1gpu-data-scaling-teacher-v1.yaml"
    config = yaml.safe_load(config_path.read_text())
    overrides = {}
    if learning_rate is not None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        overrides["lr"] = {"source": config["lr"], "value": learning_rate}
        config["lr"] = learning_rate
    if not project or "/" in project or project in {".", ".."}:
        raise ValueError("Expected a W&B project name without a path")
    teachers = []
    for name in names:
        teacher = next(run for run in original["runs"] if run["environments"] == [name] and run["kind"] == "teacher")
        eval_path = root / teacher["eval"]
        expected = manifest["sha256"][teacher["eval"]]
        if sha256(eval_path) != expected:
            raise ValueError(f"Original validation inputs changed: {name}")
        teachers.append(
            {"environment": name, "source_label": teacher["label"], "eval": teacher["eval"], "eval_sha256": expected}
        )
    if len(set(names)) != 16 or config["num-rollout"] != 500:
        raise ValueError("Expected 16 distinct teachers with 500-update caps")
    return {
        "project": project,
        "entity": "david-heineman",
        "base_model": MODEL,
        "model_args_type": "qwen2.5-1.5B",
        "source_project": manifest["teacher_project"],
        "source_panel": "n16-mopd",
        "teacher_config": config,
        "teacher_config_sha256": sha256(config_path),
        "teacher_config_overrides": overrides,
        "teachers": teachers,
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": "attn,mlp",
        "train_gpus": 2,
        "sampling_gpus": 2,
        "image": f"docker://radixark/miles:{IMAGE_TAG}",
        "tinker_version": "0.26.2",
        "optimizer_eps": 1e-8,
        "grad_clip_norm": 1.0,
        "tis_clip_low": 0.0,
        "tis_clip_high": 2.0,
        "max_tokens_per_gpu": 8192,
        "sampling_mem_fraction": 0.7,
        "attention_backend": "flash",
        "flash_attention_version": 2,
        "seed": 1234,
        "execution_changes": {
            "base_weights": "frozen; train attention/MLP LoRA adapters only",
            "placement": "two training GPUs and two sampling GPUs; multi-LoRA forbids colocation",
            "optimizer_cpu_offload": False,
            "gradient_accumulation_fusion": False,
            "attention_backend": "FlashAttention 2 selected by the Megatron init hook; avoids unsupported FA4 backward",
            "max_tokens_per_gpu": "8192 packing budget; client generation budgets are unchanged",
            "storage": "temporary sampler exports on node-local disk; copy only validated final PEFT adapters to scratch",
        },
    }


def gateway_argv(plan, *, model_path, checkpoint_root, port, router_port):
    """Runtime flags; the runner prepends the registered Qwen2 model definition."""
    config = plan["teacher_config"]
    pairs = {
        "hf-checkpoint": model_path,
        "load": model_path,
        "megatron-to-hf-mode": "bridge",
        "actor-num-nodes": 1,
        "actor-num-gpus-per-node": plan["train_gpus"],
        "rollout-num-gpus": plan["sampling_gpus"],
        "num-gpus-per-node": plan["train_gpus"] + plan["sampling_gpus"],
        "rollout-num-gpus-per-engine": 1,
        "multi-lora-n-adapters": plan.get("adapter_slots", len(plan["teachers"])),
        "lora-rank": plan["lora_rank"],
        "lora-alpha": plan["lora_alpha"],
        "lora-dropout": 0.0,
        "target-modules": plan["target_modules"],
        "tinker-server-host": "127.0.0.1",
        "tinker-server-port": port,
        "tinker-checkpoint-root": checkpoint_root,
        "tinker-base-model": plan["base_model"],
        # Miles launches the router and discovers its host through the worker provider.
        "sglang-router-port": router_port,
        "sglang-mem-fraction-static": plan["sampling_mem_fraction"],
        "sglang-max-running-requests": config["sglang-server-concurrency"],
        "sglang-lora-backend": "triton",
        "tensor-model-parallel-size": 1,
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "expert-model-parallel-size": 1,
        "expert-tensor-parallel-size": 1,
        "optimizer": config["optimizer"],
        "lr": config["lr"],
        "weight-decay": config["weight-decay"],
        "adam-beta1": config["adam-beta1"],
        "adam-beta2": config["adam-beta2"],
        "use-dynamic-batch-size": True,
        "max-tokens-per-gpu": plan["max_tokens_per_gpu"],
        "recompute-granularity": config["recompute-granularity"],
        "recompute-method": config["recompute-method"],
        "recompute-num-layers": config["recompute-num-layers"],
        "attention-dropout": config["attention-dropout"],
        "hidden-dropout": config["hidden-dropout"],
        "accumulate-allreduce-grads-in-fp32": True,
        "attention-softmax-in-fp32": True,
        "attention-backend": plan["attention_backend"],
        "custom-megatron-init-path": "lora_teachers.attention.configure_flash_attention_2",
        "no-gradient-accumulation-fusion": True,
        "micro-batch-size": 1,
        "global-batch-size": 8,
        "seed": plan["seed"],
    }
    if plan.get("decode_graph_max_batch_size") is not None:
        pairs["sglang-cuda-graph-max-bs-decode"] = plan["decode_graph_max_batch_size"]
    if plan.get("elastic_generation"):
        pairs["sglang-router-request-timeout-secs"] = plan["elastic_generation"]["router_request_timeout_seconds"]
    return [
        part for key, value in pairs.items() for part in ([f"--{key}"] if value is True else [f"--{key}", str(value)])
    ]
