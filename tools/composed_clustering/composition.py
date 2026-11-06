"""Represent simultaneous normalized LoRA updates without SVD truncation."""

import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def exact_factors(factors, scales):
    if not factors or len(factors) != len(scales):
        raise ValueError("Expected corresponding nonempty factors and scales")
    # Bcat @ Acat = mean_i (scale_i * B_i @ A_i). The saved adapter has
    # alpha/r = 1; every member contributes at every adapted model layer.
    count = len(factors)
    a = torch.cat([a.float() for a, _ in factors], dim=0)
    b = torch.cat(
        [b.float() * (scale / count) for (_, b), scale in zip(factors, scales)], dim=1
    )
    return a.contiguous(), b.contiguous()


def compose(paths, destination, *, inference_storage=False):
    paths = list(map(Path, paths))
    destination = Path(destination)
    if not paths:
        raise ValueError("Cannot compose an empty group")
    if destination.exists():
        raise FileExistsError(destination)
    configs = [json.loads((path / "adapter_config.json").read_text()) for path in paths]
    for config in configs:
        if any(
            config.get(key)
            for key in [
                "use_rslora",
                "use_dora",
                "rank_pattern",
                "alpha_pattern",
                "fan_in_fan_out",
            ]
        ):
            raise ValueError(
                "Exact composition requires standard uniformly-scaled LoRA"
            )
    for key in ["base_model_name_or_path", "target_modules", "bias"]:
        if any(config.get(key) != configs[0].get(key) for config in configs):
            raise ValueError(f"Incompatible adapters: {key}")
    temporary = destination.with_name(destination.name + ".incomplete")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    scales = [config["lora_alpha"] / config["r"] for config in configs]
    rank = sum(config["r"] for config in configs)
    tensors = {}
    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                safe_open(
                    path / "adapter_model.safetensors", framework="pt", device="cpu"
                )
            )
            for path in paths
        ]
        keys = set(handles[0].keys())
        if any(set(handle.keys()) != keys for handle in handles):
            raise ValueError("Adapter target tensors differ")
        for akey in sorted(key for key in keys if ".lora_A." in key):
            bkey = akey.replace(".lora_A.", ".lora_B.")
            factors = [
                (handle.get_tensor(akey), handle.get_tensor(bkey)) for handle in handles
            ]
            a, b = exact_factors(factors, scales)
            # The inference engine uses BF16 anyway. Saving its eventual
            # tensor dtype reduces transient IO without truncating LoRA rank.
            tensors[akey] = a.to(torch.bfloat16) if inference_storage else a
            tensors[bkey] = b.to(torch.bfloat16) if inference_storage else b
        if set(tensors) != keys:
            raise ValueError("Unexpected tensors in the source adapters")
        save_file(tensors, str(temporary / "adapter_model.safetensors"))
    config = dict(configs[0], r=rank, lora_alpha=rank, inference_mode=True)
    (temporary / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    record = {
        "members": list(map(str, paths)),
        "weights": [1 / len(paths)] * len(paths),
        "rank": rank,
        "alpha": rank,
        "compression": "none; concatenated factors",
        "dtype": "bfloat16 (inference dtype)"
        if inference_storage
        else "float32 on disk; inference engine dtype applies",
    }
    (temporary / "composition.json").write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(destination)
    return record
