"""Average weight deltas, never separately average the non-identifiable factors."""

import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from async_branching.io import write_json


def mean_factors(factors, scales, rank, output_scale):
    """Thin QR + core SVD avoids materializing an output_dim x input_dim delta."""
    if not factors or len(factors) != len(scales) or rank <= 0 or output_scale <= 0:
        raise ValueError("Invalid adapter combination")
    weights = [float(scale) / len(factors) for scale in scales]
    bs = torch.cat(
        [b.float() * weight**0.5 for (_, b), weight in zip(factors, weights)], dim=1
    )
    ats = torch.cat(
        [a.float().T * weight**0.5 for (a, _), weight in zip(factors, weights)], dim=1
    )
    qb, rb = torch.linalg.qr(bs, mode="reduced")
    qa, ra = torch.linalg.qr(ats, mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    kept = min(rank, len(s))
    root = (s[:kept] / output_scale).sqrt()
    b = (qb @ u[:, :kept]) * root
    a = root[:, None] * (vh[:kept] @ qa.T)
    if kept < rank:
        a = torch.nn.functional.pad(a, (0, 0, 0, rank - kept))
        b = torch.nn.functional.pad(b, (0, rank - kept))
    energy = float(s.square().sum())
    retained = float(s[:kept].square().sum()) / energy if energy else 1.0
    return a.contiguous(), b.contiguous(), retained


def combine_adapters(paths, destination, *, rank=32, alpha=64):
    paths, destination = list(map(Path, paths)), Path(destination)
    if not paths:
        raise ValueError("Cannot combine an empty cluster")
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.name + ".incomplete")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    configs = [json.loads((p / "adapter_config.json").read_text()) for p in paths]
    fields = (
        "base_model_name_or_path",
        "target_modules",
        "fan_in_fan_out",
        "use_rslora",
        "bias",
    )
    if any(any(c.get(k) != configs[0].get(k) for k in fields) for c in configs):
        raise ValueError("Adapters have incompatible bases or targets")
    if any(
        c.get("use_rslora")
        or c.get("rank_pattern")
        or c.get("alpha_pattern")
        or c.get("use_dora")
        for c in configs
    ):
        raise ValueError(
            "This combination supports standard, uniformly scaled LoRA only"
        )
    scales = [c["lora_alpha"] / c["r"] for c in configs]
    result, energy = {}, {}
    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                safe_open(p / "adapter_model.safetensors", framework="pt", device="cpu")
            )
            for p in paths
        ]
        keys = set(handles[0].keys())
        if any(set(h.keys()) != keys for h in handles):
            raise ValueError("Adapter targets differ")
        for key in sorted(k for k in keys if ".lora_A." in k):
            bkey = key.replace(".lora_A.", ".lora_B.")
            factors = [(h.get_tensor(key), h.get_tensor(bkey)) for h in handles]
            a, b, retained = mean_factors(factors, scales, rank, alpha / rank)
            # Keep FP32 factors; low-rank truncation is the only intended approximation.
            result[key], result[bkey], energy[key] = a, b, retained
        if set(result) != keys:
            raise ValueError("Unexpected non-factor parameters in adapter export")
    save_file(result, str(temporary / "adapter_model.safetensors"))
    config = dict(configs[0], r=rank, lora_alpha=alpha, inference_mode=True)
    write_json(temporary / "adapter_config.json", config)
    if (paths[0] / "META.json").exists():
        metadata = json.loads((paths[0] / "META.json").read_text())
        if any(
            json.loads((p / "META.json").read_text()) != metadata for p in paths[1:]
        ):
            raise ValueError(
                "Adapters do not share the same experiment ownership/base metadata"
            )
        write_json(
            temporary / "META.json", dict(metadata, lora_rank=rank, lora_alpha=alpha)
        )
    write_json(
        temporary / "merge.json",
        {
            "sources": list(map(str, paths)),
            "weights": [1 / len(paths)] * len(paths),
            "rank": rank,
            "alpha": alpha,
            "retained_energy_by_module": energy,
        },
    )
    temporary.replace(destination)
    return {
        "minimum_retained_energy": min(energy.values()),
        "mean_retained_energy": sum(energy.values()) / len(energy),
    }
