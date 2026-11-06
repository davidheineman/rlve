"""Versioned files and HTTP operations shared by the core and elastic samplers."""

import json
import math
import shlex
import socket
import sys
import urllib.request
from pathlib import Path

from async_branching.io import read_json


def http(url, payload=None, *, method=None, timeout=10):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        try:
            return json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return raw.decode()


def local_ip():
    addresses = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)
    return next(address[4][0] for address in addresses if not address[4][0].startswith("127."))


def router_workers(router):
    return http(router + "/workers")["workers"]


def unregister(router, url):
    for worker in router_workers(router):
        if worker["url"] == url:
            http(f"{router}/workers/{worker['id']}", method="DELETE")


def engine_argv(info, *, host, port, slot):
    """Clone the core engine's actual CLI, changing only worker-specific settings."""
    original = shlex.split(info["launch_command"])
    if "sglang.launch_server" in original:
        original = original[original.index("sglang.launch_server") + 1 :]
    elif not original or not original[0].startswith("--"):
        raise ValueError("Unrecognized SGLang launch command")
    replacements = {
        "--host": host,
        "--port": port,
        "--base-gpu-id": 0,
        "--gpu-id-step": 1,
        "--nnodes": 1,
        "--node-rank": 0,
        "--random-seed": int(info["random_seed"]) + 1000 + slot,
    }
    removed = {"--dist-init-addr", "--nccl-port", "--engine-info-bootstrap-port", "--grpc-port", "--gated-launch-port"}
    output, index = [], 0
    while index < len(original):
        flag = original[index]
        if flag in replacements or flag in removed:
            index += 2
        else:
            output.append(flag)
            index += 1
    for flag, value in replacements.items():
        output.extend((flag, str(value)))
    return [sys.executable, "-m", "sglang.launch_server", *output]


def validate_engine(info, *, model_path, slots, rank):
    if (
        Path(info["model_path"]).resolve() != Path(model_path).resolve()
        or not info["enable_lora"]
        or info["max_lora_rank"] != rank
        or info["max_loras_per_batch"] != slots
        or info["tp_size"] != 1
    ):
        raise ValueError("Elastic generator does not match the frozen base/LoRA configuration")


def check_probe(url, probe):
    result = http(url + "/generate", probe["request"], timeout=120)
    values = [entry[0] for entry in result["meta_info"]["input_token_logprobs"]]
    expected = probe["logprobs"]
    if len(values) != len(expected):
        raise ValueError("Elastic generator returned misaligned adapter log probabilities")
    errors = []
    for actual, reference in zip(values[1:], expected[1:], strict=True):
        if actual is None or not math.isfinite(actual):
            raise ValueError("Elastic generator returned nonfinite adapter log probabilities")
        errors.append(abs(actual - reference))
    maximum = max(errors, default=0)
    if maximum > 0.05:
        raise ValueError(f"Elastic generator adapter probe differs from the core by {maximum:.6f}")
    return {"adapter_logprob_max_abs_difference": maximum, "tokens": len(values)}


def read_optional(path):
    return read_json(path) if Path(path).exists() else None
