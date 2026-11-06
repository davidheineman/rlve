"""Validate high-rank exact composition against repeated identical adapters."""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from composed_clustering.composition import compose
from composed_clustering.server import (
    http,
    load_adapter,
    start_server,
    stop_server,
    unload_adapter,
)


def probe(source, model, output):
    torch.set_num_threads(2)
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = json.loads((source / "teachers-completed.json").read_text())
    native = Path(results[0]["hf"])
    process, log, url = start_server(model, native, output, port=17600, max_rank=1024)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        tokens = tokenizer.encode("The sum of 1 and 1 is 2.", add_special_tokens=False)

        def scores(name):
            r = http(
                url + "/generate",
                {
                    "input_ids": tokens,
                    "lora_path": name,
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
                timeout=600,
            )
            return [row[0] for row in r["meta_info"]["input_token_logprobs"]][1:]

        expected = scores("native")
        unload_adapter(url, "native")
        destination = output / "adapter"
        existing = destination / "composition.json"
        if (
            existing.exists()
            and json.loads(existing.read_text())["members"] == [str(native)] * 32
        ):
            repeated = json.loads(existing.read_text())
        else:
            shutil.rmtree(destination, ignore_errors=True)
            repeated = compose([native] * 32, destination)
        # Check the weight update independently in FP32 on the GPU. Equivalent
        # ranks can differ slightly in BF16 activations because they use
        # different reduction shapes; that is not rank truncation.
        torch.backends.cuda.matmul.allow_tf32 = False
        with (
            safe_open(
                native / "adapter_model.safetensors", framework="pt", device="cpu"
            ) as original,
            safe_open(
                destination / "adapter_model.safetensors", framework="pt", device="cpu"
            ) as combined,
        ):
            key = next(
                k for k in original.keys() if "mlp.down_proj" in k and ".lora_A." in k
            )
            bkey = key.replace(".lora_A.", ".lora_B.")
            config = json.loads((native / "adapter_config.json").read_text())
            reference_delta = (
                original.get_tensor(bkey).float().cuda()
                @ original.get_tensor(key).float().cuda()
            ) * (config["lora_alpha"] / config["r"])
            composed_delta = (
                combined.get_tensor(bkey).float().cuda()
                @ combined.get_tensor(key).float().cuda()
            )
            torch.testing.assert_close(
                composed_delta, reference_delta, rtol=1e-4, atol=1e-6
            )
            delta_relative_error = float(
                torch.linalg.vector_norm(composed_delta - reference_delta)
                / torch.linalg.vector_norm(reference_delta)
            )
            del reference_delta, composed_delta
        torch.cuda.empty_cache()
        load_adapter(url, "repeat", destination)
        actual = scores("repeat")
        if not np.isfinite(actual).all() or not np.allclose(
            actual, expected, atol=0.1, rtol=0
        ):
            raise RuntimeError(
                f"Exact composition changed repeated-adapter scores: {expected} vs {actual}"
            )
        unload_adapter(url, "repeat")
        shutil.rmtree(destination)
        destination = (
            Path("/tmp")
            / f"composed-probe-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
            / "adapter"
        )
        shutil.rmtree(destination, ignore_errors=True)
        distinct = compose(
            [r["hf"] for r in results[:32]], destination, inference_storage=True
        )
        load_adapter(url, "distinct", destination)
        actual_distinct = scores("distinct")
        if not np.isfinite(actual_distinct).all():
            raise RuntimeError("Distinct adapter composition produced nonfinite scores")
        record = {
            "repeated_adapter_rank": repeated["rank"],
            "distinct_adapter_rank": distinct["rank"],
            "repeated_adapter_logprob_max_abs_difference": float(
                np.max(np.abs(np.asarray(actual) - expected))
            ),
            "distinct_logprobs_finite": True,
            "compression": "none",
            "fp32_delta_relative_error": delta_relative_error,
            "repeated_adapter_bf16_logprob_tolerance": 0.1,
        }
        (output / "passed.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        unload_adapter(url, "distinct")
        shutil.rmtree(destination)
    finally:
        stop_server(process, log)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    probe(args.source, args.model, args.output)
