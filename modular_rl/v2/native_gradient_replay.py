"""Replay a captured native batch without sampling or taking optimizer updates.

The FP32 mode uses the native model with unfused attention and compares it with
an independent HF FP32 model. This separates architecture/objective agreement
from the production BF16 kernel comparison; it does not change training.
"""

import argparse
import json
import tempfile
from pathlib import Path


def pack_native_weight(name, converted, weights, args):
    """Invert the Qwen grouped-QKV / gate-up storage permutations explicitly."""
    import torch

    if name.endswith("self_attention.linear_qkv.weight"):
        values = [weights[key] for key, _ in converted]
        per_group = args.num_attention_heads // args.num_query_groups
        shapes = [(args.num_query_groups, heads, args.kv_channels, args.hidden_size)
                  for heads in (per_group, 1, 1)]
        return torch.cat([value.reshape(shape) for value, shape in zip(values, shapes)], dim=1).flatten(0, 2)
    if name.endswith("mlp.linear_fc1.weight"):
        return torch.cat([weights[key] for key, _ in converted], dim=0)
    if len(converted) != 1:
        raise ValueError(f"Unsupported native mapping: {name}")
    return weights[converted[0][0]]


def replay(capture_path, batch_path, native_common, output_capture, dtype="float32", packing="single"):
    import torch
    from megatron.core import mpu, tensor_parallel
    from megatron.core.transformer.enums import AttnBackend
    from megatron.training.global_vars import set_args

    from slime.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf
    from slime.backends.megatron_utils.model_provider import get_model_provider_func

    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if packing == "production" and dtype != "bfloat16":
        raise ValueError("Production THD packing requires the BF16 flash-attention reference")
    native = torch.load(capture_path, map_location="cpu", weights_only=False)
    record = torch.load(batch_path, map_location="cpu", weights_only=False)
    args = torch.load(native_common, map_location="cpu", weights_only=False)["args"]
    if record["rollout_id"] != 0 or record["rank"] != 0:
        raise ValueError("Replay the captured initial single-rank batch")
    args.params_dtype = getattr(torch, dtype)
    args.bf16, args.fp16 = dtype == "bfloat16", False
    args.gradient_accumulation_fusion = False
    args.recompute_granularity = args.recompute_method = args.recompute_num_layers = None
    if dtype == "float32":
        args.attention_backend = AttnBackend.unfused
        args.apply_rope_fusion = args.masked_softmax_fusion = False
    args.rank, args.world_size = 0, 1
    set_args(args)
    with tempfile.TemporaryDirectory(prefix="native-gradient-replay-") as temporary:
        torch.distributed.init_process_group("nccl", init_method=f"file://{temporary}/init", rank=0, world_size=1)
        mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        tensor_parallel.model_parallel_cuda_manual_seed(1234)
        model = get_model_provider_func(args)().to(device="cuda", dtype=args.params_dtype).train()
        parameters = dict(model.named_parameters())
        seen = set()
        for name, param in parameters.items():
            full_name = "module.module." + name
            converted = convert_qwen2_to_hf(args, full_name, param)
            value = pack_native_weight(full_name, converted, native["weights"], args)
            if value.shape != param.shape:
                raise ValueError(f"Native shape mismatch: {name}")
            with torch.no_grad():
                param.copy_(value)
            for key, actual in convert_qwen2_to_hf(args, full_name, param):
                torch.testing.assert_close(actual.cpu(), native["weights"][key].to(args.params_dtype), rtol=0, atol=0)
                seen.add(key)
        if seen != set(native["weights"]):
            raise ValueError(f"Unused or missing captured weights: {seen ^ set(native['weights'])}")
        accumulated = {name: torch.zeros_like(p, device="cpu", dtype=torch.float32)
                       for name, p in parameters.items()}
        batch, meta = record["rollout_data"], native["metadata"]
        count = len(batch["tokens"])
        if count != meta["global_batch_size"]:
            raise ValueError("Replay response count differs from native capture")
        errors, processed = [], []
        if packing == "production":
            from slime.backends.megatron_utils.data import get_batch
            from slime.backends.utils.data import get_data_iterator

            batch["tokens"] = [torch.as_tensor(t, dtype=torch.long, device="cuda") for t in batch["tokens"]]
            iterators, microbatches = get_data_iterator(args, [model], batch)
            if len(microbatches) != 1:
                raise ValueError("Expected exactly one captured optimizer batch")
            groups = iterators[0].micro_batch_indices
            if groups is None:
                groups = [list(range(i, i + args.micro_batch_size))
                          for i in range(0, count, args.micro_batch_size)]
        else:
            groups = [[i] for i in range(count)]
        for micro_index, indices in enumerate(groups):
            model.zero_grad(set_to_none=True)
            if packing == "production":
                micro = get_batch(iterators[0], ["tokens", "total_lengths", "response_lengths"])
                output = model(input_ids=micro["tokens"], position_ids=None, attention_mask=None,
                               packed_seq_params=micro["packed_seq_params"])
            else:
                tokens = torch.as_tensor(batch["tokens"][indices[0]], dtype=torch.long, device="cuda").reshape(1, -1)
                output = model(input_ids=tokens, position_ids=None, attention_mask=None)
            loss, offset = 0, 0
            for index in indices:
                tokens = torch.as_tensor(batch["tokens"][index], dtype=torch.long, device="cuda")
                total, length = tokens.numel(), int(batch["response_lengths"][index])
                start = total - length
                logits = output[0, offset + start - 1:offset + total - 1].float()
                if logits.shape != (length, args.padded_vocab_size):
                    raise ValueError(f"Unexpected native logit shape {logits.shape}")
                old = batch["log_probs"][index].float().cuda()
                advantage = batch["advantages"][index].float().cuda()
                mask = torch.as_tensor(batch["loss_masks"][index], dtype=torch.float32, device="cuda")
                target = tokens[start:]
                logp = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
                tis = torch.ones_like(old)
                if meta["use_tis"]:
                    sampled = batch["rollout_log_probs"][index].float().cuda()
                    tis = (old - sampled).exp().clamp(meta["tis_clip_low"], meta["tis_clip_high"])
                loss = loss - (logp * advantage * tis * mask).sum() / (count * meta["sampled_token_cap"])
                error = (logp.detach() - old)[mask.bool()].abs()
                errors.append({"sample": index, "tokens": error.numel(), "absolute_error_sum": float(error.sum())})
                processed.append(index)
                offset += total
            loss.backward()
            for name, param in parameters.items():
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise ValueError(f"Missing or nonfinite replay gradient: {name}")
                accumulated[name].add_(param.grad.detach().float().cpu())
            del loss, logits, logp, tokens, output
            print(f"NATIVE_GRADIENT_REPLAY {dtype} {packing} microbatch {micro_index + 1}/{len(groups)}", flush=True)
        if sorted(processed) != list(range(count)):
            raise ValueError("Packing duplicated or omitted a captured response")
        gradients = {}
        for name, value in accumulated.items():
            for key, grad in convert_qwen2_to_hf(args, "module.module." + name, value):
                gradients[key] = grad.contiguous()
        if set(gradients) != set(native["gradients"]):
            raise ValueError("Native replay gradient names differ from capture")
        summary = {"dtype": dtype, "responses": count, "sampling": False, "optimizer_updates": 0,
                   "packing": packing, "micro_batch_indices": groups,
                   "attention_backend": str(args.attention_backend), "recompute_granularity": None,
                   "gradient_accumulation_fusion": False, "allow_tf32": False,
                   "native_production_forward_mean_abs_logprob_error": (
                       sum(r["absolute_error_sum"] for r in errors) / sum(r["tokens"] for r in errors)),
                   "peak_cuda_allocated_GiB": torch.cuda.max_memory_allocated() / 2 ** 30,
                   "device": torch.cuda.get_device_name()}
        replay_meta = {**meta, "native_unclipped_gradient_norm": (
            sum(float(g.square().sum()) for g in gradients.values()) ** 0.5)}
        value = {"weights": native["weights"], "gradients": gradients, "metadata": replay_meta,
                 "replay": summary}
        torch.save(value, output_capture)
        Path(output_capture).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print("NATIVE_REPLAY_SUMMARY " + json.dumps(summary), flush=True)
        mpu.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--native-common", type=Path, required=True)
    parser.add_argument("--output-capture", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--packing", choices=["single", "production"], default="single")
    opts = parser.parse_args()
    replay(opts.capture, opts.batch, opts.native_common, opts.output_capture, opts.dtype, opts.packing)
