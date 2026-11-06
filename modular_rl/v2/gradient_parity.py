"""Opt-in full-model gradient comparison against independent HF teacher forcing.

Capture is inserted only into a dedicated frozen diagnostic. The
comparison runs after its native workers exit, never alongside a live trainer.
Native gradients are captured before clipping and optimizer.step().
"""

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import MethodType


@contextmanager
def reference_precision(model, mode):
    """Audit BF16 rounding boundaries without changing the Qwen equations.

    Stock HF rounds normalized activations before gamma multiplication and rounds
    RoPE cos/sin to BF16. Fused native kernels can retain these intermediates in
    FP32. Keep the modes explicit and restore every patched method afterward.
    """
    import torch
    from transformers.models.qwen3 import modeling_qwen3 as qwen

    if mode not in {"stock", "norm", "rope", "norm-rope", "fused"}:
        raise ValueError("Unknown reference precision mode")
    saved_norms = []
    saved_mlps = []
    rotary = model.model.rotary_emb
    original_rotary, original_apply = rotary.forward, qwen.apply_rotary_pos_emb

    def norm_forward(module, hidden_states):
        dtype = hidden_states.dtype
        values = hidden_states.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + module.variance_epsilon)
        return (normalized * module.weight.float()).to(dtype)

    def rotary_forward(module, values, position_ids):
        return original_rotary(values.float(), position_ids)

    def apply_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        qr, kr = original_apply(q.float(), k.float(), cos.float(), sin.float(), unsqueeze_dim=unsqueeze_dim)
        return qr.to(q.dtype), kr.to(k.dtype)

    def mlp_forward(module, hidden_states):
        gate, up = module.gate_proj(hidden_states), module.up_proj(hidden_states)
        activated = (module.act_fn(gate.float()) * up.float()).to(gate.dtype)
        return module.down_proj(activated)

    try:
        if mode in {"norm", "norm-rope", "fused"}:
            for module in model.modules():
                if isinstance(module, qwen.Qwen3RMSNorm):
                    saved_norms.append((module, module.forward))
                    module.forward = MethodType(norm_forward, module)
        if mode in {"rope", "norm-rope", "fused"}:
            rotary.forward = MethodType(rotary_forward, rotary)
            qwen.apply_rotary_pos_emb = apply_rotary
        if mode == "fused":
            if model.config.hidden_act != "silu":
                raise ValueError("The fused reference expects Qwen3 SwiGLU")
            for module in model.modules():
                if isinstance(module, qwen.Qwen3MLP):
                    saved_mlps.append((module, module.forward))
                    module.forward = MethodType(mlp_forward, module)
        yield
    finally:
        for module, original in saved_norms:
            module.forward = original
        for module, original in saved_mlps:
            module.forward = original
        rotary.forward, qwen.apply_rotary_pos_emb = original_rotary, original_apply


def instrument(source):
    marker = "    valid_step = True\n"
    if source.count(marker) != 1:
        raise ValueError("Inspect native gradient capture placement")
    return source.replace(marker, (
        "    from modular_rl.v2.gradient_parity import capture\n"
        "    capture(args, rollout_id, step_id, model, optimizer)\n\n" + marker
    ))


def capture(args, rollout_id, step_id, model, optimizer):
    import torch

    from slime.backends.megatron_utils.update_weight_utils import (
        convert_to_hf,
        named_parameters,
    )

    if torch.distributed.get_world_size() != 1:
        raise ValueError("Gradient parity requires one GPU")
    if rollout_id != 0 or step_id != 0:
        return  # Preserve the initial capture while later diagnostic steps finish.
    if args.entropy_coef != 0 or args.use_kl_loss or args.advantage_estimator != "grpo":
        raise ValueError("Compare the initial on-policy fixed-token GRPO gradient without regularizers")
    if float(optimizer.get_loss_scale()) != 1:
        raise ValueError("Expected unscaled BF16 gradients")
    weights, gradients = {}, {}
    for name, param in named_parameters(args, model):
        grad = getattr(param, "main_grad", None)
        if grad is None or grad.shape != param.shape:
            raise ValueError(f"Missing full native gradient: {name}")
        for converted, value in convert_to_hf(args, "qwen3", name, param.detach()):
            weights[converted] = value.detach().cpu().clone()
        for converted, value in convert_to_hf(args, "qwen3", name, grad.detach()):
            gradients[converted] = value.detach().float().cpu().clone()
    norm_sq = sum(float(g.square().sum()) for g in gradients.values())
    if norm_sq <= 1e-12:
        raise ValueError("The parity batch needs nonzero advantages and gradients")
    value = {"weights": weights, "gradients": gradients,
             "metadata": {"global_batch_size": args.global_batch_size,
                          "sampled_token_cap": args.rollout_max_response_len,
                          "tis_clip_low": args.tis_clip_low, "tis_clip_high": args.tis_clip,
                          "use_tis": args.use_tis, "loss_type": args.loss_type,
                          "custom_loss": args.custom_loss_function_path,
                          "native_unclipped_gradient_norm": norm_sq ** 0.5}}
    path = Path(os.environ["MODULAR_RL_GRADIENT_PARITY_CAPTURE"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    print("GRADIENT_PARITY_CAPTURE " + json.dumps(value["metadata"]), flush=True)


def compare(capture_path, batch_path, hf_checkpoint, output, precision="stock", forward_audit=False,
            reference_dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(6)
    if reference_dtype not in {"bfloat16", "float32"}:
        raise ValueError("Choose an explicit BF16 or FP32 reference")
    if reference_dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    native = torch.load(capture_path, map_location="cpu", weights_only=False)
    record = torch.load(batch_path, map_location="cpu", weights_only=False)
    if record["rollout_id"] != 0 or record["rank"] != 0:
        raise ValueError("Use the captured initial single-GPU batch")
    batch, meta = record["rollout_data"], native["metadata"]
    if meta["custom_loss"] != "modular_rl.v2.losses.fixed_token_policy_loss":
        raise ValueError("Parity formula expects the production fixed-token objective")
    model = AutoModelForCausalLM.from_pretrained(
        hf_checkpoint, torch_dtype=getattr(torch, reference_dtype), attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    missing, unexpected = model.load_state_dict(native["weights"], strict=False)
    allowed_missing = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(f"Native/HF weight mapping differs: {missing}, {unexpected}")
    model.tie_weights()
    parameters = dict(model.named_parameters())
    if set(parameters) != set(native["gradients"]):
        raise ValueError(f"Native/HF parameter sets differ: {set(parameters) ^ set(native['gradients'])}")
    accumulated = {name: torch.zeros_like(g) for name, g in native["gradients"].items()}
    samples, absolute_logprob_error, token_count = [], 0.0, 0
    count = len(batch["tokens"])
    if count != meta["global_batch_size"]:
        raise ValueError("Captured global batch differs from the actual response count")
    if model.is_gradient_checkpointing:
        raise ValueError("Precision audit requires eager backward without forward recomputation")
    if forward_audit:
        audit = []
        for mode in ("stock", "norm", "rope", "norm-rope", "fused"):
            total_error, total_tokens, details = 0.0, 0, []
            for index in range(0, count, 8):
                tokens = torch.as_tensor(batch["tokens"][index], dtype=torch.long, device="cuda").reshape(1, -1)
                length = int(batch["response_lengths"][index])
                start = tokens.shape[1] - length
                with torch.no_grad(), reference_precision(model, mode):
                    logits = model(input_ids=tokens, use_cache=False).logits[0, start - 1:-1].float()
                    logp = logits.gather(-1, tokens[0, start:].unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
                mask = torch.as_tensor(batch["loss_masks"][index], dtype=torch.bool, device="cuda")
                error = (logp - batch["log_probs"][index].float().cuda())[mask].abs()
                total_error += float(error.sum())
                total_tokens += error.numel()
                details.append({"sample": index, "mean_abs_error": float(error.mean()),
                                "max_abs_error": float(error.max())})
                del logits, logp, tokens
            audit.append({"precision": mode, "mean_abs_forward_logprob_error": total_error / total_tokens,
                          "samples": details})
        Path(output).with_name("forward-precision-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print("FORWARD_PRECISION_AUDIT " + json.dumps(audit), flush=True)
    for index in range(count):
        tokens = torch.as_tensor(batch["tokens"][index], dtype=torch.long, device="cuda").reshape(1, -1)
        length = int(batch["response_lengths"][index])
        start = tokens.shape[1] - length
        old = batch["log_probs"][index].float().cuda()
        advantage = batch["advantages"][index].float().cuda()
        mask = torch.as_tensor(batch["loss_masks"][index], dtype=torch.float32, device="cuda")
        if any(t.numel() != length for t in (old, advantage, mask)):
            raise ValueError("Captured token, advantage and mask lengths disagree")
        model.zero_grad(set_to_none=True)
        with reference_precision(model, precision):
            logits = model(input_ids=tokens, use_cache=False).logits[0, start - 1:-1].float()
        target = tokens[0, start:]
        logp = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
        # This independent REINFORCE expression is the native PPO derivative at
        # its initial ratio=1, with clipping inactive. Compare actual forward
        # probabilities separately, rather than hiding BF16 differences in ratios.
        tis = torch.ones_like(old)
        if meta["use_tis"]:
            sampled = batch["rollout_log_probs"][index].float().cuda()
            tis = (old - sampled).exp().clamp(meta["tis_clip_low"], meta["tis_clip_high"])
        loss = -(logp * advantage * tis * mask).sum() / (count * meta["sampled_token_cap"])
        loss.backward()
        valid = mask.bool()
        error = (logp.detach() - old)[valid].abs()
        absolute_logprob_error += float(error.sum())
        token_count += error.numel()
        samples.append({"sample": index, "sampled_tokens": int(valid.sum()),
                        "mean_abs_forward_logprob_error": float(error.mean()),
                        "max_abs_forward_logprob_error": float(error.max())})
        for name, param in parameters.items():
            if param.grad is None:
                raise ValueError(f"Missing HF gradient: {name}")
            accumulated[name].add_(param.grad.detach().float().cpu())
        del loss, logits, logp, tokens
        print(f"HF_GRADIENT_PARITY sample {index + 1}/{count}", flush=True)
    rows, totals = [], {"native_squared": 0.0, "hf_squared": 0.0, "dot": 0.0, "error_squared": 0.0}
    for name, reference in native["gradients"].items():
        actual = accumulated[name]
        n2, h2 = float(reference.square().sum()), float(actual.square().sum())
        dot, e2 = float((reference * actual).sum()), float((reference - actual).square().sum())
        rows.append({"parameter": name, "elements": reference.numel(),
                     "native_norm": n2 ** 0.5, "hf_norm": h2 ** 0.5,
                     "cosine": dot / max((n2 * h2) ** 0.5, 1e-30),
                     "relative_l2_error": (e2 / max(n2, 1e-30)) ** 0.5})
        for key, value in zip(totals, (n2, h2, dot, e2), strict=True):
            totals[key] += value
    summary = {"native_norm": totals["native_squared"] ** 0.5,
               "hf_norm": totals["hf_squared"] ** 0.5,
               "cosine": totals["dot"] / (totals["native_squared"] * totals["hf_squared"]) ** 0.5,
               "relative_l2_error": (totals["error_squared"] / totals["native_squared"]) ** 0.5,
               "mean_abs_forward_logprob_error": absolute_logprob_error / token_count,
               "parameter_tensors": len(rows), "parameter_elements": sum(r["elements"] for r in rows),
               "responses": count, "sampled_tokens": token_count}
    summary.update(reference_dtype=reference_dtype, device=torch.cuda.get_device_name(),
                   peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 30,
                   peak_cuda_reserved_GiB=torch.cuda.max_memory_reserved() / 2 ** 30,
                   allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    # BF16 TE vs PyTorch SDPA/wgrad differ numerically. These thresholds are
    # specified before running the comparison; record every tensor regardless.
    summary["passed"] = (summary["cosine"] >= 0.99 and summary["relative_l2_error"] <= 0.1
                          and summary["mean_abs_forward_logprob_error"] <= 0.02)
    result = {"summary": summary, "reference_precision": precision, "parameters": rows, "samples": samples,
              "capture": str(capture_path), "batch": str(batch_path), "hf_checkpoint": str(hf_checkpoint),
              "limitations": "One initial BF16 batch, TP/DP/PP=1, fixed-token loss, unclipped gradients; not a complete trajectory parity proof."}
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not summary["passed"]:
        raise ValueError("Full-model gradient comparison exceeded predefined tolerances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--hf-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["stock", "norm", "rope", "norm-rope", "fused"], default="stock")
    parser.add_argument("--forward-audit", action="store_true")
    parser.add_argument("--reference-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    compare(args.capture, args.batch, args.hf_checkpoint, args.output, args.precision, args.forward_audit,
            args.reference_dtype)
