"""GPU check of installed offloaded Adam against FP32 CPU AdamW.

This checks optimizer arithmetic, copies and state reload, not model gradients or
distributed training. Run in the same container as the RL jobs on an allocated GPU.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path


def check(output):
    import torch

    torch.set_num_threads(4)
    root = Path(importlib.util.find_spec("megatron.core").origin).parent
    source = root / "optimizer/cpu_offloading/hybrid_optimizer.py"
    spec = importlib.util.spec_from_file_location("installed_hybrid_optimizer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = []
    for overlap in (False, True):
        generator = torch.Generator().manual_seed(42)
        initial = [torch.randn(shape, generator=generator).mul_(0.02).bfloat16()
                   for shape in ((256, 256), (1024, 1024), (2048, 2048))]
        params = [torch.nn.Parameter(p.cuda()) for p in initial]
        reference = [torch.nn.Parameter(p.float()) for p in initial]
        settings = {"lr": 2e-6, "betas": (0.9, 0.98), "eps": 1e-8, "weight_decay": 0, "fused": True}

        def make_optimizer(parameters, overlap=overlap, settings=settings):
            return module.HybridDeviceOptimizer(
                parameters, offload_fraction=1.0,
                cpu_optimizer_cls=torch.optim.AdamW, gpu_optimizer_cls=torch.optim.AdamW,
                param_update_in_fp32=True, overlap_cpu_optimizer_d2h_h2d=overlap,
                pin_cpu_grads=True, pin_cpu_params=True, **settings,
            )

        optimizer = make_optimizer(params)
        ordinary = torch.optim.AdamW(reference, **settings)
        restored = restored_params = None
        records = []
        for step in range(16):
            gradients = [torch.zeros_like(p) if step % 3 == 2 else torch.randn(p.shape, generator=generator) * 0.01
                         for p in reference]
            for param, gradient in zip(params, gradients, strict=True):
                param.decoupled_grad = gradient.cuda()
            optimizer.step()
            # Take a current-stream read immediately, before CPU reference work.
            immediate = [p.detach().clone() for p in params]
            for param, gradient in zip(reference, gradients, strict=True):
                param.grad = gradient
            ordinary.step()
            if restored is not None:
                for param, gradient in zip(restored_params, gradients, strict=True):
                    param.decoupled_grad = gradient.cuda()
                restored.step()
            torch.cuda.synchronize()
            master_error = max(float((optimizer.param_to_inner_param[p] - r).abs().max())
                               for p, r in zip(params, reference, strict=True))
            synced_mismatch = sum(int((p != r.detach().bfloat16().cuda()).sum())
                                  for p, r in zip(params, reference, strict=True))
            immediate_mismatch = sum(int((p != r.detach().bfloat16().cuda()).sum())
                                     for p, r in zip(immediate, reference, strict=True))
            resumed_error = (max(float((restored.param_to_inner_param[p] - r).abs().max())
                                 for p, r in zip(restored_params, reference, strict=True))
                             if restored is not None else None)
            records.append({"step": step, "zero_gradient": step % 3 == 2,
                            "master_max_error": master_error, "synced_bf16_mismatches": synced_mismatch,
                            "immediate_bf16_mismatches": immediate_mismatch,
                            "restored_master_max_error": resumed_error})
            if step == 7:
                state = copy.deepcopy(optimizer.state_dict())
                restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
                restored = make_optimizer(restored_params)
                restored.load_state_dict(state)
            if master_error > 2e-7 or synced_mismatch or (resumed_error is not None and resumed_error > 2e-7):
                raise RuntimeError(f"Offloaded Adam arithmetic/state mismatch: {records[-1]}")
        results.append({"overlap": overlap, "records": records,
                        "immediate_read_mismatch_steps": sum(r["immediate_bf16_mismatches"] > 0 for r in records),
                        "master_dtypes": sorted({str(optimizer.param_to_inner_param[p].dtype) for p in params}),
                        "state_dtypes": sorted({str(v.dtype) for p in params for k, v in optimizer.state[p].items()
                                                 if k in ("exp_avg", "exp_avg_sq")}),
                        "adam_steps": sorted({float(optimizer.state[p]["step"]) for p in params})})
        del optimizer, restored, ordinary, params, restored_params, reference, immediate
        torch.cuda.empty_cache()
    value = {"passed": True, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
             "optimizer_source": str(source), "optimizer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "results": results}
    output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    check(parser.parse_args().output)
