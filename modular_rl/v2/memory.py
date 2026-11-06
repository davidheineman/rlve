"""Opt-in cache release between completed forward/backward microbatches.

The old memory-saver allocator can abort on a new allocation before PyTorch's
normal OOM cache-retry path runs. Large accumulation batches need a bounded cache.
"""

import os


def release_transient_cuda_cache(microbatch_index, stage="train"):
    interval = int(os.environ.get("MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL", "0"))
    if interval < 0:
        raise ValueError("CUDA cache interval must be nonnegative")
    if not interval or microbatch_index % interval:
        return
    import torch

    # Complete outstanding stream use before releasing unused cached segments.
    # Live model, gradient, optimizer, and rollout tensors remain allocated.
    report = os.environ.get("MODULAR_RL_CUDA_CACHE_LOG", "0") == "1" and microbatch_index % 16 == 0
    before = torch.cuda.memory_reserved() if report else None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if report:
        print(f"CUDA cache release stage={stage} microbatch={microbatch_index} "
              f"allocated={torch.cuda.memory_allocated()} reserved_before={before} "
              f"reserved_after={torch.cuda.memory_reserved()}", flush=True)


def instrument_model(source):
    """Add the guard only to a frozen native trainer, leaving the worktree intact."""
    if "cache_guard_microbatch" in source:
        raise ValueError("Native trainer already has a cache guard")
    for name in ("forward_only", "train_one_step"):
        marker = f"def {name}("
        before, separator, remainder = source.partition(marker)
        if not separator:
            raise ValueError(f"Native {name} changed; inspect cache guard placement")
        boundary = remainder.find("\ndef ")
        function = remainder if boundary < 0 else remainder[:boundary]
        after = "" if boundary < 0 else remainder[boundary:]
        definition = "    def forward_step(data_iterator, model: GPTModel):\n"
        batch_marker = "        # Get the batch.\n"
        if function.count(definition) != 1 or function.count(batch_marker) != 1:
            raise ValueError(f"Native {name} boundary changed; inspect cache guard placement")
        function = function.replace(definition, "    cache_guard_microbatch = 0\n\n" + definition, 1)
        guard = (
            "        nonlocal cache_guard_microbatch\n"
            "        from modular_rl.v2.memory import release_transient_cuda_cache\n"
            f"        release_transient_cuda_cache(cache_guard_microbatch, {name!r})\n"
            "        cache_guard_microbatch += 1\n\n"
        )
        source = before + separator + function.replace(batch_marker, guard + batch_marker, 1) + after
    return source
