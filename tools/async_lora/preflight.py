"""Short GPU checks before production rollouts; restore the untouched base adapter afterward."""

import shutil

import numpy as np
from tinker import types

from async_branching.io import write_json
from async_lora.storage import adapter_identity, export, save_bundle, uri_directory


async def check_student(worker, base):
    """Exercise native OPD, SGLang scoring and exact Adam checkpoint continuation."""
    root, training, service = worker.root, worker.training, worker.service
    directory = worker.experiment / "preflight"
    directory.mkdir(exist_ok=True)
    await worker.load(base["resume"], optimizer=True)
    sampler, uri = await export(training, service, root, "preflight-base")
    tokens = worker.tokenizer.encode("The sum of 1 and 1 is 2.", add_special_tokens=False)
    scores = await sampler.compute_logprobs_async(types.ModelInput.from_ints(tokens))
    if len(scores) != len(tokens) or any(x is None or not np.isfinite(x) for x in scores[1:]):
        raise RuntimeError("GPU preflight: teacher token scores are missing or misaligned")
    datums = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tokens[1:],
                "logprobs": scores[1:],
                "advantages": [0.001 / (8 * (len(tokens) - 1))] * (len(tokens) - 1),
            },
        )
        for _ in range(8)
    ]
    replay = worker.plan.get("opd_replay")
    if replay:
        for datum in datums[-3:]:
            datum.loss_fn_inputs["advantages"] = types.TensorData.from_torch(
                -0.5 * datum.loss_fn_inputs["advantages"].to_torch()
            )
    config = worker.plan["student_config"]
    adam = types.AdamParams(
        learning_rate=config["lr"],
        beta1=config["adam-beta1"],
        beta2=config["adam-beta2"],
        eps=worker.plan["optimizer_eps"],
        weight_decay=config["weight-decay"],
        grad_clip_norm=worker.plan["grad_clip_norm"],
    )

    async def update():
        fresh = datums[:-3] if replay else datums
        await (await training.forward_backward_async(fresh, "importance_sampling"))
        if replay:
            epsilon = replay["clip_epsilon"]
            await (
                await training.forward_backward_async(
                    datums[-3:], "ppo", {"clip_low_threshold": 1 - epsilon, "clip_high_threshold": 1 + epsilon}
                )
            )
        result = await (await training.optim_step_async(adam))
        if not np.isfinite(result.metrics.get("grad_norm", float("nan"))) or result.metrics["grad_norm"] <= 0:
            raise RuntimeError("GPU preflight: native OPD did not produce a finite nonzero gradient")
        return result.metrics

    first = await update()
    first_sampler, first_uri = await export(training, service, root, "preflight-first")
    if worker.plan.get("elastic_generation"):
        probe_path = worker.experiment / "flex/probe-adapter"
        shutil.rmtree(probe_path, ignore_errors=True)
        shutil.copytree(uri_directory(root, first_uri), probe_path)
        probe_scores = await first_sampler.compute_logprobs_async(types.ModelInput.from_ints(tokens))
        name = "elastic-admission-probe"
        write_json(
            worker.experiment / "flex/probe.json",
            {
                "request": {
                    "input_ids": tokens,
                    "lora_path": name,
                    "lora_backfill_paths": {name: str(probe_path)},
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
                "logprobs": list(probe_scores),
            },
        )
    shutil.rmtree(directory / "first", ignore_errors=True)
    checkpoint = await save_bundle(training, first_uri, root, directory / "first", worker.plan)
    second = await update()
    _, expected_uri = await export(training, service, root, "preflight-expected", previous=first_uri)
    await worker.load(checkpoint["resume"], optimizer=True)
    resumed = await update()
    _, actual_uri = await export(training, service, root, "preflight-resumed")
    parity = adapter_identity(uri_directory(root, expected_uri), uri_directory(root, actual_uri))
    await worker.load(base["resume"], optimizer=True)
    worker.sampler, worker.sampler_uri = await export(training, service, root, "preflight-restored")
    restored = adapter_identity(base["hf"], uri_directory(root, worker.sampler_uri))
    write_json(
        directory / "passed.json",
        {
            "native_opd_first_optimizer": first,
            "second_optimizer": second,
            "resumed_optimizer": resumed,
            "optimizer_resume_parity": parity,
            "base_restored": restored,
            "teacher_logprob_alignment": True,
            "diagnostic_updates_excluded_from_training_rollouts": True,
            "replay_ppo_backward_exercised": bool(replay),
            "obsolete_sampler_cleanup_exercised": True,
        },
    )
    shutil.rmtree(directory / "first")
    for path in (uri, first_uri, expected_uri, actual_uri):
        shutil.rmtree(uri_directory(root, path), ignore_errors=True)
