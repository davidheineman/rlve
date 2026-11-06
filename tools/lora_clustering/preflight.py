"""Check exact Adam recovery and inference with a composed adapter before RL."""

import shutil

import numpy as np
import torch
from async_lora.preflight import check_student
from async_lora.storage import export, restore_reference, save_bundle, uri_directory
from async_lora.worker import Worker
from tinker import types

from async_branching.io import read_json, write_json
from lora_clustering.merge import combine_adapters


async def check_backend(experiment, plan, model_path, base_url):
    marker = experiment / "preflight/clustering-passed.json"
    if marker.exists():
        return
    torch.set_num_threads(4)
    # A diagnostic slot is independent of the real specialists. Its updates
    # are discarded and excluded from all teacher rollout counts and W&B curves.
    probe = Worker(experiment, "teacher", len(plan["teachers"]), model_path, base_url)
    probe.plan["student_config"] = probe.plan["teacher_config"]
    await probe.initialize()
    probe.sampler, probe.sampler_uri = await export(
        probe.training, probe.service, probe.root, "diagnostic-base"
    )
    directory = experiment / "preflight/base"
    shutil.rmtree(directory, ignore_errors=True)
    base = await save_bundle(
        probe.training, probe.sampler_uri, probe.root, directory, plan
    )
    await check_student(probe, base)
    # Also pass an actual QR/SVD-composed PEFT artifact through the production
    # sampling loader, and check its token log probabilities are finite.
    adapter = experiment / "preflight/composed-adapter"
    shutil.rmtree(adapter, ignore_errors=True)
    diagnostics = combine_adapters(
        [base["hf"], base["hf"]],
        adapter,
        rank=plan["lora_rank"],
        alpha=plan["lora_alpha"],
    )
    uri = "tinker://clustering-preflight/sampler_weights/composed"
    local = uri_directory(probe.root, uri)
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(local, ignore_errors=True)
    shutil.copytree(adapter, local)
    sampler = await probe.service.create_sampling_client_async(model_path=uri)
    tokens = probe.tokenizer.encode(
        "The sum of 1 and 1 is 2.", add_special_tokens=False
    )
    reference = await probe.sampler.compute_logprobs_async(
        types.ModelInput.from_ints(tokens)
    )
    actual = await sampler.compute_logprobs_async(types.ModelInput.from_ints(tokens))
    if not np.isfinite(actual[1:]).all() or not np.allclose(
        actual[1:], reference[1:], atol=0.01, rtol=0.01
    ):
        raise RuntimeError(
            "Composed zero-delta adapter changed the base model's token scores"
        )
    imported_ok = False
    if plan.get("reuse"):
        name = plan["reuse"]["environments"][0]
        teacher = next(t for t in plan["teachers"] if t["environment"] == name)
        imported = read_json(
            experiment / "teacher" / teacher["source_label"] / "completed.json"
        )
        imported_uri = restore_reference(probe.root, imported["hf"], sampling=True)
        imported_sampler = await probe.service.create_sampling_client_async(
            model_path=imported_uri
        )
        scores = await imported_sampler.compute_logprobs_async(
            types.ModelInput.from_ints(tokens)
        )
        if not np.isfinite(scores[1:]).all():
            raise RuntimeError("Imported specialist token scores are nonfinite")
        merged = experiment / "preflight/imported-composed-adapter"
        shutil.rmtree(merged, ignore_errors=True)
        combine_adapters(
            [imported["hf"], base["hf"]],
            merged,
            rank=plan["lora_rank"],
            alpha=plan["lora_alpha"],
        )
        merged_uri = "tinker://clustering-preflight/sampler_weights/imported-composed"
        local = uri_directory(probe.root, merged_uri)
        shutil.rmtree(local, ignore_errors=True)
        shutil.copytree(merged, local)
        sampler = await probe.service.create_sampling_client_async(
            model_path=merged_uri
        )
        scores = await sampler.compute_logprobs_async(
            types.ModelInput.from_ints(tokens)
        )
        if not np.isfinite(scores[1:]).all():
            raise RuntimeError(
                "Imported/new composed adapter token scores are nonfinite"
            )
        imported_ok = True
    write_json(
        marker,
        {
            "exact_optimizer_resume": True,
            "composed_adapter_inference": True,
            "imported_adapter_and_mixed_composition_inference": imported_ok,
            "composition_diagnostics": diagnostics,
            "diagnostic_optimizer_updates_excluded_from_teacher_counts": True,
        },
    )
