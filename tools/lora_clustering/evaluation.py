"""Seeded, restartable per-adapter/per-environment empirical transfer measurements."""

import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path

import tinker
from async_lora.storage import restore_reference
from lora_teachers.plan import sha256
from lora_teachers.sampling import prompt_tokens, sample_group
from transformers import AutoTokenizer

from async_branching.io import read_json, write_json


class SeededSampler:
    def __init__(self, sampler, seed):
        self.sampler, self.seed, self.calls = sampler, seed, 0

    async def sample_async(self, **kwargs):
        seed = self.seed + self.calls * 1009
        self.calls += 1
        kwargs["sampling_params"] = kwargs["sampling_params"].model_copy(
            update={"seed": seed}
        )
        return await self.sampler.sample_async(**kwargs)


class Evaluator:
    def __init__(self, experiment, plan, model_path, base_url, root):
        self.experiment, self.plan, self.root = experiment, plan, Path(root)
        self.directory = experiment / "clustering"
        self.directory.mkdir(exist_ok=True)
        self.service = tinker.ServiceClient(
            base_url=base_url, api_key=f"tml-async-{experiment.name}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.rows = {}
        for split in ("selection", "confirmation"):
            path = experiment / "source/rlve" / plan["clustering"][split]
            if sha256(path) != plan["clustering"][f"{split}_sha256"]:
                raise ValueError(f"Changed {split} inputs")
            grouped = {name: [] for name in plan["environments"]}
            for row in read_json(path):
                meta = row["metadata"]
                meta = json.loads(meta) if isinstance(meta, str) else meta
                grouped[meta["environment"]].append(dict(row, metadata=meta))
            self.rows[split] = grouped
        self.semaphore = asyncio.Semaphore(32)

    async def sampler(self, label, bundle=None, adapter=None):
        if bundle:
            uri = restore_reference(self.root, bundle["hf"], sampling=True)
        elif adapter:
            digest = sha256(Path(adapter) / "adapter_model.safetensors")[:24]
            uri = f"tinker://clustering-{digest}/sampler_weights/final"
            destination = self.root / f"clustering-{digest}/sampler_weights/final"
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name("final.incomplete")
                shutil.rmtree(temporary, ignore_errors=True)
                shutil.copytree(adapter, temporary)
                temporary.replace(destination)
        else:
            uri = None
        return await self.service.create_sampling_client_async(
            model_path=uri, base_model=self.plan["base_model"] if uri is None else None
        )

    def cached(self, split, label, environment):
        path = self.directory / "evaluations" / split / label / f"{environment}.json"
        if path.exists():
            result = read_json(path)
            if result["eval_sha256"] != self.plan["clustering"][f"{split}_sha256"]:
                raise ValueError("Evaluation cache belongs to different inputs")
            return result
        return None

    async def cell(self, split, label, environment, sampler):
        path = self.directory / "evaluations" / split / label / f"{environment}.json"
        result = self.cached(split, label, environment)
        if result is not None:
            return result
        settings = dict(self.plan["teacher_config"])
        settings["n-samples-per-eval-prompt"] = self.plan["clustering"][
            f"{split}_samples_per_prompt"
        ]

        async def score(row):
            async with self.semaphore:
                prompt = row["user_prompt"]
                prompt_id = hashlib.sha256(prompt.encode()).hexdigest()
                seed = (
                    self.plan["clustering"]["seed"] + int(prompt_id[:8], 16)
                ) % 2**31
                completions = await sample_group(
                    SeededSampler(sampler, seed),
                    self.tokenizer,
                    prompt_tokens(self.tokenizer, prompt),
                    environment,
                    row["metadata"]["config"],
                    settings,
                    evaluation=True,
                )
                return {
                    "prompt_id": prompt_id,
                    "correct": sum(c.accuracy for c in completions),
                    "total": len(completions),
                    "forced": sum(c.forced for c in completions),
                    "truncated": sum(c.truncated for c in completions),
                }

        began = time.time()
        observations = await asyncio.gather(
            *(score(r) for r in self.rows[split][environment])
        )
        total = sum(x["total"] for x in observations)
        result = {
            "adapter": label,
            "environment": environment,
            "split": split,
            "eval_sha256": self.plan["clustering"][f"{split}_sha256"],
            "accuracy": sum(x["correct"] for x in observations) / total,
            "samples": total,
            "seconds": time.time() - began,
            "truncated_fraction": sum(x["truncated"] for x in observations) / total,
            "answer_forcing_fraction": sum(x["forced"] for x in observations) / total,
            "observations": observations,
        }
        write_json(path, result)
        print(
            f"CLUSTER_EVAL {split} {label} {environment} accuracy={result['accuracy']:.4f}",
            flush=True,
        )
        return result

    async def vector(self, split, label, *, bundle=None, adapter=None):
        cached = [self.cached(split, label, env) for env in self.plan["environments"]]
        if all(row is not None for row in cached):
            return [row["accuracy"] for row in cached]
        sampler = await self.sampler(label, bundle, adapter)
        results = await asyncio.gather(
            *(
                self.cell(split, label, env, sampler)
                for env in self.plan["environments"]
            )
        )
        return [r["accuracy"] for r in results]
