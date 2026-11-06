"""Launch independent pretrained-Base MATH/MBPP jobs and optional diagnostics."""

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from modular_rl.prepare import write_json
from slime.utils.checkpoint_state import atomic_write

from .ablate import freeze
from .base_models import BASE_MODELS, verify_initialization
from .data import stratum
from .inspect_rollouts import read_events


@dataclass(frozen=True)
class Cell:
    model: str
    cap: int
    variant: str
    lr: float = 2e-6
    responses: int = 8000
    passes: int = 1
    reference_kl: float = 0.0
    seed: int | None = None
    rollout_seed: int | None = None
    task: str = "math"
    answer_reserve: int = 256
    reward_variant: str = "binary"
    code_format: str = "chat"

    @property
    def name(self):
        task_suffix = "-code" if self.task == "code" else ""
        return f"{self.model}{task_suffix}-gen{self.cap}-{self.variant}"

    @property
    def hours(self):
        initial = (8 if self.cap == 2048 else 14) if "0.6B" in self.model else (18 if self.cap == 2048 else 30)
        return min(72, int(initial * (self.responses / 8000) * (1 + .25 * (self.passes - 1)) + 2))


def matrix():
    recipes = (("control", 2e-6, 8000, 1), ("lr4e-6", 4e-6, 8000, 1),
               ("lr1e-5", 1e-5, 8000, 1), ("rollouts16k", 2e-6, 16000, 1),
               ("passes2", 2e-6, 8000, 2), ("passes4", 2e-6, 8000, 4))
    return [Cell(model, cap, name, lr, responses, passes)
            for model in BASE_MODELS for cap in (2048, 4096)
            for name, lr, responses, passes in recipes]


def arguments(cell, *, preflight=False):
    if cell.model not in BASE_MODELS or type(cell.cap) is not int or cell.cap not in (2048, 4096):
        raise ValueError("Only the exact pretrained Base checkpoints and declared generation caps are allowed")
    if not isinstance(cell.variant, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cell.variant):
        raise ValueError("Use a simple variant name without path separators")
    if cell.task not in ("math", "code"):
        raise ValueError("Base sweep tasks must be math or code")
    if type(cell.answer_reserve) is not int or not 0 < cell.answer_reserve < cell.cap:
        raise ValueError("Answer reserve must fit inside the sampled-token cap")
    if cell.reward_variant not in ("binary", "partial") or (cell.reward_variant == "partial" and cell.task != "code"):
        raise ValueError("Partial-test rewards are only supported for code")
    if cell.code_format not in ("chat", "prefix") or (cell.code_format == "prefix" and cell.task != "code"):
        raise ValueError("A code prefix is only supported for code")
    if cell.code_format == "prefix" and cell.reward_variant != "binary":
        raise ValueError("Code-prefix diagnostics currently use strict binary rewards")
    if type(cell.lr) not in (int, float) or not math.isfinite(cell.lr) or cell.lr <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if (type(cell.reference_kl) not in (int, float) or not math.isfinite(cell.reference_kl)
            or cell.reference_kl < 0):
        raise ValueError("Reference KL coefficient must be nonnegative and finite")
    if type(cell.responses) is not int or cell.responses <= 0 or cell.responses % 8:
        raise ValueError("Response budget must be a positive multiple of eight")
    if type(cell.passes) is not int or cell.passes not in (1, 2, 4):
        raise ValueError("Use one, two or four objective passes")
    for seed in (cell.seed, cell.rollout_seed):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
            raise ValueError("Seeds must be nonnegative integers smaller than2**31")
    passes = 4 if preflight else cell.passes
    value = (f"--lr {cell.lr} --train-reuse-epochs {passes} --micro-batch-size 1 "
            "--loss-type custom_loss --custom-loss-function-path modular_rl.v2.losses.fixed_token_policy_loss "
            f"--seq-length {4096 if cell.cap == 2048 else 8192} --max-position-embeddings 32768 "
            f"--max-tokens-per-gpu {cell.cap} --sglang-mem-fraction-static 0.2 "
            "--sglang-max-running-requests 16 --sglang-cuda-graph-max-bs 16")
    if cell.reference_kl:
        value += f" --use-kl-loss --kl-loss-type k3 --kl-loss-coef {cell.reference_kl}"
    if cell.seed is not None:
        value += f" --seed {cell.seed}"
    if cell.rollout_seed is not None:
        value += f" --rollout-seed {cell.rollout_seed}"
    if cell.reward_variant == "partial":
        value += " --custom-rm-path modular_rl.v2.code_reward.partial_test_reward"
    if cell.code_format == "prefix":
        value += (" --custom-generate-function-path modular_rl.v2.code_generation.generate"
                  " --custom-rm-path modular_rl.v2.code_generation.reward")
    return value


def validate_cells(cells):
    cells = list(cells)
    if not 1 <= len(cells) <= 32:
        raise ValueError("Declare one to32 independent Base conditions")
    for cell in cells:
        arguments(cell)
        if cell.variant == "preflight":
            raise ValueError("The preflight variant name is reserved for diagnostics")
    if len({cell.name for cell in cells}) != len(cells):
        raise ValueError("Each model/cap/variant must have a unique output directory")
    return cells


def load_cells(path):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise TypeError("A cells file must contain a JSON list")
    try:
        cells = [Cell(**row) for row in rows]
    except TypeError as exc:
        raise ValueError("Each condition must contain named Cell settings") from exc
    return validate_cells(cells)


def launch(repo, base, experiment, *, dry_run=False, cells=None, preflights=True, selection=None):
    from transformers import AutoTokenizer

    cells = validate_cells(matrix() if cells is None else cells)
    repo, base, experiment = repo.resolve(), base.resolve(), experiment.resolve()
    if experiment.exists():
        raise ValueError("Use a fresh Base sweep directory")
    model_root = base.parents[1] / "models"
    models = [model for model in BASE_MODELS if any(c.model == model for c in cells)]
    identities = {model: verify_initialization(model_root / model, model_root / (model + "_torch_dist"))
                  for model in models}
    tasks = list(dict.fromkeys(c.task for c in cells))
    source = freeze(repo, experiment, base, tasks, skip_zero=True)
    if selection is not None:
        for task in tasks:
            destination = experiment / "selection" / task
            shutil.rmtree(destination)
            shutil.copytree(selection.resolve() / task, destination)
    config = source / "modular_rl/v2/config.yaml"
    contents = config.read_text()
    if contents.count("use-dynamic-batch-size: true") != 1:
        raise ValueError("Inspect the native batching configuration before launching the Base sweep")
    atomic_write(config, contents.replace("use-dynamic-batch-size: true", "use-dynamic-batch-size: false"))
    proof = []
    for model in models:
        tokenizer = AutoTokenizer.from_pretrained(model_root / model, local_files_only=True)
        if tokenizer.eos_token_id != 151643 or not tokenizer.chat_template:
            raise ValueError("Unexpected pretrained Base tokenizer assets")
        for task in tasks:
            sample = read_events(experiment / "selection" / task / "train.jsonl")[0]["metadata"]["messages"]
            rendered = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            proof.append({"model": model, "task": task, "repo_id": identities[model]["repo_id"], "eos_token_id": tokenizer.eos_token_id,
                          "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                          "prompt_prefix": rendered[:180], "tokenizer_source": str(model_root / model)})
    write_json(experiment / "prompt-rendering-proof.json", proof)
    write_json(experiment / "plan.json", {
        "kind": "pretrained-base-controlled-" + "-".join(tasks) + "-sweep", "checkpoint_family": "pretrained-base",
        "identities": identities, "cells": [asdict(c) for c in cells], "max_training_gpus": len(cells),
        "preflight_topology": f"Independent diagnostics; all{len(cells)} full jobs immediately eligible without dependencies",
        "common_recipe": "fixed-token GRPO; skip entire uniform-advantage batches, including any reference penalty; "
                         "optional explicit per-cell K3 anchoring; no entropy or weight decay",
        "answer_forcing_reserve": "per-cell", "prompts_per_step": 1, "samples_per_prompt": 8,
        "validation_prompts": 32, "validation_interval": 50, "checkpoint_keep": 1,
        "raw_rollouts_saved": True, "source": str(source), "project": "modular-mopd-rl-initial",
        "memory_recipe": "Static single-response microbatches; CUDA cache release every microbatch",
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.rglob("*")) if p.is_file()},
    })
    jobs = []

    def submit(cell, *, preflight=False):
        task = cell.task
        directory = experiment / ("preflights" if preflight else "cells") / cell.name
        variant = "fixed-base-preflight" if preflight else "fixed-base-" + cell.variant
        run_output = directory / task / variant
        run_output.mkdir(parents=True)
        if preflight:
            selection = directory / "selection"
            shutil.copytree(experiment / "selection", selection)
            train = read_events(selection / task / "train.jsonl")
            easy = sorted((r for r in train if task == "code" or r["metadata"]["level"].endswith("1")),
                          key=lambda r: (len(r["metadata"]["messages"][0]["content"]), r["metadata"]["key"]))[:16]
            if len(easy) != 16:
                raise ValueError("Expected sixteen short level-one training questions for the GPU preflight")
            val = read_events(selection / task / "val.jsonl")[:4]
            for name, rows in (("train", easy), ("val", val)):
                (selection / task / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
            info = json.loads((selection / task / "selection.json").read_text())
            info["keys"] = {name: [r["metadata"]["key"] for r in rows] for name, rows in (("train", easy), ("val", val))}
            info["sha256"] = {name: hashlib.sha256((selection / task / (name + ".jsonl")).read_bytes()).hexdigest()
                              for name in ("train", "val")}
            info["strata"] = {name: dict(Counter("/".join(stratum(r)) for r in rows))
                              for name, rows in (("train", easy), ("val", val))}
            info.update(validation_prompts=4, preflight_only=True)
            write_json(selection / task / "selection.json", info)
        else:
            selection = experiment / "selection"
            (directory / "selection").symlink_to(selection, target_is_directory=True)
        steps, interval = (2, 2) if preflight else (cell.responses // 8, 50)
        passes = 4 if preflight else cell.passes
        stage = "preflight" if preflight else ("full" if cell.responses >= 8000 else "pilot")
        plan = {"models": [cell.model], "tasks": [task], "variants": [variant], "stage": stage,
                "checkpoint_family": "pretrained-base", "hf_checkpoint": "Qwen/" + cell.model,
                "prompts_per_update": 1, "updates": steps, "training_responses": steps * 8,
                "max_sampled_tokens": cell.cap, "answer_reserve": cell.answer_reserve, "objective_passes": passes,
                "reward_variant": cell.reward_variant,
                "code_format": cell.code_format,
                "lr": cell.lr, "reference_kl": cell.reference_kl,
                "seed": cell.seed, "rollout_seed": cell.rollout_seed,
                "preflight": preflight, "identity": identities[cell.model], "selection": str(selection)}
        write_json(directory / "plan.json", plan)
        environment = {**os.environ, "RLVE_REPO": str(source), "MODULAR_RL_TASKS": task,
                       "MODULAR_RL_MODELS": cell.model, "MODULAR_RL_GROUP": experiment.name,
                       "MODULAR_RL_VARIANT": "BASE " + stage.upper() + " " + cell.name,
                       "MODULAR_RL_DATA": str(base / "data"), "MODULAR_RL_SELECTION": str(selection),
                       "MODULAR_RL_RUNTIME": str(experiment / "runtime"), "MODULAR_RL_OUTPUT": str(run_output),
                       "MODULAR_RL_STEPS": str(steps), "MODULAR_RL_EVAL_INTERVAL": str(interval),
                       "MODULAR_RL_CHECKPOINT_INTERVAL": str(interval), "MODULAR_RL_CHECKPOINT_KEEP": "1",
                       "MODULAR_RL_MAX_RESPONSE_TOKENS": str(cell.cap), "MODULAR_RL_ANSWER_FORCING_TOKENS": str(cell.answer_reserve),
                       "MODULAR_RL_PROMPTS_PER_UPDATE": "1", "MODULAR_RL_REQUIRE_PRETRAINED_BASE": "1",
                       "MODULAR_RL_EXPECTED_BASE_MODEL": cell.model, "MODULAR_RL_SKIP_ZERO_ADVANTAGE": "1",
                       "MODULAR_RL_EVAL_ONLY": "0",
                       "MODULAR_RL_FILTER_UNIFORM_REFERENCE_BATCHES": str(int(bool(cell.reference_kl))),
                       "MODULAR_RL_ALLOW_OBJECTIVE_REUSE": "1",
                       "MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL": "1", "MODULAR_RL_UPDATE_PROBE_INTERVAL": "1" if preflight else "10",
                       "MODULAR_RL_EXTRA_ARGS": arguments(cell, preflight=preflight),
                       "WANDB_PROJECT": "modular-mopd-rl-initial",
                       "MODULAR_RL_TIMEOUT": "6500" if preflight else str(cell.hours * 3600 - 600)}
        wrapper = directory / "run.sbatch"
        body = ("#!/usr/bin/env bash\n#SBATCH --account=miso\n"
                "#SBATCH --partition=jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio\n"
                "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --requeue\n#SBATCH --open-mode=append\n"
                "set -euo pipefail\n" + f"bash {shlex.quote(str(source / 'modular_rl/v2/run.sbatch'))}\n")
        if preflight:
            body += (f"PYTHONPATH={shlex.quote(str(source))} /nlp/scr/dhei/tinker-mopd/venv-rl/bin/python "
                     f"-m modular_rl.v2.base_sweep --verify-preflight {shlex.quote(str(directory))}\n")
        wrapper.write_text(body)
        command = ["sbatch", "--parsable", f"--job-name={'base-preflight' if preflight else 'base-' + task}-{cell.name}",
                   "--gres=gpu:1", "--constraint=40G|48G|80G|141G", "--cpus-per-task=8",
                   "--mem=64G" if "0.6B" in cell.model else "--mem=128G",
                   "--time=02:00:00" if preflight else f"--time={cell.hours // 24}-{cell.hours % 24:02}:00:00",
                   "--exclude=viscam10,jagupard33,jagupard34", f"--chdir={source}",
                   f"--output={run_output}/slurm-%j.out"]
        command.append(str(wrapper))
        job_id = "dry-" + str(len(jobs)) if dry_run else subprocess.check_output(command, env=environment, text=True).strip().split(";")[0]
        row = {**asdict(cell), "preflight": preflight, "variant": variant, "job_id": job_id,
               "stage": stage, "experiment": str(directory), "run": str(run_output / (cell.model + "-" + task)), "command": command}
        jobs.append(row)
        write_json(experiment / "jobs.json", jobs)
        write_json(directory / "jobs.json", [row])
        print(json.dumps({k: row[k] for k in ("model", "cap", "variant", "job_id", "preflight")}), flush=True)
        return job_id

    if preflights:
        for model, cap, task in dict.fromkeys((c.model, c.cap, c.task) for c in cells):
            submit(Cell(model, cap, "preflight", task=task), preflight=True)
    for cell in cells:
        submit(cell)
    return jobs


def verify_preflight(directory):
    from .verify_probe_alignment import verify

    plan = json.loads((directory / "plan.json").read_text())
    if not plan["preflight"] or plan["models"][0] not in BASE_MODELS:
        raise ValueError("This is not a declared pretrained Base GPU preflight")
    model, variant, task = plan["models"][0], plan["variants"][0], plan["tasks"][0]
    run = directory / task / variant / (model + "-" + task)
    completed = json.loads((run / "completed.json").read_text())
    if not completed["completed"] or completed["train_rollouts"] != 16:
        raise ValueError("Base preflight did not complete its two prompt iterations")
    alignment = verify(directory)
    samples = [r for p in (run / "rollouts/train").glob("*.jsonl") for r in read_events(p)]
    vals = [r for p in (run / "rollouts/eval").glob("*.jsonl") for r in read_events(p)]
    if not samples or not any(r["correct"] for r in samples + vals):
        raise ValueError("Base preflight has no positive math rewards; inspect prompt/forcing")
    write_json(directory / "preflight-passed.json", {"passed": True, "model": model,
               "checkpoint_family": "pretrained-base", "cap": plan["max_sampled_tokens"], "alignment": alignment})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", type=Path, default=Path("/nlp/scr/dhei/modular-mopd-rl/v2"))
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cells-json", type=Path, help="Optional JSON list of explicit Base conditions, including reference_kl")
    parser.add_argument("--skip-preflights", action="store_true", help="Submit only full conditions; diagnostics never gate them")
    parser.add_argument("--selection", type=Path, help="Optional frozen task selections to copy into this experiment")
    parser.add_argument("--verify-preflight", type=Path)
    options = parser.parse_args()
    if options.verify_preflight:
        verify_preflight(options.verify_preflight)
    elif options.experiment:
        launch(options.repo, options.base, options.experiment, dry_run=options.dry_run,
               cells=load_cells(options.cells_json) if options.cells_json else None,
               preflights=not options.skip_preflights, selection=options.selection)
    else:
        parser.error("Choose --experiment or --verify-preflight")
