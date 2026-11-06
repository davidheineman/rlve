"""Copy compatible completed specialists and matching per-task evaluation cells.

The panel-wide evaluation hash changes when tasks are added. Reusing a cell
requires identical per-environment rows, sampling settings, and adapter bytes;
the imported cell records both its original hash and the new panel hash.
"""

import hashlib
import json
import shutil
from pathlib import Path

from lora_teachers.plan import sha256


def read(path):
    return json.loads(Path(path).read_text())


def describe_reuse(source, plan):
    original = read(source / "plan.json")
    keys = (
        "project",
        "base_model",
        "teacher_config",
        "lora_rank",
        "lora_alpha",
        "target_modules",
        "seed",
        "teacher_rollouts",
        "teacher_validation_threshold",
        "optimizer_eps",
        "grad_clip_norm",
        "tis_clip_low",
        "tis_clip_high",
    )
    if any(original[key] != plan[key] for key in keys):
        raise ValueError(
            "Reused specialists must have exactly the same training recipe"
        )
    if not (source / "completed.json").exists():
        raise ValueError("Only completed source experiments can be reused")
    names = original["environments"]
    if not set(names) < set(plan["environments"]):
        raise ValueError("The reused panel must be a strict subset of the new panel")
    by_name = {t["environment"]: t for t in plan["teachers"]}
    for teacher in original["teachers"]:
        name = teacher["environment"]
        if teacher["eval_sha256"] != by_name[name]["eval_sha256"]:
            raise ValueError(f"Original teacher gate changed: {name}")
        result = read(source / "teacher" / teacher["source_label"] / "completed.json")
        if (
            result["environment"] != name
            or result["training_rollouts"] > plan["teacher_rollouts"]
        ):
            raise ValueError(f"Invalid completed specialist: {name}")
        if (
            result["training_rollouts"] < plan["teacher_rollouts"]
            and result["accuracy"] < plan["teacher_validation_threshold"]
        ):
            raise ValueError(f"Specialist did not reach a stopping condition: {name}")
        for filename in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "META.json",
        ):
            if not (Path(result["hf"]) / filename).is_file():
                raise FileNotFoundError(f"Missing source adapter: {name}/{filename}")
        if not (Path(result["hf"]).parent / "reference.json").is_file():
            raise FileNotFoundError(f"Missing source sampler reference: {name}")
    return {
        "source_experiment": str(source),
        "source_plan_sha256": sha256(source / "plan.json"),
        "teachers": len(names),
        "environments": names,
        "protocol": "copy final adapters; reuse only identical per-task evaluation inputs with explicit provenance",
    }


def grouped_rows(experiment, plan, split):
    path = experiment / "source/rlve" / plan["clustering"][split]
    if sha256(path) != plan["clustering"][f"{split}_sha256"]:
        raise ValueError("Frozen evaluation inputs changed")
    grouped = {}
    for row in read(path):
        meta = row["metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        grouped.setdefault(meta["environment"], []).append(row)
    return grouped


def import_reuse(experiment, source, plan):
    if describe_reuse(source, plan) != plan["reuse"]:
        raise ValueError("Source experiment changed before import")
    original = read(source / "plan.json")
    imported, source_hashes = {}, {}

    def copy_file(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        expected = sha256(src)
        shutil.copy2(src, dst)
        if sha256(dst) != expected:
            raise ValueError(f"Incomplete copied artifact: {dst}")
        source_hashes[str(src)] = expected
        imported[str(dst.relative_to(experiment))] = expected

    target_teachers = {t["environment"]: t for t in plan["teachers"]}
    for teacher in original["teachers"]:
        src = source / "teacher" / teacher["source_label"]
        dst = (
            experiment
            / "teacher"
            / target_teachers[teacher["environment"]]["source_label"]
        )
        result = read(src / "completed.json")
        for path in Path(result["hf"]).rglob("*"):
            if path.is_file():
                copy_file(path, dst / "final/adapter" / path.relative_to(result["hf"]))
        copy_file(
            Path(result["hf"]).parent / "reference.json", dst / "final/reference.json"
        )
        # Only the copied ownership envelope changes. Weight tensors stay
        # byte-identical, and reuse.json retains the original metadata hash.
        metadata = dst / "final/adapter/META.json"
        meta = read(metadata)
        if (
            meta["tenant_digest"]
            != hashlib.sha256(f"tml-async-{source.name}".encode()).hexdigest()
        ):
            raise ValueError("Source adapter has unexpected experiment ownership")
        meta["tenant_digest"] = hashlib.sha256(
            f"tml-async-{experiment.name}".encode()
        ).hexdigest()
        metadata.write_text(json.dumps(meta, indent=2) + "\n")
        imported[str(metadata.relative_to(experiment))] = sha256(metadata)
        result.update(
            hf=str(dst / "final/adapter"),
            resume=None,
            reused_from=str(src / "completed.json"),
        )
        dst.mkdir(parents=True, exist_ok=True)
        path = dst / "completed.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        imported[str(path.relative_to(experiment))] = sha256(path)
        source_hashes[str(src / "completed.json")] = sha256(src / "completed.json")
        for filename in ("metrics.jsonl", "identity.json", "initial-validation.json"):
            if (src / filename).is_file():
                copy_file(src / filename, dst / filename)

    cells = {}
    for split in ("selection", "confirmation"):
        old_rows, new_rows = (
            grouped_rows(source, original, split),
            grouped_rows(experiment, plan, split),
        )
        samples = plan["clustering"][f"{split}_samples_per_prompt"]
        if samples != original["clustering"][f"{split}_samples_per_prompt"]:
            raise ValueError("Evaluation sampling settings changed")
        cells[split] = 0
        for environment in original["environments"]:
            if old_rows[environment] != new_rows[environment]:
                raise ValueError(f"Per-task {split} inputs changed: {environment}")
            prompt_ids = [
                hashlib.sha256(r["user_prompt"].encode()).hexdigest()
                for r in new_rows[environment]
            ]
            labels = [("base", "base")] + [
                (
                    f"individual-{i:02d}",
                    f"individual-{plan['environments'].index(name):02d}",
                )
                for i, name in enumerate(original["environments"])
            ]
            for old_label, new_label in labels:
                src = (
                    source
                    / "clustering/evaluations"
                    / split
                    / old_label
                    / f"{environment}.json"
                )
                if not src.exists():
                    continue
                cell = read(src)
                if (
                    cell["eval_sha256"] != original["clustering"][f"{split}_sha256"]
                    or [o["prompt_id"] for o in cell["observations"]] != prompt_ids
                    or any(o["total"] != samples for o in cell["observations"])
                ):
                    raise ValueError(f"Incompatible cached evaluation: {src}")
                source_hashes[str(src)] = sha256(src)
                cell["reused_from"] = {
                    "path": str(src),
                    "sha256": sha256(src),
                    "eval_sha256": cell["eval_sha256"],
                }
                cell.update(
                    adapter=new_label, eval_sha256=plan["clustering"][f"{split}_sha256"]
                )
                dst = (
                    experiment
                    / "clustering/evaluations"
                    / split
                    / new_label
                    / f"{environment}.json"
                )
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(cell, indent=2) + "\n")
                imported[str(dst.relative_to(experiment))] = sha256(dst)
                cells[split] += 1
    record = {
        **plan["reuse"],
        "evaluation_cells": cells,
        "source_file_sha256": source_hashes,
    }
    path = experiment / "reuse.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    imported["reuse.json"] = sha256(path)
    manifest = read(experiment / "input_sha256.json")
    manifest.update(imported)
    (experiment / "input_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return record
