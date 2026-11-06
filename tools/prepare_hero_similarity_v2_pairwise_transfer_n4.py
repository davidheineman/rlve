"""Freeze four N=4 transfer sets and materialize their R1 teacher views."""

import argparse
import csv
import hashlib
import itertools
import json
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data/HERO-SIMILARITY-V2-PAIRWISE-TRANSFER-N4"
PAIRWISE = ROOT / "plot/initial-pairwise-ood-r1/pairwise/delta_curves_normalized_smoothed_pairs.csv"
SNAPSHOT = ROOT / "plot/initial-pairwise-ood-r1/snapshot.json"
ORDER = ROOT / "scripts/slurm/sweeps/mopd-v2-32.txt"
VALIDATION = ROOT / "data/EVAL-pairwise32"
SOURCE_ID = ROOT / "data/MOPD-SWEEPS-V2/id_n16_d0_x4.json"
OOD = ROOT / "data/MOPD-SWEEPS/ood_d0_16env_x24.json"
SOURCE_COSTS = "hero-similarity-v2-20261004"
TEACHER_PROJECT = "rl-data-hero-similarity-teachers-v2"


def metadata(row):
    value = row["metadata"]
    return json.loads(value) if isinstance(value, str) else value


def write_or_verify(path, content):
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Frozen input differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def select_extremes():
    snapshot = json.loads(SNAPSHOT.read_text())
    environments = sorted(snapshot["training_environments"])
    if len(environments) != 32 or len(snapshot["runs"]) != 32:
        raise RuntimeError("Need evaluations from all 32 pairwise training runs")
    scores, steps = {}, {}
    for row in csv.DictReader(PAIRWISE.open()):
        a, b = row["environment_a"], row["environment_b"]
        scores[a, b] = float(row["a_to_b_percent_points"])
        scores[b, a] = float(row["b_to_a_percent_points"])
        steps[a] = int(row["a_to_b_step"])
        steps[b] = int(row["b_to_a_step"])
    if len(scores) != 32 * 31 or set(steps) != set(environments):
        raise RuntimeError("Expected all 496 reciprocal pairs in the transfer CSV")

    positive = negative = none = oneway = None
    for names in itertools.combinations(environments, 4):
        directed = [scores[a, b] for a, b in itertools.permutations(names, 2)]
        if min(directed) > 0:
            rank = (min(directed), statistics.mean(directed))
            if positive is None or rank > positive[0]:
                positive = (rank, names)
        if max(directed) < 0:
            rank = (max(directed), statistics.mean(directed))
            if negative is None or rank < negative[0]:
                negative = (rank, names)
        rank = (max(abs(value) for value in directed),
                statistics.mean(abs(value) for value in directed))
        if none is None or rank < none[0]:
            none = (rank, names)
        for hub in names:
            others = [name for name in names if name != hub]
            outgoing = [scores[hub, name] for name in others]
            incoming = [scores[name, hub] for name in others]
            if min(outgoing) > 10 and max(map(abs, incoming)) <= 2:
                rank = (min(outgoing), statistics.mean(outgoing))
                if oneway is None or rank > oneway[0]:
                    oneway = (rank, names, hub)
    if any(value is None for value in (positive, negative, oneway, none)):
        raise RuntimeError("No quartet satisfies a required transfer category")

    picked = {
        "pairwise_transfer_positive_n4": positive[1],
        "pairwise_transfer_negative_n4": negative[1],
        "pairwise_transfer_oneway_n4": oneway[1],
        "pairwise_transfer_none_n4": none[1],
    }
    selections = {}
    states = {run["environment"]: run["state"] for run in snapshot["runs"]}
    for variant, names in picked.items():
        edges = [
            {"a": a, "b": b, "a_to_b_pp": scores[a, b], "b_to_a_pp": scores[b, a]}
            for a, b in itertools.combinations(names, 2)
        ]
        record = {"environments": list(names), "edges": edges,
                  "endpoint_steps": {name: steps[name] for name in names},
                  "run_states": {name: states[name] for name in names}}
        if variant.endswith("oneway_n4"):
            record["source_hub"] = oneway[2]
        selections[variant] = record
    return {
        "source_pairwise_csv": str(PAIRWISE.relative_to(ROOT)),
        "source_pairwise_csv_sha256": hashlib.sha256(PAIRWISE.read_bytes()).hexdigest(),
        "source_snapshot_fetched_at": snapshot["fetched_at"],
        "selection_rule": {
            "positive": "all 12 directed scores positive; maximize the minimum, then mean",
            "negative": "all 12 directed scores negative; minimize the maximum, then mean",
            "oneway": "one source transfers >10 pp to each other environment while each reverse direction is within 2 pp of zero; maximize minimum outgoing, then mean",
            "none": "minimize maximum absolute directed score, then mean absolute score",
        },
        "selections": selections,
    }


def prepare():
    order = ORDER.read_text().split()
    if len(order) != 32 or len(set(order)) != 32:
        raise RuntimeError("Expected 32 distinct environments in the v2 order")
    manifest_path = OUTPUT / "manifest.json"
    provenance = json.loads(manifest_path.read_text()) if manifest_path.exists() else select_extremes()
    selections = provenance["selections"]
    selected = {name for record in selections.values() for name in record["environments"]}
    if len(selections) != 4 or any(len(record["environments"]) != 4 for record in selections.values()):
        raise RuntimeError("Expected four selections of four environments")
    if not selected <= set(order):
        raise RuntimeError(f"Selected environment not in v2 order: {selected - set(order)}")
    ood = json.loads(OOD.read_text())
    ood_counts = Counter(metadata(row)["environment"] for row in ood)
    if len(ood) != 384 or len(ood_counts) != 16 or set(ood_counts.values()) != {24} or selected & ood_counts.keys():
        raise RuntimeError("Expected the existing disjoint 16-by-24 OOD set")

    existing = {}
    for row in json.loads(SOURCE_ID.read_text()):
        existing.setdefault(metadata(row)["environment"], []).append(row)
    new_teachers = [name for name in order if name in selected and name not in existing]
    from make_rlve_eval_set import _records_for_environment

    ids = {}
    for name in sorted(selected):
        validation = json.loads((VALIDATION / f"{name}_d0.json").read_text())
        val_seeds = {metadata(row)["config"]["seed"] for row in validation}
        if len(validation) != 128 or len(val_seeds) != 128:
            raise RuntimeError(f"Invalid frozen validation for {name}")
        if name in existing:
            ids[name] = existing[name]
        else:
            index = order.index(name)
            ids[name] = _records_for_environment(
                name, 0, 4, 6_000_000 + index * 100_000, 20261003,
                seen_prompts=set(),
            )
            write_or_verify(OUTPUT / f"{name}_d0.json", (VALIDATION / f"{name}_d0.json").read_text())
        if len(ids[name]) != 4 or len({row["user_prompt"] for row in ids[name]}) != 4:
            raise RuntimeError(f"Invalid ID prompts for {name}")
        if {metadata(row)["config"]["seed"] for row in ids[name]} & val_seeds:
            raise RuntimeError(f"ID and validation seeds overlap for {name}")
    write_or_verify(OUTPUT / "new_teachers.txt", " ".join(new_teachers) + "\n")
    for variant, record in selections.items():
        names = record["environments"]
        write_or_verify(OUTPUT / f"{variant}.txt", " ".join(names) + "\n")
        write_or_verify(OUTPUT / f"id_{variant}_d0_x4.json",
                        json.dumps([row for name in names for row in ids[name]], indent=1) + "\n")
        print(f"{variant}: {', '.join(names)}; {4 * len(names)} frozen ID prompts", flush=True)
    provenance.update({
        "source_id": str(SOURCE_ID.relative_to(ROOT)),
        "source_validation": str(VALIDATION.relative_to(ROOT)),
        "source_ood": str(OOD.relative_to(ROOT)),
        "source_teacher_costs_experiment": SOURCE_COSTS,
        "new_teachers": new_teachers,
        "id_prompts_per_environment": 4,
        "ood_prompts_per_environment": 24,
    })
    write_or_verify(manifest_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"New teachers: {', '.join(new_teachers)}", flush=True)


def teacher_allocations(array_job, count):
    result = subprocess.run(
        ["sacct", "-n", "-P", "--duplicates", "-j", str(array_job),
         "--format=JobID,ElapsedRaw,State,AllocTRES", "--allocations"],
        check=True, capture_output=True, text=True,
    )
    allocations = {index: [] for index in range(count)}
    completed = set()
    for line in result.stdout.splitlines():
        job, elapsed, state, tres = line.split("|", 3)
        match = re.fullmatch(rf"{array_job}_(\d+)", job)
        if match:
            index = int(match.group(1))
            if index not in allocations or "gres/gpu=1" not in tres:
                raise RuntimeError(f"Invalid teacher allocation: {line}")
            if state == "COMPLETED":
                completed.add(index)
            allocations[index].append({"job": job, "state": state,
                                       "gpu_hours": int(elapsed) / 3600})
    if completed != set(range(count)):
        raise RuntimeError(f"Teacher array incomplete: completed {sorted(completed)}")
    return allocations


def materialize(experiment_dir, runs_dir, teacher_array, teacher_group):
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    new_teachers = manifest["new_teachers"]
    source = json.loads((runs_dir / SOURCE_COSTS / "teacher_costs.json").read_text())
    teachers = dict(source["teachers"])
    allocations = teacher_allocations(teacher_array, len(new_teachers))
    for index, name in enumerate(new_teachers):
        run = runs_dir / teacher_group / "resumable" / f"{index:02d}-{name}"
        marker = json.loads((run / "weights_only_complete.json").read_text())
        run_id = (run / "wandb_run_id.txt").read_text().strip()
        if marker.get("format") != "hf-safetensors" or not 0 <= marker["checkpoint_step"] <= 499:
            raise RuntimeError(f"Incomplete teacher export for {name}")
        if len(run_id) != 8 or not (run / "hf-final/model.safetensors.index.json").is_file():
            raise RuntimeError(f"Missing teacher weights or W&B ID for {name}")
        teachers[name] = {
            "source_run": str(run.resolve(strict=True)),
            "source_wandb_run": f"https://wandb.ai/david-heineman/{TEACHER_PROJECT}/runs/{run_id}",
            "checkpoint_step": marker["checkpoint_step"],
            "teacher_gpu_hours": sum(item["gpu_hours"] for item in allocations[index]),
            "source_allocations": allocations[index],
        }
    views = {}
    for variant, record in manifest["selections"].items():
        view = experiment_dir / "teacher-views" / variant / "resumable"
        view.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(record["environments"]):
            target = Path(teachers[name]["source_run"]).resolve(strict=True)
            link = view / f"{index:02d}-{name}"
            if link.is_symlink():
                if link.resolve() != target:
                    raise RuntimeError(f"Wrong existing teacher link: {link}")
            elif link.exists():
                raise RuntimeError(f"Teacher link path already occupied: {link}")
            else:
                link.symlink_to(target, target_is_directory=True)
        views[variant] = str(view)
    costs = {
        "teachers": teachers,
        "per_variant_teacher_gpu_hours": {
            variant: sum(teachers[name]["teacher_gpu_hours"] for name in record["environments"])
            for variant, record in manifest["selections"].items()
        },
        "views": views,
        "source_teacher_array": teacher_array,
        "source_teacher_group": teacher_group,
        "existing_teacher_costs": str(runs_dir / SOURCE_COSTS / "teacher_costs.json"),
    }
    write_or_verify(experiment_dir / "teacher_costs.json", json.dumps(costs, indent=2, sort_keys=True) + "\n")
    print(f"Linked {len(teachers)} validated teachers into {len(views)} N=4 views", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "materialize"))
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--teacher-array", type=int)
    parser.add_argument("--teacher-group")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if not all((args.experiment_dir, args.runs_dir, args.teacher_array, args.teacher_group)):
            parser.error("materialize requires experiment, runs, teacher array, and teacher group")
        materialize(args.experiment_dir, args.runs_dir, args.teacher_array, args.teacher_group)


if __name__ == "__main__":
    main()
