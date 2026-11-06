"""Benchmark balanced clustering on cached prompt outcomes; never allocates GPUs.

Run with numpy, scipy, matplotlib and networkx installed. Defaults use four CPU
processes, resumable fits, the existing four prompt folds, and nested selection.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from analyze_balanced_lora_clustering import load_observations
from lora_clustering.grouping_methods import (
    METHODS, PARAMETERS, fit_methods, random_task_scores, task_scores, validate,
)

ROOT = Path(__file__).resolve().parents[1]
PLOT = ROOT / "plot/mopd-clustering-v2"
FOLDS = [[i, i + 4] for i in range(4)]
COUNTS = [2, 4, 8, 16, 32]
SEED = 20261105


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_indices(mask):
    return [p for fold, held in enumerate(FOLDS) if mask & (1 << fold) for p in held]


def fit_context(arguments):
    directory, mask, k, fingerprint = arguments
    directory = Path(directory)
    path = directory / "fits" / f"folds-{mask:04b}-k{k:02d}.json"
    record = read(path) if path.exists() else {"fingerprint": fingerprint, "groups": {}}
    if record["fingerprint"] != fingerprint:
        raise ValueError("Fit cache belongs to a different frozen protocol")
    if record.get("complete"):
        return mask, k, 0.0
    started = time.monotonic()
    cube = np.load(directory / "observations.npz")["cube"][:, :, training_indices(mask)]

    def checkpoint(method, groups):
        record["groups"][method] = groups
        write(path, record)

    fit_methods(cube, k, SEED + mask * 1000003 + k * 101,
                existing=record["groups"], callback=checkpoint)
    record.update(complete=True, training_folds=[i for i in range(4) if mask & (1 << i)],
                  elapsed_seconds=record.get("elapsed_seconds", 0) + time.monotonic() - started)
    write(path, record)
    return mask, k, record["elapsed_seconds"]


def choose_nested(cube, fits, k, outer):
    """Select among new methods using only the three outer-training folds."""
    scores = {}
    for method in METHODS:
        values = []
        for inner in range(4):
            if inner == outer:
                continue
            mask = 15 ^ (1 << outer) ^ (1 << inner)
            groups = fits[mask, k][method]
            values.append(float(task_scores(cube[:, :, FOLDS[inner]].mean(axis=-1), groups).mean()))
        scores[method] = float(np.mean(values))
    selected = max(METHODS, key=lambda method: scores[method])
    return selected, scores


def summarize(values):
    return {"mean": float(np.mean(values)), "se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "fold_means": [float(v) for v in values]}


def analyze(directory, cube, original, low, names, fingerprint):
    fits = {}
    for path in (directory / "fits").glob("*.json"):
        row = read(path)
        if row["fingerprint"] != fingerprint or not row.get("complete"):
            raise ValueError("Incomplete or incompatible fits")
        mask = sum(1 << i for i in row["training_folds"])
        k = int(path.stem.split("-k")[-1])
        fits[mask, k] = row["groups"]
    nested = {}
    for k in COUNTS:
        nested[k] = []
        for outer in range(4):
            selected, scores = choose_nested(cube, fits, k, outer)
            nested[k].append({"outer_fold": outer, "method": selected, "inner_cv_means": scores})
    labels = {"original": "Original maximin", **METHODS,
              "nested": "Nested method selector", "random": "Random expectation", "low": "Original low-transfer"}
    records = []
    for method, label in labels.items():
        points = []
        for k in [1, *COUNTS, len(names)]:
            evaluations = []
            for outer, held in enumerate(FOLDS):
                matrix = cube[:, :, held].mean(axis=-1)
                groups = None
                if k == 1:
                    groups = [list(range(len(names)))]
                elif k == len(names):
                    groups = [[i] for i in range(len(names))]
                elif method in ("original", "low"):
                    source = original if method == "original" else low
                    row = next(r for r in source["candidates"] if r["k"] == k)
                    fold = row["folds"][outer]
                    if fold["held_prompt_indices"] != held:
                        raise ValueError("Original folds differ from the frozen benchmark folds")
                    groups = fold["groups"]
                elif method != "random":
                    chosen = nested[k][outer]["method"] if method == "nested" else method
                    groups = fits[15 ^ (1 << outer), k][chosen]
                if groups is None:
                    scores = random_task_scores(matrix, k)
                    off_diagonal = None
                else:
                    validate(groups, len(names), k)
                    scores = task_scores(matrix, groups)
                    off_diagonal = (float(task_scores(matrix, groups, include_self=False).mean())
                                    if k < len(names) else None)
                evaluations.append({"held_prompt_indices": held, "groups": groups,
                                    "mean": float(scores.mean()), "per_task": scores.tolist(),
                                    "off_diagonal_mean": off_diagonal})
            points.append({"k": k, "size": len(names) // k,
                           **summarize([r["mean"] for r in evaluations]), "folds": evaluations})
        records.append({"method": method, "label": label, "points": points})
    baseline = records[0]
    for record in records:
        for point, previous in zip(record["points"], baseline["points"]):
            point["delta_vs_original"] = summarize(np.array(point["fold_means"]) - previous["fold_means"])
    best = {}
    for k in COUNTS:
        chosen = max((r for r in records if r["method"] in METHODS),
                     key=lambda r: next(p["mean"] for p in r["points"] if p["k"] == k))
        best[str(k)] = {"method": chosen["method"], "label": chosen["label"],
                        "groups": fits[15, k][chosen["method"]],
                        "mean": next(p["mean"] for p in chosen["points"] if p["k"] == k),
                        "scope": "Best observed CV method; selection across methods is exploratory. Groups refit on all eight selection prompts."}
    result = {
        "fingerprint": fingerprint, "environments": names, "records": records,
        "nested_choices": nested, "exploratory_best_by_k": best,
        "metric": "Macro pass@1 improvement, averaging individual group adapters including own-task adapter",
        "evaluation": "Four difficulty-stratified outer folds: fit on six prompts/task, score on two; two completions/prompt",
        "nested_selection": "For each outer fold and k, select one of 12 new methods using three inner folds; refit using all six outer-training prompts",
        "uncertainty": "Error bars are +/-1 descriptive SE across four prompt folds. Delta SE uses paired fold differences.",
        "scope": "Exploratory comparison conditional on the 64 cached specialists; no disjoint confirmation outcomes or new GPU evaluations used",
        "full_fit_assignments": {str(k): fits[15, k] for k in COUNTS},
    }
    write(directory / "results.json", result)
    with (directory / "assignments.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "k", "cluster", "environment"])
        for k in COUNTS:
            for method, groups in fits[15, k].items():
                for cluster, group in enumerate(groups, 1):
                    writer.writerows((method, k, cluster, names[i]) for i in group)
    write(directory / "selected-groupings.json", best)
    report(result, directory)
    return result


def report(result, directory):
    lines = ["# Pairwise clustering method comparison", "", result["metric"] + ".", "",
             result["evaluation"] + ".", "", result["nested_selection"] + ".", "",
             result["uncertainty"], "", result["scope"] + ".", "",
             "All groups are exactly balanced and cover all 64 tasks. k=1 and k=64 are shared endpoints.", "",
             "The original baseline optimizes weakest-cluster internal/external quality; the new mean-swap method directly maximizes the plotted average. These objectives differ.", "",
             "| Method | k=2 | k=4 | k=8 | k=16 | k=32 |", "|---|---:|---:|---:|---:|---:|"]
    for record in result["records"]:
        cells = [f"{100*p['mean']:.2f}" for p in record["points"] if p["k"] in COUNTS]
        lines.append("| " + " | ".join([record["label"], *cells]) + " |")
    lines += ["", "Values above are absolute percentage-point gains over base.", "",
              "## Paired improvement over original maximin (percentage points)", "",
              "| Method | k=2 | k=4 | k=8 | k=16 | k=32 |", "|---|---:|---:|---:|---:|---:|"]
    for record in result["records"]:
        if record["method"] not in METHODS and record["method"] != "nested":
            continue
        cells = [f"{100*p['delta_vs_original']['mean']:+.2f} ± {100*p['delta_vs_original']['se']:.2f}"
                 for p in record["points"] if p["k"] in COUNTS]
        lines.append("| " + " | ".join([record["label"], *cells]) + " |")
    lines += ["", "The best-looking method at each k is exploratory after comparing many methods. Nested selection evaluates the choice without reading that outer fold. Four folds do not provide a strong significance claim.", "",
              "`selected-groupings.json` and `assignments.csv` contain full-data fits for follow-up; they have no new independent confirmation.", "",
              "Balanced k-means uses [SciPy linear assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) to enforce capacities. Hierarchies use disjoint maximum-weight block matching at each level, rather than unconstrained agglomeration.", ""]
    (directory / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PLOT / ".cache/groupings")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plot-python", type=Path,
                        default=ROOT.parent / "nano-omni/.venv-midtraining/bin/python")
    args = parser.parse_args()
    directory = args.output.resolve()
    positive_path = ROOT / "plot/mopd-clustering/n64/balanced/results.json"
    low_path = ROOT / "plot/mopd-clustering/n64/low-transfer/results.json"
    protocol = {
        "methods": METHODS, "parameters": PARAMETERS, "counts": COUNTS,
        "folds": FOLDS, "seed": SEED, "self_transfer_included": True,
        "balanced": True, "gpu_compute": False,
        "selection_rule": "All new method choices use nested prompt CV; display all outer-fold method results as exploratory",
        "positive_sha256": digest(positive_path), "low_sha256": digest(low_path),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), ROOT / "tools/lora_clustering/grouping_methods.py"]},
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "networkx"]},
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (directory / "study-plan.json").exists() and read(directory / "study-plan.json") != protocol:
        raise ValueError("Existing output has a different frozen plan; choose a new --output")
    write(directory / "study-plan.json", protocol)
    original, low = read(positive_path), read(low_path)
    if (directory / "observations.npz").exists():
        inputs = read(directory / "input-provenance.json")
        if inputs["observation_sha256"] != low["observation_sha256"]:
            raise ValueError("Input cache has different observations")
        if digest(directory / "observations.npz") != inputs["array_sha256"]:
            raise ValueError("Cached prompt outcomes changed")
        cube = np.load(directory / "observations.npz")["cube"]
        names = inputs["environments"]
    else:
        plan, cube, hashes = load_observations(Path(low["source_experiment"]))
        if hashes != low["observation_sha256"] or hashes != original["observation_sha256"]:
            raise ValueError("Original high/low and study observations differ")
        names = plan["environments"]
        np.savez_compressed(directory / "observations.npz", cube=cube)
        write(directory / "input-provenance.json", {
            "environments": names, "source_experiment": low["source_experiment"],
            "observation_sha256": hashes, "array_sha256": digest(directory / "observations.npz"),
        })
    print(f"Loaded {cube.shape}: 4,160 hash-verified cached cells. Frozen {len(METHODS)} new methods.", flush=True)
    masks = [m for m in range(1, 16) if m.bit_count() == 3]
    masks += [m for m in range(1, 16) if m.bit_count() == 2] + [15]
    jobs = [(str(directory), mask, k, fingerprint) for mask in masks for k in COUNTS]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_context, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            mask, k, seconds = future.result()
            print(f"FIT {count}/{len(jobs)} folds={mask:04b} k={k} seconds={seconds:.1f} elapsed={time.monotonic()-started:.1f}", flush=True)
            write(directory / "progress.json", {"complete_contexts": count, "total_contexts": len(jobs),
                                                "elapsed_seconds": time.monotonic()-started})
    result = analyze(directory, cube, original, low, names, fingerprint)
    write(directory / "completion.json", {"fingerprint": fingerprint, "contexts": len(jobs),
                                          "elapsed_seconds": time.monotonic()-started, "gpu_compute": False})
    figure_data = read(PLOT / ".cache/figure_data.json")
    figure_data["pairwise_groupings"] = result
    write(PLOT / ".cache/figure_data.json", figure_data)
    subprocess.run([str(args.plot_python), str(PLOT / ".plotting/groupings_plot.py"),
                    str(directory / "results.json"), "--output", str(PLOT / "pairwise_transfer_groupings.png")], check=True)
    print(f"DONE {directory / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
