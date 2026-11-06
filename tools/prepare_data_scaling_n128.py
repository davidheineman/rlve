"""Freeze an additive 128-env panel without changing the running 64-env sweep."""

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prepare_data_scaling import ROOT, OUTPUT as BASE, freeze, generate
from make_rlve_eval_set import _records_for_environment

OUTPUT = ROOT / "data/DATA-SCALING-V1-N128"
SEED = 20261024


def build_plan(names):
    original = json.loads((BASE / "run_plan.json").read_text())
    runs = copy.deepcopy(original["runs"][:64])
    for index, name in enumerate(names[64:], 64):
        run = copy.deepcopy(runs[0])
        run.update(label=f"{index:02d}-{name}", environments=[name],
                   eval=f"data/DATA-SCALING-V1-N128/{name}_d01.json")
        runs.append(run)
    for kind in ("rl", "mopd"):
        run = copy.deepcopy(next(r for r in original["runs"] if r["label"] == f"n64-{kind}"))
        run.update(label=f"n128-{kind}", size=128, environments=names,
                   eval="data/DATA-SCALING-V1-N128/id_n128_d01.json")
        if kind == "mopd":
            # 128 BF16 teachers need ~424 GiB before GPU-cache residency,
            # plus student optimizer state and temporary loading buffers.
            run["memory_gib"] = 512
        runs.append(run)
    return {"runs": runs, "shared_teacher_count": 64,
            "new_jobs": 66, "initialization": original["initialization"],
            "scheduling": "64 shared current-sweep teachers; 64 new independent 1-GPU teachers; independent 4-GPU RL; 4-GPU MOPD depends on its 128 teachers only"}


def check():
    from collections import Counter
    from data_scaling_plan import build_plan as base_plan, validate

    validate(base_plan())
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    original = json.loads((BASE / "manifest.json").read_text())
    names = manifest["environments"]
    assert len(names) == len(set(names)) == 128
    assert names[:64] == original["environments"]
    assert not set(names) & set(original["test_environments"] + original["excluded_original_pool"])
    assert manifest["test_environments"] == original["test_environments"]
    for relative, digest in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    assert (OUTPUT / "n128.txt").read_text().split() == names
    for name in names[64:]:
        rows = json.loads((OUTPUT / f"{name}_d01.json").read_text())
        counts = Counter((m["environment"], m["difficulty"]) for row in rows for m in [json.loads(row["metadata"])])
        assert counts == {(name, 0): 16, (name, 1): 16}
    rows = json.loads((OUTPUT / "id_n128_d01.json").read_text())
    metadata = [json.loads(row["metadata"]) for row in rows]
    assert len(rows) == 32
    assert len({m["environment"] for m in metadata}) == 32
    assert {m["environment"] for m in metadata} <= set(names)
    assert Counter(m["difficulty"] for m in metadata) == {0: 16, 1: 16}
    plan = json.loads((OUTPUT / "run_plan.json").read_text())
    assert plan == build_plan(names)
    assert len(plan["runs"]) == 130
    print("Verified 64 additional teachers, nested N128 panel, 32 ID + unchanged 32 unseen-test prompts, and unchanged 4k/16k/64k budgets")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--candidate-timeout", type=int, default=90)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(generate(args.worker[0], int(args.worker[1]))))
        return
    if args.check_only or (OUTPUT / "manifest.json").exists():
        check()
        return
    from Gym.environments import identifier2environment
    from Gym.parameter_controllers import identifier2controller

    original = json.loads((BASE / "manifest.json").read_text())
    excluded = set(original["environments"] + original["test_environments"] + original["excluded_original_pool"])
    candidates = sorted(set(identifier2environment) & set(identifier2controller) - excluded)
    random.Random(SEED).shuffle(candidates)
    selected, skipped, records = [], [], {}

    def worker(item):
        name, index = item
        try:
            result = subprocess.run([sys.executable, __file__, "--worker", name, str(index)],
                                    capture_output=True, text=True, check=True, timeout=args.candidate_timeout)
            rows = json.loads(result.stdout.splitlines()[-1])
            assert len(rows) == 32
            return name, rows, None
        except (subprocess.SubprocessError, ValueError, AssertionError) as error:
            return name, None, str(error)[-1000:]

    # Bound CPU work; consume in shuffled order so completion timing cannot
    # alter which environments enter the panel.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(candidates), 4):
            batch = [(name, 128 + start + offset) for offset, name in enumerate(candidates[start:start + 4])]
            for name, rows, error in pool.map(worker, batch):
                if error:
                    skipped.append({"environment": name, "reason": error})
                    print(f"Skipped {name}: could not freeze scoreable d0/d1 eval", flush=True)
                else:
                    selected.append(name)
                    records[name] = rows
                    print(f"Prepared {len(selected)}/64 additional envs: {name}", flush=True)
                if len(selected) == 64:
                    break
            if len(selected) == 64:
                break
    if len(selected) != 64:
        raise RuntimeError(f"Only {len(selected)} additional scoreable environments available")
    names = original["environments"] + selected
    for name in selected:
        freeze(OUTPUT / f"{name}_d01.json", records[name])
    freeze(OUTPUT / "n128.txt", " ".join(names) + "\n")
    rng = random.Random(SEED + 128)
    draws = rng.sample(names, 32)
    levels = [0] * 16 + [1] * 16
    rng.shuffle(levels)
    rows = []
    for index, (name, difficulty) in enumerate(zip(draws, levels, strict=True)):
        generated = _records_for_environment(name, difficulty, 1, 100_000_000 + index * 1000, SEED + index)
        meta = json.loads(generated[0]["metadata"])
        meta["difficulty"] = difficulty
        generated[0]["metadata"] = json.dumps(meta)
        rows.extend(generated)
    freeze(OUTPUT / "id_n128_d01.json", rows)
    freeze(OUTPUT / "run_plan.json", build_plan(names))
    paths = [BASE / "manifest.json", BASE / "run_plan.json", BASE / "test_32env_d01.json",
             *OUTPUT.glob("*.json"), *OUTPUT.glob("*.txt")]
    manifest = {"seed": SEED, "environments": names, "additional_environments": selected,
                "test_environments": original["test_environments"], "excluded_candidates": skipped,
                "selection": "Keep current 64 training environments and 32 unseen test environments; seeded shuffle of remaining registered scoreable environments for 64 additional teachers",
                "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}}
    freeze(OUTPUT / "manifest.json", manifest)
    check()


if __name__ == "__main__":
    main()
