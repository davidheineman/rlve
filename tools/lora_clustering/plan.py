"""Keep the established teacher recipe, changing only panel, budget and project."""

import json

from lora_teachers.plan import ROOT, build_plan, sha256

PROJECT = "rl-data-scaling-lora-teachers-v3"
DATA = "data/MOPD-CLUSTERING-V1"


def build_clustering_plan(root=ROOT, *, size=32):
    if size not in (32, 64):
        raise ValueError("Expected the original N32 or N64 MOPD panel")
    plan = build_plan(root, project=PROJECT, learning_rate=2e-5)
    source = root / "data/DATA-SCALING-V1"
    manifest = json.loads((source / "manifest.json").read_text())
    runs = json.loads((source / "run_plan.json").read_text())["runs"]
    panel = next(r for r in runs if r["label"] == f"n{size}-mopd")
    names = panel["environments"]
    if (
        names != manifest["environments"][:size]
        or len(set(names)) != size
        or names != (source / f"n{size}.txt").read_text().split()
    ):
        raise ValueError(f"Expected the original N{size} MOPD panel")
    data = DATA if size == 32 else f"{DATA}/n{size}"
    teachers = []
    for index, environment in enumerate(names):
        teacher = next(
            r
            for r in runs
            if r["kind"] == "teacher" and r["environments"] == [environment]
        )
        expected = manifest["sha256"][teacher["eval"]]
        if sha256(root / teacher["eval"]) != expected:
            raise ValueError(f"Teacher gate changed: {environment}")
        teachers.append(
            dict(
                environment=environment,
                source_label=f"teacher-{index:02d}",
                eval=teacher["eval"],
                eval_sha256=expected,
            )
        )
    plan["teacher_config"]["num-rollout"] = 1000
    plan["teacher_config_overrides"]["num-rollout"] = {"source": 500, "value": 1000}
    plan.update(
        method="lora-transfer-clustering-v1",
        size=size,
        environments=names,
        teachers=teachers,
        source_panel=f"n{size}-mopd",
        teacher_project=PROJECT,
        adapter_slots=size + 1,
        teacher_eval={t["environment"]: t["eval"] for t in teachers},
        teacher_rollouts=8000,
        teacher_validation_threshold=0.98,
        evaluation_concurrency=2,
        decode_graph_max_batch_size=64,
        checkpointing="retain latest adapter, Adam, prompt-source state and client RNG at every teacher evaluation",
        clustering={
            "selection": f"{data}/selection.json",
            "confirmation": f"{data}/confirmation.json",
            "selection_prompts_per_environment": 8,
            "selection_samples_per_prompt": 2,
            "confirmation_prompts_per_environment": 32,
            "confirmation_samples_per_prompt": 4,
            "candidate_k": [2, 4, 8],
            "seed": 1234,
            "algorithm": "greedy signed correlation clustering of symmetrized transfer relative to the base",
            "merge": "uniform mean of scaled B@A deltas, compressed by QR/SVD to rank 32",
            "selection_objective": "macro within-cluster transfer minus macro between-cluster transfer",
            "confirmation_protocol": "fresh disjoint prompts; no assignment or k changes after selection",
            "probe_difficulty_protocol": "start at levels 0/1; advance both levels only when the distinct problem space is exhausted",
            "mode": "balanced" if size == 64 else "signed",
            "balanced_restarts": 256,
            "balanced_candidate_k": [k for k in (2, 4, 8, 16, 32) if k < size],
            "balanced_confirmation": "fresh original confirmation prompts; compare selected leave-one-task-out merges, random balanced leave-one-task-out merges, and equal-size outside-cluster merges; also evaluate assigned full merges and own specialists",
        },
    )
    return plan


def prepare_evaluation(root=ROOT, *, size=32):
    # Generation/verifier dependencies are supplied by the pinned training image.
    from make_rlve_eval_set import _records_for_environment

    plan = build_clustering_plan(root, size=size)
    target = (root / plan["clustering"]["selection"]).parent
    target.mkdir(parents=True, exist_ok=True)
    outputs, levels = {"selection": [], "confirmation": []}, {}
    for index, teacher in enumerate(plan["teachers"]):
        original = {
            r["user_prompt"] for r in json.loads((root / teacher["eval"]).read_text())
        }
        for start in range(10):
            seen, generated = set(original), {}
            try:
                for split, count, seed in [
                    ("selection", 8, 20_000_000),
                    ("confirmation", 32, 30_000_000),
                ]:
                    rows = []
                    for difficulty in (start, start + 1):
                        added = _records_for_environment(
                            teacher["environment"],
                            difficulty,
                            count // 2,
                            seed + index * 10_000 + difficulty * 1000,
                            plan["seed"] + index * 100 + difficulty,
                            seen_prompts=seen,
                        )
                        for row in added:
                            row["probe_difficulty"] = difficulty
                        rows.extend(added)
                    generated[split] = rows
            except AssertionError:
                continue
            break
        else:
            raise ValueError(
                f"Cannot generate disjoint probes for {teacher['environment']}"
            )
        levels[teacher["environment"]] = [start, start + 1]
        print(
            f"CLUSTER_PROBES {teacher['environment']} levels={levels[teacher['environment']]}",
            flush=True,
        )
        for split in outputs:
            outputs[split].extend(generated[split])
    for split, rows in outputs.items():
        path = target / f"{split}.json"
        text = json.dumps(rows, indent=2) + "\n"
        if path.exists() and path.read_text() != text:
            raise ValueError(
                f"Refusing to overwrite different frozen evaluation data: {path}"
            )
        path.write_text(text)
    record = {
        split: {"rows": len(rows), "sha256": sha256(target / f"{split}.json")}
        for split, rows in outputs.items()
    }
    record["probe_difficulties"] = levels
    (target / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def validate_evaluation(plan, root=ROOT):
    original = {
        r["user_prompt"]
        for t in plan["teachers"]
        for r in json.loads((root / t["eval"]).read_text())
    }
    seen = set(original)
    plan["clustering"]["probe_difficulties"] = json.loads(
        (root / plan["clustering"]["selection"]).with_name("manifest.json").read_text()
    )["probe_difficulties"]
    for split in ("selection", "confirmation"):
        path = root / plan["clustering"][split]
        rows = json.loads(path.read_text())
        grouped = {name: [] for name in plan["environments"]}
        for row in rows:
            meta = row["metadata"]
            meta = json.loads(meta) if isinstance(meta, str) else meta
            grouped[meta["environment"]].append(row)
        expected = plan["clustering"][f"{split}_prompts_per_environment"]
        if any(len(v) != expected for v in grouped.values()):
            raise ValueError(f"Wrong per-environment count in {split}")
        prompts = {r["user_prompt"] for r in rows}
        if len(prompts) != len(rows) or prompts & seen:
            raise ValueError(f"Evaluation leakage or duplicates in {split}")
        seen.update(prompts)
        plan["clustering"][f"{split}_sha256"] = sha256(path)
    return plan
