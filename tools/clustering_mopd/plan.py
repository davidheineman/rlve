"""Freeze assignment choices and matched data without creating W&B runs."""

import copy
import hashlib
import json
import random
from pathlib import Path

from lora_teachers.plan import ROOT, sha256

REFERENCE = Path("/juice5b/scr5b/dhei/runs/async-lora-n32-20261025-141924")
CLUSTERING = Path("/juice5b/scr5b/dhei/runs/mopd-clustering-n64-20261031-165345")
ASSIGNMENTS = ROOT / "plot/mopd-clustering/n64/balanced/results.json"
ASSIGNMENTS32 = ROOT / "plot/mopd-clustering/balanced/results.json"
COUNTS = (1, 4, 16, 64)


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def metadata(row):
    value = row["metadata"]
    return json.loads(value) if isinstance(value, str) else value


def validate_partition(groups, count, environments):
    n = len(environments)
    if len(groups) != count or any(len(g) != n // count for g in groups):
        raise ValueError("Expected equal-size clusters")
    if sorted(i for group in groups for i in group) != list(range(n)):
        raise ValueError("Assignments must cover every environment exactly once")


def partition(count, assignment, selection, seed):
    environments = selection["environments"]
    n = len(environments)
    if n not in (32, 64) or count < 1 or count > n or n % count or count & (count - 1):
        raise ValueError(
            "Expected a power-of-two partition of the 32- or 64-environment panel"
        )
    if selection.get("method") == "composed-search-assignments":
        record = next(r for r in selection["candidates"] if r["k"] == count)
        if assignment not in record["groups_by_assignment"]:
            raise ValueError("Unknown composed-search assignment")
        groups = copy.deepcopy(record["groups_by_assignment"][assignment])
    elif count == 1:
        groups = [list(range(n))]
    elif count == n:
        groups = [[i] for i in range(n)]
    elif assignment == "optimized":
        record = next(r for r in selection["candidates"] if r["k"] == count)
        groups = copy.deepcopy(record["fit"]["groups"])
    elif assignment == "random":
        order = list(range(n))
        random.Random(seed).shuffle(order)
        groups = [order[i : i + n // count] for i in range(0, n, n // count)]
    else:
        raise ValueError("Assignment must be optimized or random")
    groups = sorted(sorted(group) for group in groups)
    validate_partition(groups, count, environments)
    return groups


def configurations(size=64, *, composed=False, composed_counts=(8,)):
    if composed:
        if size != 64:
            raise ValueError("The composed-search comparison uses the 64-task panel")
        counts = list(composed_counts)
        if (
            not counts
            or len(set(counts)) != len(counts)
            or any(k not in (2, 4, 8, 16, 32) for k in counts)
        ):
            raise ValueError("Expected unique composed-search cluster counts")
        return [
            (k, name)
            for k in counts
            for name in ("high-transfer", "random", "low-transfer")
        ]
    if size == 32:
        # Joint assignments coincide; these are independent training repeats.
        return [
            (32, "singleton"),
            (8, "random"),
            (8, "optimized"),
            (2, "random"),
            (2, "optimized"),
            (1, "random"),
            (1, "optimized"),
        ]
    if size != 64:
        raise ValueError("Supported panels are n32 and n64")
    return [
        (1, "single"),
        (4, "random"),
        (4, "optimized"),
        (16, "random"),
        (16, "optimized"),
        (64, "singleton"),
    ]


def build_plan(
    count,
    assignment,
    *,
    devices=16,
    horizon=16000,
    seed=1234,
    reference=REFERENCE,
    clustering=CLUSTERING,
    assignments=ASSIGNMENTS,
    size=64,
    assignment_selection=None,
    student_project="rl-clustering-students",
    teacher_project="rl-clustering-teachers",
):
    if devices < 2 or horizon <= 0 or horizon % 8:
        raise ValueError(
            "Need at least two devices and an eight-aligned positive horizon"
        )
    old = json.loads((reference / "plan.json").read_text())
    if old["size"] != 32 or old.get("single_question") or old.get("opd_replay"):
        raise ValueError(
            "Reference must be the original random-question, zero-replay N32 run"
        )
    selected = (
        json.loads(assignments.read_text())
        if assignment_selection is None
        else copy.deepcopy(assignment_selection)
    )
    discovery = json.loads((clustering / "plan.json").read_text())
    environments = selected["environments"]
    hero = next(
        r
        for r in json.loads((ROOT / "data/DATA-SCALING-V1/run_plan.json").read_text())[
            "runs"
        ]
        if r["label"] == f"n{size}-mopd"
    )
    if (
        len(environments) != size
        or environments != hero["environments"]
        or environments[:32] != old["environments"]
    ):
        raise ValueError("Clustering panel and original MOPD panels differ")
    if discovery["environments"][:size] != environments:
        raise ValueError("Clustering provenance uses different environments")
    groups = partition(count, assignment, selected, seed)
    label = f"k{count:02d}-{assignment}"
    plan = copy.deepcopy(old)
    by_name = {t["environment"]: t for t in discovery["teachers"]}
    teachers = [copy.deepcopy(by_name[e]) for e in environments]
    source = clustering / "source/rlve"
    gates = {}
    for environment in environments:
        record = by_name[environment]
        path = source / record["eval"]
        if sha256(path) != record["eval_sha256"]:
            raise ValueError(f"Original teacher gate changed: {environment}")
        rows = json.loads(path.read_text())
        if len(rows) != 32 or {metadata(r)["environment"] for r in rows} != {
            environment
        }:
            raise ValueError("Expected the original 32 prompts per environment")
        gates[environment] = rows
    legacy_path = ROOT / hero["eval"]
    manifest = json.loads((ROOT / "data/DATA-SCALING-V1/manifest.json").read_text())
    if sha256(legacy_path) != manifest["sha256"][hero["eval"]]:
        raise ValueError("Original ID inputs changed")
    legacy = json.loads(legacy_path.read_text())
    legacy_by_env = {metadata(row)["environment"]: row for row in legacy}
    if len(legacy_by_env) != len(legacy):
        raise ValueError("Expected one legacy ID prompt per represented environment")
    balanced_id = (
        legacy
        if size == 32 and set(legacy_by_env) == set(environments)
        else [legacy_by_env.get(e, gates[e][0]) for e in environments]
    )
    test = json.loads((reference / "source/rlve" / old["test_eval"]).read_text())
    if set(environments) & {metadata(r)["environment"] for r in test}:
        raise ValueError("Unseen test environments overlap the training panel")
    id_eval = f"data/RL-CLUSTERING-STUDENTS-V1/id_n{size}_balanced.json"
    generated = {id_eval: balanced_id}
    clusters = []
    for index, group in enumerate(groups):
        members = [environments[i] for i in group]
        # Keep the baseline's 32-prompt budget per policy, with at least one
        # prompt per member. All supported sizes divide these budgets exactly.
        prompts = max(old["eval_prompts"], len(members))
        per_env = prompts // len(members)
        path = f"data/RL-CLUSTERING-STUDENTS-V1/{label}/cluster-{index:02d}.json"
        panel = [row for e in members for row in gates[e][:per_env]]
        generated[path] = panel
        clusters.append(
            {
                "id": f"cluster-{index:02d}",
                "index": index,
                "environment_indices": group,
                "environments": members,
                "eval": path,
                "eval_sha256": hashlib.sha256(encoded(panel)).hexdigest(),
                "eval_prompts": len(panel),
            }
        )
    lanes = min(devices - 1, count)
    initial = json.loads((reference / "base.json").read_text())
    for name in (
        "adapter/adapter_model.safetensors",
        "state/.metadata",
        "reference.json",
    ):
        if not (Path(initial["hf"]).parent / name).is_file():
            raise FileNotFoundError(f"Untrained reference bundle missing: {name}")
    plan.update(
        method="clustered-async-mopd-composed-v2"
        if selected.get("method") == "composed-search-assignments"
        else "clustered-async-mopd-v1",
        size=size,
        source_panel=f"n{size}-mopd",
        environments=environments,
        teachers=teachers,
        clusters=clusters,
        cluster_count=count,
        environments_per_cluster=size // count,
        assignment=assignment,
        assignment_aliases=["random", "optimized"]
        if count in (1, size)
        else [assignment],
        assignment_seed=seed,
        seed=old["seed"],
        schedule_seed=seed + 10000,
        teacher_worker_count=lanes,
        simulated_devices=devices,
        simulated_teacher_devices=lanes,
        simulated_student_devices=1,
        simulated_idle_devices=devices - 1 - lanes,
        simulated_clock_budget=horizon,
        student_rollouts=horizon,
        source_student_rollouts=old["student_rollouts"],
        teacher_project=teacher_project,
        student_project=student_project,
        project=teacher_project,
        physical_gpus=4,
        train_gpus=2,
        sampling_gpus=2,
        adapter_slots=lanes + 1,
        decode_graph_max_batch_size=64,
        teacher_eval={e: by_name[e]["eval"] for e in environments},
        id_eval=id_eval,
        id_eval_prompts=len(balanced_id),
        test_eval_prompts=len(test),
        id_eval_sha256=hashlib.sha256(encoded(balanced_id)).hexdigest(),
        legacy_id_eval=hero["eval"],
        legacy_id_environments=list(legacy_by_env),
        cluster_gate_policy="32 prompts per policy, or one per member for the 64-task cluster; macro accuracy",
        clock_definition="One training rollout per simulated device per tick. Evaluation is excluded, matching async-lora-n32. Common clock deadline, not equal completed student updates.",
        simulation_scope=f"Fixed one-device student and up to {devices - 1} one-device teachers; spare devices remain idle. No assumed multi-GPU speedup within one model.",
        reference_async_experiment=str(reference),
        reference_async_plan_sha256=sha256(reference / "plan.json"),
        assignment_source=str(assignments),
        assignment_source_sha256=sha256(assignments)
        if assignment_selection is None
        else hashlib.sha256(encoded(selected)).hexdigest(),
        assignment_method=selected.get("method", "pairwise-transfer-balanced-search"),
        assignment_provenance=copy.deepcopy(selected.get("provenance", {})),
        assignment_discovery_experiment=selected.get(
            "source_experiment", str(clustering)
        ),
        runtime_source_experiment=str(clustering),
        offline_assignment_cost_in_primary_clock=False,
        initial_student_reference={
            "bundle": "initial-base",
            "source_experiment": str(reference),
            "source_bundle": str(Path(initial["hf"]).parent),
            "adapter_sha256": sha256(Path(initial["hf"]) / "adapter_model.safetensors"),
            "optimizer": "fresh; import the common untrained adapter only",
        },
        checkpoint_retention="Keep all potentially needed causal student snapshots and active parents; retire consumed teacher adapters",
    )
    for key in ("elastic_generation", "single_question", "opd_replay"):
        plan.pop(key, None)
    if size == 32 and count == 1:
        repeat = int(assignment == "optimized")
        plan.update(
            assignment=f"joint-repeat{repeat + 1}",
            requested_comparison_slot=assignment,
            assignment_is_identical_across_repeats=True,
            training_repeat=repeat,
            training_seed_offset=repeat * 100_000_000,
            seed=old["seed"] + repeat,
        )
    return plan, generated
