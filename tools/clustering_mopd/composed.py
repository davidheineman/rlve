"""Import selection-frozen composed-search groups without reading held scores."""

import json
from pathlib import Path

from lora_teachers.plan import sha256

from clustering_mopd.plan import validate_partition

COMPOSED_SEARCH = Path(
    "/juice5b/scr5b/dhei/runs/composed-clustering-n64-20261102-150018"
)


def assignments_from_composed_search(
    directory, *, count=8, counts=None, random_replicate=0
):
    directory = Path(directory).resolve()
    plan_path, assignments_path = (
        directory / "plan.json",
        directory / "assignments.json",
    )
    plan = json.loads(plan_path.read_text())
    frozen = json.loads(assignments_path.read_text())
    if frozen.get("used_confirmation") is not False:
        raise ValueError(
            "Composed assignments must be frozen without confirmation outcomes"
        )
    environments = plan["environments"]
    if len(environments) != 64 or len(set(environments)) != 64:
        raise ValueError("Expected the original 64-environment panel")
    counts = [count] if counts is None else list(counts)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("Cluster counts must be nonempty and unique")
    candidates = []
    for count in counts:
        records = [r for r in frozen["candidates"] if r["k"] == count]
        if len(records) != 1:
            raise ValueError("Expected exactly one frozen assignment for this k")
        row = records[0]
        if not 0 <= random_replicate < len(row["random_groups"]):
            raise ValueError("Random replicate is outside the predeclared draws")
        groups = {
            "high-transfer": row["best"]["groups"],
            "random": row["random_groups"][random_replicate],
            "low-transfer": row["worst"]["groups"],
        }
        for name, value in groups.items():
            validate_partition(value, count, environments)
            groups[name] = sorted(sorted(group) for group in value)
        candidates.append({"k": count, "groups_by_assignment": groups})
    return {
        "method": "composed-search-assignments",
        "source_experiment": str(directory),
        "environments": environments,
        "provenance": {
            "plan_sha256": sha256(plan_path),
            "assignments_sha256": sha256(assignments_path),
            "frozen_at": frozen["frozen_at"],
            "confirmation_used_for_selection": False,
            "random_replicate": random_replicate,
            "random_seed": plan["random_seeds"][random_replicate],
            "random_selection": "predeclared index; no outcome-based selection",
        },
        "candidates": candidates,
    }
