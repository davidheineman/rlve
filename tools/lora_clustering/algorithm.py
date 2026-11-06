"""A small signed graph objective, with no assumed number of task clusters."""

import random

import numpy as np


def members_label(members):
    return (
        f"individual-{members[0]:02d}"
        if len(members) == 1
        else "members-" + "-".join(map(str, sorted(members)))
    )


def signed_hierarchy(transfer):
    """Maximize sum of within-cluster signed edges by greedy agglomeration.

    transfer[i,j] is adapter i's accuracy on task j minus base accuracy on j.
    A positive edge rewards grouping; a negative edge penalizes grouping.
    Diagonal improvements do not affect assignments. Stop before a merge
    decreases the objective, but retain the remaining path as fixed-k controls.
    """
    transfer = np.asarray(transfer, dtype=float)
    if (
        transfer.ndim != 2
        or transfer.shape[0] != transfer.shape[1]
        or not np.isfinite(transfer).all()
    ):
        raise ValueError("Expected a finite square transfer matrix")
    edges = (transfer + transfer.T) / 2
    np.fill_diagonal(edges, 0)
    groups = [(i,) for i in range(len(edges))]
    partitions = {len(groups): [list(g) for g in groups]}
    merges, automatic = [], None
    while len(groups) > 1:
        gain, i, j = max(
            (float(edges[np.ix_(groups[i], groups[j])].sum()), i, j)
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
        )
        if automatic is None and gain <= 1e-12:
            automatic = [list(g) for g in groups]
        merged = tuple(sorted(groups[i] + groups[j]))
        merges.append({"left": list(groups[i]), "right": list(groups[j]), "gain": gain})
        groups = sorted([g for n, g in enumerate(groups) if n not in (i, j)] + [merged])
        partitions[len(groups)] = [list(g) for g in groups]
    return {
        "automatic": automatic if automatic is not None else partitions[1],
        "partitions": partitions,
        "merges": merges,
        "affinity": edges.tolist(),
    }


def random_partition(groups, seed):
    indices = list(range(sum(map(len, groups))))
    random.Random(seed).shuffle(indices)
    result, offset = [], 0
    for group in groups:
        result.append(sorted(indices[offset : offset + len(group)]))
        offset += len(group)
    return sorted(result)


def partition_metrics(groups, matrix, base):
    """Use equal task weights; outside transfer is first averaged per task."""
    matrix, base = np.asarray(matrix), np.asarray(base)
    n = len(base)
    if sorted(i for g in groups for i in g) != list(range(n)) or matrix.shape != (
        len(groups),
        n,
    ):
        raise ValueError("Partition must cover every environment exactly once")
    within, outside, assigned = [], [], []
    for task in range(n):
        own = next(i for i, g in enumerate(groups) if task in g)
        assigned.append(float(matrix[own, task]))
        within.append(float(matrix[own, task] - base[task]))
        other = [matrix[i, task] - base[task] for i in range(len(groups)) if i != own]
        if other:
            outside.append(float(np.mean(other)))
    within_gain = float(np.mean(within))
    between_gain = float(np.mean(outside)) if outside else None
    return {
        "k": len(groups),
        "assigned_accuracy": float(np.mean(assigned)),
        "within_transfer": within_gain,
        "between_transfer": between_gain,
        "separation": within_gain - between_gain if between_gain is not None else None,
        "per_environment_assigned_accuracy": assigned,
    }
