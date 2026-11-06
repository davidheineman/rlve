"""Budgeted balanced swaps accepted only by measured composed-group accuracy."""

import hashlib
import json

import numpy as np


def canonical(groups):
    groups = sorted(sorted(map(int, group)) for group in groups)
    n = sum(map(len, groups))
    if (
        not groups
        or len({len(g) for g in groups}) != 1
        or sorted(i for g in groups for i in g) != list(range(n))
    ):
        raise ValueError("Expected equal groups covering every task exactly once")
    return groups


def partition_key(groups):
    return hashlib.sha256(
        json.dumps(canonical(groups), separators=(",", ":")).encode()
    ).hexdigest()[:20]


def random_partition(n, k, seed):
    if n % k:
        raise ValueError("k must divide the task count")
    order = np.random.default_rng(seed).permutation(n)
    return canonical([order[i : i + n // k].tolist() for i in range(0, n, n // k)])


def proposals(groups, transfer, *, maximize, rng, seen, count=4):
    groups = canonical(groups)
    affinity = (np.asarray(transfer) + np.asarray(transfer).T) / 2
    candidates = []
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            for x in groups[a]:
                for y in groups[b]:
                    delta = sum(
                        affinity[y, z] - affinity[x, z] for z in groups[a] if z != x
                    )
                    delta += sum(
                        affinity[x, z] - affinity[y, z] for z in groups[b] if z != y
                    )
                    candidates.append((float(delta), a, b, x, y))
    # Half the proposals exploit the pairwise surrogate; half explore at
    # random. Acceptance always uses actual simultaneous-composition scores.
    candidates.sort(reverse=maximize)
    ordered = candidates[: max(32, count)] + [
        candidates[i] for i in rng.permutation(len(candidates))
    ]
    selected = []
    for index, (_, a, b, x, y) in enumerate(ordered):
        if index < max(32, count) and len(selected) >= count // 2:
            continue
        swapped = [list(g) for g in groups]
        swapped[a][swapped[a].index(x)] = y
        swapped[b][swapped[b].index(y)] = x
        swapped = canonical(swapped)
        key = partition_key(swapped)
        if key in seen:
            continue
        seen.add(key)
        selected.append(swapped)
        if len(selected) == count:
            break
    return selected


def refine(
    initial,
    evaluator,
    transfer,
    *,
    maximize,
    rounds=4,
    batch=4,
    seed=1234,
    on_round=None,
    evaluate_many=None,
):
    evaluate_many = evaluate_many or (lambda groups: [evaluator(g) for g in groups])
    initial = [canonical(g) for g in initial]
    measured = list(zip(evaluate_many(initial), initial))
    current_score, current = (max if maximize else min)(
        measured, key=lambda x: (x[0], partition_key(x[1]))
    )
    seen = {partition_key(g) for g in initial}
    rng = np.random.default_rng(seed)
    trace = [{"round": 0, "score": current_score, "groups": current}]
    for iteration in range(rounds):
        choices = proposals(
            current, transfer, maximize=maximize, rng=rng, seen=seen, count=batch
        )
        if not choices:
            break
        tested = list(zip(evaluate_many(choices), choices))
        score, groups = (max if maximize else min)(
            [(current_score, current), *tested],
            key=lambda x: (x[0], partition_key(x[1])),
        )
        current_score, current = score, groups
        record = {
            "round": iteration + 1,
            "score": score,
            "groups": groups,
            "proposals": [{"score": s, "groups": g} for s, g in tested],
        }
        trace.append(record)
        if on_round:
            on_round(record)
    return {"groups": current, "selection_accuracy": current_score, "trace": trace}
