"""Balanced groups minimizing mean transfer between distinct member tasks."""

import numpy as np

from lora_clustering.balanced import _matrix, balanced_metrics, candidate_ks


def summarize(transfer, groups):
    matrix = _matrix(transfer)
    result = balanced_metrics(matrix, groups)
    within = [row["within"] for row in result["per_cluster"]]
    pairs = [
        (i, j) for group in groups for p, i in enumerate(group) for j in group[p + 1 :]
    ]
    result.update(
        maximum_within=float(max(within)),
        negative_clusters=sum(value < 0 for value in within),
        both_directions_negative_pair_fraction=float(
            np.mean([matrix[i, j] < 0 and matrix[j, i] < 0 for i, j in pairs])
        ),
    )
    return result


def fit_low_transfer(transfer, k, *, restarts=256, seed=1234):
    """Minimize average within transfer, with equal sizes and no self-transfer.

    Pairs use exact minimum-weight perfect matching on the mutual mean. Larger
    groups use multi-start best-improving swaps; they have no optimality claim.
    This mean objective can sacrifice some groups; per-group extremes and the
    fraction of negative groups are therefore reported separately.
    """
    matrix = _matrix(transfer)
    n = len(matrix)
    if k not in candidate_ks(n) or restarts < 1:
        raise ValueError(
            "Expected a nontrivial balanced partition and positive restarts"
        )
    size = n // k
    affinity = (matrix + matrix.T) / 2
    if size == 2:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        # Integer weights avoid floating point blossom arithmetic. This
        # resolution is much finer than the saved evaluations' quantization.
        graph.add_weighted_edges_from(
            (i, j, int(round(1e9 * affinity[i, j])))
            for i in range(n)
            for j in range(i + 1, n)
        )
        groups = sorted(
            sorted(edge) for edge in nx.min_weight_matching(graph, weight="weight")
        )
        if len(groups) != k:
            raise RuntimeError("Pair matching did not cover every task")
        solver = "minimum-weight perfect matching; integer weight precision 1e-9"
    else:
        rng = np.random.default_rng(seed)
        all_i, all_j = np.triu_indices(n, 1)
        best, best_value = None, np.inf
        for _ in range(restarts):
            labels = np.repeat(np.arange(k), size)
            rng.shuffle(labels)
            sums = np.stack(
                [affinity[:, labels == g].sum(axis=1) for g in range(k)], axis=1
            )
            for _iteration in range(200):
                different = labels[all_i] != labels[all_j]
                i, j = all_i[different], all_j[different]
                a, b = labels[i], labels[j]
                delta = (
                    sums[j, a]
                    - affinity[i, j]
                    - sums[i, a]
                    + sums[i, b]
                    - affinity[i, j]
                    - sums[j, b]
                )
                chosen = int(delta.argmin())
                if delta[chosen] >= -1e-12:
                    break
                x, y, first, second = i[chosen], j[chosen], a[chosen], b[chosen]
                sums[:, first] += affinity[:, y] - affinity[:, x]
                sums[:, second] += affinity[:, x] - affinity[:, y]
                labels[x], labels[y] = second, first
            value = float(sum(sums[labels == g, g].sum() for g in range(k)) / 2)
            if value < best_value - 1e-12:
                best_value = value
                best = sorted(
                    sorted(np.flatnonzero(labels == g).tolist()) for g in range(k)
                )
        groups = best
        solver = "multi-start best-improving balanced swaps"
    return {"groups": groups, "solver": solver, **summarize(matrix, groups)}


def evaluate_fold(cube, k, held, *, restarts=256, seed=1234):
    """Freeze assignments using training prompts before scoring held prompts."""
    cube = np.asarray(cube, dtype=float)
    held = list(held)
    training = [i for i in range(cube.shape[-1]) if i not in held]
    if (
        not held
        or not training
        or len(set(held)) != len(held)
        or not set(held) <= set(range(cube.shape[-1]))
    ):
        raise ValueError(
            "Expected distinct held prompt indices and remaining training prompts"
        )
    fit = fit_low_transfer(
        cube[:, :, training].mean(axis=-1), k, restarts=restarts, seed=seed
    )
    return {
        "held_prompt_indices": held,
        "groups": fit["groups"],
        "train": fit,
        "held_out": summarize(cube[:, :, held].mean(axis=-1), fit["groups"]),
    }
