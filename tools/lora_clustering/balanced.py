"""Equal-size task groups with strong mutual transfer relative to their exterior."""

import numpy as np


def candidate_ks(n):
    return [k for k in range(2, n) if n % k == 0 and n // k >= 2]


def _matrix(transfer):
    matrix = np.array(transfer, dtype=float, copy=True)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Expected a finite square transfer matrix")
    np.fill_diagonal(matrix, 0)
    return matrix


def _scores(internal, outgoing, incoming, size, n):
    within = 2 * internal / (size * (size - 1))
    out = (outgoing - 2 * internal) / (size * (n - size))
    inc = (incoming - 2 * internal) / (size * (n - size))
    # Both directions must be weaker outside. Negative exterior transfer is
    # welcome but unnecessary. The raw internal gain must also be positive.
    gap = within - np.maximum(out, inc)
    quality = np.minimum(within, gap).min(axis=-1)
    average_gap = (within - (out + inc) / 2).mean(axis=-1)
    return quality, average_gap


def balanced_metrics(transfer, groups):
    matrix = _matrix(transfer)
    n, k = len(matrix), len(groups)
    if k not in candidate_ks(n) or sorted(i for g in groups for i in g) != list(
        range(n)
    ):
        raise ValueError("Expected a nontrivial partition covering every task once")
    if len({len(g) for g in groups}) != 1:
        raise ValueError("All clusters must have exactly the same size")
    size = len(groups[0])
    block = np.empty((k, k))
    details = []
    for c, group in enumerate(groups):
        for d, other in enumerate(groups):
            values = matrix[np.ix_(group, other)]
            block[c, d] = (
                values.sum() / (size * (size - 1)) if c == d else values.mean()
            )
        outside = [i for i in range(n) if i not in group]
        within = block[c, c]
        outgoing = float(matrix[np.ix_(group, outside)].mean())
        incoming = float(matrix[np.ix_(outside, group)].mean())
        gap = within - max(outgoing, incoming)
        details.append(
            {
                "within": float(within),
                "outgoing_between": outgoing,
                "incoming_between": incoming,
                "gap": float(gap),
            }
        )
    within = float(np.trace(block) / k)
    between = float(block[~np.eye(k, dtype=bool)].mean())
    return {
        "k": k,
        "size": size,
        "within": within,
        "between": between,
        "mean_gap": within - between,
        "weakest_quality": min(min(d["within"], d["gap"]) for d in details),
        "minimum_within": min(d["within"] for d in details),
        "minimum_gap": min(d["gap"] for d in details),
        "block_matrix": block.tolist(),
        "per_cluster": details,
    }


def random_balanced_baseline(transfer_folds, k, *, draws=10000, seed=1234):
    """Unoptimized equal-size partitions, independently drawn for each fold.

    Intervals describe variation across random assignments of these fixed
    observations, not sampling uncertainty or an independent confirmation.
    """
    matrices = np.stack([_matrix(fold) for fold in transfer_folds])
    n = matrices.shape[1]
    if n < 2 or k < 2 or k > n or n % k or draws < 1:
        raise ValueError(
            "Expected a valid equal-size partition and positive draw count"
        )
    size = n // k
    within_count, between_count = n * (size - 1), n * (n - size)
    expected = float(matrices.sum(axis=(1, 2)).mean() / (n * (n - 1)))
    rng = np.random.default_rng(seed)
    internal = np.zeros(draws)
    external = np.zeros(draws)
    for matrix in matrices:
        for start in range(0, draws, 256):
            stop = min(start + 256, draws)
            labels = np.stack([rng.permutation(n) // size for _ in range(stop - start)])
            same = labels[:, :, None] == labels[:, None, :]
            sums = np.einsum("bij,ij->b", same, matrix)
            internal[start:stop] += sums
            external[start:stop] += matrix.sum() - sums
    external /= len(matrices) * between_count

    def summarize(values, mean):
        return {
            "expected_mean": mean,
            "sampled_mean": float(values.mean()),
            "assignment_interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
        }

    result = {
        "draws": draws,
        "seed": seed,
        "folds": len(matrices),
        "method": "independent uniform equal-size random partition per held-out fold; average fold scores",
        "within": None,
        "between": summarize(external, expected),
        "gap": None,
    }
    if within_count:
        internal /= len(matrices) * within_count
        result["within"] = summarize(internal, expected)
        result["gap"] = summarize(internal - external, 0.0)
    return result


def fit_balanced(transfer, k, *, restarts=256, seed=1234):
    """Multi-start improving swaps; optimize the weakest cluster, then mean gap.

    Each move swaps two tasks, preserving exact sizes. Self-transfer is zeroed
    before any score is computed. This is a heuristic, not a global optimum claim.
    """
    matrix = _matrix(transfer)
    n = len(matrix)
    if k not in candidate_ks(n) or restarts < 1:
        raise ValueError("Invalid k or restart count")
    size = n // k
    affinity = (matrix + matrix.T) / 2
    row, column = matrix.sum(axis=1), matrix.sum(axis=0)
    rng = np.random.default_rng(seed)
    best = None
    best_score = (-np.inf, -np.inf)
    all_i, all_j = np.triu_indices(n, 1)
    for _ in range(restarts):
        labels = np.repeat(np.arange(k), size)
        rng.shuffle(labels)
        sums = np.stack(
            [affinity[:, labels == g].sum(axis=1) for g in range(k)], axis=1
        )
        internal = np.array([sums[labels == g, g].sum() / 2 for g in range(k)])
        outgoing = np.array([row[labels == g].sum() for g in range(k)])
        incoming = np.array([column[labels == g].sum() for g in range(k)])
        score = tuple(float(x) for x in _scores(internal, outgoing, incoming, size, n))
        for _iteration in range(200):
            different = labels[all_i] != labels[all_j]
            i, j = all_i[different], all_j[different]
            a, b = labels[i], labels[j]
            index = np.arange(len(i))
            u = np.tile(internal, (len(i), 1))
            out = np.tile(outgoing, (len(i), 1))
            inc = np.tile(incoming, (len(i), 1))
            u[index, a] += sums[j, a] - affinity[i, j] - sums[i, a]
            u[index, b] += sums[i, b] - affinity[i, j] - sums[j, b]
            for totals, values in [(out, row), (inc, column)]:
                change = values[j] - values[i]
                totals[index, a] += change
                totals[index, b] -= change
            primary, secondary = _scores(u, out, inc, size, n)
            choices = np.flatnonzero(primary >= primary.max() - 1e-12)
            chosen = choices[secondary[choices].argmax()]
            proposed = (float(primary[chosen]), float(secondary[chosen]))
            improves = proposed[0] > score[0] + 1e-12 or (
                abs(proposed[0] - score[0]) <= 1e-12 and proposed[1] > score[1] + 1e-12
            )
            if not improves:
                break
            x, y, first, second = i[chosen], j[chosen], a[chosen], b[chosen]
            sums[:, first] += affinity[:, y] - affinity[:, x]
            sums[:, second] += affinity[:, x] - affinity[:, y]
            labels[x], labels[y] = second, first
            internal, outgoing, incoming = u[chosen], out[chosen], inc[chosen]
            score = proposed
        if score > best_score:
            best_score = score
            best = sorted(
                [sorted(np.flatnonzero(labels == g).tolist()) for g in range(k)]
            )
    return {"groups": best, **balanced_metrics(matrix, best)}


def select_k(records):
    """Prefer fewer groups when their held-out scores are within one SE of best."""
    best = max(records, key=lambda r: r["cv_gap_mean"])
    threshold = best["cv_gap_mean"] - best["cv_gap_se"]
    eligible = [
        r
        for r in records
        if r["cv_gap_mean"] >= threshold and r["fit"]["weakest_quality"] > 0
    ]
    selected = min(eligible, key=lambda r: r["k"]) if eligible else best
    return {
        "k": selected["k"],
        "best_cv_k": best["k"],
        "one_se_threshold": threshold,
        "supported_by_positive_cv_gap": selected["cv_gap_mean"] > 0,
        "all_clusters_consistent_in_cv": selected["cv_quality_mean"] > 0,
        "rule": "smallest k within one SE of the best held-out mean gap; require positive weakest-cluster quality in the development fit",
    }
