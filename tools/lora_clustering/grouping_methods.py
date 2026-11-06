"""CPU-only equal-size task grouping methods, fitted without held-out outcomes."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from lora_clustering.low_transfer import fit_low_transfer


METHODS = {
    "mean_swaps": "Mean-transfer swaps",
    "annealed_swaps": "Simulated annealing",
    "spectral_signed": "Signed spectral",
    "spectral_positive": "Positive spectral",
    "spectral_recursive": "Spectral bisection",
    "profile_kmeans": "Transfer-profile k-means",
    "profile_cosine": "Cosine-profile k-means",
    "profile_svd": "SVD-profile k-means",
    "hierarchy_average": "Average-link hierarchy",
    "hierarchy_complete": "Complete-link hierarchy",
    "hierarchy_single": "Single-link hierarchy",
    "risk_swaps": "Uncertainty-penalized swaps",
}
PARAMETERS = {
    "mean_swap_restarts": 256,
    "risk_swap_restarts": 128,
    "kmeans_restarts": 16,
    "kmeans_iterations": 40,
    "spectral_dimensions_cap": 16,
    "svd_dimensions": 8,
    "annealing_restarts": 12,
    "annealing_moves_per_restart": 6000,
    "risk_prompt_standard_error_multiplier": 1.0,
}


def canonical(groups):
    return sorted(sorted(int(i) for i in group) for group in groups)


def validate(groups, n, k):
    if (len(groups) != k or any(len(g) != n // k for g in groups)
            or sorted(i for g in groups for i in g) != list(range(n))):
        raise ValueError("Every method must return an exactly balanced full partition")


def affinity(matrix):
    result = (np.asarray(matrix) + np.asarray(matrix).T) / 2
    result = result.copy()
    np.fill_diagonal(result, 0)
    return result


def labels_for(groups, n):
    labels = np.empty(n, dtype=int)
    for c, group in enumerate(groups):
        labels[group] = c
    return labels


def swap_polish(weights, groups):
    """Best-improving capacity-preserving swaps for a symmetric edge objective."""
    n, k = len(weights), len(groups)
    labels = labels_for(groups, n)
    sums = np.stack([weights[:, labels == c].sum(axis=1) for c in range(k)], axis=1)
    left, right = np.triu_indices(n, 1)
    for _ in range(500):
        different = labels[left] != labels[right]
        i, j = left[different], right[different]
        a, b = labels[i], labels[j]
        changes = (sums[j, a] - weights[i, j] - sums[i, a]
                   + sums[i, b] - weights[i, j] - sums[j, b])
        at = int(changes.argmax())
        if changes[at] <= 1e-12:
            break
        x, y, first, second = i[at], j[at], a[at], b[at]
        sums[:, first] += weights[:, y] - weights[:, x]
        sums[:, second] += weights[:, x] - weights[:, y]
        labels[x], labels[y] = second, first
    return canonical([np.flatnonzero(labels == c).tolist() for c in range(k)])


def balanced_kmeans(features, k, seed, *, restarts=None):
    """Lloyd iterations with exact-capacity linear assignment at each step."""
    features = np.asarray(features, dtype=float)
    n, size = len(features), len(features) // k
    slots = np.repeat(np.arange(k), size)
    rng = np.random.default_rng(seed)
    best, best_loss = None, np.inf
    for _ in range(restarts or PARAMETERS["kmeans_restarts"]):
        chosen = [int(rng.integers(n))]
        distance = ((features - features[chosen[0]]) ** 2).sum(axis=1)
        while len(chosen) < k:
            distance[chosen] = 0
            next_point = (int(rng.choice(n, p=distance / distance.sum()))
                          if distance.sum() > 1e-15 else
                          int(rng.choice([i for i in range(n) if i not in chosen])))
            chosen.append(next_point)
            distance = np.minimum(distance, ((features - features[next_point]) ** 2).sum(axis=1))
        centers = features[chosen].copy()
        previous = None
        for _iteration in range(PARAMETERS["kmeans_iterations"]):
            costs = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            rows, columns = linear_sum_assignment(costs[:, slots])
            labels = np.empty(n, dtype=int)
            labels[rows] = slots[columns]
            centers = np.stack([features[labels == c].mean(axis=0) for c in range(k)])
            loss = float(((features - centers[labels]) ** 2).sum())
            if loss < best_loss - 1e-12:
                best_loss = loss
                best = canonical([np.flatnonzero(labels == c).tolist() for c in range(k)])
            if previous is not None and np.array_equal(previous, labels):
                break
            previous = labels.copy()
    return best


def spectral(weights, k, seed, *, positive=False):
    n = len(weights)
    if positive:
        graph = np.maximum(weights, 0)
        degree = np.maximum(graph.sum(axis=1), 1e-12)
        graph = graph / np.sqrt(degree[:, None] * degree[None, :])
    else:
        center = np.eye(n) - np.ones((n, n)) / n
        graph = center @ weights @ center
    _, vectors = np.linalg.eigh(graph)
    dimensions = min(k, PARAMETERS["spectral_dimensions_cap"])
    features = vectors[:, -dimensions:]
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return balanced_kmeans(features, k, seed)


def spectral_bisection(weights, k):
    groups = [list(range(len(weights)))]
    while len(groups) < k:
        children = []
        for group in groups:
            matrix = weights[np.ix_(group, group)]
            n = len(group)
            center = np.eye(n) - np.ones((n, n)) / n
            _, vectors = np.linalg.eigh(center @ matrix @ center)
            order = np.argsort(vectors[:, -1], kind="stable")
            children.extend([[group[i] for i in order[:n // 2]],
                             [group[i] for i in order[n // 2:]]])
        groups = children
    return canonical(groups)


def hierarchy(weights, k, linkage):
    """Balanced agglomeration: match equal-size blocks at every tree level."""
    import networkx as nx

    groups = [[i] for i in range(len(weights))]
    reducers = {"average": np.mean, "complete": np.min, "single": np.max}
    while len(groups) > k:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(groups)))
        graph.add_weighted_edges_from(
            (i, j, int(round(1e9 * reducers[linkage](weights[np.ix_(groups[i], groups[j])]))))
            for i in range(len(groups)) for j in range(i + 1, len(groups))
        )
        matching = nx.max_weight_matching(graph, maxcardinality=True)
        if len(matching) != len(groups) // 2:
            raise RuntimeError("Balanced hierarchical matching is incomplete")
        groups = canonical([groups[i] + groups[j] for i, j in matching])
    return groups


def anneal(weights, groups, seed):
    n, k = len(weights), len(groups)
    size = n // k
    rng = np.random.default_rng(seed)
    best = canonical(groups)
    best_value = sum(weights[np.ix_(g, g)].sum() for g in best) / 2
    moves = PARAMETERS["annealing_moves_per_restart"]
    spread = max(float(weights[np.triu_indices(n, 1)].std()) * np.sqrt(size), 1e-6)
    temperatures = np.geomspace(spread, spread * .002, moves)
    for restart in range(PARAMETERS["annealing_restarts"]):
        labels = (labels_for(best, n) if restart % 3 == 0 else
                  rng.permutation(n) // size)
        sums = np.stack([weights[:, labels == c].sum(axis=1) for c in range(k)], axis=1)
        value = sum(sums[labels == c, c].sum() for c in range(k)) / 2
        for temperature in temperatures:
            x, y = rng.integers(n, size=2)
            a, b = labels[x], labels[y]
            if a == b:
                continue
            delta = (sums[y, a] - weights[x, y] - sums[x, a]
                     + sums[x, b] - weights[x, y] - sums[y, b])
            if delta >= 0 or rng.random() < np.exp(max(-700, delta / temperature)):
                sums[:, a] += weights[:, y] - weights[:, x]
                sums[:, b] += weights[:, x] - weights[:, y]
                labels[x], labels[y] = b, a
                value += delta
                if value > best_value + 1e-12:
                    best_value = value
                    best = canonical([np.flatnonzero(labels == c).tolist() for c in range(k)])
        candidate = swap_polish(weights, [np.flatnonzero(labels == c).tolist() for c in range(k)])
        candidate_value = sum(weights[np.ix_(g, g)].sum() for g in candidate) / 2
        if candidate_value > best_value + 1e-12:
            best_value, best = candidate_value, candidate
    return best


def fit_methods(cube, k, seed, *, existing=None, callback=None):
    """Input contains training prompts only; all model selection here is in-sample."""
    cube = np.asarray(cube, dtype=float)
    matrix = cube.mean(axis=-1)
    weights = affinity(matrix)
    n = len(weights)
    # Preserve source/target direction in profile methods; remove own-task cells.
    directed = matrix.copy()
    np.fill_diagonal(directed, 0)
    profiles = np.concatenate((directed, directed.T), axis=1)
    groups = dict(existing or {})
    for index, method in enumerate(METHODS):
        if method in groups:
            validate(groups[method], n, k)
            continue
        rng_seed = seed + index * 100003
        if method == "mean_swaps":
            value = fit_low_transfer(-weights, k, restarts=PARAMETERS["mean_swap_restarts"], seed=rng_seed)["groups"]
        elif method == "annealed_swaps":
            value = anneal(weights, groups["mean_swaps"], rng_seed)
        elif method in ("spectral_signed", "spectral_positive"):
            value = spectral(weights, k, rng_seed, positive=method == "spectral_positive")
        elif method == "spectral_recursive":
            value = spectral_bisection(weights, k)
        elif method.startswith("profile_"):
            features = profiles - profiles.mean(axis=0)
            if method == "profile_cosine":
                features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            elif method == "profile_svd":
                u, s, _ = np.linalg.svd(features, full_matrices=False)
                features = u[:, :PARAMETERS["svd_dimensions"]] * s[:PARAMETERS["svd_dimensions"]]
            value = balanced_kmeans(features, k, rng_seed)
        elif method.startswith("hierarchy_"):
            value = hierarchy(weights, k, method.removeprefix("hierarchy_"))
        elif method == "risk_swaps":
            samples = (cube + cube.transpose(1, 0, 2)) / 2
            error = samples.std(axis=-1, ddof=1) / np.sqrt(samples.shape[-1])
            penalized = weights - PARAMETERS["risk_prompt_standard_error_multiplier"] * error
            np.fill_diagonal(penalized, 0)
            value = fit_low_transfer(-penalized, k, restarts=PARAMETERS["risk_swap_restarts"], seed=rng_seed)["groups"]
        else:
            raise ValueError(method)
        groups[method] = canonical(value)
        validate(groups[method], n, k)
        if callback:
            callback(method, groups[method])
    return groups


def task_scores(matrix, groups, *, include_self=True):
    n = len(matrix)
    scores = np.empty(n)
    for group in groups:
        for target in group:
            sources = group if include_self else [i for i in group if i != target]
            scores[target] = np.mean(matrix[sources, target]) if sources else np.nan
    return scores


def random_task_scores(matrix, k):
    n = len(matrix)
    size = n // k
    diagonal = np.diag(matrix)
    others = (matrix.sum(axis=0) - diagonal) / (n - 1)
    return (diagonal + (size - 1) * others) / size
