def rollouts_for_accuracy(initial_size, thresholds, sizes, accuracy):
    """Return the rollout count selected by a measured validation accuracy."""
    if accuracy is None:
        return initial_size
    selected = initial_size
    for threshold, size in zip(thresholds, sizes):
        if accuracy >= threshold:
            selected = size
    return selected
