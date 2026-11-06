"""Validation plateau bookkeeping shared by the training driver and tests."""


def update_plateau(best, stale_evals, accuracy, min_delta):
    if accuracy is None:
        return best, stale_evals, False
    if best is None or accuracy > best + min_delta:
        return accuracy, 0, True
    return best, stale_evals + 1, False
