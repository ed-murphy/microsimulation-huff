"""The Huff model: distances, pulls, and choice probabilities."""

import numpy as np

# A household standing on a store would have distance 0 and infinite pull.
MIN_DISTANCE = 0.1


def distances(households, stores):
    """Euclidean distance from every household to every store, shape (n_households, n_stores)."""
    diff = households[:, None, :] - stores[None, :, :]
    return np.maximum(np.sqrt((diff ** 2).sum(axis=2)), MIN_DISTANCE)


def pulls(dist, sizes, alpha=1.0, beta=2.0):
    """Attraction of each store to each household: size^alpha / distance^beta."""
    return sizes[None, :] ** alpha / dist ** beta


def probabilities(pull):
    """Normalize each household's pulls so its row sums to 1."""
    return pull / pull.sum(axis=1, keepdims=True)
