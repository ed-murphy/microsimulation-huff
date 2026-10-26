"""Trip simulation and aggregation to store sales."""

import numpy as np


def expected_sales(probs, trips, spend):
    """Deterministic forecast: probabilities x trips x spend, summed over households."""
    return (probs * trips * spend).sum(axis=0)


def simulate_trips(probs, trips, rng):
    """For each household, draw `trips` store choices from its probabilities.

    Returns an (n_households, n_stores) array of trip counts.
    """
    return np.array([rng.multinomial(trips, row) for row in probs])


def simulated_sales(probs, trips, spend, rng):
    """One random realization of store sales."""
    return (simulate_trips(probs, trips, rng) * spend).sum(axis=0)
