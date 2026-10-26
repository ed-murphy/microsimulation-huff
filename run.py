"""Run one scenario end to end and print a small results table.

Usage: python run.py [--trips 50] [--alpha 1] [--beta 2] [--seed 42]
"""

import argparse

import numpy as np
import pandas as pd

import data
import huff
import sim


def hand_check():
    """One household, two stores: A (size 10, 2 away) vs B (size 20, 4 away) -> 2/3, 1/3."""
    dist = np.array([[2.0, 4.0]])
    probs = huff.probabilities(huff.pulls(dist, np.array([10.0, 20.0])))
    assert np.allclose(probs, [[2 / 3, 1 / 3]]), probs
    print("Hand check passed: P(A)=2/3, P(B)=1/3\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=int, default=data.TRIPS_PER_YEAR)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hand_check()

    households = data.make_households()
    stores = data.make_stores()

    # The Huff model and simulation work on plain arrays; labels are added back afterwards.
    dist = huff.distances(households[["x", "y"]].to_numpy(), stores[["x", "y"]].to_numpy())
    probs = huff.probabilities(huff.pulls(dist, stores["size"].to_numpy(), args.alpha, args.beta))

    rng = np.random.default_rng(args.seed)
    expected = sim.expected_sales(probs, args.trips, data.SPEND_PER_TRIP)
    simulated = sim.simulated_sales(probs, args.trips, data.SPEND_PER_TRIP, rng)

    probs = pd.DataFrame(probs, index=households.index, columns=stores.index)
    results = stores[["size"]].assign(expected=expected, simulated=simulated)
    results["gap"] = results["simulated"] / results["expected"] - 1

    print(f"alpha={args.alpha}  beta={args.beta}  trips/household={args.trips}\n")
    print("Household probabilities:")
    table = households.round(1).join(probs.round(2).add_prefix("P(").add_suffix(")"))
    print(table.to_string())

    print("\nStore sales:")
    print(results.to_string(formatters={
        "size": "{:.0f}".format,
        "expected": "{:,.0f}".format,
        "simulated": "{:,.0f}".format,
        "gap": "{:+.1%}".format,
    }))
    print(f"Total expected {results['expected'].sum():,.0f}, simulated {results['simulated'].sum():,.0f}")


if __name__ == "__main__":
    main()
