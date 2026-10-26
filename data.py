"""Creates the tiny synthetic market: 10 households and 3 stores on a 10x10 grid."""

import numpy as np
import pandas as pd

TRIPS_PER_YEAR = 50
SPEND_PER_TRIP = 40.0


def make_households(n=10, seed=0):
    """Return a DataFrame of household (x, y) locations on a 10x10 grid, indexed by household."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, size=(n, 2))
    return pd.DataFrame(xy, columns=["x", "y"]).rename_axis("household")


def make_stores():
    """Return a DataFrame of store (x, y) locations and sizes, indexed by store name."""
    return pd.DataFrame(
        {
            "x": [2.0, 8.0, 5.0],
            "y": [2.0, 3.0, 8.0],
            "size": [10.0, 20.0, 15.0],
        },
        index=pd.Index(["A", "B", "C"], name="store"),
    )
