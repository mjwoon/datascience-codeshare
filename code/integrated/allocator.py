"""
allocator.py — Phase 4: Route aggregation of commodity-level predictions.

route_aggregate(comm_preds) → pd.Series indexed (origin, destination)
"""
from __future__ import annotations

import pandas as pd

KEY_ROUTE = ["origin", "destination"]


def route_aggregate(comm_preds: pd.Series) -> pd.Series:
    """
    Sum OD×commodity predictions to route level.

    Parameters
    ----------
    comm_preds : pd.Series indexed (origin, destination, commodity)

    Returns
    -------
    pd.Series indexed (origin, destination)
    """
    return comm_preds.clip(lower=0).groupby(level=[0, 1]).sum()
