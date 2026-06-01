"""
context_features.py — A features from context.csv.

build_context_features(context_df) → DataFrame indexed (origin, destination, commodity)
with ~6 features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COMM = ["origin", "destination", "commodity"]
KEY_ROUTE = ["origin", "destination"]


def build_context_features(context_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute context-year features at OD×commodity level.

    Features
    --------
    ctx_tons_last      : tons in context year
    ctx_tons_pct_route : commodity share of route total
    ctx_tons_log1p     : log1p(ctx_tons_last)
    ctx_value_per_ton  : value / tons (0 if tons==0)
    ctx_zero_flag      : 1 if tons == 0
    ctx_route_total    : route-level total context tons

    Returns
    -------
    DataFrame indexed (origin, destination, commodity)
    """
    # Aggregate to OD×commodity level (sum over sub-rows if any)
    ctx = (
        context_df.groupby(KEY_COMM, as_index=False)
        .agg(tons=("tons", "sum"), value=("value", "sum"))
    )

    # Route total
    route_total = (
        ctx.groupby(KEY_ROUTE, as_index=False)["tons"]
        .sum()
        .rename(columns={"tons": "ctx_route_total"})
    )
    ctx = ctx.merge(route_total, on=KEY_ROUTE, how="left")
    ctx["ctx_route_total"] = ctx["ctx_route_total"].fillna(0.0)

    ctx["ctx_tons_last"] = ctx["tons"].fillna(0.0).clip(lower=0)
    ctx["ctx_tons_pct_route"] = (
        ctx["ctx_tons_last"] / ctx["ctx_route_total"].replace(0, np.nan)
    ).fillna(0.0)
    ctx["ctx_tons_log1p"] = np.log1p(ctx["ctx_tons_last"])
    ctx["ctx_value_per_ton"] = (
        ctx["value"].fillna(0.0) / ctx["tons"].replace(0, np.nan)
    ).fillna(0.0)
    ctx["ctx_zero_flag"] = (ctx["ctx_tons_last"] == 0).astype(int)

    result = ctx.set_index(KEY_COMM)[
        ["ctx_tons_last", "ctx_tons_pct_route", "ctx_tons_log1p",
         "ctx_value_per_ton", "ctx_zero_flag", "ctx_route_total"]
    ]
    return result
