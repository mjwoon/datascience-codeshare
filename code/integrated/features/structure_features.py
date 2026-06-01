"""
structure_features.py — C features: structural route/commodity features.

build_structure_features(train_df) → DataFrame indexed (origin, destination, commodity)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COMM = ["origin", "destination", "commodity"]
KEY_ROUTE = ["origin", "destination"]


def build_structure_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute structural features at OD×commodity level.

    Features
    --------
    n_commodities    : number of active commodities on this route
    modal_dist_band  : label-encoded modal distance band for the route
    tons_cv_route    : route-level coefficient of variation (std/mean across years)
    large_route_flag : 1 if route total tons in top 20%
    intrastate_flag  : 1 if origin == destination

    Returns
    -------
    DataFrame indexed (origin, destination, commodity)
    """
    # All (origin, destination, commodity) combinations
    entities = train_df[KEY_COMM].drop_duplicates().copy()

    # ── n_commodities (route level, broadcast)
    n_comm = (
        train_df.groupby(KEY_ROUTE)["commodity"]
        .nunique()
        .reset_index()
        .rename(columns={"commodity": "n_commodities"})
    )

    # ── modal_dist_band (route modal by total tons, label-encoded)
    db = (
        train_df.groupby(KEY_ROUTE + ["distance_band"], as_index=False)["tons"]
        .sum()
        .sort_values("tons", ascending=False)
        .groupby(KEY_ROUTE, as_index=False)
        .first()[KEY_ROUTE + ["distance_band"]]
        .rename(columns={"distance_band": "modal_dist_band"})
    )
    cats = sorted(db["modal_dist_band"].astype(str).fillna("_na_").unique())
    ct = pd.CategoricalDtype(categories=cats, ordered=False)
    db["modal_dist_band"] = (
        db["modal_dist_band"].astype(str).fillna("_na_").astype(ct).cat.codes
    )

    # ── tons_cv_route: route-level CV across years
    route_year = (
        train_df.groupby(KEY_ROUTE + ["year"], as_index=False)["tons"].sum()
    )
    route_stats = route_year.groupby(KEY_ROUTE)["tons"].agg(["mean", "std"]).reset_index()
    route_stats.columns = KEY_ROUTE + ["_route_mean", "_route_std"]
    route_stats["tons_cv_route"] = (
        route_stats["_route_std"] / route_stats["_route_mean"].replace(0, np.nan)
    ).fillna(0.0)

    # ── large_route_flag: top 20% by total tons
    route_total = (
        train_df.groupby(KEY_ROUTE, as_index=False)["tons"]
        .sum()
        .rename(columns={"tons": "_route_total"})
    )
    threshold = route_total["_route_total"].quantile(0.80)
    route_total["large_route_flag"] = (
        route_total["_route_total"] >= threshold
    ).astype(int)

    # ── intrastate_flag
    entities["intrastate_flag"] = (
        entities["origin"] == entities["destination"]
    ).astype(int)

    # ── Merge everything
    out = entities.copy()
    out = out.merge(n_comm, on=KEY_ROUTE, how="left")
    out = out.merge(db, on=KEY_ROUTE, how="left")
    out = out.merge(
        route_stats[KEY_ROUTE + ["tons_cv_route"]], on=KEY_ROUTE, how="left"
    )
    out = out.merge(
        route_total[KEY_ROUTE + ["large_route_flag"]], on=KEY_ROUTE, how="left"
    )

    out["n_commodities"]   = out["n_commodities"].fillna(1).astype(int)
    out["modal_dist_band"] = out["modal_dist_band"].fillna(-1).astype(int)
    out["tons_cv_route"]   = out["tons_cv_route"].fillna(0.0)
    out["large_route_flag"] = out["large_route_flag"].fillna(0).astype(int)
    out["intrastate_flag"]  = out["intrastate_flag"].fillna(0).astype(int)

    return out.set_index(KEY_COMM)[
        ["n_commodities", "modal_dist_band", "tons_cv_route",
         "large_route_flag", "intrastate_flag"]
    ]
