"""
lag_trend_features.py — B features: lag/trend at OD×commodity level.

build_lag_trend_features(history_df, base_year) → DataFrame indexed (origin, destination, commodity)

Full history used (no year branching — design decision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

KEY_COMM = ["origin", "destination", "commodity"]
KEY_ROUTE = ["origin", "destination"]


def _ols_slope(values: np.ndarray, years: np.ndarray) -> float:
    """OLS slope of values ~ years."""
    if len(years) < 2:
        return 0.0
    x = years.astype(float)
    xm = x - x.mean()
    denom = (xm * xm).sum()
    if denom == 0:
        return 0.0
    return float((values.astype(float) @ xm) / denom)


def build_lag_trend_features(history_df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """
    Build lag/trend features at OD×commodity level.

    Parameters
    ----------
    history_df : DataFrame with (origin, destination, commodity, year, tons).
                 Should contain ALL available history (train + context up to base_year).
    base_year  : the "t" year (most recent year available).
                 tons_lag1 = base_year, tons_lag2 = base_year-1, tons_lag3 = base_year-2.

    Features
    --------
    tons_lag1, tons_lag2, tons_lag3
    tons_mean_3yr, tons_std_3yr
    tons_mean_all, tons_cv_all
    tons_yoy_last
    tons_trend, tons_trend_norm
    n_years_hist
    tons_lag1_route
    tons_ratio_lag1
    trend5_d05, trend5_d10

    Returns
    -------
    DataFrame indexed (origin, destination, commodity)
    """
    # Aggregate to OD×commodity×year
    hist = (
        history_df[history_df["year"] <= base_year]
        .groupby(KEY_COMM + ["year"], as_index=False)["tons"]
        .sum()
    )

    # Pivot to wide: index=(origin,destination,commodity), columns=years
    pivot = hist.pivot_table(
        index=KEY_COMM, columns="year", values="tons", aggfunc="sum"
    ).fillna(0.0)

    all_years = sorted(pivot.columns)

    out = pivot.reset_index()[KEY_COMM].copy()

    # Lag features — fill 0 for missing years
    for i, yr in enumerate([base_year, base_year - 1, base_year - 2], 1):
        if yr in pivot.columns:
            out[f"tons_lag{i}"] = pivot[yr].values
        else:
            out[f"tons_lag{i}"] = 0.0

    # Last 3 years stats
    last3 = [y for y in [base_year - 2, base_year - 1, base_year] if y in pivot.columns]
    if last3:
        rv = pivot[last3]
        out["tons_mean_3yr"] = rv.mean(axis=1).values
        out["tons_std_3yr"]  = rv.std(axis=1).fillna(0.0).values
    else:
        out["tons_mean_3yr"] = 0.0
        out["tons_std_3yr"]  = 0.0

    # All-history stats
    if all_years:
        av = pivot[all_years]
        ma = av.mean(axis=1).values
        sa = av.std(axis=1).fillna(0.0).values
        out["tons_mean_all"] = ma
        out["tons_cv_all"]   = np.divide(sa, ma, out=np.zeros_like(sa), where=(ma != 0))
    else:
        out["tons_mean_all"] = 0.0
        out["tons_cv_all"]   = 0.0

    # YoY last
    lag1 = out["tons_lag1"].values.astype(float)
    lag2 = out["tons_lag2"].values.astype(float)
    out["tons_yoy_last"] = (lag1 - lag2) / (lag2 + 1.0)

    # OLS trend over all years
    n = len(all_years)
    if n >= 2:
        # Vectorized OLS across all rows
        arr = pivot[all_years].values.astype(float)
        x = np.array(all_years, dtype=float)
        xm = x - x.mean()
        denom = (xm * xm).sum()
        slopes = (arr @ xm) / denom
        intercepts = arr.mean(axis=1) - slopes * x.mean()
        out["tons_trend"] = slopes
        ma_vals = out["tons_mean_all"].values
        out["tons_trend_norm"] = np.divide(slopes, ma_vals, out=np.zeros_like(slopes), where=(ma_vals != 0))
    else:
        out["tons_trend"]      = 0.0
        out["tons_trend_norm"] = 0.0

    out["n_years_hist"] = len(all_years)

    # Route-level lag1 (sum over commodities at base_year)
    route_lag1 = (
        hist[hist["year"] == base_year]
        .groupby(KEY_ROUTE)["tons"]
        .sum()
        .rename("tons_lag1_route")
    )
    out = out.merge(route_lag1.reset_index(), on=KEY_ROUTE, how="left")
    out["tons_lag1_route"] = out["tons_lag1_route"].fillna(0.0)
    out["tons_ratio_lag1"] = (
        out["tons_lag1"] / out["tons_lag1_route"].replace(0, np.nan)
    ).fillna(0.0)

    # Damped trend: 5yr OLS projected 3 years ahead
    # trend5_d05 (damp=0.5), trend5_d10 (damp=1.0)
    last5_yrs = sorted([y for y in all_years if y >= base_year - 4])
    if len(last5_yrs) >= 2:
        k5 = min(5, len(last5_yrs))
        use5 = sorted(last5_yrs)[-k5:]
        arr5 = pivot[use5].values.astype(float)
        x5 = np.array(use5, dtype=float)
        xm5 = x5 - x5.mean()
        denom5 = (xm5 * xm5).sum()
        slopes5 = (arr5 @ xm5) / denom5 if denom5 != 0 else np.zeros(len(arr5))
        intercepts5 = arr5.mean(axis=1) - slopes5 * x5.mean()
        projection5 = slopes5 * (base_year + 3) + intercepts5

        med3_vals = out["tons_mean_3yr"].values  # using mean_3yr as median proxy
        # Actual median for damping
        if last3:
            med3_actual = pivot[last3].median(axis=1).values
        else:
            med3_actual = np.zeros(len(out))

        proj5 = pd.Series(projection5, index=pivot.index)
        med3_s = pd.Series(med3_actual, index=pivot.index)

        trend5_d05 = (med3_s + 0.5 * (proj5 - med3_s)).clip(lower=0)
        trend5_d10 = (med3_s + 1.0 * (proj5 - med3_s)).clip(lower=0)

        out["trend5_d05"] = trend5_d05.values
        out["trend5_d10"] = trend5_d10.values
    else:
        # fallback: use lag1
        out["trend5_d05"] = out["tons_lag1"].clip(lower=0)
        out["trend5_d10"] = out["tons_lag1"].clip(lower=0)

    result = out.set_index(KEY_COMM)
    return result
