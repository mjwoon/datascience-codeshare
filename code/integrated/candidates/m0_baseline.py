"""
m0_baseline.py — M0: Pure statistical candidates (no external data).

CANDIDATES_M0 = ["M0_median_all", "M0_median_3", "M0_median_4",
                 "M0_trend5_d05", "M0_trend5_d10", "M0_trend3_d05"]

predict_m0(train_df, context_df, context_year, target_year)
    → dict[str, pd.Series] indexed by (origin, destination, commodity)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COMM = ["origin", "destination", "commodity"]

CANDIDATES_M0 = [
    "M0_median_all",
    "M0_median_3",
    "M0_median_4",
    "M0_trend5_d05",
    "M0_trend5_d10",
    "M0_trend3_d05",
]


def _pivot_comm_years(df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """
    Aggregate to OD×commodity×year, pivot wide.
    Returns DataFrame indexed (origin, destination, commodity), columns = sorted years.
    """
    hist = (
        df[df["year"] <= base_year]
        .groupby(KEY_COMM + ["year"], as_index=False)["tons"]
        .sum()
    )
    pivot = hist.pivot_table(
        index=KEY_COMM, columns="year", values="tons", aggfunc="sum"
    ).fillna(0.0)
    return pivot


def _ols_trend_projection(pivot: pd.DataFrame, use_years: list, target_year: int) -> pd.Series:
    """OLS slope on use_years, projected to target_year. Returns Series (same index)."""
    if len(use_years) < 2:
        if use_years:
            return pivot[use_years].iloc[:, -1].fillna(0).clip(lower=0)
        return pd.Series(0.0, index=pivot.index)

    arr = pivot[use_years].values.astype(float)
    x = np.array(use_years, dtype=float)
    xm = x - x.mean()
    denom = (xm * xm).sum()
    if denom == 0:
        return pd.Series(0.0, index=pivot.index)
    slopes = (arr @ xm) / denom
    intercepts = arr.mean(axis=1) - slopes * x.mean()
    projection = slopes * target_year + intercepts
    return pd.Series(projection, index=pivot.index)


def predict_m0(
    train_df: pd.DataFrame,
    context_df: pd.DataFrame,
    context_year: int,
    target_year: int,
) -> dict[str, pd.Series]:
    """
    Compute all M0 candidates.

    Parameters
    ----------
    train_df     : training data
    context_df   : context year data
    context_year : most recent year available (= base_year for features)
    target_year  : year to predict (= context_year + 3)

    Returns
    -------
    dict: {candidate_name: pd.Series indexed (origin, destination, commodity)}
    """
    all_data = pd.concat([train_df, context_df], ignore_index=True)
    base_year = context_year

    pivot = _pivot_comm_years(all_data, base_year)
    all_years = sorted(pivot.columns)
    
    from config import COVID_SKIP_YEARS
    valid_years = [y for y in all_years if y not in COVID_SKIP_YEARS]

    results: dict[str, pd.Series] = {}

    # ── M0_median_all
    if valid_years:
        results["M0_median_all"] = pivot[valid_years].median(axis=1).clip(lower=0)
    else:
        results["M0_median_all"] = pd.Series(0.0, index=pivot.index)

    # ── M0_median_3
    last3 = [y for y in valid_years if y <= base_year][-3:]
    if last3:
        results["M0_median_3"] = pivot[last3].median(axis=1).clip(lower=0)
    else:
        results["M0_median_3"] = pd.Series(0.0, index=pivot.index)

    # ── M0_median_4
    last4 = [y for y in valid_years if y <= base_year][-4:]
    if last4:
        results["M0_median_4"] = pivot[last4].median(axis=1).clip(lower=0)
    else:
        results["M0_median_4"] = pd.Series(0.0, index=pivot.index)

    # Compute median3 once for damping anchor
    med3 = results["M0_median_3"].copy()

    # ── M0_trend5_d05 (5yr OLS, damp=0.5)
    last5 = [y for y in valid_years if y <= base_year][-5:]
    proj5 = _ols_trend_projection(pivot, last5, target_year)
    results["M0_trend5_d05"] = (med3 + 0.5 * (proj5 - med3)).clip(lower=0)

    # ── M0_trend5_d10 (5yr OLS, damp=1.0 = full trend)
    results["M0_trend5_d10"] = (med3 + 1.0 * (proj5 - med3)).clip(lower=0)

    # ── M0_trend3_d05 (3yr OLS, damp=0.5)
    proj3 = _ols_trend_projection(pivot, last3, target_year)
    results["M0_trend3_d05"] = (med3 + 0.5 * (proj3 - med3)).clip(lower=0)

    # Ensure all results are named Series
    for name in CANDIDATES_M0:
        if name not in results:
            results[name] = pd.Series(0.0, index=pivot.index)
        results[name].name = name

    return results
