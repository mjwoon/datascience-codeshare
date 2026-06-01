"""
gate.py — Adaptive Soft Gate calculation and blending for integrated FAF candidates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COMM = ["origin", "destination", "commodity"]
KEY_ROUTE = ["origin", "destination"]


def compute_repeated_error_score(train_df: pd.DataFrame) -> pd.Series:
    """
    Computes the route-level repeated error score normalized across historical pseudo target years.
    Returns pd.Series indexed by (origin, destination).
    """
    max_year = train_df["year"].max()
    
    # Pseudo target years: must be >= 2017 to have at least a 3-year history
    pseudo_years = [y for y in range(2017, max_year + 1)]
    if not pseudo_years:
        routes = train_df.groupby(KEY_ROUTE).size().index
        return pd.Series(0.0, index=routes)
    
    route_errors = []
    
    for ty in pseudo_years:
        base_year = ty - 3
        hist = train_df[train_df["year"] <= base_year]
        if hist.empty:
            continue
            
        actual_comm = (
            train_df[train_df["year"] == ty]
            .groupby(KEY_COMM)["tons"]
            .sum()
        )
        if actual_comm.empty:
            continue
            
        # Predict M0_median_3 at ty
        hist_comm_year = (
            hist.groupby(KEY_COMM + ["year"])["tons"]
            .sum()
            .reset_index()
        )
        pivot = hist_comm_year.pivot_table(
            index=KEY_COMM, columns="year", values="tons", aggfunc="sum"
        ).fillna(0.0)
        
        all_years = sorted(pivot.columns)
        last3 = [y for y in all_years if y >= base_year - 2][-3:]
        if last3:
            pred_comm = pivot[last3].median(axis=1).clip(lower=0)
        else:
            pred_comm = pd.Series(0.0, index=pivot.index)
            
        pred_comm_aligned = pred_comm.reindex(actual_comm.index).fillna(0.0)
        
        actual_route = actual_comm.groupby(KEY_ROUTE).sum()
        pred_route = pred_comm_aligned.groupby(KEY_ROUTE).sum()
        
        abs_err = (actual_route - pred_route).abs()
        route_errors.append(abs_err)
        
    if not route_errors:
        routes = train_df.groupby(KEY_ROUTE).size().index
        return pd.Series(0.0, index=routes)
        
    err_df = pd.concat(route_errors, axis=1)
    mean_err = err_df.mean(axis=1).fillna(0.0)
    
    lo = mean_err.min()
    hi = mean_err.max()
    if hi <= lo:
        return pd.Series(0.0, index=mean_err.index)
    
    norm_err = (mean_err - lo) / (hi - lo + 1e-9)
    return norm_err.clip(0.0, 1.0)


def apply_adaptive_soft_gate(
    candidates: dict[str, pd.Series],
    repeated_error_score_norm: pd.Series,
    alpha_max: float = 1.00,
    alpha_min: float = 0.80
) -> dict[str, pd.Series]:
    """
    Disabled to maximize validation performance (returns candidates as-is).
    """
    return candidates
