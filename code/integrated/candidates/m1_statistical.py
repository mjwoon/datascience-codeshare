"""
m1_statistical.py — M1: Statistical baseline + external ratio factor.

CANDIDATES_M1 = ["M1_median_3_ext", "M1_median_4_ext", "M1_trend5_d05_ext",
                 "M1_trend5_d10_ext", "M1_trend3_d05_ext", "M1_median_all_ext"]

predict_m1(train_df, context_df, context_year, target_year, data_dir)
    → dict[str, pd.Series] indexed by (origin, destination, commodity)
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from pathlib import Path

import numpy as np
import pandas as pd

from candidates.m0_baseline import predict_m0
from features.external_ratio_features import build_external_ratio_features

KEY_COMM = ["origin", "destination", "commodity"]

CANDIDATES_M1 = [
    "M1_median_3_ext",
    "M1_median_4_ext",
    "M1_trend5_d05_ext",
    "M1_trend5_d10_ext",
    "M1_trend3_d05_ext",
    "M1_median_all_ext",
]

# Commodity group → ratio feature key mapping
# FUEL: dest_hdd_ratio3
# AG: orig_crop_prod_ratio3
# others (including BULK): mean(orig_gdp_ratio3, dest_gdp_ratio3)
FUEL_COMMODITIES = {"Fuel oils", "Gasoline"}
AG_COMMODITIES   = {"Animal feed", "Cereal grains", "Other ag prods.", "Live animals/fish"}


def _get_factor_series(
    comm_df: pd.DataFrame,
    ratios: dict,
) -> pd.Series:
    """
    Compute per-row external factor based on commodity group.

    Parameters
    ----------
    comm_df : DataFrame with columns (origin, destination, commodity)
    ratios  : dict from build_external_ratio_features

    Returns
    -------
    pd.Series of factor values (same index as comm_df)
    """
    orig_gdp = comm_df["origin"].map(
        ratios.get("gdp_ratio3", pd.Series(dtype=float))
    ).fillna(1.0)
    dest_gdp = comm_df["destination"].map(
        ratios.get("gdp_ratio3", pd.Series(dtype=float))
    ).fillna(1.0)
    orig_crop = comm_df["origin"].map(
        ratios.get("crop_prod_ratio3", pd.Series(dtype=float))
    ).fillna(1.0)
    dest_hdd = comm_df["destination"].map(
        ratios.get("hdd_ratio3", pd.Series(dtype=float))
    ).fillna(1.0)

    gdp_avg = (orig_gdp + dest_gdp) / 2.0

    # Default factor = GDP average
    factor = gdp_avg.copy()

    # Override for FUEL
    fuel_mask = comm_df["commodity"].isin(FUEL_COMMODITIES)
    factor[fuel_mask] = dest_hdd[fuel_mask]

    # Override for AG
    ag_mask = comm_df["commodity"].isin(AG_COMMODITIES)
    factor[ag_mask] = orig_crop[ag_mask]

    return factor


def predict_m1(
    train_df: pd.DataFrame,
    context_df: pd.DataFrame,
    context_year: int,
    target_year: int,
    data_dir: Path,
) -> dict[str, pd.Series]:
    """
    Compute all M1 candidates (M0 baseline × external ratio factor).

    Parameters
    ----------
    train_df     : training data
    context_df   : context year data
    context_year : most recent year available
    target_year  : year to predict
    data_dir     : path to additional_dataset directory

    Returns
    -------
    dict: {candidate_name: pd.Series indexed (origin, destination, commodity)}
    """
    # Get M0 base predictions
    m0_preds = predict_m0(train_df, context_df, context_year, target_year)

    # Load ratio features at context_year
    try:
        ratios = build_external_ratio_features(context_year, data_dir)
    except Exception:
        # Fallback: all factors = 1.0
        ratios = {}

    # Build mapping from M0 name → M1 name
    name_map = {
        "M0_median_3":    "M1_median_3_ext",
        "M0_median_4":    "M1_median_4_ext",
        "M0_trend5_d05":  "M1_trend5_d05_ext",
        "M0_trend5_d10":  "M1_trend5_d10_ext",
        "M0_trend3_d05":  "M1_trend3_d05_ext",
        "M0_median_all":  "M1_median_all_ext",
    }

    results: dict[str, pd.Series] = {}

    for m0_name, m1_name in name_map.items():
        base = m0_preds[m0_name]
        # base is indexed (origin, destination, commodity)
        comm_df = base.reset_index()

        # Compute factor
        try:
            factor = _get_factor_series(comm_df, ratios)
            factor.index = base.index
        except Exception:
            factor = pd.Series(1.0, index=base.index)

        results[m1_name] = (base * factor).clip(lower=0)
        results[m1_name].name = m1_name

    return results
