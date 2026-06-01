"""
data_loader.py — Split CSV loader and commodity-level prediction helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from typing import Tuple

from config import SPLITS_DIR, SPLITS, KEY_COMM, KEY_ROUTE


def load_split(
    split_name: str,
    splits_dir: Path = SPLITS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all CSV files for a given split.

    Returns
    -------
    train_df, context_df, val_comm, val_route, test_comm, test_route
    """
    d = splits_dir / split_name
    train_df   = pd.read_csv(d / "train.csv")
    context_df = pd.read_csv(d / "context.csv")
    val_comm   = pd.read_csv(d / "val_commodity.csv")
    val_route  = pd.read_csv(d / "val.csv")
    test_comm  = pd.read_csv(d / "test_commodity.csv")
    test_route = pd.read_csv(d / "test.csv")
    return train_df, context_df, val_comm, val_route, test_comm, test_route


def cm3_predict(df: pd.DataFrame, base_year: int) -> pd.Series:
    """
    CommodityMedian3: median of last 3 valid years per OD×commodity.
    """
    from config import COVID_SKIP_YEARS
    all_years = sorted([y for y in df["year"].unique() if y <= base_year and y not in COVID_SKIP_YEARS])
    years = all_years[-3:] if all_years else [base_year - 2, base_year - 1, base_year]
    sub = df[df["year"].isin(years)]
    if sub.empty:
        sub = df

    comm_med = (
        sub.groupby(KEY_COMM + ["year"], as_index=False)["tons"]
           .sum()
           .groupby(KEY_COMM)["tons"]
           .median()
    )
    return comm_med.clip(lower=0)


def medmean_predict(df: pd.DataFrame, base_year: int, alpha: float = 0.20) -> pd.Series:
    """
    MedMean: (1-alpha)*median + alpha*mean over last 3 valid years per OD×commodity.
    """
    from config import COVID_SKIP_YEARS
    all_years = sorted([y for y in df["year"].unique() if y <= base_year and y not in COVID_SKIP_YEARS])
    years = all_years[-3:] if all_years else [base_year - 2, base_year - 1, base_year]
    sub = df[df["year"].isin(years)]
    if sub.empty:
        sub = df

    agg = (
        sub.groupby(KEY_COMM + ["year"], as_index=False)["tons"]
           .sum()
           .groupby(KEY_COMM)["tons"]
    )
    med = agg.median()
    mean = agg.mean()
    result = (1.0 - alpha) * med + alpha * mean
    return result.clip(lower=0)


def get_all_comm_index(train_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.MultiIndex:
    """Return the union of all (origin, destination, commodity) combinations."""
    all_data = pd.concat([train_df, context_df], ignore_index=True)
    return pd.MultiIndex.from_frame(
        all_data[KEY_COMM].drop_duplicates().sort_values(KEY_COMM).reset_index(drop=True)
    )
