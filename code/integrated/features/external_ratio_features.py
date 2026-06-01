"""
external_ratio_features.py — D_ratio: 14 ratio3 features for M5.

ratio3 = val_t / val_{t-3}, clipped [0.5, 1.8], fillna 1.0.

Functions
---------
build_external_ratio_features(base_year, data_dir) → dict of ratio DataFrames
attach_ratio_features(comm_df, base_year, data_dir) → comm_df + 14 new columns
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY_ROUTE = ["origin", "destination"]

_STATE_RENAME = {
    "District of Columbia": "Washington DC",
}

RATIO_CLIP_LO = 0.5
RATIO_CLIP_HI = 1.8


def _norm_state(df: pd.DataFrame, col: str = "state_name") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].replace(_STATE_RENAME)
    return df


def _state_ratio3(df: pd.DataFrame, value_col: str, t: int) -> pd.Series:
    """
    Compute val_t / val_{t-3} per state, clipped and fillna=1.0.
    Returns Series indexed by state_name.
    """
    now  = df[df["year"] == t].set_index("state_name")[value_col]
    past = df[df["year"] == t - 3].set_index("state_name")[value_col]
    ratio = (now / past.replace(0, np.nan)).clip(RATIO_CLIP_LO, RATIO_CLIP_HI).fillna(1.0)
    return ratio


def _national_ratio3(df: pd.DataFrame, value_col: str, t: int) -> float:
    """Compute national val_t / val_{t-3}, scalar."""
    now_rows  = df[df["year"] == t][value_col]
    past_rows = df[df["year"] == t - 3][value_col]
    if now_rows.empty or past_rows.empty:
        return 1.0
    now_val  = float(now_rows.iloc[0])
    past_val = float(past_rows.iloc[0])
    if past_val == 0:
        return 1.0
    return float(np.clip(now_val / past_val, RATIO_CLIP_LO, RATIO_CLIP_HI))


def build_external_ratio_features(
    base_year: int,
    data_dir: Path,
) -> dict[str, pd.Series | float]:
    """
    Build the 14 ratio3 features keyed by feature name.

    base_year: the "t" year for ratio3 (= context_year when predicting val/test).

    Returns dict mapping feature name → state Series or scalar float.
    """
    d = Path(data_dir)
    t = base_year

    result: dict[str, pd.Series | float] = {}

    # GDP ratio3 (state-level)
    gdp = _norm_state(pd.read_csv(d / "state_unify_gdp.csv").rename(columns={"real_GDP": "real_gdp"}))
    result["gdp_ratio3"] = _state_ratio3(gdp, "real_gdp", t)

    # Population ratio3
    pop = _norm_state(pd.read_csv(d / "population_by_states.csv"))
    result["pop_ratio3"] = _state_ratio3(pop, "population", t)

    # HDD ratio3
    hdd = _norm_state(pd.read_csv(d / "eia_heating_degree_days_by_state.csv"))
    result["hdd_ratio3"] = _state_ratio3(hdd, "hdd_annual", t)

    # Coal production ratio3
    coal_p = _norm_state(pd.read_csv(d / "eia_coal_production_by_state.csv"))
    result["coal_prod_ratio3"] = _state_ratio3(coal_p, "coal_production_ktons", t)

    # Coal electricity generation ratio3
    coal_e = _norm_state(pd.read_csv(d / "eia_coal_electricity_gen_by_state.csv"))
    result["coal_gen_ratio3"] = _state_ratio3(coal_e, "coal_elec_gen_gwh", t)

    # PCE ratios (state-level by category)
    pce_raw = _norm_state(pd.read_csv(d / "pce_by_state_real_2017.csv"))
    pce_map = {
        "Goods": "pce_goods_ratio3",
        "Gasoline and other energy goods": "pce_energy_ratio3",
        "Food and beverages purchased for off-premises consumption": "pce_food_ratio3",
        "Motor vehicles and parts": "pce_motor_ratio3",
    }
    for cat, feat_name in pce_map.items():
        sub = pce_raw[pce_raw["category"] == cat][["state_name", "year", "pce_value"]].copy()
        result[feat_name] = _state_ratio3(sub, "pce_value", t)

    # Crop production ratio3 (sum all crops per state)
    crop = _norm_state(pd.read_csv(d / "usda_crop_production_by_state.csv"))
    crop_agg = crop.groupby(["state_name", "year"], as_index=False)["production_bu"].sum()
    result["crop_prod_ratio3"] = _state_ratio3(crop_agg, "production_bu", t)

    # Natgas national ratio3
    natgas = pd.read_csv(d / "eia_natgas_electricity_gen_national.csv")
    result["natgas_ratio3"] = _national_ratio3(natgas, "natgas_elec_gen_gwh", t)

    # Fertilizer price ratio3 (average of DAP + urea)
    fert = pd.read_csv(d / "fao_fertilizer_price_index.csv")
    fert["fert_price"] = fert[["dap_price_usd_mt", "urea_price_usd_mt"]].mean(axis=1)
    result["fert_price_ratio3"] = _national_ratio3(fert, "fert_price", t)

    return result


def attach_ratio_features(
    comm_df: pd.DataFrame,
    base_year: int,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Attach the 14 D_ratio features to a commodity-level DataFrame.

    Feature names:
    orig_gdp_ratio3, dest_gdp_ratio3, orig_pop_ratio3, dest_pop_ratio3,
    dest_hdd_ratio3, orig_coal_prod_ratio3, dest_coal_gen_ratio3,
    dest_pce_goods_ratio3, dest_pce_energy_ratio3, dest_pce_food_ratio3,
    dest_pce_motor_ratio3, orig_crop_prod_ratio3, natgas_ratio3, fert_price_ratio3

    Parameters
    ----------
    comm_df  : DataFrame with 'origin', 'destination' columns
    base_year: the "t" year for ratio computation (= context_year)
    data_dir : path to external data directory

    Returns
    -------
    comm_df with 14 new columns
    """
    ratios = build_external_ratio_features(base_year, data_dir)
    out = comm_df.copy()

    # State-level ratios
    state_mappings = {
        "orig_gdp_ratio3":        ("origin",      "gdp_ratio3"),
        "dest_gdp_ratio3":        ("destination",  "gdp_ratio3"),
        "orig_pop_ratio3":        ("origin",      "pop_ratio3"),
        "dest_pop_ratio3":        ("destination",  "pop_ratio3"),
        "dest_hdd_ratio3":        ("destination",  "hdd_ratio3"),
        "orig_coal_prod_ratio3":  ("origin",      "coal_prod_ratio3"),
        "dest_coal_gen_ratio3":   ("destination",  "coal_gen_ratio3"),
        "dest_pce_goods_ratio3":  ("destination",  "pce_goods_ratio3"),
        "dest_pce_energy_ratio3": ("destination",  "pce_energy_ratio3"),
        "dest_pce_food_ratio3":   ("destination",  "pce_food_ratio3"),
        "dest_pce_motor_ratio3":  ("destination",  "pce_motor_ratio3"),
        "orig_crop_prod_ratio3":  ("origin",      "crop_prod_ratio3"),
    }

    for feat_name, (side, ratio_key) in state_mappings.items():
        ratio_series = ratios.get(ratio_key, pd.Series(dtype=float))
        if isinstance(ratio_series, pd.Series) and not ratio_series.empty:
            out[feat_name] = out[side].map(ratio_series).fillna(1.0)
        else:
            out[feat_name] = 1.0

    # National scalar ratios
    out["natgas_ratio3"]    = float(ratios.get("natgas_ratio3", 1.0))
    out["fert_price_ratio3"] = float(ratios.get("fert_price_ratio3", 1.0))

    return out
