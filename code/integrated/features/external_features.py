"""
external_features.py — D_abs: absolute external features (~35 features) for M1–M4.

Reimplements the logic from code/mjwoon/load_external.py inline to avoid path issues.

Functions
---------
build_external_features(context_year, data_dir) → DataFrame indexed state_name
attach_external_features(comm_df, context_year, data_dir) → comm_df + ~35 new columns
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY_ROUTE = ["origin", "destination"]

_STATE_RENAME = {
    "District of Columbia": "Washington DC",
}


def _norm(df: pd.DataFrame, col: str = "state_name") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].replace(_STATE_RENAME)
    return df


def _interpolate_year_gaps(df: pd.DataFrame, value_cols: list) -> pd.DataFrame:
    df = df.copy().sort_values(["state_name", "year"])
    for col in value_cols:
        df[col] = df.groupby("state_name")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
    return df


# ── Individual loaders ──────────────────────────────────────────────────────

def _load_gdp(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "state_unify_gdp.csv")
    df = df.rename(columns={"real_GDP": "real_gdp"})
    return _norm(df)[["state_name", "year", "real_gdp"]]


def _load_unemployment(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "state_unify_ue-rate.csv")
    return _norm(df)[["state_name", "year", "unemployment_rate"]]


def _load_population(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "population_by_states.csv")
    return _norm(df)[["state_name", "year", "population"]]


def _load_income(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "income_annual.csv")
    return _norm(df)[["state_name", "year", "median_income"]]


def _load_pce(d: Path) -> pd.DataFrame:
    df = _norm(pd.read_csv(d / "pce_by_state_real_2017.csv"))
    total = (
        df[df["category"] == "Personal consumption expenditures"]
        .rename(columns={"pce_value": "pce_total"})
        [["state_name", "year", "pce_total"]]
    )
    goods = (
        df[df["category"] == "Goods"]
        .rename(columns={"pce_value": "pce_goods"})
        [["state_name", "year", "pce_goods"]]
    )
    return total.merge(goods, on=["state_name", "year"], how="outer")


def _load_pce_changes(d: Path) -> pd.DataFrame:
    df = _norm(pd.read_csv(d / "pce_by_state_real_2017_changes.csv"))
    total = df[df["category"] == "Personal consumption expenditures"].copy()
    total = total.rename(columns={"current_year": "year"})
    return total[["state_name", "year", "pce_growth_rate"]]


def _load_coal_production(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "eia_coal_production_by_state.csv")
    return _norm(df)[["state_name", "year", "coal_production_ktons"]]


def _load_coal_electricity(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "eia_coal_electricity_gen_by_state.csv")
    return _norm(df)[["state_name", "year", "coal_elec_gen_gwh"]]


def _load_hdd(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "eia_heating_degree_days_by_state.csv")
    return _norm(df)[["state_name", "year", "hdd_annual"]]


def _load_pdsi(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "noaa_pdsi_by_state.csv")
    return _norm(df)[["state_name", "year", "pdsi_annual_avg"]]


def _load_crop(d: Path) -> pd.DataFrame:
    df = _norm(pd.read_csv(d / "usda_crop_production_by_state.csv"))
    pivot = df.pivot_table(
        index=["state_name", "year"],
        columns="commodity",
        values="production_bu",
        aggfunc="sum",
    ).reset_index()
    rename_map = {"CORN": "corn_prod_bu", "WHEAT": "wheat_prod_bu", "SOYBEANS": "soy_prod_bu"}
    pivot = pivot.rename(columns=rename_map)
    keep = ["state_name", "year"] + [v for v in rename_map.values() if v in pivot.columns]
    return pivot[keep]


def _load_migration(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "migration_od_annual.csv")
    df["origin"]      = df["origin"].replace(_STATE_RENAME)
    df["destination"] = df["destination"].replace(_STATE_RENAME)
    inflow  = df.groupby(["destination", "year"])["migrants"].sum().reset_index()
    inflow  = inflow.rename(columns={"destination": "state_name", "migrants": "inflow"})
    outflow = df.groupby(["origin", "year"])["migrants"].sum().reset_index()
    outflow = outflow.rename(columns={"origin": "state_name", "migrants": "outflow"})
    net = inflow.merge(outflow, on=["state_name", "year"], how="outer").fillna(0)
    net["net_migration"] = net["inflow"] - net["outflow"]
    return net[["state_name", "year", "net_migration"]]


def _load_natgas(d: Path) -> pd.DataFrame:
    return pd.read_csv(d / "eia_natgas_electricity_gen_national.csv")[["year", "natgas_elec_gen_gwh"]]


def _load_fertilizer(d: Path) -> pd.DataFrame:
    return pd.read_csv(d / "fao_fertilizer_price_index.csv")[["year", "dap_price_usd_mt", "urea_price_usd_mt"]]


# ── Main loader ──────────────────────────────────────────────────────────────

STATE_FEATURE_COLS = [
    "real_gdp", "unemployment_rate", "population", "pop_growth_rate",
    "median_income", "pce_total", "pce_goods", "pce_growth_rate",
    "coal_production_ktons", "coal_elec_gen_gwh", "hdd_annual", "pdsi_annual_avg",
    "corn_prod_bu", "wheat_prod_bu", "soy_prod_bu", "net_migration",
]

NATIONAL_FEATURE_COLS = ["natgas_elec_gen_gwh", "dap_price_usd_mt", "urea_price_usd_mt"]


def build_external_features(
    context_year: int,
    data_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Load and join all external datasets, filtered to context_year.

    Returns
    -------
    state_ctx : DataFrame indexed by state_name — state-level features
    national_dict : dict of national-level scalar features
    """
    d = Path(data_dir)

    gdp     = _load_gdp(d)
    ue      = _load_unemployment(d)
    pop     = _load_population(d)
    income  = _load_income(d)
    pce     = _load_pce(d)
    pce_chg = _load_pce_changes(d)
    coal_p  = _load_coal_production(d)
    coal_e  = _load_coal_electricity(d)
    hdd     = _load_hdd(d)
    pdsi    = _load_pdsi(d)
    crop    = _load_crop(d)
    mig     = _load_migration(d)

    # Interpolate for 2020 gaps
    income = _interpolate_year_gaps(income, ["median_income"])
    mig    = _interpolate_year_gaps(mig, ["net_migration"])

    # Population growth rate
    pop = pop.sort_values(["state_name", "year"])
    pop["pop_growth_rate"] = pop.groupby("state_name")["population"].pct_change()

    # Merge all state-level
    state_all = gdp.copy()
    for df in [ue, pop, income, pce, pce_chg, coal_p, coal_e, hdd, pdsi, crop, mig]:
        state_all = state_all.merge(df, on=["state_name", "year"], how="left")

    # Fill NAs
    state_all["coal_production_ktons"] = state_all["coal_production_ktons"].fillna(0)
    state_all["coal_elec_gen_gwh"]     = state_all["coal_elec_gen_gwh"].fillna(0)
    for col in ["corn_prod_bu", "wheat_prod_bu", "soy_prod_bu"]:
        if col in state_all.columns:
            state_all[col] = state_all[col].fillna(0)
    for col in ["hdd_annual", "pdsi_annual_avg"]:
        median_by_year = state_all.groupby("year")[col].transform("median")
        state_all[col] = state_all[col].fillna(median_by_year)
    state_all["net_migration"] = state_all["net_migration"].fillna(0)
    state_all["pop_growth_rate"] = state_all["pop_growth_rate"].fillna(0)

    # Log1p transform for skewed columns
    skew_cols = ["coal_production_ktons", "coal_elec_gen_gwh",
                 "corn_prod_bu", "wheat_prod_bu", "soy_prod_bu"]
    for col in skew_cols:
        if col in state_all.columns:
            state_all[col] = np.log1p(state_all[col].clip(lower=0))

    # Filter to context_year
    state_ctx = (
        state_all[state_all["year"] == context_year]
        .drop(columns=["year"])
        .set_index("state_name")
    )

    # National features
    natgas = _load_natgas(d)
    fert   = _load_fertilizer(d)
    national = natgas.merge(fert, on="year", how="outer")
    nat_row = national[national["year"] == context_year]
    if len(nat_row) == 0:
        national_dict = {}
    else:
        national_dict = nat_row.drop(columns=["year"]).iloc[0].to_dict()
        if "natgas_elec_gen_gwh" in national_dict:
            national_dict["natgas_elec_gen_gwh"] = np.log1p(national_dict["natgas_elec_gen_gwh"])

    return state_ctx, national_dict


def attach_external_features(
    comm_df: pd.DataFrame,
    context_year: int,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Attach external features to a commodity-level DataFrame.

    Origin state features get 'orig_' prefix.
    Destination state features get 'dest_' prefix.
    National features added as constants.

    Parameters
    ----------
    comm_df : DataFrame with columns including 'origin', 'destination'
    context_year : context year
    data_dir : path to external data directory

    Returns
    -------
    comm_df with ~35 new columns added
    """
    state_ctx, national_dict = build_external_features(context_year, data_dir)

    out = comm_df.copy()

    feat_cols = [c for c in STATE_FEATURE_COLS if c in state_ctx.columns]

    # orig_ prefix
    orig_feat = (
        state_ctx[feat_cols]
        .rename(columns={c: f"orig_{c}" for c in feat_cols})
    )
    out = out.merge(
        orig_feat.reset_index().rename(columns={"state_name": "origin"}),
        on="origin", how="left",
    )

    # dest_ prefix
    dest_feat = (
        state_ctx[feat_cols]
        .rename(columns={c: f"dest_{c}" for c in feat_cols})
    )
    out = out.merge(
        dest_feat.reset_index().rename(columns={"state_name": "destination"}),
        on="destination", how="left",
    )

    # Fill NAs for state features
    for c in feat_cols:
        out[f"orig_{c}"] = out[f"orig_{c}"].fillna(0.0)
        out[f"dest_{c}"] = out[f"dest_{c}"].fillna(0.0)

    # National features as constants
    for col, val in national_dict.items():
        out[col] = float(val) if not pd.isna(val) else 0.0

    return out
