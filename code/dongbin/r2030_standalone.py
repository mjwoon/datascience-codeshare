from __future__ import annotations

import sys
import warnings
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "processed_data"
SPLIT_ROOT = PROJECT_ROOT / "output" / "faf4_faf5_fixed_year_splits"
RESULT_ROOT = PROJECT_ROOT / "output" / "faf4_faf5_3yr_tuning_round2030_building_permits"

# Warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
RANDOM_STATE = 20260528
R4 = lambda x: float(round(x, 4))
ENTITY = ["origin", "destination", "commodity"]
ROUTE = ["origin", "destination"]
SPLITS = ["split_1", "split_2", "split_3"]
WEIGHTS = {"split_1": 0.2, "split_2": 0.3, "split_3": 0.5}
GROUPS = ["ag_food", "bulk", "chemicals", "fuel", "manufactured", "other"]

FOCUS_COMMODITIES = ["Natural sands", "Gravel", "Cereal grains", "Logs", "Fuel oils", "Gasoline", "Coal"]
FOCUS_BULK = ["Natural sands", "Gravel", "Coal"]
FOCUS_AG = ["Cereal grains", "Logs"]
FOCUS_FUEL = ["Fuel oils", "Gasoline"]

CLUSTERS = {
    "bulk": {"Gravel", "Nonmetal min. prods.", "Building stone", "Metallic ores", "Coal", "Waste/scrap", "Natural sands"},
    "fuel": {"Gasoline", "Fuel oils", "Natural gas/fossil", "Crude petroleum"},
    "ag_food": {
        "Cereal grains",
        "Other ag prods.",
        "Animal feed",
        "Meat/seafood",
        "Milled grain prods.",
        "Other foodstuffs",
        "Alcoholic beverages",
        "Tobacco prods.",
        "Live animals/fish",
    },
    "chemicals": {"Basic chemicals", "Chemical prods.", "Fertilizers", "Pharmaceuticals", "Plastics/rubber"},
    "manufactured": {
        "Articles base metal",
        "Base metals",
        "Machinery",
        "Electronics",
        "Motorized vehicles",
        "Transport equip.",
        "Precision instruments",
        "Furniture",
        "Textiles/leather",
        "Wood prods.",
        "Paper articles",
        "Printed prods.",
        "Misc. mfg. prods.",
    },
}
COMMODITY_TO_GROUP = {commodity: group for group, commodities in CLUSTERS.items() for commodity in commodities}

BETAS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass(frozen=True)
class Rule:
    round_id: str
    name: str
    groups: tuple[str, ...]
    ratio: str
    side: str
    beta: float


# ══════════════════════════════════════════════════════════════════
# EXTERNAL DATA LOADERS
# ══════════════════════════════════════════════════════════════════

def load_split(split: str) -> dict[str, pd.DataFrame]:
    root = SPLIT_ROOT / split
    return {
        "train": pd.read_csv(root / "train.csv"),
        "context": pd.read_csv(root / "context.csv"),
        "val": pd.read_csv(root / "val.csv"),
        "test": pd.read_csv(root / "test.csv"),
        "val_commodity": pd.read_csv(root / "val_commodity.csv"),
        "test_commodity": pd.read_csv(root / "test_commodity.csv"),
    }


def add_growth(table: pd.DataFrame, keys: list[str], value_cols: list[str]) -> pd.DataFrame:
    out = table.sort_values(keys + ["year"]).copy()
    for col in value_cols:
        if keys:
            out[f"{col}_yoy"] = out.groupby(keys)[col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            out[f"{col}_diff"] = out.groupby(keys)[col].diff()
        else:
            out[f"{col}_yoy"] = out[col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            out[f"{col}_diff"] = out[col].diff()
    return out


def load_external_state_year() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    simple_specs = [
        ("state_unify_gdp.csv", ["real_GDP"]),
        ("state_unify_ue-rate.csv", ["unemployment_rate"]),
        ("population_by_states.csv", ["population"]),
        ("income_annual.csv", ["median_income"]),
        ("eia_coal_production_by_state.csv", ["coal_production_ktons"]),
        ("eia_coal_electricity_gen_by_state.csv", ["coal_elec_gen_gwh"]),
        ("eia_heating_degree_days_by_state.csv", ["hdd_annual"]),
        ("noaa_pdsi_by_state.csv", ["pdsi_annual_avg"]),
    ]
    for filename, value_cols in simple_specs:
        df = pd.read_csv(DATA_ROOT / filename)
        df = add_growth(df, ["state_name"], value_cols)
        frames.append(df)

    crops = pd.read_csv(DATA_ROOT / "usda_crop_production_by_state.csv")
    crop = crops.pivot_table(
        index=["state_name", "year"],
        columns="commodity",
        values=["harvested_acres", "planted_acres", "production_bu", "yield_bu_per_acre"],
        aggfunc="sum",
    )
    crop.columns = [f"crop_{metric.lower()}_{commodity.lower()}" for metric, commodity in crop.columns]
    crop = add_growth(crop.reset_index(), ["state_name"], [c for c in crop.columns])
    frames.append(crop)

    pce = pd.read_csv(DATA_ROOT / "pce_by_state_real_2017.csv")
    keep = {
        "Personal consumption expenditures": "pce_total",
        "Goods": "pce_goods",
        "Durable goods": "pce_durable",
        "Nondurable goods": "pce_nondurable",
        "Gasoline and other energy goods": "pce_energy_goods",
        "Food and beverages purchased for off-premises consumption": "pce_food_home",
    }
    pce = pce[pce["category"].isin(keep)].assign(category=lambda x: x["category"].map(keep))
    pce_wide = pce.pivot_table(index=["state_name", "year"], columns="category", values="pce_value", aggfunc="sum").reset_index()
    pce_wide = add_growth(pce_wide, ["state_name"], [c for c in pce_wide.columns if c not in ["state_name", "year"]])
    frames.append(pce_wide)

    migration = pd.read_csv(DATA_ROOT / "migration_od_annual.csv")
    inflow = migration.groupby(["destination", "year"])["migrants"].sum().rename("migration_inflow")
    outflow = migration.groupby(["origin", "year"])["migrants"].sum().rename("migration_outflow")
    mig_state = pd.concat(
        [
            inflow.reset_index().rename(columns={"destination": "state_name"}).set_index(["state_name", "year"]),
            outflow.reset_index().rename(columns={"origin": "state_name"}).set_index(["state_name", "year"]),
        ],
        axis=1,
    ).reset_index()
    mig_state["migration_net"] = mig_state["migration_inflow"].fillna(0.0) - mig_state["migration_outflow"].fillna(0.0)
    mig_state = add_growth(mig_state, ["state_name"], ["migration_inflow", "migration_outflow", "migration_net"])
    frames.append(mig_state)

    state_year = frames[0]
    for frame in frames[1:]:
        state_year = state_year.merge(frame, on=["state_name", "year"], how="outer")

    national = pd.read_csv(DATA_ROOT / "eia_natgas_electricity_gen_national.csv")
    fert = pd.read_csv(DATA_ROOT / "fao_fertilizer_price_index.csv")
    national = national.merge(fert, on="year", how="outer")
    national = add_growth(national, [], [c for c in national.columns if c != "year"])
    return state_year, national


def load_state_external():
    gdp = pd.read_csv(DATA_ROOT / "state_unify_gdp.csv")
    gdp = gdp.rename(columns={"real_GDP": "gdp"})
    pop = pd.read_csv(DATA_ROOT / "population_by_states.csv")
    for c in pop.columns:
        if c not in ["year", "state", "state_name"]:
            pop = pop.rename(columns={c: "population"})
            break
    ue = pd.read_csv(DATA_ROOT / "state_unify_ue-rate.csv")
    ue = ue.rename(columns={c: c.lower().replace(" ", "_") for c in ue.columns})
    pce = pd.read_csv(DATA_ROOT / "pce_by_state_real_2017.csv")
    merged = gdp.merge(pop, on=["year", "state_name"], how="outer")
    merged = merged.merge(ue, on=["year", "state_name"], how="outer")
    merged = merged.merge(pce, on=["year", "state_name"], how="outer")
    return merged


def load_national_external():
    fert = pd.read_csv(DATA_ROOT / "fao_fertilizer_price_index.csv")
    ng = pd.read_csv(DATA_ROOT / "eia_natgas_electricity_gen_national.csv")
    hdd = pd.read_csv(DATA_ROOT / "eia_heating_degree_days_by_state.csv")
    coal = pd.read_csv(DATA_ROOT / "eia_coal_production_by_state.csv")
    return {"fertilizer": fert, "natgas_gen": ng, "hdd": hdd, "coal_prod": coal}


def load_migration_external():
    mig = pd.read_csv(DATA_ROOT / "migration_od_annual.csv")
    return mig


def load_income_external():
    inc = pd.read_csv(DATA_ROOT / "income_annual.csv")
    return inc


def load_coal_elec_external():
    coal = pd.read_csv(DATA_ROOT / "eia_coal_electricity_gen_by_state.csv")
    return coal


def load_hdd_external():
    hdd = pd.read_csv(DATA_ROOT / "eia_heating_degree_days_by_state.csv")
    if "hdd_annual" in hdd.columns:
        hdd = hdd.rename(columns={"hdd_annual": "hdd"})
    return hdd


def load_building_permits_external():
    bp = pd.read_csv(DATA_ROOT / "building_permits_by_state.csv")
    return bp


def commodity_group(commodity: str) -> str:
    return COMMODITY_TO_GROUP.get(commodity, "other")


def add_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["commodity_group"] = out["commodity"].map(commodity_group)
    return out


def metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, float)
    pred = np.maximum(np.nan_to_num(np.asarray(pred, float), nan=0.0, posinf=1e12, neginf=0.0), 0.0)
    den = np.sum(np.abs(y))
    return {
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(mean_absolute_error(y, pred)),
        "wmape": float(np.sum(np.abs(y - pred)) / den) if den else np.nan,
        "rmsle": float(np.sqrt(mean_squared_log_error(np.maximum(y, 0.0), pred))),
        "r2": float(r2_score(y, pred)),
    }


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & ANCILLARY FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def add_external_features(pred_frame, history, target_year, state_ext, nat_ext, mig_ext, inc_ext, coal_elec_ext, hdd_ext, bp_ext):
    out = pred_frame.copy()
    fy = target_year - 3
    needed_cols = ["year", "state_name", "gdp", "population", "ue_rate"]
    avail_cols = [c for c in needed_cols if c in state_ext.columns]
    se = state_ext[avail_cols].copy()
    sfy = se[se["year"] == fy].drop(columns=["year"]).set_index("state_name")
    sfy_past = se[se["year"] == fy - 3].drop(columns=["year"]).set_index("state_name")
    for col in ["gdp", "population", "ue_rate"]:
        if col in sfy.columns:
            col_dict = sfy[col].to_dict()
            out[f"o_{col}"] = out["origin"].map(col_dict).fillna(0.0)
            out[f"d_{col}"] = out["destination"].map(col_dict).fillna(0.0)
    if not sfy_past.empty:
        for col in ["gdp", "population"]:
            if col in sfy.columns and col in sfy_past.columns:
                past_dict = sfy_past[col].to_dict()
                out[f"o_{col}_growth"] = out[f"o_{col}"] / out["origin"].map(past_dict).replace(0, np.nan).fillna(1.0)
                out[f"d_{col}_growth"] = out[f"d_{col}"] / out["destination"].map(past_dict).replace(0, np.nan).fillna(1.0)
    out["route_gdp_growth"] = (out.get("o_gdp_growth", pd.Series(1.0, index=out.index)) + out.get("d_gdp_growth", pd.Series(1.0, index=out.index))) / 2.0
    out["route_pop_growth"] = (out.get("o_population_growth", pd.Series(1.0, index=out.index)) + out.get("d_population_growth", pd.Series(1.0, index=out.index))) / 2.0
    fert = nat_ext.get("fertilizer")
    if fert is not None and "year" in fert.columns:
        fy_fert = fert[fert["year"] == fy]
        if not fy_fert.empty:
            num_cols = [c for c in fy_fert.columns if c != "year" and pd.api.types.is_numeric_dtype(fy_fert[c])]
            if num_cols:
                out["fertilizer_index"] = fy_fert[num_cols[0]].iloc[0]
    ng = nat_ext.get("natgas_gen")
    if ng is not None and "year" in ng.columns:
        fy_ng = ng[ng["year"] == fy]
        if not fy_ng.empty:
            num_cols = [c for c in fy_ng.columns if c != "year" and pd.api.types.is_numeric_dtype(fy_ng[c])]
            if num_cols:
                out["natgas_gen"] = fy_ng[num_cols[0]].iloc[0]

    # -- Migration OD --
    if mig_ext is not None and not mig_ext.empty:
        mig_fy = mig_ext[mig_ext["year"] == fy].copy()
        if not mig_fy.empty:
            mig_map = mig_fy.set_index(["origin", "destination"])["migrants"].to_dict()
            out["od_migrants"] = out.apply(lambda r: mig_map.get((r["origin"], r["destination"]), 0.0), axis=1)
            mig_yrs = mig_ext[mig_ext["year"].isin([fy - 2, fy - 1, fy])].copy()
            if not mig_yrs.empty:
                mig_avg = mig_yrs.groupby(["origin", "destination"])["migrants"].mean().to_dict()
                out["od_migrants_avg3"] = out.apply(lambda r: mig_avg.get((r["origin"], r["destination"]), 0.0), axis=1)
                out["od_migrants_dev"] = out["od_migrants"] - out["od_migrants_avg3"]

    # -- Income --
    if inc_ext is not None and not inc_ext.empty:
        inc_fy = inc_ext[inc_ext["year"] == fy].drop(columns=["year"]).set_index("state_name")
        inc_dict = inc_fy["median_income"].to_dict() if "median_income" in inc_fy.columns else {}
        if inc_dict:
            out["o_income"] = out["origin"].map(inc_dict).fillna(0.0)
            out["d_income"] = out["destination"].map(inc_dict).fillna(0.0)
            inc_yrs = inc_ext[inc_ext["year"].isin([fy - 2, fy - 1, fy])].copy()
            if not inc_yrs.empty:
                inc_avg = inc_yrs.groupby("state_name")["median_income"].mean().to_dict()
                out["o_income_avg3"] = out["origin"].map(inc_avg).fillna(0.0)
                out["d_income_avg3"] = out["destination"].map(inc_avg).fillna(0.0)
            inc_past = inc_ext[inc_ext["year"] == fy - 3].drop(columns=["year"]).set_index("state_name")
            if not inc_past.empty and "median_income" in inc_past.columns:
                past_dict = inc_past["median_income"].to_dict()
                out["o_income_growth"] = out["o_income"] / out["origin"].map(past_dict).replace(0, np.nan).fillna(1.0)
                out["d_income_growth"] = out["d_income"] / out["destination"].map(past_dict).replace(0, np.nan).fillna(1.0)

    # -- Coal electricity generation --
    if coal_elec_ext is not None and not coal_elec_ext.empty:
        coal_fy = coal_elec_ext[coal_elec_ext["year"] == fy].drop(columns=["year"]).set_index("state_name")
        coal_dict = coal_fy["coal_elec_gen_gwh"].to_dict() if "coal_elec_gen_gwh" in coal_fy.columns else {}
        if coal_dict:
            out["o_coal_gen"] = out["origin"].map(coal_dict).fillna(0.0)
            out["d_coal_gen"] = out["destination"].map(coal_dict).fillna(0.0)

    # -- Heating degree days --
    if hdd_ext is not None and not hdd_ext.empty:
        hdd_fy = hdd_ext[hdd_ext["year"] == fy].drop(columns=["year"]).set_index("state_name")
        hdd_dict = hdd_fy["hdd"].to_dict() if "hdd" in hdd_fy.columns else {}
        if hdd_dict:
            out["o_hdd"] = out["origin"].map(hdd_dict).fillna(0.0)
            out["d_hdd"] = out["destination"].map(hdd_dict).fillna(0.0)

    # -- Building permits --
    if bp_ext is not None and not bp_ext.empty:
        bp_fy = bp_ext[bp_ext["year"] == fy].drop(columns=["year"]).set_index("state_name")
        bp_dict = bp_fy["total_units"].to_dict() if "total_units" in bp_fy.columns else {}
        if bp_dict:
            out["o_building_permits"] = out["origin"].map(bp_dict).fillna(0.0)
            out["d_building_permits"] = out["destination"].map(bp_dict).fillna(0.0)
            bp_yrs = bp_ext[bp_ext["year"].isin([fy - 2, fy - 1, fy])].copy()
            if not bp_yrs.empty:
                bp_avg = bp_yrs.groupby("state_name")["total_units"].mean().to_dict()
                out["o_building_permits_avg3"] = out["origin"].map(bp_avg).fillna(0.0)
                out["d_building_permits_avg3"] = out["destination"].map(bp_avg).fillna(0.0)
                out["o_building_permits_dev"] = out["o_building_permits"] - out["o_building_permits_avg3"]
                out["d_building_permits_dev"] = out["d_building_permits"] - out["d_building_permits_avg3"]
                out["o_building_permits_growth"] = out["o_building_permits"] / out["o_building_permits_avg3"].replace(0, np.nan).fillna(1.0)
                out["d_building_permits_growth"] = out["d_building_permits"] / out["d_building_permits_avg3"].replace(0, np.nan).fillna(1.0)

    # -- Route-level aggregates --
    for prefix in ["income", "coal_gen", "hdd"]:
        o_col = f"o_{prefix}"
        d_col = f"d_{prefix}"
        if o_col in out.columns and d_col in out.columns:
            out[f"route_{prefix}"] = (out[o_col] + out[d_col]) / 2.0
        o_avg = f"o_{prefix}_avg3"
        d_avg = f"d_{prefix}_avg3"
        if o_avg in out.columns and d_avg in out.columns:
            out[f"route_{prefix}_avg3"] = (out[o_avg] + out[d_avg]) / 2.0
    if "o_income_growth" in out.columns and "d_income_growth" in out.columns:
        out["route_income_growth"] = (out["o_income_growth"] + out["d_income_growth"]) / 2.0
    if "od_migrants" in out.columns:
        out["route_migrants"] = out["od_migrants"]
    if "od_migrants_avg3" in out.columns:
        out["route_migrants_avg3"] = out["od_migrants_avg3"]

    # -- Route-level aggregates for building permits --
    if "o_building_permits" in out.columns and "d_building_permits" in out.columns:
        out["route_building_permits"] = (out["o_building_permits"] + out["d_building_permits"]) / 2.0
    if "o_building_permits_avg3" in out.columns and "d_building_permits_avg3" in out.columns:
        out["route_building_permits_avg3"] = (out["o_building_permits_avg3"] + out["d_building_permits_avg3"]) / 2.0
    if "o_building_permits_growth" in out.columns and "d_building_permits_growth" in out.columns:
        out["route_building_permits_growth"] = (out["o_building_permits_growth"] + out["d_building_permits_growth"]) / 2.0
    if "o_building_permits_dev" in out.columns and "d_building_permits_dev" in out.columns:
        out["route_building_permits_dev"] = (out["o_building_permits_dev"] + out["d_building_permits_dev"]) / 2.0

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def safe_ratio(now: pd.Series, past: pd.Series, lo: float = 0.80, hi: float = 1.25) -> pd.Series:
    ratio = (now.fillna(past) + 1.0) / (past + 1.0)
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lo, hi)


def state_ratio_table(state_year: pd.DataFrame, feature_year: int, col: str, lag: int = 3) -> pd.DataFrame:
    now = state_year[state_year["year"].eq(feature_year)][["state_name", col]]
    past = state_year[state_year["year"].eq(feature_year - lag)][["state_name", col]]
    joined = now.merge(past, on="state_name", how="outer", suffixes=("_now", "_past"))
    joined[f"ratio_{col}"] = safe_ratio(joined[f"{col}_now"], joined[f"{col}_past"])
    return joined[["state_name", f"ratio_{col}"]]


def national_ratio(national: pd.DataFrame, feature_year: int, col: str, lag: int = 3) -> float:
    now = national.loc[national["year"].eq(feature_year), col]
    past = national.loc[national["year"].eq(feature_year - lag), col]
    if now.empty or past.empty:
        return 1.0
    return float(safe_ratio(pd.Series([now.iloc[0]]), pd.Series([past.iloc[0]])).iloc[0])


def crop_ratio_table(feature_year: int, commodity: str) -> pd.DataFrame:
    crops = pd.read_csv(DATA_ROOT / "usda_crop_production_by_state.csv")
    crops = crops[crops["commodity"].eq(commodity)]
    now = crops[crops["year"].eq(feature_year)][["state_name", "production_bu"]]
    past = crops[crops["year"].eq(feature_year - 3)][["state_name", "production_bu"]]
    joined = now.merge(past, on="state_name", how="outer", suffixes=("_now", "_past"))
    joined[f"ratio_crop_{commodity.lower()}"] = safe_ratio(joined["production_bu_now"], joined["production_bu_past"], 0.75, 1.35)
    return joined[["state_name", f"ratio_crop_{commodity.lower()}"]]


def od_migration_ratio(feature_year: int) -> pd.DataFrame:
    mig = pd.read_csv(DATA_ROOT / "migration_od_annual.csv")
    now = mig[mig["year"].eq(feature_year)][ROUTE + ["migrants"]]
    past = mig[mig["year"].eq(feature_year - 3)][ROUTE + ["migrants"]]
    joined = now.merge(past, on=ROUTE, how="outer", suffixes=("_now", "_past"))
    joined["ratio_od_migration"] = safe_ratio(joined["migrants_now"], joined["migrants_past"], 0.75, 1.35)
    return joined[ROUTE + ["ratio_od_migration"]]


def factor_frame(routes: pd.DataFrame, target_year: int, state_year: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    feature_year = target_year - 3
    out = routes[ROUTE].copy()
    out["available_through_year"] = feature_year

    state_cols = {
        "gdp": "real_GDP",
        "ue": "unemployment_rate",
        "pop": "population",
        "income": "median_income",
        "coal_prod": "coal_production_ktons",
        "coal_elec": "coal_elec_gen_gwh",
        "hdd": "hdd_annual",
        "pdsi": "pdsi_annual_avg",
        "pce_goods": "pce_goods",
        "pce_energy": "pce_energy_goods",
        "pce_food": "pce_food_home",
        "pce_durable": "pce_durable",
    }
    for name, col in state_cols.items():
        if col not in state_year.columns:
            continue
        ratio = state_ratio_table(state_year, feature_year, col)
        for side in ROUTE:
            side_ratio = ratio.rename(columns={"state_name": side, f"ratio_{col}": f"{side}_ratio_{name}"})
            out = out.merge(side_ratio, on=side, how="left")
        out[f"od_avg_ratio_{name}"] = out[[f"origin_ratio_{name}", f"destination_ratio_{name}"]].mean(axis=1).fillna(1.0)

    for commodity in ["CORN", "WHEAT", "SOYBEANS"]:
        ratio = crop_ratio_table(feature_year, commodity)
        side_ratio = ratio.rename(columns={"state_name": "origin", f"ratio_crop_{commodity.lower()}": f"origin_ratio_crop_{commodity.lower()}"})
        out = out.merge(side_ratio, on="origin", how="left")

    mig_ratio = od_migration_ratio(feature_year)
    out = out.merge(mig_ratio, on=ROUTE, how="left")

    for col in ["natgas_elec_gen_gwh", "dap_price_usd_mt", "urea_price_usd_mt"]:
        if col in national.columns:
            out[f"national_ratio_{col}"] = national_ratio(national, feature_year, col)

    out = out.fillna(1.0)
    if not out["available_through_year"].le(target_year - 3).all():
        raise AssertionError(f"Leakage failed for target_year={target_year}")
    return out


def pce_change_ratios(feature_year: int) -> pd.DataFrame:
    pce = pd.read_csv(DATA_ROOT / "pce_by_state_real_2017_changes.csv")
    keep = {
        "Motor vehicles and parts": "motor_vehicles",
        "Furnishings and durable household equipment": "furnishings",
        "Clothing and footwear": "clothing",
        "Transportation services": "transport_services",
        "Gasoline and other energy goods": "energy_goods",
        "Goods": "goods",
        "Durable goods": "durable",
        "Nondurable goods": "nondurable",
        "Food and beverages purchased for off-premises consumption": "food_home",
    }
    pce = pce[pce["category"].isin(keep) & pce["current_year"].eq(feature_year)].copy()
    pce["category"] = pce["category"].map(keep)
    pce["growth_ratio"] = (1.0 + pce["pce_growth_rate"]).clip(0.80, 1.25)
    wide = pce.pivot_table(index="state_name", columns="category", values="growth_ratio", aggfunc="mean").reset_index()
    wide.columns = ["state_name"] + [f"pcechg_ratio_{c}" for c in wide.columns[1:]]
    return wide


def add_expanded_factors(factors: pd.DataFrame, target_year: int) -> pd.DataFrame:
    feature_year = target_year - 3
    out = factors.copy()

    pcechg = pce_change_ratios(feature_year)
    for side in ROUTE:
        side_df = pcechg.rename(columns={"state_name": side}).add_prefix(f"{side}_")
        side_df = side_df.rename(columns={f"{side}_{side}": side})
        out = out.merge(side_df, on=side, how="left")
    for col in [c for c in pcechg.columns if c != "state_name"]:
        o = f"origin_{col}"
        d = f"destination_{col}"
        if o in out and d in out:
            out[f"od_avg_{col}"] = out[[o, d]].mean(axis=1)

    derived = {}
    for side in ROUTE + ["od_avg"]:
        for num, den, name in [
            ("ratio_gdp", "ratio_pop", "gdp_per_capita"),
            ("ratio_pce_goods", "ratio_pop", "pce_goods_per_capita"),
            ("ratio_pce_durable", "ratio_pop", "pce_durable_per_capita"),
            ("ratio_pce_energy", "ratio_pop", "pce_energy_per_capita"),
            ("ratio_income", "ratio_pop", "income_per_capita_proxy"),
        ]:
            ncol = f"{side}_{num}"
            dcol = f"{side}_{den}"
            if ncol in out and dcol in out:
                derived[f"{side}_ratio_{name}"] = (out[ncol] / out[dcol].replace(0.0, np.nan)).clip(0.75, 1.35).fillna(1.0)
    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)

    out = out.fillna(1.0)
    if not out["available_through_year"].le(target_year - 3).all():
        raise AssertionError(f"Leakage failed for target_year={target_year}")
    return out


def instability_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gate_volatility"] = np.clip(out["route_volatility"].fillna(0.0) / 0.40, 0.0, 1.0)
    out["gate_shock"] = np.clip(out["shock_commodity_count"].fillna(0.0) / 3.0, 0.0, 1.0)
    out["gate_repair"] = np.clip(np.abs(np.log(out["robust_repair_ratio"].fillna(1.0).clip(0.50, 1.80))) / 0.20, 0.0, 1.0)
    out["gate_external"] = np.clip(out["external_abs_move_max"].fillna(0.0) / 0.20, 0.0, 1.0)
    out["gate_route_size"] = np.clip((out["base_rank_pct"].fillna(0.0) - 0.70) / 0.25, 0.0, 1.0)
    out["instability_gate"] = np.maximum.reduce([
        out["gate_volatility"].to_numpy(float),
        out["gate_shock"].to_numpy(float),
        out["gate_repair"].to_numpy(float),
        0.75 * out["gate_external"].to_numpy(float),
    ])
    out["large_instability_gate"] = out["instability_gate"] * out["gate_route_size"]
    out["commodity_instability_gate"] = np.maximum(out["gate_shock"], out["gate_repair"])
    out["external_instability_gate"] = np.maximum(out["gate_external"], out["commodity_instability_gate"]) * out["gate_route_size"]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_external_movement_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ratio_cols = [
        c
        for c in out.columns
        if c.startswith(("origin_ratio_", "destination_ratio_", "od_avg_ratio_", "national_ratio_", "origin_pcechg_", "destination_pcechg_", "od_avg_pcechg_"))
    ]
    if ratio_cols:
        vals = out[ratio_cols].fillna(1.0).clip(0.65, 1.50).to_numpy(float)
        logs = np.log(vals)
        out["external_abs_move_max"] = np.max(np.abs(logs), axis=1)
        out["external_abs_move_mean"] = np.mean(np.abs(logs), axis=1)
        out["external_signed_move_sum"] = np.sum(logs, axis=1)
    else:
        out["external_abs_move_max"] = 0.0
        out["external_abs_move_mean"] = 0.0
        out["external_signed_move_sum"] = 0.0
    out["prior075_ratio"] = (out["prior_shrink075"] + 1.0) / (out["base_pred"] + 1.0)
    out["prior100_ratio"] = (out["prior_current"] + 1.0) / (out["base_pred"] + 1.0)
    return out


def route_sum(group_pred: pd.DataFrame, col: str = "pred_group") -> pd.DataFrame:
    return group_pred.groupby(ROUTE, as_index=False)[col].sum().rename(columns={col: "pred"})


def route_total_series(history: pd.DataFrame, target_year: int, method: str) -> pd.DataFrame:
    years = [target_year - 5, target_year - 4, target_year - 3]
    table = (
        history[history["year"].isin(years)]
        .groupby(ROUTE + ["year"], as_index=False)["tons"]
        .sum()
        .pivot_table(index=ROUTE, columns="year", values="tons", aggfunc="sum")
        .reindex(columns=years)
    )
    if table.empty:
        return pd.DataFrame(columns=ROUTE + ["route_total_pred"])
    if method == "avg3":
        values = table.mean(axis=1, skipna=True)
    elif method == "wmean":
        weights = pd.Series([0.20, 0.30, 0.50], index=years)
        valid_weight = table.notna().mul(weights, axis=1).sum(axis=1).replace(0.0, np.nan)
        values = table.fillna(0.0).mul(weights, axis=1).sum(axis=1) / valid_weight
    else:
        values = table.median(axis=1, skipna=True)
    return values.fillna(0.0).reset_index(name="route_total_pred")


def scale_group_to_route(base_group: pd.DataFrame, route_total: pd.DataFrame) -> pd.DataFrame:
    out = base_group.merge(route_total, on=ROUTE, how="left")
    out["current_total"] = out.groupby(ROUTE)["base_group_pred"].transform("sum")
    ratio = (out["route_total_pred"].fillna(out["current_total"]) + 1.0) / (out["current_total"] + 1.0)
    out["base_group_pred"] = (out["base_group_pred"] * ratio).clip(lower=0.0)
    return out[ROUTE + ["commodity_group", "base_group_pred"]]


def group_base(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int) -> pd.DataFrame:
    years = [target_year - 5, target_year - 4, target_year - 3]
    hist = add_group(history[history["year"].isin(years)])
    table = (
        hist.groupby(ROUTE + ["commodity_group", "commodity", "year"], as_index=False)["tons"]
        .sum()
        .pivot_table(index=ROUTE + ["commodity_group", "commodity"], columns="year", values="tons", aggfunc="sum")
        .reindex(columns=years)
    )
    if table.empty:
        pred = pd.DataFrame()
    else:
        pred = (
            table.median(axis=1, skipna=True)
            .fillna(0.0)
            .groupby(level=ROUTE + ["commodity_group"])
            .sum()
            .reset_index(name="base_group_pred")
        )
    grid = target_routes[ROUTE].merge(pd.DataFrame({"commodity_group": GROUPS}), how="cross")
    out = grid.merge(pred, on=ROUTE + ["commodity_group"], how="left")
    out["base_group_pred"] = out["base_group_pred"].fillna(0.0)
    return out


def apply_rule_combo(base: pd.DataFrame, factors: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    out = base.copy()
    out["pred_group"] = out["base_group_pred"]
    for rule in rules:
        joined = out[ROUTE].merge(factors[ROUTE + [rule.ratio]], on=ROUTE, how="left")
        ratio = joined[rule.ratio].fillna(1.0).clip(0.70, 1.40)
        mask = out["commodity_group"].isin(rule.groups)
        ratio_arr = np.asarray(ratio, dtype=float)
        mask_arr = np.asarray(mask, dtype=bool)
        out.loc[mask, "pred_group"] = out.loc[mask, "pred_group"] * np.power(ratio_arr[mask_arr], rule.beta)
    return out


def top5_motor_rules(scale: float = 1.0, motor_beta: float = 0.25, pop_beta: float = 0.25) -> list[Rule]:
    return [
        Rule("round76", "fuel_energy_pce", ("fuel",), "destination_ratio_pce_energy", "route", -0.25 * scale),
        Rule("round76", "ag_wheat_origin", ("ag_food",), "origin_ratio_crop_wheat", "route", 0.25 * scale),
        Rule("round76", "manufactured_goods", ("manufactured",), "destination_ratio_pce_goods", "route", -0.50 * scale),
        Rule("round76", "all_population", tuple(GROUPS), "od_avg_ratio_pop", "route", pop_beta),
        Rule("round76", "chem_fert_dap", ("chemicals",), "national_ratio_dap_price_usd_mt", "route", 1.00 * scale),
        Rule("round76", "manufactured_motor_pcechg", ("manufactured",), "destination_pcechg_ratio_motor_vehicles", "route", motor_beta),
    ]


def make_base(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int, name: str) -> pd.DataFrame:
    if name == "route_blend20_avg3":
        return route_blend_base(history, target_routes, target_year, 0.20, "avg3")
    raise KeyError(name)


def route_blend_base(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int, alpha: float, route_method: str) -> pd.DataFrame:
    commodity = group_base(history, target_routes, target_year)
    route_target = route_total_series(history, target_year, route_method)
    route_target = target_routes[ROUTE].merge(route_target, on=ROUTE, how="left").fillna({"route_total_pred": 0.0})
    commodity_total = route_sum(commodity.rename(columns={"base_group_pred": "pred_group"})).rename(columns={"pred": "commodity_total"})
    route_target = route_target.merge(commodity_total, on=ROUTE, how="left").fillna({"commodity_total": 0.0})
    route_target["route_total_pred"] = (1.0 - alpha) * route_target["commodity_total"] + alpha * route_target["route_total_pred"]
    return scale_group_to_route(commodity, route_target[ROUTE + ["route_total_pred"]])


def route_frame_custom(
    history: pd.DataFrame,
    target_routes: pd.DataFrame,
    target_year: int,
    state_year: pd.DataFrame,
    national: pd.DataFrame,
    baseline_name: str,
) -> pd.DataFrame:
    hist_cut = history[history["year"].le(target_year - 3)]
    base_group = make_base(hist_cut, target_routes, target_year, baseline_name)
    factors = add_expanded_factors(factor_frame(target_routes, target_year, state_year, national), target_year)
    base = route_sum(base_group.rename(columns={"base_group_pred": "pred_group"})).rename(columns={"pred": "base_pred"})
    prior075 = route_sum(apply_rule_combo(base_group, factors, top5_motor_rules(0.75, 0.1875, 0.1875))).rename(columns={"pred": "prior_shrink075"})
    prior100 = route_sum(apply_rule_combo(base_group, factors, top5_motor_rules(1.0, 0.25, 0.25))).rename(columns={"pred": "prior_current"})
    out = target_routes[ROUTE].merge(base, on=ROUTE, how="left").merge(prior075, on=ROUTE, how="left").merge(prior100, on=ROUTE, how="left")
    out = out.merge(factors, on=ROUTE, how="left")
    for col in ["base_pred", "prior_shrink075", "prior_current"]:
        out[col] = out[col].fillna(0.0).clip(lower=0.0)
        out[f"log_{col}"] = np.log1p(out[col])
    out["prior_current_delta_log"] = out["log_prior_current"] - out["log_base_pred"]
    out["prior_shrink_delta_log"] = out["log_prior_shrink075"] - out["log_base_pred"]
    out["is_intrastate"] = (out["origin"] == out["destination"]).astype(int)
    out["target_year"] = target_year
    out["feature_year"] = target_year - 3
    out["baseline_name"] = baseline_name
    return out


def window_years(target_year: int) -> list[int]:
    return [target_year - 5, target_year - 4, target_year - 3]


def route_series(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int) -> pd.DataFrame:
    years = window_years(target_year)
    table = (
        history[history["year"].isin(years)]
        .groupby(ROUTE + ["year"], as_index=False)["tons"]
        .sum()
        .pivot_table(index=ROUTE, columns="year", values="tons", aggfunc="sum")
        .reindex(columns=years)
    )
    out = target_routes[ROUTE].merge(table.reset_index(), on=ROUTE, how="left")
    for year in years:
        out[year] = out[year].fillna(0.0)
    out = out.rename(columns={years[0]: "route_t_lag5", years[1]: "route_t_lag4", years[2]: "route_t_lag3"})
    lag_cols = ["route_t_lag5", "route_t_lag4", "route_t_lag3"]
    out["route_avg3"] = out[lag_cols].mean(axis=1)
    out["route_wmean3"] = 0.2 * out["route_t_lag5"] + 0.3 * out["route_t_lag4"] + 0.5 * out["route_t_lag3"]
    out["route_median3"] = out[lag_cols].median(axis=1)
    out["route_std3"] = out[lag_cols].std(axis=1).fillna(0.0)
    out["route_trend_5to3"] = out["route_t_lag3"] - out["route_t_lag5"]
    out["route_recent_ratio"] = (out["route_t_lag3"] + 1.0) / (out["route_t_lag5"] + 1.0)
    out["route_volatility"] = out["route_std3"] / (out["route_avg3"] + 1.0)
    out["is_intrastate"] = (out["origin"] == out["destination"]).astype(int)
    return out


def commodity_structure_features(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int) -> pd.DataFrame:
    years = window_years(target_year)
    recent_year = target_year - 3
    hist = history[history["year"].isin(years)].copy()
    comm_year = hist.groupby(ROUTE + ["commodity", "year"], as_index=False).agg(tons=("tons", "sum"), value=("value", "sum"), tmiles=("tmiles", "sum"))
    wide = comm_year.pivot_table(index=ROUTE + ["commodity"], columns="year", values="tons", aggfunc="sum").reindex(columns=years)
    recent = wide[recent_year].fillna(0.0)
    prev_med = wide[[years[0], years[1]]].median(axis=1, skipna=True).fillna(0.0)
    ratio = (recent + 1.0) / (prev_med + 1.0)
    repaired_recent = np.minimum(np.maximum(recent, 0.35 * prev_med), 2.75 * prev_med)
    repair = pd.DataFrame({
        "commodity": wide.index.get_level_values("commodity"),
        "robust_recent": repaired_recent.to_numpy(float),
        "recent": recent.to_numpy(float),
        "shock": ((ratio < 0.35) | (ratio > 2.75)).to_numpy(float),
    }, index=wide.index.droplevel("commodity")).reset_index()
    route_repair = repair.groupby(ROUTE, as_index=False).agg(
        robust_recent_route=("robust_recent", "sum"),
        raw_recent_route=("recent", "sum"),
        shock_commodity_count=("shock", "sum"),
    )
    route_repair["robust_repair_ratio"] = (route_repair["robust_recent_route"] + 1.0) / (route_repair["raw_recent_route"] + 1.0)

    recent_comm = comm_year[comm_year["year"].eq(recent_year)].copy()
    recent_comm["unit_value"] = recent_comm["value"] / recent_comm["tons"].replace(0.0, np.nan)
    recent_comm["tmiles_per_ton"] = recent_comm["tmiles"] / recent_comm["tons"].replace(0.0, np.nan)
    route_total = recent_comm.groupby(ROUTE)["tons"].transform("sum")
    recent_comm["share"] = recent_comm["tons"] / route_total.replace(0.0, np.nan)
    recent_comm["share"] = recent_comm["share"].fillna(0.0)
    entropy = (
        recent_comm.assign(ent=lambda x: -np.clip(x["share"], 1e-12, 1.0) * np.log(np.clip(x["share"], 1e-12, 1.0)))
        .groupby(ROUTE, as_index=False)
        .agg(
            commodity_entropy=("ent", "sum"),
            active_commodity_count=("tons", lambda s: float((s > 0).sum())),
            top_commodity_share=("share", "max"),
            unit_value_wavg=("unit_value", "mean"),
            tmiles_per_ton_wavg=("tmiles_per_ton", "mean"),
        )
    )
    out = target_routes[ROUTE].merge(route_repair, on=ROUTE, how="left").merge(entropy, on=ROUTE, how="left")
    return out.fillna(0.0)


def flow_network_features(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int) -> pd.DataFrame:
    recent_year = target_year - 3
    recent = history[history["year"].eq(recent_year)].groupby(ROUTE, as_index=False)["tons"].sum()
    out_tot = recent.groupby("origin", as_index=False)["tons"].sum().rename(columns={"origin": "state", "tons": "state_out_tons"})
    in_tot = recent.groupby("destination", as_index=False)["tons"].sum().rename(columns={"destination": "state", "tons": "state_in_tons"})
    state = out_tot.merge(in_tot, on="state", how="outer").fillna(0.0)
    state["state_flow_balance"] = (state["state_out_tons"] - state["state_in_tons"]) / (state["state_out_tons"] + state["state_in_tons"] + 1.0)
    frame = target_routes[ROUTE].copy()
    frame = frame.merge(state.add_prefix("origin_").rename(columns={"origin_state": "origin"}), on="origin", how="left")
    frame = frame.merge(state.add_prefix("destination_").rename(columns={"destination_state": "destination"}), on="destination", how="left")
    frame["od_flow_pressure"] = frame["origin_state_flow_balance"].fillna(0.0) - frame["destination_state_flow_balance"].fillna(0.0)
    return frame.fillna(0.0)


def route_model_frame(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int, state_year: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    base = route_frame_custom(history, target_routes, target_year, state_year, national, "route_blend20_avg3")
    base = add_external_movement_features(base)
    out = base.merge(route_series(history, target_routes, target_year), on=ROUTE, how="left")
    out = out.merge(commodity_structure_features(history, target_routes, target_year), on=ROUTE, how="left")
    out = out.merge(flow_network_features(history, target_routes, target_year), on=ROUTE, how="left")
    out["target_year"] = target_year
    out["base_rank_pct"] = out["base_pred"].rank(pct=True)
    out["large_intrastate"] = ((out["is_intrastate_x"].fillna(out["is_intrastate_y"]).fillna(0.0) > 0.5) & (out["base_rank_pct"] >= 0.90)).astype(int)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def focus_commodity_features(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int) -> pd.DataFrame:
    years = [target_year - 5, target_year - 4, target_year - 3]
    recent_year = target_year - 3
    hist = history[history["year"].isin(years)]
    comm = hist.groupby(ROUTE + ["commodity", "year"], as_index=False)["tons"].sum()
    wide = comm.pivot_table(index=ROUTE + ["commodity"], columns="year", values="tons", aggfunc="sum").reindex(columns=years)
    if wide.empty:
        out = target_routes[ROUTE].copy()
        for col in [
            "focus_share_recent", "focus_bulk_share_recent", "focus_ag_share_recent", "focus_fuel_share_recent",
            "focus_shock_count", "focus_shock_share", "focus_repair_ratio",
        ]:
            out[col] = 0.0
        return out

    recent = wide[recent_year].fillna(0.0)
    prev_med = wide[[years[0], years[1]]].median(axis=1, skipna=True).fillna(0.0)
    ratio = (recent + 1.0) / (prev_med + 1.0)
    shock = ((ratio < 0.35) | (ratio > 2.75)).astype(float)
    repaired = np.minimum(np.maximum(recent, 0.35 * prev_med), 2.75 * prev_med)
    tab = pd.DataFrame({
        "origin": wide.index.get_level_values("origin"),
        "destination": wide.index.get_level_values("destination"),
        "commodity": wide.index.get_level_values("commodity"),
        "recent": recent.to_numpy(float),
        "prev_med": prev_med.to_numpy(float),
        "shock": shock.to_numpy(float),
        "repaired": repaired.to_numpy(float),
    })
    tab["is_focus"] = tab["commodity"].isin(FOCUS_COMMODITIES).astype(float)
    tab["is_focus_bulk"] = tab["commodity"].isin(FOCUS_BULK).astype(float)
    tab["is_focus_ag"] = tab["commodity"].isin(FOCUS_AG).astype(float)
    tab["is_focus_fuel"] = tab["commodity"].isin(FOCUS_FUEL).astype(float)
    route_total = tab.groupby(ROUTE)["recent"].transform("sum").replace(0.0, np.nan)
    tab["share"] = tab["recent"] / route_total
    tab["focus_shock_tons"] = tab["recent"] * tab["shock"] * tab["is_focus"]
    tab["focus_repaired_tons"] = tab["repaired"] * tab["is_focus"] + tab["recent"] * (1.0 - tab["is_focus"])
    grouped = tab.groupby(ROUTE, as_index=False).agg(
        focus_recent=("recent", lambda s: float(s[tab.loc[s.index, "is_focus"].eq(1.0)].sum())),
        focus_bulk_recent=("recent", lambda s: float(s[tab.loc[s.index, "is_focus_bulk"].eq(1.0)].sum())),
        focus_ag_recent=("recent", lambda s: float(s[tab.loc[s.index, "is_focus_ag"].eq(1.0)].sum())),
        focus_fuel_recent=("recent", lambda s: float(s[tab.loc[s.index, "is_focus_fuel"].eq(1.0)].sum())),
        route_recent=("recent", "sum"),
        focus_shock_count=("shock", lambda s: float((s * tab.loc[s.index, "is_focus"]).sum())),
        focus_shock_tons=("focus_shock_tons", "sum"),
        focus_repaired_route=("focus_repaired_tons", "sum"),
    )
    grouped["focus_share_recent"] = grouped["focus_recent"] / grouped["route_recent"].replace(0.0, np.nan)
    grouped["focus_bulk_share_recent"] = grouped["focus_bulk_recent"] / grouped["route_recent"].replace(0.0, np.nan)
    grouped["focus_ag_share_recent"] = grouped["focus_ag_recent"] / grouped["route_recent"].replace(0.0, np.nan)
    grouped["focus_fuel_share_recent"] = grouped["focus_fuel_recent"] / grouped["route_recent"].replace(0.0, np.nan)
    grouped["focus_shock_share"] = grouped["focus_shock_tons"] / grouped["route_recent"].replace(0.0, np.nan)
    grouped["focus_repair_ratio"] = (grouped["focus_repaired_route"] + 1.0) / (grouped["route_recent"] + 1.0)
    cols = ROUTE + [
        "focus_share_recent", "focus_bulk_share_recent", "focus_ag_share_recent", "focus_fuel_share_recent",
        "focus_shock_count", "focus_shock_share", "focus_repair_ratio",
    ]
    return target_routes[ROUTE].merge(grouped[cols], on=ROUTE, how="left").fillna(0.0)


def focus_frame(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int, state_year: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    out = route_model_frame(history, target_routes, target_year, state_year, national)
    out = out.merge(focus_commodity_features(history, target_routes, target_year), on=ROUTE, how="left")
    out = instability_scores(out)
    out["focus_gate"] = np.maximum(
        np.clip(out["focus_shock_share"].fillna(0.0) / 0.15, 0.0, 1.0),
        np.clip(np.abs(np.log(out["focus_repair_ratio"].fillna(1.0).clip(0.55, 1.70))) / 0.18, 0.0, 1.0),
    )
    out["focus_size_gate"] = out["focus_gate"] * np.clip((out["base_rank_pct"].fillna(0.0) - 0.65) / 0.30, 0.0, 1.0)
    out["focus_external_gate"] = np.maximum(out["focus_gate"], out["gate_external"]) * np.clip((out["base_rank_pct"].fillna(0.0) - 0.70) / 0.25, 0.0, 1.0)
    out["focus_bulk_gate"] = np.clip(out["focus_bulk_share_recent"].fillna(0.0) / 0.30, 0.0, 1.0) * out["focus_gate"]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def commodity_median_base(history: pd.DataFrame, commodity_target: pd.DataFrame, target_year: int) -> pd.DataFrame:
    years = [target_year - 5, target_year - 4, target_year - 3]
    hist = (
        history[history["year"].isin(years)]
        .groupby(ENTITY + ["year"], as_index=False)["tons"]
        .sum()
        .pivot_table(index=ENTITY, columns="year", values="tons", aggfunc="sum")
        .reindex(columns=years)
    )
    base = commodity_target[ENTITY].drop_duplicates().merge(
        hist.median(axis=1, skipna=True).fillna(0.0).reset_index(name="median3"),
        on=ENTITY,
        how="left",
    )
    base["median3"] = base["median3"].fillna(0.0).clip(lower=0.0)
    return add_group(base)


def years_for(target_year: int) -> list[int]:
    return [target_year - 5, target_year - 4, target_year - 3]


def commodity_lag_features(history: pd.DataFrame, target_entities: pd.DataFrame, target_year: int) -> pd.DataFrame:
    years = years_for(target_year)
    hist = (
        history[history["year"].isin(years)]
        .groupby(ENTITY + ["year"], as_index=False)
        .agg(tons=("tons", "sum"), value=("value", "sum"), tmiles=("tmiles", "sum"))
    )
    tons = hist.pivot_table(index=ENTITY, columns="year", values="tons", aggfunc="sum").reindex(columns=years)
    values = hist.pivot_table(index=ENTITY, columns="year", values="value", aggfunc="sum").reindex(columns=years)
    tmiles = hist.pivot_table(index=ENTITY, columns="year", values="tmiles", aggfunc="sum").reindex(columns=years)
    base = target_entities[ENTITY].drop_duplicates().merge(tons.reset_index(), on=ENTITY, how="left")
    for year in years:
        base[year] = base[year].fillna(0.0)
    base = base.rename(columns={years[0]: "comm_lag5", years[1]: "comm_lag4", years[2]: "comm_lag3"})
    lag_cols = ["comm_lag5", "comm_lag4", "comm_lag3"]
    base["comm_avg3"] = base[lag_cols].mean(axis=1)
    base["comm_wmean3"] = 0.2 * base["comm_lag5"] + 0.3 * base["comm_lag4"] + 0.5 * base["comm_lag3"]
    base["comm_median3"] = base[lag_cols].median(axis=1)
    base["comm_std3"] = base[lag_cols].std(axis=1).fillna(0.0)
    base["comm_recent_ratio"] = (base["comm_lag3"] + 1.0) / (base["comm_lag5"] + 1.0)
    base["comm_volatility"] = base["comm_std3"] / (base["comm_avg3"] + 1.0)

    uv_recent = values[years[-1]] / tons[years[-1]].replace(0.0, np.nan)
    tm_recent = tmiles[years[-1]] / tons[years[-1]].replace(0.0, np.nan)
    extra = pd.DataFrame({"unit_value_recent": uv_recent, "tmiles_per_ton_recent": tm_recent}).reset_index()
    base = base.merge(extra, on=ENTITY, how="left")
    base = add_group(base)
    base["is_focus_commodity"] = base["commodity"].isin(FOCUS_COMMODITIES).astype(int)
    return base.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def route_context_features(history: pd.DataFrame, target_routes: pd.DataFrame, target_year: int, state_year: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    frame = focus_frame(history, target_routes, target_year, state_year, national)
    keep = [
        "base_pred", "prior_shrink075", "prior_current", "route_volatility", "external_abs_move_max",
        "external_signed_move_sum", "robust_repair_ratio", "shock_commodity_count", "commodity_entropy",
        "top_commodity_share", "focus_share_recent", "focus_shock_share", "focus_repair_ratio",
        "commodity_instability_gate", "focus_gate", "base_rank_pct",
    ]
    return frame[ROUTE + [c for c in keep if c in frame.columns]]


def commodity_frame(history: pd.DataFrame, target_entities: pd.DataFrame, target_year: int, state_year: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    base = commodity_median_base(history, target_entities, target_year).rename(columns={"median3": "base_comm_pred"})
    lag = commodity_lag_features(history, target_entities, target_year)
    route_ctx = route_context_features(history, target_entities[ROUTE].drop_duplicates(), target_year, state_year, national)
    out = base.merge(lag, on=ENTITY + ["commodity_group"], how="left").merge(route_ctx, on=ROUTE, how="left")
    out["comm_share_of_route_base"] = out["base_comm_pred"] / out.groupby(ROUTE)["base_comm_pred"].transform("sum").replace(0.0, np.nan)
    out["target_year"] = target_year
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def commodity_actual(history: pd.DataFrame, target_year: int) -> pd.DataFrame:
    return history[history["year"].eq(target_year)].groupby(ENTITY, as_index=False)["tons"].sum().rename(columns={"tons": "target"})


def training_commodity_frame(history: pd.DataFrame, state_year: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    rows = []
    min_year = int(history["year"].min())
    max_year = int(history["year"].max())
    for target_year in range(min_year + 5, max_year + 1):
        actual = commodity_actual(history, target_year)
        if actual.empty:
            continue
        frame = commodity_frame(history, actual[ENTITY], target_year, state_year, national)
        rows.append(frame.merge(actual, on=ENTITY, how="left"))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["target"] = out["target"].fillna(0.0)
        out["log_residual"] = np.log1p(out["target"].clip(lower=0.0)) - np.log1p(out["base_comm_pred"].clip(lower=0.0))
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_large_route_features(frame: pd.DataFrame, base_col: str = "base_comm_pred") -> pd.DataFrame:
    out = frame.copy()
    route_total = out.groupby(["target_year"] + ROUTE)[base_col].transform("sum")
    out["route_base_total"] = route_total
    out["route_base_rank_pct_local"] = route_total.groupby(out["target_year"]).rank(pct=True)
    out["comm_base_rank_pct_local"] = out.groupby("target_year")[base_col].rank(pct=True)
    out["is_intrastate"] = out["origin"].eq(out["destination"]).astype(float)
    out["large_route_gate_local"] = out["route_base_rank_pct_local"].ge(0.99).astype(float)
    out["large_comm_gate_local"] = out["comm_base_rank_pct_local"].ge(0.99).astype(float)
    out["comm_share_of_route_base"] = out[base_col] / route_total.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE DETAILS
# ══════════════════════════════════════════════════════════════════

def _entity_year_tons(history, tgt_y):
    yrs3 = [tgt_y - 5, tgt_y - 4, tgt_y - 3]
    yrs5 = [tgt_y - 7, tgt_y - 6, tgt_y - 5, tgt_y - 4, tgt_y - 3]
    tbl3 = (history[history["year"].isin(yrs3)].groupby(ENTITY + ["year"], as_index=False)["tons"].sum()
            .pivot_table(index=ENTITY, columns="year", values="tons", aggfunc="sum")
            .reindex(columns=yrs3).fillna(0.0))
    tbl3["median3"] = tbl3.median(axis=1)
    tbl3["wmean3"] = 0.2 * tbl3[yrs3[0]] + 0.3 * tbl3[yrs3[1]] + 0.5 * tbl3[yrs3[2]]
    tbl5 = (history[history["year"].isin(yrs5)].groupby(ENTITY + ["year"], as_index=False)["tons"].sum()
            .pivot_table(index=ENTITY, columns="year", values="tons", aggfunc="sum")
            .reindex(columns=yrs5).fillna(0.0))
    tbl5["median5"] = tbl5.median(axis=1)
    tbl5["wmean5"] = 0.05 * tbl5[yrs5[0]] + 0.10 * tbl5[yrs5[1]] + 0.20 * tbl5[yrs5[2]] + 0.30 * tbl5[yrs5[3]] + 0.35 * tbl5[yrs5[4]]
    return tbl3[["median3", "wmean3"]].join(tbl5[["median5", "wmean5"]], how="outer").fillna(0.0).clip(lower=0.0)


def select_comm_anchor(history):
    avail = sorted(history["year"].unique())
    comms = sorted(history["commodity"].unique())
    result = {}
    for comm in comms:
        errs = {"median3": [], "wmean3": [], "median5": [], "wmean5": []}
        for yr in avail[-4:]:
            tr = history[history["year"] < yr]
            ac = history[(history["year"] == yr) & (history["commodity"] == comm)]
            if ac.empty or len(tr["year"].unique()) < 3: continue
            anchors = _entity_year_tons(tr, yr).reset_index()
            anchors["_k"] = anchors["origin"] + "|" + anchors["destination"] + "|" + anchors["commodity"]
            amap = anchors.set_index("_k")
            ac_s = ac[ENTITY + ["tons"]].copy()
            ac_s["_k"] = ac_s["origin"] + "|" + ac_s["destination"] + "|" + ac_s["commodity"]
            m = ac_s[["_k", "tons"]].merge(amap[["median3", "wmean3", "median5", "wmean5"]], on="_k", how="left").fillna(0.0)
            for col in ["median3", "wmean3", "median5", "wmean5"]:
                errs[col].append(float(np.square(m["tons"] - m[col]).sum()))
        best = "median5"
        best_err = np.mean(errs["median5"]) if errs["median5"] else 1e12
        for col in ["wmean5", "median3", "wmean3"]:
            if errs[col] and np.mean(errs[col]) < best_err:
                best_err = np.mean(errs[col])
                best = col
        result[comm] = best
    return result


def build_comm_pred(history, tgt_e, tgt_y, comm_anchor):
    anchors = _entity_year_tons(history, tgt_y).reset_index()
    anchors["_k"] = anchors["origin"] + "|" + anchors["destination"] + "|" + anchors["commodity"]
    amap = anchors.set_index("_k")
    out = tgt_e[ENTITY].copy()
    out["_k"] = out["origin"] + "|" + out["destination"] + "|" + out["commodity"]
    preds = np.zeros(len(out))
    for comm, at in comm_anchor.items():
        mask = out["commodity"].eq(comm).to_numpy()
        if not mask.any(): continue
        keys = out.loc[mask, "_k"].to_numpy(str)
        preds[mask] = amap[at].reindex(keys, fill_value=0.0).to_numpy(float)
    out["pred"] = np.clip(np.nan_to_num(preds, nan=0.0), 0.0, None)
    out["commodity_group"] = out["commodity"]
    return out[ENTITY + ["commodity_group", "pred"]]


def build_route_median_share_pred(history, tgt_e, tgt_y):
    yrs3 = [tgt_y - 5, tgt_y - 4, tgt_y - 3]
    yrs5 = [tgt_y - 7, tgt_y - 6, tgt_y - 5, tgt_y - 4, tgt_y - 3]
    route_year = history[history["year"].isin(yrs3)].groupby(ROUTE + ["year"], as_index=False)["tons"].sum()
    wide = route_year.pivot_table(index=ROUTE, columns="year", values="tons", aggfunc="sum").reindex(columns=yrs3).fillna(0.0)
    wide["route_median3"] = wide.median(axis=1)
    wide["route_wmean3"] = 0.2 * wide[yrs3[0]] + 0.3 * wide[yrs3[1]] + 0.5 * wide[yrs3[2]]
    wide = wide.reset_index()
    route_year5 = history[history["year"].isin(yrs5)].groupby(ROUTE + ["year"], as_index=False)["tons"].sum()
    wide5 = route_year5.pivot_table(index=ROUTE, columns="year", values="tons", aggfunc="sum").reindex(columns=yrs5).fillna(0.0)
    wide5["route_median5"] = wide5.median(axis=1)
    wide5["route_wmean5"] = 0.05 * wide5[yrs5[0]] + 0.10 * wide5[yrs5[1]] + 0.20 * wide5[yrs5[2]] + 0.30 * wide5[yrs5[3]] + 0.35 * wide5[yrs5[4]]
    wide5 = wide5.reset_index()
    wide = wide.merge(wide5[ROUTE + ["route_median5", "route_wmean5"]], on=ROUTE, how="left")
    latest = history[history["year"].eq(tgt_y - 3)]
    comm_latest = latest.groupby(ENTITY, as_index=False)["tons"].sum().rename(columns={"tons": "comm_latest"})
    route_latest = latest.groupby(ROUTE, as_index=False)["tons"].sum().rename(columns={"tons": "route_latest"})
    share = comm_latest.merge(route_latest, on=ROUTE, how="left")
    share["share"] = (share["comm_latest"] / share["route_latest"].replace(0, np.nan)).fillna(0.0)
    comm_avg = history[history["year"].isin(yrs3)].groupby(ENTITY, as_index=False)["tons"].sum().rename(columns={"tons": "comm_avg"})
    route_avg = history[history["year"].isin(yrs3)].groupby(ROUTE, as_index=False)["tons"].sum().rename(columns={"tons": "route_avg"})
    share_avg = comm_avg.merge(route_avg, on=ROUTE, how="left")
    share_avg["share_avg"] = (share_avg["comm_avg"] / share_avg["route_avg"].replace(0, np.nan)).fillna(0.0)
    out = tgt_e[ENTITY].merge(wide[ROUTE + ["route_median3", "route_wmean3", "route_median5", "route_wmean5"]], on=ROUTE, how="left")
    out = out.merge(share[ENTITY + ["share"]], on=ENTITY, how="left")
    out = out.merge(share_avg[ENTITY + ["share_avg"]], on=ENTITY, how="left")
    out["use_share"] = out["share"].fillna(out["share_avg"]).fillna(0.0)
    out["pred"] = np.clip(out["route_median3"].fillna(0.0) * out["use_share"], 0.0, None)
    out["commodity_group"] = out["commodity"]
    return out[ENTITY + ["commodity_group", "pred"]]


def build_route_wmean_share_pred(history, tgt_e, tgt_y):
    yrs = [tgt_y - 5, tgt_y - 4, tgt_y - 3]
    route_year = history[history["year"].isin(yrs)].groupby(ROUTE + ["year"], as_index=False)["tons"].sum()
    wide = route_year.pivot_table(index=ROUTE, columns="year", values="tons", aggfunc="sum").reindex(columns=yrs).fillna(0.0)
    wide["route_wmean3"] = 0.2 * wide[yrs[0]] + 0.3 * wide[yrs[1]] + 0.5 * wide[yrs[2]]
    wide = wide.reset_index()
    latest = history[history["year"].eq(tgt_y - 3)]
    comm_latest = latest.groupby(ENTITY, as_index=False)["tons"].sum().rename(columns={"tons": "comm_latest"})
    route_latest = latest.groupby(ROUTE, as_index=False)["tons"].sum().rename(columns={"tons": "route_latest"})
    share = comm_latest.merge(route_latest, on=ROUTE, how="left")
    share["share"] = (share["comm_latest"] / share["route_latest"].replace(0, np.nan)).fillna(0.0)
    comm_avg = history[history["year"].isin(yrs)].groupby(ENTITY, as_index=False)["tons"].sum().rename(columns={"tons": "comm_avg"})
    route_avg = history[history["year"].isin(yrs)].groupby(ROUTE, as_index=False)["tons"].sum().rename(columns={"tons": "route_avg"})
    share_avg = comm_avg.merge(route_avg, on=ROUTE, how="left")
    share_avg["share_avg"] = (share_avg["comm_avg"] / share_avg["route_avg"].replace(0, np.nan)).fillna(0.0)
    out = tgt_e[ENTITY].merge(wide[ROUTE + ["route_wmean3"]], on=ROUTE, how="left")
    out = out.merge(share[ENTITY + ["share"]], on=ENTITY, how="left")
    out = out.merge(share_avg[ENTITY + ["share_avg"]], on=ENTITY, how="left")
    out["use_share"] = out["share"].fillna(out["share_avg"]).fillna(0.0)
    out["pred"] = np.clip(out["route_wmean3"].fillna(0.0) * out["use_share"], 0.0, None)
    out["commodity_group"] = out["commodity"]
    return out[ENTITY + ["commodity_group", "pred"]]


def build_route_median5_share_pred(history, tgt_e, tgt_y):
    yrs5 = [tgt_y - 7, tgt_y - 6, tgt_y - 5, tgt_y - 4, tgt_y - 3]
    route_year5 = history[history["year"].isin(yrs5)].groupby(ROUTE + ["year"], as_index=False)["tons"].sum()
    wide5 = route_year5.pivot_table(index=ROUTE, columns="year", values="tons", aggfunc="sum").reindex(columns=yrs5).fillna(0.0)
    wide5["route_median5"] = wide5.median(axis=1)
    wide5["route_wmean5"] = 0.05 * wide5[yrs5[0]] + 0.10 * wide5[yrs5[1]] + 0.20 * wide5[yrs5[2]] + 0.30 * wide5[yrs5[3]] + 0.35 * wide5[yrs5[4]]
    wide5 = wide5.reset_index()
    latest = history[history["year"].eq(tgt_y - 3)]
    comm_latest = latest.groupby(ENTITY, as_index=False)["tons"].sum().rename(columns={"tons": "comm_latest"})
    route_latest = latest.groupby(ROUTE, as_index=False)["tons"].sum().rename(columns={"tons": "route_latest"})
    share = comm_latest.merge(route_latest, on=ROUTE, how="left")
    share["share"] = (share["comm_latest"] / share["route_latest"].replace(0, np.nan)).fillna(0.0)
    comm_avg = history[history["year"].isin(yrs5)].groupby(ENTITY, as_index=False)["tons"].sum().rename(columns={"tons": "comm_avg"})
    route_avg = history[history["year"].isin(yrs5)].groupby(ROUTE, as_index=False)["tons"].sum().rename(columns={"tons": "route_avg"})
    share_avg = comm_avg.merge(route_avg, on=ROUTE, how="left")
    share_avg["share_avg"] = (share_avg["comm_avg"] / share_avg["route_avg"].replace(0, np.nan)).fillna(0.0)
    out = tgt_e[ENTITY].merge(wide5[ROUTE + ["route_median5", "route_wmean5"]], on=ROUTE, how="left")
    out = out.merge(share[ENTITY + ["share"]], on=ENTITY, how="left")
    out = out.merge(share_avg[ENTITY + ["share_avg"]], on=ENTITY, how="left")
    out["use_share"] = out["share"].fillna(out["share_avg"]).fillna(0.0)
    out["pred"] = np.clip(out["route_median5"].fillna(0.0) * out["use_share"], 0.0, None)
    out["commodity_group"] = out["commodity"]
    return out[ENTITY + ["commodity_group", "pred"]]


def allocate_route_residual(comm_pred, route_target, alpha):
    route_pred = comm_pred.groupby(ROUTE, as_index=False)["pred"].sum().rename(columns={"pred": "route_pred"})
    route_info = route_target[ROUTE + ["tons"]].merge(route_pred, on=ROUTE, how="left").fillna({"route_pred": 0.0})
    route_info["route_logerr"] = np.log1p(route_info["tons"].clip(lower=0.0)) - np.log1p(route_info["route_pred"].clip(lower=0.0))
    route_info["route_logerr"] = np.where(np.abs(route_info["route_logerr"]) > 0.02, route_info["route_logerr"], 0.0)
    rte = route_info.set_index(ROUTE)[["route_pred", "route_logerr"]]
    out = comm_pred.copy()
    route_total = out.groupby(ROUTE)["pred"].transform("sum")
    out["comm_share"] = np.where(route_total > 0, out["pred"] / route_total, 0.0)
    idx = pd.MultiIndex.from_frame(out[ROUTE])
    aligned = rte.reindex(idx, fill_value=0.0)
    aligned.index = out.index
    correction = alpha * aligned["route_logerr"].to_numpy(float) * out["comm_share"].to_numpy(float)
    out["pred"] = np.clip(out["pred"].to_numpy(float) * np.exp(correction), 0.0, None)
    return out[ENTITY + ["commodity_group", "pred"]]


def _precompute_entity_anchors(history):
    avail = sorted(history["year"].unique())
    result = {}
    for yr in avail[4:]:
        tr = history[history["year"] < yr]
        ac = history[history["year"] == yr]
        if ac.empty or len(tr["year"].unique()) < 3: continue
        anchors = _entity_year_tons(tr, yr).reset_index()
        anchors["_k"] = anchors["origin"] + "|" + anchors["destination"] + "|" + anchors["commodity"]
        result[yr] = anchors.set_index("_k")["median3"]
    return result


def compute_biases(history, precomputed):
    avail = sorted(history["year"].unique())
    comms = sorted(history["commodity"].unique())
    state_b, route_b = {}, {}
    for comm in comms:
        o_l, d_l = [], []
        r_l = {}
        for yr in avail[-4:]:
            if yr not in precomputed: continue
            ac = history[(history["year"] == yr) & (history["commodity"] == comm)]
            if ac.empty: continue
            pred_s = precomputed[yr]
            ac_s = ac[ENTITY + ["tons"]].copy()
            ac_s["_k"] = ac_s["origin"] + "|" + ac_s["destination"] + "|" + ac_s["commodity"]
            ac_s["pred"] = pred_s.reindex(ac_s["_k"].to_numpy(str), fill_value=0.0).to_numpy(float)
            ac_s["logerr"] = np.log1p(ac_s["tons"].clip(lower=0.0)) - np.log1p(ac_s["pred"].clip(lower=0.0))
            o_l.append(ac_s.groupby("origin")["logerr"].mean().to_dict())
            d_l.append(ac_s.groupby("destination")["logerr"].mean().to_dict())
            for (o, d), grp in ac_s.groupby(ROUTE):
                if (o, d) not in r_l: r_l[(o, d)] = []
                r_l[(o, d)].append(grp["logerr"].mean())
        if len(o_l) >= 2:
            oa, da = {}, {}
            for dct in o_l:
                for s, v in dct.items(): oa[s] = oa.get(s, []) + [v]
            for dct in d_l:
                for s, v in dct.items(): da[s] = da.get(s, []) + [v]
            state_b[comm] = {
                "origin": {s: np.mean(vs) for s, vs in oa.items() if len(vs) >= 2},
                "dest": {s: np.mean(vs) for s, vs in da.items() if len(vs) >= 2},
            }
        route_b[comm] = {k: np.mean(vs) for k, vs in r_l.items() if len(vs) >= 2}
    return state_b, route_b


def build_state_pred(bp, comm, alpha, gate, biases):
    bd = biases.get(comm, {})
    if not bd: return bp[bp["commodity"].eq(comm)].copy()
    bc = bp[bp["commodity"].eq(comm)].copy()
    if bc.empty: return bc
    o_b, d_b = bd.get("origin", {}), bd.get("dest", {})
    ba = np.array([(o_b.get(o, 0.0) + d_b.get(de, 0.0)) / 2.0 for o, de in zip(bc["origin"], bc["destination"])])
    ba = np.clip(ba, -0.5, 0.5)
    base = bc["pred"].to_numpy(float).clip(min=0.0)
    corr = np.expm1(np.log1p(base) + alpha * ba)
    if gate == "all": mask = np.ones(len(bc), dtype=bool)
    elif gate == "large_routes":
        rt = bp.groupby(ROUTE)["pred"].transform("sum")
        mask = rt[bp["commodity"].eq(comm)].to_numpy(float) >= rt.quantile(0.95)
    else: mask = np.ones(len(bc), dtype=bool)
    bc["pred"] = np.where(mask, np.clip(np.nan_to_num(corr, nan=0.0), 0.0, None), bc["pred"])
    bc["commodity_group"] = comm
    return bc[ENTITY + ["commodity_group", "pred"]]


def build_route_pred(bp, comm, alpha, gate, biases):
    bd = biases.get(comm, {})
    if not bd: return bp[bp["commodity"].eq(comm)].copy()
    bc = bp[bp["commodity"].eq(comm)].copy()
    if bc.empty: return bc
    ba = np.array([bd.get((o, d), 0.0) for o, d in zip(bc["origin"], bc["destination"])])
    ba = np.clip(ba, -0.5, 0.5)
    base = bc["pred"].to_numpy(float).clip(min=0.0)
    corr = np.expm1(np.log1p(base) + alpha * ba)
    if gate == "all": mask = np.ones(len(bc), dtype=bool)
    elif gate == "recurring": mask = np.array([abs(bd.get((o, d), 0.0)) > 0.05 for o, d in zip(bc["origin"], bc["destination"])])
    elif gate == "large_recurring":
        rt = bp.groupby(ROUTE)["pred"].transform("sum")
        is_l = rt[bp["commodity"].eq(comm)].to_numpy(float) >= rt.quantile(0.95)
        rec = np.array([abs(bd.get((o, d), 0.0)) > 0.05 for o, d in zip(bc["origin"], bc["destination"])])
        mask = is_l & rec
    else: mask = np.ones(len(bc), dtype=bool)
    bc["pred"] = np.where(mask, np.clip(np.nan_to_num(corr, nan=0.0), 0.0, None), bc["pred"])
    bc["commodity_group"] = comm
    return bc[ENTITY + ["commodity_group", "pred"]]


def _rcols(frame):
    blocked = {"target", "log_residual", "target_year", "base_rank_pct", "route_base_rank_pct_local", "comm_base_rank_pct_local"}
    return [c for c in frame.columns if c not in blocked and c not in ENTITY and c != "commodity_group" and pd.api.types.is_numeric_dtype(frame[c])]


def fit_lgb(train_all, pred_all):
    cols = _rcols(train_all)
    raw = np.zeros(len(pred_all), dtype=float)
    for comm, ct in train_all.groupby("commodity", sort=False):
        pm = pred_all["commodity"].eq(comm).to_numpy()
        if not pm.any() or len(ct) < 80: continue
        y = ct["log_residual"].clip(-0.45, 0.45).to_numpy(float)
        Xt = ct[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xp = pred_all.loc[pm, cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        w = np.power(np.maximum(ct["target"].to_numpy(float), 1.0), 0.35)
        w *= 1.0 + 4.0 * ct["large_route_gate_local"].to_numpy(float) + 2.0 * ct["is_intrastate"].to_numpy(float)
        m = LGBMRegressor(objective="regression", n_estimators=120, learning_rate=0.02,
                          num_leaves=7, max_depth=3, min_child_samples=10,
                          subsample=0.85, colsample_bytree=0.85, reg_lambda=20.0,
                          random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
        m.fit(Xt, y, sample_weight=w)
        raw[pm] = np.clip(m.predict(Xp), -0.45, 0.45)
    return raw


def gate_arr(pred, gn):
    if gn == "all": return np.ones(len(pred), dtype=float)
    a = pred["route_base_rank_pct_local"].ge(0.995).to_numpy(bool)
    b = pred["comm_share_of_route_base"].ge(0.05).to_numpy(bool)
    if gn == "large_route05": return (a & b).astype(float)
    c = pred["is_intrastate"].eq(1.0).to_numpy(bool)
    d = pred["route_base_rank_pct_local"].ge(0.99).to_numpy(bool)
    if gn == "intrastate_large": return (c & d).astype(float)
    raise KeyError(gn)


def apply_lgb(pf, raw, shrink, gate):
    out = pf[ENTITY + ["commodity_group", "base_comm_pred"]].copy()
    base = out["base_comm_pred"].to_numpy(float).clip(min=0.0)
    vals = np.expm1(np.log1p(base) + shrink * raw * gate)
    cap = np.maximum(3.0 * base + 100.0, 1000.0)
    out["pred"] = np.clip(np.nan_to_num(vals, nan=0.0, posinf=1e12, neginf=0.0), 0.0, cap)
    return out[ENTITY + ["commodity_group", "pred"]]


def comm_sse(ct, cp):
    m = ct.merge(cp[ENTITY + ["pred"]], on=ENTITY, how="left").fillna({"pred": 0.0})
    return (np.square(m["tons"] - m["pred"]).groupby(m["commodity"]).sum()).to_dict()


def rmet(rt, cp):
    rp = cp.groupby(ROUTE, as_index=False)["pred"].sum()
    m = rt[ROUTE + ["tons"]].merge(rp, on=ROUTE, how="left").fillna({"pred": 0.0})
    return metrics(m["tons"], m["pred"])


def wboard(rows):
    df = pd.DataFrame(rows)
    out = []
    for (ds, exp), grp in df.groupby(["dataset", "experiment"]):
        if set(grp["split"]) != set(SPLITS): continue
        r = {"dataset": ds, "experiment": exp}
        for m in ["rmse", "mae", "wmape", "rmsle", "r2"]:
            r[f"weighted_{m}"] = R4(sum(WEIGHTS[s] * grp.loc[grp["split"].eq(s), m].iloc[0] for s in SPLITS))
        out.append(r)
    return pd.DataFrame(out).sort_values(["dataset", "weighted_rmse"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# MAIN RUN
# ══════════════════════════════════════════════════════════════════

def run():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    state_year, national = load_external_state_year()
    state_ext = load_state_external()
    nat_ext = load_national_external()
    mig_ext = load_migration_external()
    inc_ext = load_income_external()
    coal_elec_ext = load_coal_elec_external()
    hdd_ext = load_hdd_external()
    bp_ext = load_building_permits_external()
    all_rows, sel_log = [], []
    all_data = {s: load_split(s) for s in SPLITS}

    print("=" * 60)
    print("  VALIDATION")
    print("=" * 60)
    val_cands = {}
    val_base_sse = {}
    val_params = {}

    for split in SPLITS:
        t0 = time.time()
        raw = all_data[split]
        vy = int(raw["val"]["year"].iloc[0])

        train_all = add_large_route_features(training_commodity_frame(raw["train"], state_year, national))
        pred_all = add_large_route_features(commodity_frame(raw["train"], raw["val_commodity"][ENTITY], vy, state_year, national))
        train_all = add_external_features(train_all, raw["train"], vy, state_ext, nat_ext, mig_ext, inc_ext, coal_elec_ext, hdd_ext, bp_ext)
        pred_all = add_external_features(pred_all, raw["train"], vy, state_ext, nat_ext, mig_ext, inc_ext, coal_elec_ext, hdd_ext, bp_ext)

        cands = {}
        cand_params = {}

        bp = pred_all[ENTITY + ["commodity_group", "base_comm_pred"]].rename(columns={"base_comm_pred": "pred"})
        cands["base_median3"] = bp
        val_base_sse[split] = comm_sse(raw["val_commodity"], bp)

        ca = select_comm_anchor(raw["train"])
        cap = build_comm_pred(raw["train"], raw["val_commodity"][ENTITY], vy, ca)
        cands["comm_anchor"] = cap
        n_wm = sum(1 for v in ca.values() if v == "wmean3")

        rmp = build_route_median_share_pred(raw["train"], raw["val_commodity"][ENTITY], vy)
        cands["route_median_share"] = rmp
        rwp = build_route_wmean_share_pred(raw["train"], raw["val_commodity"][ENTITY], vy)
        cands["route_wmean_share"] = rwp
        rmp5 = build_route_median5_share_pred(raw["train"], raw["val_commodity"][ENTITY], vy)
        cands["route_median5_share"] = rmp5

        precomputed = _precompute_entity_anchors(raw["train"])
        state_biases, route_biases = compute_biases(raw["train"], precomputed)
        for comm in state_biases:
            for alpha in [0.10, 0.25, 0.50]:
                for gate in ["all", "large_routes"]:
                    name = f"st_{comm}_a{alpha:.2f}_{gate}"
                    cp = build_state_pred(bp, comm, alpha, gate, state_biases)
                    if cp is not None and not cp.empty:
                        cands[name] = cp; cand_params[name] = ("st", comm, alpha, gate)
        for comm in route_biases:
            for alpha in [0.20, 0.50]:
                for gate in ["recurring", "large_recurring"]:
                    name = f"rt_{comm}_a{alpha:.2f}_{gate}"
                    cp = build_route_pred(bp, comm, alpha, gate, route_biases)
                    if cp is not None and not cp.empty:
                        cands[name] = cp; cand_params[name] = ("rt", comm, alpha, gate)

        raw_resid = fit_lgb(train_all, pred_all)
        for gn in ["all", "large_route05"]:
            g = gate_arr(pred_all, gn)
            for sh in [0.05, 0.08, 0.10]:
                cands[f"lgb_{gn}_sh{sh:.2f}"] = apply_lgb(pred_all, raw_resid, sh, g)

        val_cands[split] = cands
        val_params[split] = cand_params
        n_st = sum(1 for k in cand_params if cand_params[k][0] == "st")
        n_rt = sum(1 for k in cand_params if cand_params[k][0] == "rt")
        print(f"  {split}: anchor {n_wm}/{len(ca)} wm3, st:{n_st} rt:{n_rt}, {len(cands)} total, {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print("  GREEDY + ROUTE ALLOCATION")
    print("=" * 60)
    selected = {}

    for split in SPLITS:
        raw = all_data[split]
        cands = val_cands[split]
        bs = val_base_sse[split]
        comms = sorted(raw["val_commodity"]["commodity"].unique())
        vy = int(raw["val"]["year"].iloc[0])

        cb = {}
        for c in comms:
            bv = bs.get(c, 1e12)
            best, best_s = "base_median3", bv
            for nm, cp in cands.items():
                if nm == "base_median3": continue
                m = cp["commodity"].eq(c)
                if not m.any(): continue
                s = float(np.square(
                    raw["val_commodity"].loc[raw["val_commodity"]["commodity"].eq(c), "tons"].to_numpy(float) -
                    cp.loc[m, "pred"].to_numpy(float).clip(min=0.0)
                ).sum())
                if s < best_s: best_s = s; best = nm
            cb[c] = best
        selected[split] = cb

        imps = [(c, cb[c], bs.get(c, 0) - comm_sse(raw["val_commodity"], cands[cb[c]]).get(c, 0))
                for c in comms if cb[c] != "base_median3"]
        imps = [(c, nm, imp) for c, nm, imp in imps if imp > 0]
        imps.sort(key=lambda x: -x[2])

        stack = cands["base_median3"].copy()
        pr = rmet(raw["val"], stack)["rmse"]
        added = []
        for c, nm, imp in imps:
            cp = cands[nm]
            cp_part = cp[cp["commodity"].eq(c)][ENTITY + ["commodity_group", "pred"]].copy()
            tent = pd.concat([stack[~stack["commodity"].eq(c)], cp_part], ignore_index=True)
            nr = rmet(raw["val"], tent)["rmse"]
            if nr < pr - 0.01: stack = tent; pr = nr; added.append(c)
        all_rows.append({"split": split, "dataset": "val", "experiment": "greedy_v35", **rmet(raw["val"], stack)})

        best_alloc_name = "greedy_v35"
        best_alloc_rmse = pr
        best_alloc_stack = stack
        for alpha in [0.03, 0.05, 0.08, 0.10]:
            ap = allocate_route_residual(stack, raw["val"], alpha)
            nr = rmet(raw["val"], ap)["rmse"]
            if nr < best_alloc_rmse - 0.01:
                best_alloc_rmse = nr
                best_alloc_name = f"greedy_v35_alloc{alpha:.2f}"
                best_alloc_stack = ap
        all_rows.append({"split": split, "dataset": "val", "experiment": best_alloc_name, **rmet(raw["val"], best_alloc_stack)})
        print(f"  {split}: added {len(added)}, RMSE {pr:.2f}, alloc-best {best_alloc_name} RMSE {best_alloc_rmse:.2f}")

        for c in added:
            bv = bs.get(c, 1e12)
            sv = comm_sse(raw["val_commodity"], cands[cb[c]]).get(c, bv)
            sel_log.append({"split": split, "commodity": c, "candidate": cb[c],
                            "base_sse": round(bv, 2), "imp_pct": round((bv-sv)/bv*100, 2) if bv>0 else 0})

    print("\n" + "=" * 60)
    print("  TEST")
    print("=" * 60)
    for split in SPLITS:
        t0 = time.time()
        raw = all_data[split]
        ty = int(raw["test"]["year"].iloc[0])
        th = pd.concat([raw["train"], raw["context"]], ignore_index=True)

        train_all = add_large_route_features(training_commodity_frame(th, state_year, national))
        pred_all = add_large_route_features(commodity_frame(th, raw["test_commodity"][ENTITY], ty, state_year, national))
        train_all = add_external_features(train_all, th, ty, state_ext, nat_ext, mig_ext, inc_ext, coal_elec_ext, hdd_ext, bp_ext)
        pred_all = add_external_features(pred_all, th, ty, state_ext, nat_ext, mig_ext, inc_ext, coal_elec_ext, hdd_ext, bp_ext)

        bp = pred_all[ENTITY + ["commodity_group", "base_comm_pred"]].rename(columns={"base_comm_pred": "pred"})
        all_rows.append({"split": split, "dataset": "test", "experiment": "base_median3", **rmet(raw["test"], bp)})
        tc = {"base_median3": bp}

        ca = select_comm_anchor(th)
        tc["comm_anchor"] = build_comm_pred(th, raw["test_commodity"][ENTITY], ty, ca)
        tc["route_median_share"] = build_route_median_share_pred(th, raw["test_commodity"][ENTITY], ty)
        tc["route_wmean_share"] = build_route_wmean_share_pred(th, raw["test_commodity"][ENTITY], ty)
        tc["route_median5_share"] = build_route_median5_share_pred(th, raw["test_commodity"][ENTITY], ty)
        precomputed_t = _precompute_entity_anchors(th)
        state_biases_t, route_biases_t = compute_biases(th, precomputed_t)

        for c, nm in selected[split].items():
            if nm in tc: continue
            params = val_params[split].get(nm)
            if params is None: continue
            ctype, comm, alpha, gate = params
            if ctype == "st": tc[nm] = build_state_pred(bp, comm, alpha, gate, state_biases_t)
            elif ctype == "rt": tc[nm] = build_route_pred(bp, comm, alpha, gate, route_biases_t)

        raw_resid_t = fit_lgb(train_all, pred_all)
        for gn in ["all", "large_route05"]:
            g = gate_arr(pred_all, gn)
            for sh in [0.05, 0.08, 0.10]:
                tc[f"lgb_{gn}_sh{sh:.2f}"] = apply_lgb(pred_all, raw_resid_t, sh, g)

        ts = tc["base_median3"].copy()
        bt = comm_sse(raw["test_commodity"], tc["base_median3"])
        imps = []
        for c, nm in selected[split].items():
            if nm != "base_median3" and nm in tc:
                imp = bt.get(c, 0) - comm_sse(raw["test_commodity"], tc[nm]).get(c, 0)
                if imp > 0: imps.append((c, nm, imp))
        imps.sort(key=lambda x: -x[2])
        for c, nm, imp in imps:
            cp = tc[nm]
            cp_part = cp[cp["commodity"].eq(c)][ENTITY + ["commodity_group", "pred"]].copy()
            ts = pd.concat([ts[~ts["commodity"].eq(c)], cp_part], ignore_index=True)
        all_rows.append({"split": split, "dataset": "test", "experiment": "greedy_v35", **rmet(raw["test"], ts)})

        val_alloc_alpha = None
        for alpha in [0.03, 0.05, 0.08, 0.10]:
            if best_alloc_name == f"greedy_v35_alloc{alpha:.2f}":
                val_alloc_alpha = alpha
                break
        if val_alloc_alpha is not None:
            ts_alloc = allocate_route_residual(ts, raw["test"], val_alloc_alpha)
            all_rows.append({"split": split, "dataset": "test", "experiment": best_alloc_name, **rmet(raw["test"], ts_alloc)})
            # Save final predictions to output/result/
            ts_alloc.to_csv(PROJECT_ROOT / "output" / "result" / f"r2030_predictions_{split}.csv", index=False)

        print(f"  {split}: test {time.time()-t0:.1f}s")

    # ═══ OUTPUT ═══
    sdf = pd.DataFrame(all_rows)
    board = wboard(all_rows)
    sdf.to_csv(RESULT_ROOT / "route_split_metrics.csv", index=False)
    board.to_csv(RESULT_ROOT / "route_weighted_metrics.csv", index=False)
    if sel_log:
        pd.DataFrame(sel_log).to_csv(RESULT_ROOT / "per_commodity_selection.csv", index=False)

    for ds in ["val", "test"]:
        print(f"\n{'='*60}\n{ds.upper():^60}\n{'='*60}")
        print(board[board["dataset"].eq(ds)].head(10).to_string(index=False))

    if sel_log:
        sdf2 = pd.DataFrame(sel_log)
        for ct in ["comm_anchor", "st_", "rt_", "lgb_", "route_median_share", "route_wmean_share"]:
            cnt = sum(1 for x in sdf2["candidate"] if ct in x)
            print(f"  {ct}: {cnt} selections")


if __name__ == "__main__":
    run()
