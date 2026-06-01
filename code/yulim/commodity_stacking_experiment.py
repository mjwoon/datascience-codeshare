from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


RANDOM_SEED = 42
SPLIT_WEIGHTS = {"split_1": 0.2, "split_2": 0.3, "split_3": 0.5}
METRIC_COLUMNS = ["wRMSE", "RMSE", "MAE", "WMAPE", "RMSLE", "R2_Score"]
PRIMARY_METRIC = "wRMSE"
KEY_ROUTE = ["origin", "destination"]
KEY_COMM = ["origin", "destination", "commodity"]
DETAIL_COLS = ["origin", "destination", "commodity", "distance_band", "year", "tons", "value", "tmiles"]
GLOBAL_DATA_ROOT: Path | None = None
EXTERNAL_RATIO_CACHE: dict[str, pd.DataFrame] | None = None


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    commodity_stat: str = "median"
    median_mean_alpha: float = 0.0
    route_blend_stat: str | None = None
    route_blend_alpha: float = 0.0
    external_strength: float = 0.0


@dataclass(frozen=True)
class ResidualSpec:
    name: str
    baseline: BaselineSpec
    shrink: float
    gate: str = "directional_under"


def set_seed(seed: int = RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--artifact-dir", default="artifacts_commodity_stacking")
    parser.add_argument("--skip-residual", action="store_true")
    parser.add_argument(
        "--residual-baseline",
        default="auto",
        help="auto uses the best train-internal commodity baseline. Otherwise pass a BaselineSpec name.",
    )
    parser.add_argument("--max-pseudo-years", type=int, default=0, help="0 means use all train-internal pseudo years.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_data_root(explicit: str | None = None) -> Path:
    if explicit:
        candidates = [Path(explicit)]
    else:
        root = repo_root()
        candidates = [
            root / "data",
            root.parents[1] / "datascience" / "datascience" / "data",
            root.parents[1] / "data" / "DS_Truck_Route_Volume_Prediction" / "project" / "data",
        ]
    for candidate in candidates:
        split_root = candidate / "faf4_faf5_fixed_year_splits"
        if (split_root / "split_1" / "train.csv").exists():
            return candidate
    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No usable data root found. Checked:\n{checked}")


def split_root(data_root: Path) -> Path:
    return data_root / "faf4_faf5_fixed_year_splits"


def additional_data_root(data_root: Path) -> Path:
    candidates = [
        data_root / "additional_dataset",
        data_root / "additional_data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No usable additional data root found. Checked:\n{checked}")


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def split_dir(data_root: Path, split: str) -> Path:
    path = split_root(data_root) / split
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_detail(data_root: Path, split: str, include_context: bool) -> pd.DataFrame:
    sp = split_dir(data_root, split)
    train = read_csv(sp / "train.csv", usecols=DETAIL_COLS)
    frames = [train]
    if include_context:
        frames.append(read_csv(sp / "context.csv", usecols=DETAIL_COLS))
    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["year"].astype(int)
    return out


def load_target(data_root: Path, split: str, part: str, commodity: bool = False) -> pd.DataFrame:
    suffix = "_commodity" if commodity else ""
    path = split_dir(data_root, split) / f"{part}{suffix}.csv"
    out = read_csv(path)
    out["year"] = out["year"].astype(int)
    return out


def target_year(target: pd.DataFrame) -> int:
    years = sorted(target["year"].dropna().astype(int).unique().tolist())
    if len(years) != 1:
        raise ValueError(f"Expected one target year, got {years}")
    return int(years[0])


def metric_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    yp_clip = np.clip(yp, 0, None)
    yt_clip = np.clip(yt, 0, None)
    weights = yt_clip / (yt_clip.sum() + 1e-12)
    return {
        "wRMSE": float(np.sqrt(np.sum(weights * (yt - yp) ** 2))),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "WMAPE": float(np.sum(np.abs(yt - yp)) / (np.sum(np.abs(yt)) + 1e-12)),
        "RMSLE": float(np.sqrt(np.mean((np.log1p(yt_clip) - np.log1p(yp_clip)) ** 2))),
        "R2_Score": float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else math.nan,
    }


def weighted_metrics(rows: pd.DataFrame, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in METRIC_COLUMNS:
        total = 0.0
        for split, weight in SPLIT_WEIGHTS.items():
            value = rows.loc[rows["split"] == split, metric].iloc[0]
            total += weight * float(value)
        out[f"{prefix}_weighted_{metric}"] = total
    return out


def detail_to_route_year(detail: pd.DataFrame) -> pd.DataFrame:
    return detail.groupby(KEY_ROUTE + ["year"], as_index=False)["tons"].sum()


def detail_to_comm_year(detail: pd.DataFrame) -> pd.DataFrame:
    return detail.groupby(KEY_COMM + ["year"], as_index=False)["tons"].sum()


def available_window(detail: pd.DataFrame, tgt_year: int, window: int = 3) -> tuple[pd.DataFrame, int]:
    base_year = int(tgt_year) - 3
    start_year = base_year - window + 1
    hist = detail[(detail["year"] >= start_year) & (detail["year"] <= base_year)].copy()
    return hist, base_year


def add_year_weight(df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    year_weights = {base_year - 2: 0.2, base_year - 1: 0.3, base_year: 0.5}
    out = df.copy()
    out["_w"] = out["year"].map(year_weights).fillna(0.0).astype(float)
    out["_wx"] = out["tons"].astype(float) * out["_w"]
    return out


def grouped_weighted_mean(df: pd.DataFrame, keys: list[str], base_year: int, name: str) -> pd.DataFrame:
    weighted = add_year_weight(df, base_year)
    out = weighted.groupby(keys, as_index=False).agg(_wx=("_wx", "sum"), _w=("_w", "sum"))
    out[name] = out["_wx"] / out["_w"].replace(0, np.nan)
    out[name] = out[name].fillna(0.0)
    return out[keys + [name]]


def ratio_panel(df: pd.DataFrame, keys: list[str], value_col: str, out_col: str) -> pd.DataFrame:
    use = df[keys + ["year", value_col]].copy()
    use["year"] = use["year"].astype(int)
    use = use.sort_values(keys + ["year"])
    if keys:
        use[value_col] = use.groupby(keys)[value_col].ffill()
        lag = use.groupby(keys)[value_col].shift(3)
    else:
        use[value_col] = use[value_col].ffill()
        lag = use[value_col].shift(3)
    use[out_col] = use[value_col] / lag.replace(0, np.nan)
    use[out_col] = use[out_col].replace([np.inf, -np.inf], np.nan).clip(0.5, 1.8)
    return use[keys + ["year", out_col]]


def load_external_ratios(data_root: Path) -> dict[str, pd.DataFrame]:
    global EXTERNAL_RATIO_CACHE
    if EXTERNAL_RATIO_CACHE is not None:
        return EXTERNAL_RATIO_CACHE
    root = additional_data_root(data_root)
    panels: dict[str, pd.DataFrame] = {}

    def read(name: str) -> pd.DataFrame:
        return read_csv(root / name)

    gdp = read("state_unify_gdp.csv")
    panels["gdp"] = ratio_panel(gdp, ["state_name"], "real_GDP", "gdp_ratio3")
    pop = read("population_by_states.csv")
    panels["pop"] = ratio_panel(pop, ["state_name"], "population", "pop_ratio3")
    hdd = read("eia_heating_degree_days_by_state.csv")
    panels["hdd"] = ratio_panel(hdd, ["state_name"], "hdd_annual", "hdd_ratio3")
    coal_prod = read("eia_coal_production_by_state.csv")
    panels["coal_prod"] = ratio_panel(coal_prod, ["state_name"], "coal_production_ktons", "coal_prod_ratio3")
    coal_gen = read("eia_coal_electricity_gen_by_state.csv")
    panels["coal_gen"] = ratio_panel(coal_gen, ["state_name"], "coal_elec_gen_gwh", "coal_gen_ratio3")

    pce = read("pce_by_state_real_2017.csv")
    wanted = {
        "Goods": "pce_goods_ratio3",
        "Gasoline and other energy goods": "pce_energy_ratio3",
        "Food and beverages purchased for off-premises consumption": "pce_food_ratio3",
        "Motor vehicles and parts": "pce_motor_ratio3",
    }
    pce_frames = []
    for category, out_col in wanted.items():
        one = pce[pce["category"] == category].rename(columns={"pce_value": out_col.replace("_ratio3", "_value")})
        pce_frames.append(ratio_panel(one, ["state_name"], out_col.replace("_ratio3", "_value"), out_col))
    pce_panel = pce_frames[0]
    for frame in pce_frames[1:]:
        pce_panel = pce_panel.merge(frame, on=["state_name", "year"], how="outer")
    panels["pce"] = pce_panel

    crops = read("usda_crop_production_by_state.csv")
    crop_total = crops.groupby(["state_name", "year"], as_index=False)["production_bu"].sum()
    panels["crop"] = ratio_panel(crop_total, ["state_name"], "production_bu", "crop_prod_ratio3")

    natgas = read("eia_natgas_electricity_gen_national.csv")
    panels["natgas"] = ratio_panel(natgas, [], "natgas_elec_gen_gwh", "natgas_ratio3")
    fert = read("fao_fertilizer_price_index.csv")
    fert["fert_price"] = fert[["dap_price_usd_mt", "urea_price_usd_mt"]].mean(axis=1)
    panels["fert"] = ratio_panel(fert, [], "fert_price", "fert_price_ratio3")

    EXTERNAL_RATIO_CACHE = panels
    return panels


def attach_state_ratio(rows: pd.DataFrame, panel: pd.DataFrame, state_col: str, prefix: str) -> pd.DataFrame:
    renamed = panel.rename(columns={"state_name": state_col})
    value_cols = [c for c in renamed.columns if c not in [state_col, "year"]]
    renamed = renamed.rename(columns={c: f"{prefix}_{c}" for c in value_cols})
    return rows.merge(renamed, left_on=[state_col, "base_year"], right_on=[state_col, "year"], how="left").drop(columns=["year"], errors="ignore")


def apply_external_prior(rows: pd.DataFrame, base_year: int, strength: float) -> pd.DataFrame:
    if strength <= 0 or GLOBAL_DATA_ROOT is None:
        return rows
    panels = load_external_ratios(GLOBAL_DATA_ROOT)
    out = rows.copy()
    out = attach_state_ratio(out, panels["gdp"], "origin", "orig")
    out = attach_state_ratio(out, panels["gdp"], "destination", "dest")
    out = attach_state_ratio(out, panels["pop"], "origin", "orig")
    out = attach_state_ratio(out, panels["pop"], "destination", "dest")
    out = attach_state_ratio(out, panels["hdd"], "destination", "dest")
    out = attach_state_ratio(out, panels["coal_prod"], "origin", "orig")
    out = attach_state_ratio(out, panels["coal_gen"], "destination", "dest")
    out = attach_state_ratio(out, panels["pce"], "destination", "dest")
    out = attach_state_ratio(out, panels["crop"], "origin", "orig")
    out = out.merge(panels["natgas"], left_on="base_year", right_on="year", how="left").drop(columns=["year"], errors="ignore")
    out = out.merge(panels["fert"], left_on="base_year", right_on="year", how="left").drop(columns=["year"], errors="ignore")

    def col(name: str) -> pd.Series:
        if name in out.columns:
            return out[name].astype(float)
        return pd.Series(np.nan, index=out.index, dtype=float)

    macro = pd.concat(
        [
            col("orig_gdp_ratio3"),
            col("dest_gdp_ratio3"),
            col("orig_pop_ratio3"),
            col("dest_pop_ratio3"),
        ],
        axis=1,
    ).mean(axis=1)
    fuel = pd.concat([col("dest_pce_energy_ratio3"), col("dest_hdd_ratio3"), macro], axis=1).mean(axis=1)
    ag_food = pd.concat([col("orig_crop_prod_ratio3"), col("dest_pce_food_ratio3"), macro], axis=1).mean(axis=1)
    manufactured = pd.concat([col("dest_pce_goods_ratio3"), col("dest_pce_motor_ratio3"), macro], axis=1).mean(axis=1)
    chemicals = pd.concat([1.0 / col("fert_price_ratio3"), macro], axis=1).mean(axis=1)
    coal_bulk = pd.concat([col("orig_coal_prod_ratio3"), col("dest_coal_gen_ratio3"), 1.0 / col("natgas_ratio3")], axis=1).mean(axis=1)

    group = out["commodity"].map(commodity_group)
    ratio = macro.copy()
    ratio[group == "fuel"] = fuel[group == "fuel"]
    ratio[group == "ag_food"] = ag_food[group == "ag_food"]
    ratio[group == "manufactured"] = manufactured[group == "manufactured"]
    ratio[group == "chemicals"] = chemicals[group == "chemicals"]
    ratio[group == "coal_bulk"] = coal_bulk[group == "coal_bulk"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.75, 1.35)
    out["external_ratio"] = ratio
    out["external_multiplier"] = np.power(ratio, strength)
    out["pred"] = (out["pred"] * out["external_multiplier"]).clip(lower=0)
    keep_cols = [c for c in rows.columns] + ["external_ratio", "external_multiplier"]
    return out[keep_cols]


def route_stat(history: pd.DataFrame, base_year: int, stat: str) -> pd.DataFrame:
    route_year = detail_to_route_year(history)
    if route_year.empty:
        return pd.DataFrame(columns=KEY_ROUTE + ["route_ref"])
    if stat == "avg3":
        out = route_year.groupby(KEY_ROUTE, as_index=False)["tons"].mean().rename(columns={"tons": "route_ref"})
    elif stat == "wmean":
        out = grouped_weighted_mean(route_year, KEY_ROUTE, base_year, "route_ref")
    elif stat == "lag1":
        out = (
            route_year.sort_values(KEY_ROUTE + ["year"])
            .groupby(KEY_ROUTE, as_index=False)
            .tail(1)[KEY_ROUTE + ["tons"]]
            .rename(columns={"tons": "route_ref"})
        )
    else:
        raise ValueError(stat)
    return out


def make_comm_baseline(
    history: pd.DataFrame,
    tgt_year: int,
    spec: BaselineSpec,
) -> pd.DataFrame:
    hist, base_year = available_window(history, tgt_year)
    comm_year = detail_to_comm_year(hist)
    if comm_year.empty:
        return pd.DataFrame(columns=KEY_COMM + ["pred", "base_year", "route_prior_total", "route_ref", "route_gap_ratio"])

    grouped = comm_year.sort_values(KEY_COMM + ["year"]).groupby(KEY_COMM)
    stats = grouped["tons"].agg(
        base_median="median",
        base_mean="mean",
        base_std="std",
        base_min="min",
        base_max="max",
        base_last="last",
        base_count="count",
    ).reset_index()
    stats = stats.merge(grouped_weighted_mean(comm_year, KEY_COMM, base_year, "base_wmean"), on=KEY_COMM, how="left")
    stats["base_wmean"] = stats["base_wmean"].fillna(0.0)
    stats["base_std"] = stats["base_std"].fillna(0.0)

    if spec.commodity_stat == "median":
        stats["pred"] = stats["base_median"]
    elif spec.commodity_stat == "mean":
        stats["pred"] = stats["base_mean"]
    elif spec.commodity_stat == "wmean":
        stats["pred"] = stats["base_wmean"]
    elif spec.commodity_stat == "median_mean":
        a = spec.median_mean_alpha
        stats["pred"] = (1.0 - a) * stats["base_median"] + a * stats["base_mean"]
    else:
        raise ValueError(spec.commodity_stat)

    route_prior = stats.groupby(KEY_ROUTE, as_index=False)["pred"].sum().rename(columns={"pred": "route_prior_total"})
    stats = stats.merge(route_prior, on=KEY_ROUTE, how="left")
    stats["route_ref"] = stats["route_prior_total"]

    if spec.route_blend_stat and spec.route_blend_alpha > 0:
        ref = route_stat(hist, base_year, spec.route_blend_stat)
        stats = stats.merge(ref, on=KEY_ROUTE, how="left", suffixes=("", "_new"))
        stats["route_ref"] = stats["route_ref_new"].fillna(stats["route_prior_total"])
        stats = stats.drop(columns=["route_ref_new"])
        blended_total = (1.0 - spec.route_blend_alpha) * stats["route_prior_total"] + spec.route_blend_alpha * stats["route_ref"]
        scale = blended_total / stats["route_prior_total"].replace(0, np.nan)
        stats["pred"] = (stats["pred"] * scale.fillna(1.0)).clip(lower=0)
        stats["route_prior_total"] = blended_total

    stats["base_year"] = base_year
    if spec.external_strength > 0:
        stats = apply_external_prior(stats, base_year, spec.external_strength)
        adjusted_total = stats.groupby(KEY_ROUTE, as_index=False)["pred"].sum().rename(columns={"pred": "route_prior_total_adjusted"})
        stats = stats.merge(adjusted_total, on=KEY_ROUTE, how="left")
        stats["route_prior_total"] = stats["route_prior_total_adjusted"]
        stats = stats.drop(columns=["route_prior_total_adjusted"])

    stats["route_gap_ratio"] = (stats["route_ref"] - stats["route_prior_total"]) / (stats["route_prior_total"].abs() + 1e-12)
    return stats


def make_route_prediction(history: pd.DataFrame, target: pd.DataFrame, spec: BaselineSpec) -> pd.DataFrame:
    tgt_year = target_year(target)
    comm = make_comm_baseline(history, tgt_year, spec)
    pred = comm.groupby(KEY_ROUTE, as_index=False)["pred"].sum().rename(columns={"pred": "pred"})
    out = target[KEY_ROUTE + ["year", "tons"]].merge(pred, on=KEY_ROUTE, how="left")
    out["pred"] = out["pred"].fillna(0.0).clip(lower=0)
    out["model"] = spec.name
    return out


def make_route_stat_prediction(history: pd.DataFrame, target: pd.DataFrame, stat: str, model_name: str) -> pd.DataFrame:
    tgt_year = target_year(target)
    hist, base_year = available_window(history, tgt_year)
    pred = route_stat(hist, base_year, stat).rename(columns={"route_ref": "pred"})
    out = target[KEY_ROUTE + ["year", "tons"]].merge(pred, on=KEY_ROUTE, how="left")
    out["pred"] = out["pred"].fillna(0.0).clip(lower=0)
    out["model"] = model_name
    return out


def baseline_specs() -> list[BaselineSpec]:
    specs = [
        BaselineSpec("commodity_median3", "median"),
        BaselineSpec("commodity_mean3", "mean"),
        BaselineSpec("commodity_wmean3", "wmean"),
    ]
    for alpha in [0.1, 0.2, 0.3]:
        specs.append(BaselineSpec(f"commodity_medmean_a{alpha:.2f}", "median_mean", median_mean_alpha=alpha))
    for stat in ["avg3", "wmean"]:
        for alpha in [0.05, 0.10, 0.20]:
            specs.append(
                BaselineSpec(
                    f"route_blend{int(alpha * 100):02d}_{stat}",
                    "median",
                    route_blend_stat=stat,
                    route_blend_alpha=alpha,
                )
            )
    specs.extend(
        [
            BaselineSpec("commodity_median3_prior075", "median", external_strength=0.75),
            BaselineSpec(
                "route_blend10_avg3_prior075",
                "median",
                route_blend_stat="avg3",
                route_blend_alpha=0.10,
                external_strength=0.75,
            ),
            BaselineSpec(
                "route_blend20_avg3_prior075",
                "median",
                route_blend_stat="avg3",
                route_blend_alpha=0.20,
                external_strength=0.75,
            ),
            BaselineSpec(
                "route_blend10_wmean_prior075",
                "median",
                route_blend_stat="wmean",
                route_blend_alpha=0.10,
                external_strength=0.75,
            ),
        ]
    )
    return specs


def evaluate_predictions(pred: pd.DataFrame, split: str, part: str) -> dict[str, object]:
    metrics = metric_dict(pred["tons"], pred["pred"])
    return {
        "split": split,
        "part": part,
        "model": pred["model"].iloc[0],
        "rows": int(len(pred)),
        **metrics,
    }


def official_baseline_evaluation(data_root: Path, specs: list[BaselineSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_frames = []
    route_stat_models = [("lag1", "NaiveLag1"), ("avg3", "route_Avg3")]
    for split in SPLIT_WEIGHTS:
        detail = load_detail(data_root, split, include_context=True)
        for part in ["val", "test"]:
            target = load_target(data_root, split, part, commodity=False)
            for stat, model_name in route_stat_models:
                pred = make_route_stat_prediction(detail, target, stat, model_name)
                metric_rows.append(evaluate_predictions(pred, split, part))
                pred_frames.append(pred.assign(split=split, part=part))
            for spec in specs:
                pred = make_route_prediction(detail, target, spec)
                metric_rows.append(evaluate_predictions(pred, split, part))
                pred_frames.append(pred.assign(split=split, part=part))

    split_metrics = pd.DataFrame(metric_rows)
    leaderboard_rows = []
    for model in split_metrics["model"].unique():
        model_rows = split_metrics[split_metrics["model"] == model]
        item: dict[str, object] = {"model": model}
        for part, prefix in [("val", "val"), ("test", "test")]:
            part_rows = model_rows[model_rows["part"] == part]
            item.update(weighted_metrics(part_rows, prefix))
        leaderboard_rows.append(item)
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).reset_index(drop=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    return leaderboard, split_metrics, predictions


def pseudo_target_years(train_detail: pd.DataFrame, max_years: int = 0) -> list[int]:
    min_year = int(train_detail["year"].min())
    max_year = int(train_detail["year"].max())
    years = [year for year in range(min_year + 5, max_year + 1) if year - 5 >= min_year]
    if max_years and len(years) > max_years:
        years = years[-max_years:]
    return years


def target_from_detail(detail: pd.DataFrame, year: int, commodity: bool = False) -> pd.DataFrame:
    keys = KEY_COMM if commodity else KEY_ROUTE
    return (
        detail[detail["year"] == year]
        .groupby(keys + ["year"], as_index=False)["tons"]
        .sum()
    )


def pseudo_baseline_screening(data_root: Path, specs: list[BaselineSpec], max_years: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    detail_rows = []
    for split in SPLIT_WEIGHTS:
        train = load_detail(data_root, split, include_context=False)
        years = pseudo_target_years(train, max_years=max_years)
        for pseudo_year in years:
            history = train[train["year"] <= pseudo_year - 3].copy()
            target = target_from_detail(train, pseudo_year, commodity=False)
            if target.empty or history.empty:
                continue
            candidates = [("lag1", "NaiveLag1"), ("avg3", "route_Avg3")]
            for stat, model_name in candidates:
                pred = make_route_stat_prediction(history, target, stat, model_name)
                item = evaluate_predictions(pred, split, "pseudo")
                item["pseudo_year"] = pseudo_year
                detail_rows.append(item)
            for spec in specs:
                pred = make_route_prediction(history, target, spec)
                item = evaluate_predictions(pred, split, "pseudo")
                item["pseudo_year"] = pseudo_year
                detail_rows.append(item)

    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for model in detail["model"].unique():
        model_rows = detail[detail["model"] == model]
        per_split = model_rows.groupby("split")[METRIC_COLUMNS].mean().reset_index()
        item: dict[str, object] = {"model": model, "pseudo_folds": int(len(model_rows))}
        for metric in METRIC_COLUMNS:
            item[f"pseudo_weighted_{metric}"] = float(
                sum(SPLIT_WEIGHTS[s] * per_split.loc[per_split["split"] == s, metric].iloc[0] for s in SPLIT_WEIGHTS)
            )
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows).sort_values(f"pseudo_weighted_{PRIMARY_METRIC}").reset_index(drop=True)
    return summary, detail


def commodity_group(name: object) -> str:
    text = str(name).lower()
    if "coal" in text:
        return "coal_bulk"
    if any(k in text for k in ["fuel", "gasoline", "petroleum", "natural gas"]):
        return "fuel"
    if any(k in text for k in ["cereal", "grain", "ag ", "animal feed", "live animals"]):
        return "ag_food"
    if any(k in text for k in ["food", "meat", "alcohol", "tobacco"]):
        return "ag_food"
    if any(k in text for k in ["chemical", "fertil"]):
        return "chemicals"
    if any(k in text for k in ["vehicle", "transport equip", "machinery", "electronics"]):
        return "manufactured"
    return "other"


def build_comm_feature_frame(history: pd.DataFrame, tgt_year: int, spec: BaselineSpec) -> pd.DataFrame:
    hist, base_year = available_window(history, tgt_year)
    base = make_comm_baseline(history, tgt_year, spec)
    if base.empty:
        return base
    comm_year = detail_to_comm_year(hist)
    route_year = detail_to_route_year(hist)

    comm_stats = (
        comm_year.groupby(KEY_COMM)
        .agg(
            hist_tons_sum=("tons", "sum"),
            hist_tons_mean=("tons", "mean"),
            hist_tons_std=("tons", "std"),
            hist_tons_min=("tons", "min"),
            hist_tons_max=("tons", "max"),
            hist_year_count=("year", "nunique"),
        )
        .reset_index()
    )
    comm_stats["hist_tons_std"] = comm_stats["hist_tons_std"].fillna(0.0)

    route_stats = (
        route_year.groupby(KEY_ROUTE)
        .agg(
            route_hist_sum=("tons", "sum"),
            route_hist_mean=("tons", "mean"),
            route_hist_std=("tons", "std"),
            route_hist_min=("tons", "min"),
            route_hist_max=("tons", "max"),
        )
        .reset_index()
    )
    route_stats["route_hist_std"] = route_stats["route_hist_std"].fillna(0.0)

    origin_stats = hist.groupby(["origin"], as_index=False)["tons"].sum().rename(columns={"tons": "origin_hist_total"})
    dest_stats = hist.groupby(["destination"], as_index=False)["tons"].sum().rename(columns={"tons": "dest_hist_total"})

    out = base.merge(comm_stats, on=KEY_COMM, how="left")
    out = out.merge(route_stats, on=KEY_ROUTE, how="left")
    out = out.merge(origin_stats, on="origin", how="left")
    out = out.merge(dest_stats, on="destination", how="left")
    out["commodity_group"] = out["commodity"].map(commodity_group)
    out["prior_share"] = out["pred"] / out["route_prior_total"].replace(0, np.nan)
    out["prior_share"] = out["prior_share"].fillna(0.0)
    out["is_intrastate"] = (out["origin"] == out["destination"]).astype(int)
    out["target_year"] = tgt_year
    out["base_year"] = base_year

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_actual_to_comm_features(features: pd.DataFrame, target_comm: pd.DataFrame) -> pd.DataFrame:
    actual = target_comm.groupby(KEY_COMM, as_index=False)["tons"].sum().rename(columns={"tons": "actual_tons"})
    out = features.merge(actual, on=KEY_COMM, how="left")
    out["actual_tons"] = out["actual_tons"].fillna(0.0)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "origin",
        "destination",
        "commodity",
        "commodity_group",
        "actual_tons",
        "pred",
        "target_year",
        "base_year",
    }
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    cols.extend(["origin_code", "destination_code", "commodity_code", "commodity_group_code"])
    return list(dict.fromkeys(cols))


def encode_categories(train: pd.DataFrame, apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    apply = apply.copy()
    for col in ["origin", "destination", "commodity", "commodity_group"]:
        categories = pd.Index(pd.concat([train[col], apply[col]], ignore_index=True).astype(str).unique())
        mapping = {value: idx for idx, value in enumerate(categories)}
        train[f"{col}_code"] = train[col].astype(str).map(mapping).fillna(-1).astype(int)
        apply[f"{col}_code"] = apply[col].astype(str).map(mapping).fillna(-1).astype(int)
    return train, apply


def make_model(model_backend: str = "lgbm") -> object | None:
    if model_backend == "baseline_only":
        return None
    if model_backend == "lgbm":
        if LGBMRegressor is None:
            return make_model("hist_gb")
        return LGBMRegressor(
            objective="regression",
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=60,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED,
            n_jobs=1,
            verbose=-1,
        )
    if model_backend == "hist_gb":
        return HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_SEED,
        )
    if model_backend == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED))])
    if model_backend == "huber":
        return Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=300))])
    if model_backend == "elasticnet":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=RANDOM_SEED, max_iter=5000)),
            ]
        )
    if model_backend == "catboost":
        if CatBoostRegressor is None:
            return make_model("lgbm")
        return CatBoostRegressor(
            loss_function="RMSE",
            iterations=180,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_SEED,
            verbose=False,
            thread_count=1,
        )
    raise ValueError(f"Unsupported model_backend: {model_backend}")


def apply_gate(frame: pd.DataFrame, pred_log_resid: np.ndarray, gate: str, large_cutoff: float) -> np.ndarray:
    if gate == "none":
        return np.ones(len(frame), dtype=float)
    if gate == "large_route":
        return (frame["route_prior_total"].to_numpy() >= large_cutoff).astype(float)
    if gate == "directional_under":
        large = frame["route_prior_total"].to_numpy() >= large_cutoff
        model_up = pred_log_resid > 0
        route_ref_up = frame["route_ref"].to_numpy() > frame["route_prior_total"].to_numpy() * 1.02
        return (large & model_up & route_ref_up).astype(float)
    raise ValueError(gate)


def corrected_comm_prediction(frame: pd.DataFrame, pred_log_resid: np.ndarray, spec: ResidualSpec, large_cutoff: float) -> pd.DataFrame:
    out = frame[KEY_COMM + ["target_year", "actual_tons", "pred", "route_prior_total", "route_ref"]].copy()
    gate = apply_gate(frame, pred_log_resid, spec.gate, large_cutoff)
    clipped = np.clip(pred_log_resid, -1.0, 1.0)
    out["gate"] = gate
    out["pred_log_residual"] = pred_log_resid
    out["corrected_pred"] = np.expm1(np.log1p(np.clip(out["pred"].to_numpy(), 0, None)) + spec.shrink * gate * clipped)
    out["corrected_pred"] = out["corrected_pred"].clip(lower=0)
    return out


def comm_to_route_pred(comm_pred: pd.DataFrame, target_route: pd.DataFrame, model_name: str) -> pd.DataFrame:
    pred = comm_pred.groupby(KEY_ROUTE, as_index=False)["corrected_pred"].sum().rename(columns={"corrected_pred": "pred"})
    out = target_route[KEY_ROUTE + ["year", "tons"]].merge(pred, on=KEY_ROUTE, how="left")
    out["pred"] = out["pred"].fillna(0.0).clip(lower=0)
    out["model"] = model_name
    return out


def train_frames_for_split(train: pd.DataFrame, years: list[int], spec: BaselineSpec) -> pd.DataFrame:
    frames = []
    for year in years:
        history = train[train["year"] <= year - 3].copy()
        features = build_comm_feature_frame(history, year, spec)
        target = target_from_detail(train, year, commodity=True)
        sample = add_actual_to_comm_features(features, target)
        frames.append(sample)
    return pd.concat(frames, ignore_index=True)


def residual_oof_screening(
    data_root: Path,
    baseline: BaselineSpec,
    shrink_values: list[float],
    max_years: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    detail_rows = []
    for split in SPLIT_WEIGHTS:
        train = load_detail(data_root, split, include_context=False)
        years = pseudo_target_years(train, max_years=max_years)
        if len(years) < 2:
            continue
        for holdout_year in years:
            train_years = [y for y in years if y != holdout_year]
            train_frame = train_frames_for_split(train, train_years, baseline)
            history = train[train["year"] <= holdout_year - 3].copy()
            hold_features = build_comm_feature_frame(history, holdout_year, baseline)
            hold_target_comm = target_from_detail(train, holdout_year, commodity=True)
            hold_frame = add_actual_to_comm_features(hold_features, hold_target_comm)
            train_enc, hold_enc = encode_categories(train_frame, hold_frame)
            cols = feature_columns(train_enc)
            y = np.log1p(train_enc["actual_tons"].clip(lower=0)) - np.log1p(train_enc["pred"].clip(lower=0))
            model = make_model()
            model.fit(train_enc[cols], y)
            pred_lr = np.asarray(model.predict(hold_enc[cols]), dtype=float)
            large_cutoff = float(train_enc["route_prior_total"].quantile(0.75))
            target_route = target_from_detail(train, holdout_year, commodity=False)
            for shrink in shrink_values:
                for gate in ["none", "large_route", "directional_under"]:
                    rspec = ResidualSpec(f"{baseline.name}__lgb_logresid__{gate}__shrink_{shrink:.2f}", baseline, shrink, gate)
                    comm_pred = corrected_comm_prediction(hold_enc, pred_lr, rspec, large_cutoff)
                    route_pred = comm_to_route_pred(comm_pred, target_route, rspec.name)
                    metrics = evaluate_predictions(route_pred, split, "pseudo_residual")
                    metrics["pseudo_year"] = holdout_year
                    metrics["baseline"] = baseline.name
                    metrics["shrink"] = shrink
                    metrics["gate"] = gate
                    detail_rows.append(metrics)

    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for model in detail["model"].unique():
        model_rows = detail[detail["model"] == model]
        per_split = model_rows.groupby("split")[METRIC_COLUMNS].mean().reset_index()
        item: dict[str, object] = {"model": model, "pseudo_folds": int(len(model_rows))}
        first = model_rows.iloc[0]
        item.update({"baseline": first["baseline"], "shrink": float(first["shrink"]), "gate": first["gate"]})
        for metric in METRIC_COLUMNS:
            item[f"pseudo_weighted_{metric}"] = float(
                sum(SPLIT_WEIGHTS[s] * per_split.loc[per_split["split"] == s, metric].iloc[0] for s in SPLIT_WEIGHTS)
            )
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows).sort_values(f"pseudo_weighted_{PRIMARY_METRIC}").reset_index(drop=True)
    return summary, detail


def official_residual_evaluation(data_root: Path, rspec: ResidualSpec, max_years: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    route_pred_rows = []
    comm_pred_rows = []
    for split in SPLIT_WEIGHTS:
        train = load_detail(data_root, split, include_context=False)
        train_years = pseudo_target_years(train, max_years=max_years)
        train_frame = train_frames_for_split(train, train_years, rspec.baseline)
        detail_full = load_detail(data_root, split, include_context=True)
        for part in ["val", "test"]:
            target_route = load_target(data_root, split, part, commodity=False)
            target_comm = load_target(data_root, split, part, commodity=True)
            tgt_year = target_year(target_route)
            history = detail_full[detail_full["year"] <= tgt_year - 3].copy()
            apply_frame = build_comm_feature_frame(history, tgt_year, rspec.baseline)
            apply_frame = add_actual_to_comm_features(apply_frame, target_comm)
            train_enc, apply_enc = encode_categories(train_frame, apply_frame)
            cols = feature_columns(train_enc)
            y = np.log1p(train_enc["actual_tons"].clip(lower=0)) - np.log1p(train_enc["pred"].clip(lower=0))
            model = make_model()
            model.fit(train_enc[cols], y)
            pred_lr = np.asarray(model.predict(apply_enc[cols]), dtype=float)
            large_cutoff = float(train_enc["route_prior_total"].quantile(0.75))
            comm_pred = corrected_comm_prediction(apply_enc, pred_lr, rspec, large_cutoff)
            route_pred = comm_to_route_pred(comm_pred, target_route, rspec.name)
            metric_rows.append(evaluate_predictions(route_pred, split, part))
            route_pred_rows.append(route_pred.assign(split=split, part=part))
            comm_pred_rows.append(comm_pred.assign(split=split, part=part, model=rspec.name))

    split_metrics = pd.DataFrame(metric_rows)
    item: dict[str, object] = {"model": rspec.name}
    for part, prefix in [("val", "val"), ("test", "test")]:
        item.update(weighted_metrics(split_metrics[split_metrics["part"] == part], prefix))
    leaderboard = pd.DataFrame([item])
    return leaderboard, split_metrics, pd.concat(route_pred_rows, ignore_index=True), pd.concat(comm_pred_rows, ignore_index=True)


def choose_baseline_spec(name: str, specs: list[BaselineSpec]) -> BaselineSpec:
    mapping = {spec.name: spec for spec in specs}
    if name in mapping:
        return mapping[name]
    raise KeyError(name)


def high_loss_improvement(predictions: pd.DataFrame, baseline_model: str, model_name: str) -> pd.DataFrame:
    base = predictions[predictions["model"] == baseline_model].copy()
    model = predictions[predictions["model"] == model_name].copy()
    keys = ["split", "part"] + KEY_ROUTE + ["year"]
    merged = base[keys + ["tons", "pred"]].rename(columns={"pred": "baseline_pred"}).merge(
        model[keys + ["pred"]].rename(columns={"pred": "model_pred"}), on=keys, how="inner"
    )
    merged["baseline_abs_error"] = (merged["tons"] - merged["baseline_pred"]).abs()
    merged["model_abs_error"] = (merged["tons"] - merged["model_pred"]).abs()
    merged["abs_error_delta"] = merged["baseline_abs_error"] - merged["model_abs_error"]
    train_like = merged[merged["part"] == "val"]
    cutoff = train_like["baseline_abs_error"].quantile(0.95) if not train_like.empty else merged["baseline_abs_error"].quantile(0.95)
    return merged[merged["baseline_abs_error"] >= cutoff].sort_values("abs_error_delta", ascending=False).reset_index(drop=True)


def write_report(
    artifact_dir: Path,
    data_root: Path,
    baseline_screen: pd.DataFrame,
    official_leaderboard: pd.DataFrame,
    residual_screen: pd.DataFrame | None,
    combined_leaderboard: pd.DataFrame,
) -> None:
    best_screen = baseline_screen.iloc[0].to_dict()
    commodity_like = baseline_screen[~baseline_screen["model"].isin(["NaiveLag1", "route_Avg3"])]
    best_commodity_screen = commodity_like.iloc[0].to_dict() if not commodity_like.empty else best_screen
    best_val = combined_leaderboard.sort_values(f"val_weighted_{PRIMARY_METRIC}").iloc[0].to_dict()
    best_test = combined_leaderboard.sort_values(f"test_weighted_{PRIMARY_METRIC}").iloc[0].to_dict()
    lines = [
        "# Commodity Stacking Experiment Report",
        "",
        f"- data_root: `{data_root}`",
        "- selection rule: train-internal pseudo backtest first, validation only for limited review",
        "- test rule: final confirmation and sensitivity only",
        "",
        "## Train-Internal Baseline Screen",
        "",
        f"Best pseudo baseline: `{best_screen['model']}`",
        f"- pseudo weighted wRMSE: {best_screen['pseudo_weighted_wRMSE']:.4f}",
        f"- pseudo weighted RMSE: {best_screen['pseudo_weighted_RMSE']:.4f}",
        f"- pseudo weighted WMAPE: {best_screen['pseudo_weighted_WMAPE']:.6f}",
        "",
        f"Best pseudo commodity anchor: `{best_commodity_screen['model']}`",
        f"- pseudo weighted wRMSE: {best_commodity_screen['pseudo_weighted_wRMSE']:.4f}",
        f"- pseudo weighted RMSE: {best_commodity_screen['pseudo_weighted_RMSE']:.4f}",
        f"- pseudo weighted WMAPE: {best_commodity_screen['pseudo_weighted_WMAPE']:.6f}",
        "",
        "## Official Leaderboard",
        "",
        f"Validation-selected review candidate: `{best_val['model']}`",
        f"- val weighted wRMSE: {best_val['val_weighted_wRMSE']:.4f}",
        f"- val weighted RMSE: {best_val['val_weighted_RMSE']:.4f}",
        f"- test weighted wRMSE: {best_val['test_weighted_wRMSE']:.4f}",
        f"- test weighted RMSE: {best_val['test_weighted_RMSE']:.4f}",
        "",
        f"Best test sensitivity candidate: `{best_test['model']}`",
        f"- val weighted wRMSE: {best_test['val_weighted_wRMSE']:.4f}",
        f"- val weighted RMSE: {best_test['val_weighted_RMSE']:.4f}",
        f"- test weighted wRMSE: {best_test['test_weighted_wRMSE']:.4f}",
        f"- test weighted RMSE: {best_test['test_weighted_RMSE']:.4f}",
        "",
    ]
    if residual_screen is not None and not residual_screen.empty:
        best_resid = residual_screen.iloc[0].to_dict()
        lines.extend(
            [
                "## Residual Screen",
                "",
                f"Best train-internal residual config: `{best_resid['model']}`",
                f"- pseudo weighted wRMSE: {best_resid['pseudo_weighted_wRMSE']:.4f}",
                f"- pseudo weighted RMSE: {best_resid['pseudo_weighted_RMSE']:.4f}",
                f"- gate: `{best_resid['gate']}`",
                f"- shrink: `{best_resid['shrink']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Files",
            "",
            "- `official_leaderboard.csv`",
            "- `official_split_metrics.csv`",
            "- `pseudo_baseline_leaderboard.csv`",
            "- `pseudo_baseline_split_metrics.csv`",
            "- `combined_official_leaderboard.csv`",
            "- `route_predictions.csv`",
            "- `high_loss_route_improvement.csv`",
        ]
    )
    (artifact_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed()
    args = parse_args()
    data_root = discover_data_root(args.data_root)
    global GLOBAL_DATA_ROOT
    GLOBAL_DATA_ROOT = data_root
    artifact_dir = repo_root() / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    specs = baseline_specs()
    baseline_screen, baseline_screen_detail = pseudo_baseline_screening(data_root, specs, max_years=args.max_pseudo_years)
    official_leaderboard, official_split_metrics, route_predictions = official_baseline_evaluation(data_root, specs)

    baseline_screen.to_csv(artifact_dir / "pseudo_baseline_leaderboard.csv", index=False, encoding="utf-8-sig")
    baseline_screen_detail.to_csv(artifact_dir / "pseudo_baseline_split_metrics.csv", index=False, encoding="utf-8-sig")
    official_leaderboard.to_csv(artifact_dir / "official_leaderboard.csv", index=False, encoding="utf-8-sig")
    official_split_metrics.to_csv(artifact_dir / "official_split_metrics.csv", index=False, encoding="utf-8-sig")

    combined = official_leaderboard.copy()
    residual_screen = None
    residual_route_predictions = []
    residual_comm_predictions = []
    residual_split_metrics = []

    if not args.skip_residual:
        spec_names = {spec.name for spec in specs}
        if args.residual_baseline == "auto":
            screened_specs = baseline_screen[baseline_screen["model"].isin(spec_names)]
            if screened_specs.empty:
                raise RuntimeError("No train-internal screened commodity baseline is available for residual modeling.")
            best_baseline_name = str(screened_specs.iloc[0]["model"])
        else:
            best_baseline_name = str(args.residual_baseline)
            if best_baseline_name not in spec_names:
                raise KeyError(f"Unknown residual baseline `{best_baseline_name}`. Available: {sorted(spec_names)}")
        residual_baseline = choose_baseline_spec(best_baseline_name, specs)
        residual_screen, residual_screen_detail = residual_oof_screening(
            data_root,
            residual_baseline,
            shrink_values=[0.03, 0.06, 0.10, 0.15],
            max_years=args.max_pseudo_years,
        )
        residual_screen.to_csv(artifact_dir / "pseudo_residual_leaderboard.csv", index=False, encoding="utf-8-sig")
        residual_screen_detail.to_csv(artifact_dir / "pseudo_residual_split_metrics.csv", index=False, encoding="utf-8-sig")

        best_resid = residual_screen.iloc[0]
        rspec = ResidualSpec(
            name=str(best_resid["model"]),
            baseline=residual_baseline,
            shrink=float(best_resid["shrink"]),
            gate=str(best_resid["gate"]),
        )
        resid_leaderboard, resid_split, resid_route_pred, resid_comm_pred = official_residual_evaluation(
            data_root,
            rspec,
            max_years=args.max_pseudo_years,
        )
        combined = pd.concat([combined, resid_leaderboard], ignore_index=True)
        residual_split_metrics.append(resid_split)
        residual_route_predictions.append(resid_route_pred)
        residual_comm_predictions.append(resid_comm_pred)

    combined = combined.sort_values([f"val_weighted_{PRIMARY_METRIC}", f"test_weighted_{PRIMARY_METRIC}"]).reset_index(drop=True)
    combined.to_csv(artifact_dir / "combined_official_leaderboard.csv", index=False, encoding="utf-8-sig")

    all_route_predictions = [route_predictions]
    if residual_route_predictions:
        all_route_predictions.extend(residual_route_predictions)
    all_preds = pd.concat(all_route_predictions, ignore_index=True)
    all_preds.to_csv(artifact_dir / "route_predictions.csv", index=False, encoding="utf-8-sig")
    if residual_comm_predictions:
        pd.concat(residual_comm_predictions, ignore_index=True).to_csv(
            artifact_dir / "commodity_residual_predictions.csv", index=False, encoding="utf-8-sig"
        )
    if residual_split_metrics:
        pd.concat(residual_split_metrics, ignore_index=True).to_csv(
            artifact_dir / "official_residual_split_metrics.csv", index=False, encoding="utf-8-sig"
        )

    selected_model = str(combined.sort_values(f"val_weighted_{PRIMARY_METRIC}").iloc[0]["model"])
    baseline_model = "commodity_median3"
    improvement = high_loss_improvement(all_preds, baseline_model, selected_model)
    improvement.to_csv(artifact_dir / "high_loss_route_improvement.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "data_root": str(data_root),
        "artifact_dir": str(artifact_dir),
        "baseline_specs": [asdict(s) for s in specs],
        "skip_residual": bool(args.skip_residual),
        "residual_baseline": str(args.residual_baseline),
        "model_backend": "lightgbm" if LGBMRegressor is not None else "hist_gradient_boosting",
        "selection_policy": "train_internal_screen_then_validation_review",
        "primary_metric": f"split_weighted_{PRIMARY_METRIC}",
    }
    (artifact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(artifact_dir, data_root, baseline_screen, official_leaderboard, residual_screen, combined)

    print("Data root:", data_root)
    print("Artifact dir:", artifact_dir)
    print("Top combined leaderboard:")
    print(combined.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
