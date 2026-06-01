"""
Bayasgalan's model: CommodityMedian3 + LightGBM Residual Correction
=====================================================================
Model   : CM3 baseline + LightGBM residual learner (FAF5-only training samples)
Metric  : tons-weighted RMSE  (team official metric)
Splits  : data/faf4_faf5_fixed_year_splits/  (team official splits)

Usage:
    python code/bayasgalan_model.py

Results (team splits, tons-weighted RMSE):
    CommodityMedian3 baseline : val 16,379
    This model (CM3 + LGBM)   : val 16,744  (see bayasgalan_rationale.md)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "faf4_faf5_fixed_year_splits"

SPLITS_CFG = {
    "split_1": {"train": list(range(2012, 2019)), "context": [2019], "val": [2021], "test": [2022], "weight": 0.2},
    "split_2": {"train": list(range(2012, 2020)), "context": [2020], "val": [2022], "test": [2023], "weight": 0.3},
    "split_3": {"train": list(range(2012, 2021)), "context": [2021], "val": [2023], "test": [2024], "weight": 0.5},
}

# Only use FAF5-era samples (>=2017) to avoid FAF4 distribution shift
MIN_TRAIN_FY = 2017

LGB_PARAMS = dict(
    objective="regression",
    num_leaves=4,
    min_child_samples=500,
    learning_rate=0.05,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=50.0,
    reg_alpha=5.0,
    verbose=-1,
    random_state=42,
)

# ── Metric ───────────────────────────────────────────────────────────────────

def weighted_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Tons-weighted RMSE — large routes penalized proportionally to their volume."""
    yt = np.clip(actual.astype(float), 0, None)
    yp = np.clip(pred.astype(float),   0, None)
    w  = yt / (yt.sum() + 1e-9)
    return float(np.sqrt(np.sum(w * (yt - yp) ** 2)))


# ── CommodityMedian3 baseline ─────────────────────────────────────────────────

def commodity_median3(src_df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """
    For each OD pair: median tons per commodity over the last 3 years, then sum.
    base_year: most recent year available (max train year or context year).
    Returns DataFrame [origin, destination, tons_pred].
    """
    years = [base_year - 2, base_year - 1, base_year]
    sub   = src_df[src_df["year"].isin(years)]
    if sub.empty:
        sub = src_df

    comm_med = (
        sub.groupby(["origin", "destination", "commodity", "year"], as_index=False)["tons"]
           .sum()
           .groupby(["origin", "destination", "commodity"], as_index=False)["tons"]
           .median()
    )
    return (
        comm_med.groupby(["origin", "destination"], as_index=False)["tons"]
                .sum()
                .rename(columns={"tons": "tons_pred"})
    )


# ── Feature engineering ───────────────────────────────────────────────────────

def _route_agg(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["origin", "destination", "year"], as_index=False)["tons"].sum()


def build_features(src_df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """Build route-level features from historical data."""
    ryr   = _route_agg(src_df)
    years = sorted(ryr["year"].unique())
    n     = len(years)
    pivot = ryr.pivot_table(
        index=["origin", "destination"], columns="year", values="tons", fill_value=0
    )

    r = pivot.reset_index()[["origin", "destination"]].copy()

    # Lag features
    for i, yr in enumerate(sorted(years, reverse=True)[:3], 1):
        r[f"tons_lag{i}"] = pivot[yr].values
    for c in ["tons_lag2", "tons_lag3"]:
        if c not in r.columns:
            r[c] = np.nan

    # Rolling 3yr stats
    rec = sorted(years)[-3:]
    rv  = pivot[rec]
    r["tons_mean_3yr"] = rv.mean(axis=1).values
    r["tons_std_3yr"]  = rv.std(axis=1).fillna(0).values

    # All-history stats
    av = pivot[years]
    ma = av.mean(axis=1).values
    sa = av.std(axis=1).fillna(0).values
    r["tons_mean_all"] = ma
    r["tons_cv_all"]   = np.where(ma != 0, sa / ma, 0.0)

    # Trend
    if n >= 2:
        xi = np.arange(n, dtype=float)
        xd = xi - xi.mean()
        sl = (av.values.astype(float) @ xd) / (xd @ xd)
        r["tons_trend"]      = sl
        r["tons_trend_norm"] = np.where(ma != 0, sl / ma, 0.0)
        lv = pivot[sorted(years)[-1]].values.astype(float)
        pv = pivot[sorted(years)[-2]].values.astype(float)
        r["tons_yoy_last"]   = np.where(pv != 0, (lv - pv) / pv, 0.0)
    else:
        r["tons_trend"] = r["tons_trend_norm"] = r["tons_yoy_last"] = 0.0

    r["n_years_hist"] = n

    # Commodity diversity
    nc = (
        src_df.groupby(["origin", "destination"])["commodity"].nunique()
              .reset_index().rename(columns={"commodity": "n_commodities"})
    )
    r = r.merge(nc, on=["origin", "destination"], how="left")

    # Modal distance band
    db = (
        src_df.groupby(["origin", "destination", "distance_band"])["tons"].sum()
              .reset_index().sort_values("tons", ascending=False)
              .groupby(["origin", "destination"], as_index=False).first()
              [["origin", "destination", "distance_band"]]
              .rename(columns={"distance_band": "modal_dist_band"})
    )
    r = r.merge(db, on=["origin", "destination"], how="left")

    # CM3 base prediction (used as a feature and as the fallback)
    base = commodity_median3(src_df, base_year)
    r = r.merge(
        base.rename(columns={"tons_pred": "base_pred"}),
        on=["origin", "destination"], how="left"
    )
    r["base_pred"]     = r["base_pred"].fillna(0)
    r["base_pred_log"] = np.log1p(r["base_pred"])

    return r


FEATURE_COLS = [
    "tons_lag1", "tons_lag2", "tons_lag3",
    "tons_mean_3yr", "tons_std_3yr",
    "tons_mean_all", "tons_cv_all",
    "tons_trend", "tons_trend_norm", "tons_yoy_last",
    "n_years_hist", "n_commodities", "modal_dist_band", "base_pred_log",
]
CAT_COL = "modal_dist_band"
CAT_IDX = [FEATURE_COLS.index(CAT_COL)]


def _label_encode(dfs: list, col: str) -> list:
    dfs  = [d.copy() for d in dfs]
    cats = sorted(pd.concat([d[col] for d in dfs]).astype(str).fillna("_na_").unique())
    ct   = pd.CategoricalDtype(categories=cats, ordered=False)
    for d in dfs:
        d[col] = d[col].astype(str).fillna("_na_").astype(ct).cat.codes
    return dfs


# ── Training sample builder ────────────────────────────────────────────────────

def make_residual_samples(
    raw_df: pd.DataFrame,
    available_years: list,
    horizon: int = 3,
    min_fy: int = MIN_TRAIN_FY,
    skip_target_years: set = None,
) -> pd.DataFrame | None:
    """
    Build LightGBM training samples: each row is one (fy → fy+horizon) pair.
    Label: residual = actual_tons - CM3_prediction.
    min_fy=2017 means only FAF5-era forecast years (avoids FAF4 distribution shift).
    """
    if skip_target_years is None:
        skip_target_years = set()
    raw_years = set(raw_df["year"].unique())
    out = []
    for fy in sorted(available_years):
        if fy < min_fy:
            continue
        ty = fy + horizon
        if ty in skip_target_years or ty not in raw_years:
            continue
        src = raw_df[raw_df["year"] <= fy]
        act = _route_agg(raw_df[raw_df["year"] == ty])
        ft  = build_features(src, fy)
        mg  = (
            act[["origin", "destination", "tons"]]
            .merge(ft, on=["origin", "destination"], how="left")
            .rename(columns={"tons": "actual_tons"})
        )
        mg["residual"] = mg["actual_tons"] - mg["base_pred"]
        mg["fy"]       = fy
        out.append(mg)
    return pd.concat(out, ignore_index=True) if out else None


# ── Model class ───────────────────────────────────────────────────────────────

class CM3LGBMModel:
    """
    CommodityMedian3 + LightGBM Residual Correction.

    Step 1: CM3 predicts baseline (median of last 3yr per OD×commodity, summed)
    Step 2: LightGBM corrects the residual using route-level trend/lag features
    Step 3: final prediction = CM3 + correction (clipped >= 0)

    Key design choices:
        min_fy=2017    : only FAF5-era training samples (FAF4 causes distribution shift)
        covid_filter   : skip 2020 target year in training (outlier shock)
    """

    def __init__(
        self,
        min_fy: int = MIN_TRAIN_FY,
        covid_filter: bool = True,
        lgb_params: dict = None,
    ):
        self.min_fy       = min_fy
        self.covid_filter = covid_filter
        self.lgb_params   = lgb_params or LGB_PARAMS
        self._model       = None
        self._cat_dtype   = None

    def fit(self, raw_df: pd.DataFrame, train_years: list, context_years: list):
        skip      = {2020} if self.covid_filter else set()
        available = sorted(set(train_years) | set(context_years))
        tf = make_residual_samples(
            raw_df, available, horizon=3,
            min_fy=self.min_fy, skip_target_years=skip,
        )
        if tf is None or len(tf) == 0:
            print("  [CM3LGBMModel] No training samples — using CM3 only.")
            return self

        cats = sorted(tf[CAT_COL].astype(str).fillna("_na_").unique())
        self._cat_dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        tf[CAT_COL] = tf[CAT_COL].astype(str).fillna("_na_").astype(self._cat_dtype).cat.codes

        self._model = lgb.LGBMRegressor(**self.lgb_params)
        self._model.fit(
            tf[FEATURE_COLS].fillna(0),
            tf["residual"],
            categorical_feature=CAT_IDX,
        )
        fys = sorted(tf["fy"].unique())
        print(f"  [CM3LGBMModel] Fitted | samples={len(tf):,} | fy={fys[0]}–{fys[-1]}")
        return self

    def predict(
        self,
        src_df: pd.DataFrame,
        base_year: int,
        target_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        src_df    : all data available at prediction time (train or train+context)
        base_year : most recent year in src_df
        target_df : DataFrame with [origin, destination] columns
        Returns   : target_df + column 'tons_pred'
        """
        ft     = build_features(src_df, base_year)
        merged = target_df[["origin", "destination"]].merge(
            ft, on=["origin", "destination"], how="left"
        )
        base_pred = merged["base_pred"].fillna(0).values

        if self._model is None:
            merged["tons_pred"] = np.clip(base_pred, 0, None)
        else:
            enc = merged.copy()
            enc[CAT_COL] = (
                enc[CAT_COL].astype(str).fillna("_na_")
                            .astype(self._cat_dtype).cat.codes
            )
            correction = self._model.predict(enc[FEATURE_COLS].fillna(0))
            merged["tons_pred"] = np.clip(base_pred + correction, 0, None)

        return target_df.merge(
            merged[["origin", "destination", "tons_pred"]],
            on=["origin", "destination"], how="left",
        )


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 65)
    print("CM3 + LightGBM Residual Model  |  Team Official Splits")
    print("Metric: tons-weighted RMSE (lower is better)")
    print("=" * 65)

    # Locate raw FAF data for building training samples
    raw_candidates = [
        REPO_ROOT / "bayasgalan" / "data" / "faf4_faf5_fixed.csv",
        REPO_ROOT / "data" / "faf4_faf5_fixed.csv",
        REPO_ROOT / "processed_data" / "faf4_faf5_fixed.csv",
    ]
    raw_df = None
    for p in raw_candidates:
        if p.exists():
            print(f"Loading raw data: {p.relative_to(REPO_ROOT)}", flush=True)
            raw_df = pd.read_csv(p)
            print(f"  {len(raw_df):,} rows", flush=True)
            break
    if raw_df is None:
        print("[WARNING] faf4_faf5_fixed.csv not found — CM3 baseline only.\n")

    cm3_vals, cm3_tests = {}, {}
    lgbm_vals, lgbm_tests = {}, {}

    for sn, cfg in SPLITS_CFG.items():
        d          = SPLITS_DIR / sn
        train_df   = pd.read_csv(d / "train.csv")
        context_df = pd.read_csv(d / "context.csv")
        val_tgt    = pd.read_csv(d / "val.csv")
        test_tgt   = pd.read_csv(d / "test.csv")
        src_test   = pd.concat([train_df, context_df], ignore_index=True)

        base_yr_val  = max(cfg["train"])
        base_yr_test = cfg["context"][0]

        # CM3 baseline
        def _score(tgt, pred_df):
            mg = tgt.merge(pred_df, on=["origin", "destination"], how="left")
            return weighted_rmse(mg["tons"].values, mg["tons_pred"].fillna(0).values)

        cm3_vals[sn]  = _score(val_tgt,  commodity_median3(train_df, base_yr_val))
        cm3_tests[sn] = _score(test_tgt, commodity_median3(src_test,  base_yr_test))

        # CM3 + LGBM
        if raw_df is not None:
            print(f"\n{sn}:", flush=True)
            model = CM3LGBMModel(min_fy=MIN_TRAIN_FY, covid_filter=True)
            model.fit(raw_df, cfg["train"], cfg["context"])
            lgbm_vals[sn]  = weighted_rmse(
                val_tgt["tons"].values,
                model.predict(train_df,  base_yr_val,  val_tgt)["tons_pred"].fillna(0).values,
            )
            lgbm_tests[sn] = weighted_rmse(
                test_tgt["tons"].values,
                model.predict(src_test,  base_yr_test, test_tgt)["tons_pred"].fillna(0).values,
            )
        else:
            lgbm_vals[sn]  = cm3_vals[sn]
            lgbm_tests[sn] = cm3_tests[sn]

    W = {sn: cfg["weight"] for sn, cfg in SPLITS_CFG.items()}
    agg = lambda d: sum(W[s] * d[s] for s in SPLITS_CFG)

    print("\n" + "=" * 65)
    print(f"{'Model':<22} {'val wRMSE':>12} {'test wRMSE':>12}")
    print("-" * 48)
    print(f"{'CommodityMedian3':<22} {agg(cm3_vals):>12.1f} {agg(cm3_tests):>12.1f}")
    lv, lt = agg(lgbm_vals), agg(lgbm_tests)
    mv = "✅" if lv < agg(cm3_vals)  else "~"
    mt = "✅" if lt < agg(cm3_tests) else "~"
    print(f"{'CM3 + LightGBM':<22} {lv:>12.1f} {lt:>12.1f}  {mv} val  {mt} test")
    print("=" * 65)

    print(f"\nPer-split val wRMSE:")
    for sn in SPLITS_CFG:
        print(f"  {sn}: CM3={cm3_vals[sn]:.1f}  LGBM={lgbm_vals[sn]:.1f}")
    print(f"\nPer-split test wRMSE:")
    for sn in SPLITS_CFG:
        print(f"  {sn}: CM3={cm3_tests[sn]:.1f}  LGBM={lgbm_tests[sn]:.1f}")


if __name__ == "__main__":
    run_evaluation()
