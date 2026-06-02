"""
m4_dongbin.py — M4: Per-commodity-group LGBM with sample weights + log-ratio.

CANDIDATES_M4 = ["M4_lgbm", "M4_resid_a003", "M4_resid_a005", "M4_resid_a008", "M4_resid_a010"]

Architecture
------------
- Per-commodity-group LGBM model (6 groups)
- Sample weights: tons^0.35 × (1 + 4×large_route_flag + 2×intrastate_flag)
- Features: A + B + C + D_abs
- Target: log1p(actual) - log1p(cm3_pred) (log-ratio)
- COVID filter: skip 2020
- M4_resid_a*: apply route residual reallocation with alpha after commodity predictions
- Correction clipped ±0.10
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    CORRECTION_CLIP, MIN_TARGET_YEAR, COVID_SKIP_YEARS, RANDOM_SEED,
    KEY_COMM, COMM_GROUPS, get_commodity_group, SHRINKAGE_FACTOR
)
from data_loader import cm3_predict
from candidates.m2_bayasgalan import (
    _build_feature_matrix, _get_feature_cols
)

KEY_ROUTE = ["origin", "destination"]

CANDIDATES_M4 = [
    "M4_lgbm",
    "M4_resid_a003",
    "M4_resid_a005",
    "M4_resid_a008",
    "M4_resid_a010",
]

COMM_GROUP_KEYS = ["ag_food", "bulk", "chemicals", "fuel", "manufactured", "other"]

LGB_PARAMS_M4 = dict(
    objective="regression",
    num_leaves=31,
    min_child_samples=200,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10.0,
    random_state=RANDOM_SEED,
    verbose=-1,
)


def _sample_weights(tons: np.ndarray, large_route: np.ndarray, intrastate: np.ndarray) -> np.ndarray:
    """tons^0.35 × (1 + 4×large_route_flag + 2×intrastate_flag)"""
    w = np.power(np.clip(tons, 0, None) + 1.0, 0.35)
    multiplier = 1.0 + 4.0 * large_route + 2.0 * intrastate
    return w * multiplier


def _make_training_samples_m4(
    all_data: pd.DataFrame,
    feature_builder,
    min_target_year: int = MIN_TARGET_YEAR,
    covid_skip: set = None,
    horizon: int = 3,
) -> Optional[dict[str, tuple]]:
    """
    Generate training samples per commodity group.

    Returns dict[group_name → (X, y, weights)] or None.
    """
    if covid_skip is None:
        covid_skip = COVID_SKIP_YEARS

    avail_years = sorted(all_data["year"].unique())

    group_rows: dict[str, list] = {g: [] for g in COMM_GROUP_KEYS}

    for ty in avail_years:
        if ty < min_target_year:
            continue
        if ty in covid_skip:
            continue

        base_year = ty - horizon
        hist = all_data[all_data["year"] <= base_year]
        if hist.empty:
            continue

        actual = (
            all_data[all_data["year"] == ty]
            .groupby(KEY_COMM, as_index=False)["tons"]
            .sum()
        )
        if actual.empty:
            continue

        # CM3 at base_year
        cm3 = cm3_predict(hist, base_year)

        # Build features
        try:
            feat = feature_builder(hist, base_year)
        except Exception:
            continue

        idx = pd.MultiIndex.from_frame(actual[KEY_COMM])
        cm3_aligned = cm3.reindex(idx).fillna(0.0)
        feat_aligned = _build_feature_matrix(feat, idx)

        # Log-ratio target
        log_ratio = np.log1p(actual["tons"].values) - np.log1p(cm3_aligned.values)
        log_ratio = np.clip(log_ratio, -CORRECTION_CLIP * 5, CORRECTION_CLIP * 5)

        # Sample weights from structure features
        large_route = feat_aligned.get("large_route_flag", pd.Series(0, index=idx)).values
        intrastate  = feat_aligned.get("intrastate_flag",  pd.Series(0, index=idx)).values
        w = _sample_weights(actual["tons"].values, large_route, intrastate)

        # Split by commodity group
        actual_reset = actual.reset_index(drop=True)
        actual_reset["_group"] = actual_reset["commodity"].map(get_commodity_group)

        for grp in COMM_GROUP_KEYS:
            mask = actual_reset["_group"] == grp
            if not mask.any():
                continue
            sub_idx = idx[mask.values]
            group_rows[grp].append((
                feat_aligned.loc[sub_idx],
                pd.Series(log_ratio[mask.values], index=sub_idx),
                pd.Series(w[mask.values], index=sub_idx),
            ))

    # Concatenate per group
    result = {}
    for grp in COMM_GROUP_KEYS:
        if not group_rows[grp]:
            continue
        X_parts = pd.concat([r[0] for r in group_rows[grp]])
        y_parts = pd.concat([r[1] for r in group_rows[grp]])
        w_parts = pd.concat([r[2] for r in group_rows[grp]])
        result[grp] = (X_parts, y_parts, w_parts)

    return result if result else None


class M4Model:
    """Per-commodity-group LGBM with sample weights + route residual reallocation."""

    def __init__(self, lgb_params: dict = None):
        self.lgb_params = lgb_params or LGB_PARAMS_M4
        self._models: dict[str, lgb.LGBMRegressor] = {}
        self._feature_cols: list[str] = []

    def fit(
        self,
        all_data: pd.DataFrame,
        feature_builder,
        context_year: int,
    ) -> "M4Model":
        """Fit per-group models."""
        group_data = _make_training_samples_m4(
            all_data,
            feature_builder=feature_builder,
            min_target_year=MIN_TARGET_YEAR,
            covid_skip=COVID_SKIP_YEARS,
        )
        if group_data is None:
            print("  [M4] No training samples available.")
            return self

        all_feat_cols = None
        for grp, (X, y, w) in group_data.items():
            if all_feat_cols is None:
                all_feat_cols = _get_feature_cols(X)
            feat_cols = [c for c in all_feat_cols if c in X.columns]
            if not feat_cols:
                continue

            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(
                X[feat_cols].fillna(0.0),
                y.values,
                sample_weight=w.values,
            )
            self._models[grp] = model
            print(f"  [M4] Fitted group={grp} | samples={len(X):,}")

        if all_feat_cols:
            self._feature_cols = all_feat_cols

        return self

    def _predict_base(
        self,
        hist_df: pd.DataFrame,
        base_year: int,
        feature_builder,
        idx: pd.MultiIndex,
    ) -> tuple[pd.Series, pd.Series]:
        """Returns (cm3, lgbm_corrected) as Series on idx."""
        cm3 = cm3_predict(hist_df, base_year).reindex(idx).fillna(0.0)

        if not self._models or not self._feature_cols:
            return cm3, cm3.clip(lower=0)

        try:
            feat = feature_builder(hist_df, base_year)
            X = _build_feature_matrix(feat, idx)
        except Exception as e:
            print(f"  [M4] Feature build failed: {e}")
            return cm3, cm3.clip(lower=0)

        available_cols = [c for c in self._feature_cols if c in X.columns]

        # Predict per group
        correction = np.zeros(len(idx))
        comm_arr = np.array(idx.get_level_values("commodity"))

        for grp, model in self._models.items():
            mask = np.array([get_commodity_group(c) == grp for c in comm_arr])
            if not mask.any():
                continue
            X_grp = X.iloc[mask][available_cols].fillna(0.0) if available_cols else X.iloc[mask].fillna(0.0)
            correction[mask] = np.clip(
                model.predict(X_grp) * SHRINKAGE_FACTOR,
                -CORRECTION_CLIP, CORRECTION_CLIP
            )

        cm3_vals = cm3.values
        lgbm_vals = np.expm1(np.log1p(cm3_vals) + correction).clip(0)
        return cm3, pd.Series(lgbm_vals, index=idx)

    def _apply_route_residual(
        self,
        comm_preds: pd.Series,
        cm3_route: pd.Series,
        alpha: float,
    ) -> pd.Series:
        """
        Route residual reallocation.

        comm_preds : Series indexed (origin, destination, commodity)
        cm3_route  : Series indexed (origin, destination) — route-level CM3
        alpha      : reallocation weight
        """
        # Route sum of commodity predictions
        route_sum = comm_preds.groupby(level=[0, 1]).transform("sum")

        # Weights: log1p(comm) / log1p(route_sum)
        log_comm  = np.log1p(comm_preds.clip(lower=0))
        log_route = np.log1p(route_sum.clip(lower=0)).replace(0, np.nan)
        w = (log_comm / log_route).fillna(0.0)

        # Normalize weights per route to sum to 1
        w_sum = w.groupby(level=[0, 1]).transform("sum").replace(0, np.nan)
        w = (w / w_sum).fillna(0.0)

        # Route error: cm3_route - current route sum
        route_index = comm_preds.index.droplevel("commodity")
        route_err_mapped = route_index.map(
            lambda t: cm3_route.get(t, 0.0) if isinstance(t, tuple) else 0.0
        )
        # Use proper mapping
        route_error = cm3_route.reindex(
            pd.MultiIndex.from_tuples(
                [(o, d) for o, d, _ in comm_preds.index],
                names=["origin", "destination"]
            )
        ).fillna(0.0).values - route_sum.values

        result = (comm_preds.values + alpha * route_error * w.values).clip(0)
        return pd.Series(result, index=comm_preds.index)

    def predict(
        self,
        hist_df: pd.DataFrame,
        base_year: int,
        feature_builder,
        target_entities: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """
        Generate M4 predictions.

        Returns
        -------
        dict: {name: Series indexed (origin, destination, commodity)}
        """
        idx = pd.MultiIndex.from_frame(
            target_entities[KEY_COMM].drop_duplicates()
        )
        cm3_comm, lgbm_comm = self._predict_base(hist_df, base_year, feature_builder, idx)

        # Route-level CM3 for residual reallocation
        cm3_route = cm3_comm.groupby(level=[0, 1]).sum()

        results = {
            "M4_lgbm": lgbm_comm.rename("M4_lgbm"),
        }

        alphas = {"M4_resid_a003": 0.03, "M4_resid_a005": 0.05,
                  "M4_resid_a008": 0.08, "M4_resid_a010": 0.10}

        for name, alpha in alphas.items():
            adjusted = self._apply_route_residual(lgbm_comm, cm3_route, alpha)
            results[name] = adjusted.rename(name)

        return results
