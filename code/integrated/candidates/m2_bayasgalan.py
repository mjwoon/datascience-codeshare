"""
m2_bayasgalan.py — M2: CM3 baseline + LightGBM log-ratio residual.

CANDIDATES_M2 = ["M2_cm3only", "M2_cm3_lgbm"]

Architecture
------------
- CM3 = commodity median 3yr per OD×commodity
- Features: A + B + C + D_abs
- Target: log1p(actual) - log1p(cm3_pred)  (log-ratio residual)
- Training: target_year >= 2017, skip COVID year 2020
- Correction clipped ±0.10
- final = expm1(log1p(cm3_pred) + correction).clip(lower=0)
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CORRECTION_CLIP, MIN_TARGET_YEAR, COVID_SKIP_YEARS, RANDOM_SEED, KEY_COMM
from data_loader import cm3_predict

KEY_ROUTE = ["origin", "destination"]

CANDIDATES_M2 = ["M2_cm3only", "M2_cm3_lgbm"]

LGB_PARAMS_M2 = dict(
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


def _build_feature_matrix(
    feature_df: pd.DataFrame,
    index: pd.MultiIndex,
) -> pd.DataFrame:
    """
    Extract feature matrix aligned to the given MultiIndex.
    feature_df may be indexed by (origin, destination, commodity) or have those as columns.
    """
    if isinstance(feature_df.index, pd.MultiIndex):
        X = feature_df.reindex(index)
    else:
        X = feature_df.set_index(KEY_COMM).reindex(index)
    return X.fillna(0.0)


def _get_feature_cols(feature_df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (exclude KEY_COMM if present as columns)."""
    exclude = set(KEY_COMM + ["origin", "destination", "commodity"])
    if isinstance(feature_df.index, pd.MultiIndex):
        cols = [c for c in feature_df.columns if c not in exclude]
    else:
        cols = [c for c in feature_df.columns if c not in exclude]
    # Only numeric
    return [c for c in cols if pd.api.types.is_numeric_dtype(feature_df[c])]


def _make_training_samples(
    all_data: pd.DataFrame,
    feature_builder,  # callable(history_df, base_year) -> feature_df
    min_target_year: int = MIN_TARGET_YEAR,
    covid_skip: set = None,
    horizon: int = 3,
) -> Optional[tuple[pd.DataFrame, pd.Series]]:
    """
    Generate training samples by pseudo-fold over available target years.

    For each target year ty:
      - base_year = ty - 3
      - features from history up to base_year
      - actual tons at ty
      - log-ratio target = log1p(actual) - log1p(cm3_pred)
    """
    if covid_skip is None:
        covid_skip = COVID_SKIP_YEARS

    avail_years = sorted(all_data["year"].unique())
    rows_X = []
    rows_y = []

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

        # CM3 prediction at base_year
        cm3 = cm3_predict(hist, base_year)

        # Build features
        try:
            feat = feature_builder(hist, base_year)
        except Exception:
            continue

        # Align
        idx = pd.MultiIndex.from_frame(actual[KEY_COMM])
        cm3_aligned = cm3.reindex(idx).fillna(0.0)
        feat_aligned = _build_feature_matrix(feat, idx)

        # Log-ratio target
        log_ratio = np.log1p(actual["tons"].values) - np.log1p(cm3_aligned.values)
        log_ratio = np.clip(log_ratio, -CORRECTION_CLIP * 5, CORRECTION_CLIP * 5)

        rows_X.append(feat_aligned)
        rows_y.append(pd.Series(log_ratio, index=idx))

    if not rows_X:
        return None

    X_train = pd.concat(rows_X, axis=0)
    y_train = pd.concat(rows_y, axis=0)

    # Align
    common_idx = X_train.index.intersection(y_train.index)
    return X_train.loc[common_idx], y_train.loc[common_idx]


class M2Model:
    """CM3 + LightGBM log-ratio residual model."""

    def __init__(self, lgb_params: dict = None):
        self.lgb_params = lgb_params or LGB_PARAMS_M2
        self._model: Optional[lgb.LGBMRegressor] = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        all_data: pd.DataFrame,
        feature_builder,
        context_year: int,
    ) -> "M2Model":
        """
        Fit the LightGBM residual model.

        Parameters
        ----------
        all_data       : concatenated train + context data
        feature_builder: callable(history_df, base_year) -> DataFrame indexed (origin,destination,commodity)
        context_year   : the last available year (context year)
        """
        result = _make_training_samples(
            all_data,
            feature_builder=feature_builder,
            min_target_year=MIN_TARGET_YEAR,
            covid_skip=COVID_SKIP_YEARS,
        )
        if result is None:
            print("  [M2] No training samples available.")
            return self

        X_train, y_train = result
        feat_cols = _get_feature_cols(X_train)
        self._feature_cols = feat_cols

        if not feat_cols:
            print("  [M2] No features available.")
            return self

        model = lgb.LGBMRegressor(**self.lgb_params)
        model.fit(
            X_train[feat_cols].fillna(0.0),
            y_train.values,
        )
        self._model = model
        print(f"  [M2] Fitted | samples={len(X_train):,} | features={len(feat_cols)}")
        return self

    def predict(
        self,
        hist_df: pd.DataFrame,
        base_year: int,
        feature_builder,
        target_entities: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """
        Generate M2 predictions for target_entities.

        Parameters
        ----------
        hist_df         : history DataFrame available at prediction time
        base_year       : most recent year in hist_df (context_year when predicting val/test)
        feature_builder : callable(hist_df, base_year) -> feature_df
        target_entities : DataFrame with (origin, destination, commodity) columns

        Returns
        -------
        dict: {name: Series indexed (origin,destination,commodity)}
        """
        idx = pd.MultiIndex.from_frame(
            target_entities[KEY_COMM].drop_duplicates()
        )

        # CM3 prediction
        cm3 = cm3_predict(hist_df, base_year).reindex(idx).fillna(0.0)

        results = {"M2_cm3only": cm3.clip(lower=0).rename("M2_cm3only")}

        if self._model is None or not self._feature_cols:
            results["M2_cm3_lgbm"] = cm3.clip(lower=0).rename("M2_cm3_lgbm")
            return results

        # Build features
        try:
            feat = feature_builder(hist_df, base_year)
            X = _build_feature_matrix(feat, idx)
        except Exception as e:
            print(f"  [M2] Feature build failed: {e}")
            results["M2_cm3_lgbm"] = cm3.clip(lower=0).rename("M2_cm3_lgbm")
            return results

        # Predict correction
        available_cols = [c for c in self._feature_cols if c in X.columns]
        if not available_cols:
            results["M2_cm3_lgbm"] = cm3.clip(lower=0).rename("M2_cm3_lgbm")
            return results

        X_pred = X[available_cols].fillna(0.0)
        correction = self._model.predict(X_pred)
        correction = np.clip(correction, -CORRECTION_CLIP, CORRECTION_CLIP)

        cm3_vals = cm3.values
        final = np.expm1(np.log1p(cm3_vals) + correction).clip(0)
        results["M2_cm3_lgbm"] = pd.Series(final, index=idx, name="M2_cm3_lgbm")

        return results
