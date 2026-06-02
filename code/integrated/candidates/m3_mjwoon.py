"""
m3_mjwoon.py — M3: TwoStage Hurdle model with CM3 baseline + log-ratio.

CANDIDATES_M3 = ["M3_lgbm", "M3_hurdle"]

Architecture
------------
- M3_lgbm   : single LGBM log-ratio regressor (similar to M2 but different hyperparams)
- M3_hurdle : Stage1 LGBMClassifier P(tons>0) × Stage2 log-ratio regressor

Features: A + B + C + D_abs
Target: log1p(actual) - log1p(cm3_pred)
COVID filter: skip 2020
Correction clipped ±0.10
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CORRECTION_CLIP, MIN_TARGET_YEAR, COVID_SKIP_YEARS, RANDOM_SEED, KEY_COMM, SHRINKAGE_FACTOR
from data_loader import cm3_predict
from candidates.m2_bayasgalan import (
    _build_feature_matrix, _get_feature_cols, _make_training_samples
)

KEY_ROUTE = ["origin", "destination"]

CANDIDATES_M3 = ["M3_lgbm", "M3_hurdle"]

LGB_PARAMS_M3_REGRESSOR = dict(
    objective="regression",
    num_leaves=31,
    n_estimators=200,
    learning_rate=0.05,
    min_child_samples=100,
    random_state=RANDOM_SEED,
    verbose=-1,
)

LGB_PARAMS_M3_CLASSIFIER = dict(
    num_leaves=15,
    n_estimators=100,
    learning_rate=0.1,
    min_child_samples=50,
    random_state=RANDOM_SEED,
    verbose=-1,
)


class M3Model:
    """TwoStage Hurdle model: CM3 + LightGBM log-ratio."""

    def __init__(self):
        self._lgbm_model: Optional[lgb.LGBMRegressor] = None
        self._clf_model: Optional[lgb.LGBMClassifier] = None
        self._reg_model: Optional[lgb.LGBMRegressor] = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        all_data: pd.DataFrame,
        feature_builder,
        context_year: int,
    ) -> "M3Model":
        """
        Fit both M3 variants.

        Parameters
        ----------
        all_data       : concatenated train + context data
        feature_builder: callable(history_df, base_year) -> feature_df
        context_year   : the last available year (context year)
        """
        result = _make_training_samples(
            all_data,
            feature_builder=feature_builder,
            min_target_year=MIN_TARGET_YEAR,
            covid_skip=COVID_SKIP_YEARS,
        )
        if result is None:
            print("  [M3] No training samples available.")
            return self

        X_train, y_train = result
        feat_cols = _get_feature_cols(X_train)
        self._feature_cols = feat_cols

        if not feat_cols:
            print("  [M3] No features available.")
            return self

        X = X_train[feat_cols].fillna(0.0)
        y = y_train.values

        # M3_lgbm: single log-ratio regressor
        lgbm_reg = lgb.LGBMRegressor(**LGB_PARAMS_M3_REGRESSOR)
        lgbm_reg.fit(X, y)
        self._lgbm_model = lgbm_reg

        # M3_hurdle: Stage 1 classifier
        # Need actual tons to build binary target
        # We approximate: if log_ratio > very low threshold, consider positive
        # Use original actual tons via the index (we need to reconstruct)
        # For the hurdle, we build binary from the original data
        binary_target = self._build_binary_labels(all_data)
        if binary_target is not None:
            X_clf, y_clf = self._align_binary(X_train, feat_cols, binary_target)
            if X_clf is not None and len(y_clf.unique()) == 2:
                clf = lgb.LGBMClassifier(**LGB_PARAMS_M3_CLASSIFIER)
                clf.fit(X_clf, y_clf)
                self._clf_model = clf

                # Stage 2: log-ratio only for positive samples
                pos_mask = y_clf.values == 1
                if pos_mask.sum() >= 10:
                    reg2 = lgb.LGBMRegressor(**LGB_PARAMS_M3_REGRESSOR)
                    reg2.fit(X_clf[pos_mask], y_train.reindex(X_clf.index)[pos_mask].fillna(0))
                    self._reg_model = reg2

        print(f"  [M3] Fitted | samples={len(X_train):,} | features={len(feat_cols)}")
        return self

    def _build_binary_labels(self, all_data: pd.DataFrame) -> Optional[pd.Series]:
        """Build binary labels (1 if tons > 0) from all_data."""
        avail = sorted(all_data["year"].unique())
        rows = []
        for ty in avail:
            if ty < MIN_TARGET_YEAR or ty in COVID_SKIP_YEARS:
                continue
            actual = (
                all_data[all_data["year"] == ty]
                .groupby(KEY_COMM, as_index=False)["tons"].sum()
            )
            if actual.empty:
                continue
            actual["binary"] = (actual["tons"] > 0).astype(int)
            rows.append(actual.set_index(KEY_COMM)["binary"])
        if not rows:
            return None
        return pd.concat(rows)

    def _align_binary(
        self, X_train: pd.DataFrame, feat_cols: list, binary_target: pd.Series
    ) -> tuple:
        """Align X_train with binary labels."""
        common = X_train.index.intersection(binary_target.index)
        if len(common) == 0:
            return None, None
        return X_train.loc[common, feat_cols].fillna(0.0), binary_target.loc[common]

    def predict(
        self,
        hist_df: pd.DataFrame,
        base_year: int,
        feature_builder,
        target_entities: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """
        Generate M3 predictions.

        Returns
        -------
        dict: {"M3_lgbm": Series, "M3_hurdle": Series}
        """
        idx = pd.MultiIndex.from_frame(
            target_entities[KEY_COMM].drop_duplicates()
        )
        cm3 = cm3_predict(hist_df, base_year).reindex(idx).fillna(0.0)
        cm3_vals = cm3.values

        results = {}
        fallback = pd.Series(cm3_vals.clip(0), index=idx)

        # Build features
        if not self._feature_cols:
            results["M3_lgbm"]   = fallback.rename("M3_lgbm")
            results["M3_hurdle"] = fallback.rename("M3_hurdle")
            return results

        try:
            feat = feature_builder(hist_df, base_year)
            X = _build_feature_matrix(feat, idx)
        except Exception as e:
            print(f"  [M3] Feature build failed: {e}")
            results["M3_lgbm"]   = fallback.rename("M3_lgbm")
            results["M3_hurdle"] = fallback.rename("M3_hurdle")
            return results

        available_cols = [c for c in self._feature_cols if c in X.columns]
        X_pred = X[available_cols].fillna(0.0) if available_cols else X.fillna(0.0)

        # M3_lgbm
        if self._lgbm_model is not None and available_cols:
            corr = np.clip(
                self._lgbm_model.predict(X_pred) * SHRINKAGE_FACTOR,
                -CORRECTION_CLIP, CORRECTION_CLIP
            )
            lgbm_final = np.expm1(np.log1p(cm3_vals) + corr).clip(0)
        else:
            lgbm_final = cm3_vals.clip(0)
        results["M3_lgbm"] = pd.Series(lgbm_final, index=idx, name="M3_lgbm")

        # M3_hurdle: P(>0) × stage2 correction
        if (self._clf_model is not None and self._reg_model is not None
                and available_cols):
            p_pos = self._clf_model.predict_proba(X_pred)[:, 1]
            corr2 = np.clip(
                self._reg_model.predict(X_pred) * SHRINKAGE_FACTOR,
                -CORRECTION_CLIP, CORRECTION_CLIP
            )
            hurdle_final = p_pos * np.expm1(np.log1p(cm3_vals) + corr2).clip(0)
        else:
            hurdle_final = lgbm_final

        results["M3_hurdle"] = pd.Series(hurdle_final, index=idx, name="M3_hurdle")
        return results
