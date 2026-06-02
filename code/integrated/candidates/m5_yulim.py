"""
m5_yulim.py — M5: Medmean baseline + log-ratio LGBM.

CANDIDATES_M5 = ["M5_medmean_base", "M5_medmean_lgbm", "M5_medmean_lgbm_extfeat"]

Architecture
------------
- M5_medmean_base        : medmean baseline only (0.8×median3 + 0.2×mean3 per OD×comm)
- M5_medmean_lgbm        : medmean + log-ratio correction (features: B+C only)
- M5_medmean_lgbm_extfeat: medmean + log-ratio correction (features: B+C+D_ratio)

feature_builder receives B+C+D_ratio features all together.
Columns ending in _ratio3 are treated as D_ratio; the rest are B+C.

Training: target_year >= 2017, NO COVID skip
One LGBM per commodity_group (6 groups)
Correction clipped ±0.10
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

from config import (
    CORRECTION_CLIP, MIN_TARGET_YEAR, RANDOM_SEED, KEY_COMM, get_commodity_group, SHRINKAGE_FACTOR
)
from data_loader import medmean_predict
from candidates.m2_bayasgalan import _build_feature_matrix, _get_feature_cols

KEY_ROUTE = ["origin", "destination"]

CANDIDATES_M5 = [
    "M5_medmean_base",
    "M5_medmean_lgbm",
    "M5_medmean_lgbm_extfeat",
]

COMM_GROUP_KEYS = ["ag_food", "bulk", "chemicals", "fuel", "manufactured", "other"]

# D_ratio column suffix — used to split BC vs full feature sets
_RATIO_SUFFIX = "_ratio3"

LGB_PARAMS_M5 = dict(
    objective="regression",
    num_leaves=31,
    n_estimators=200,
    learning_rate=0.05,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    random_state=RANDOM_SEED,
    verbose=-1,
)


def _split_bc_ext_cols(feature_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Split numeric columns into BC (no _ratio3 suffix) and full (all numeric).
    Returns (cols_bc, cols_full).
    """
    all_cols = _get_feature_cols(feature_df)
    cols_bc   = [c for c in all_cols if not c.endswith(_RATIO_SUFFIX)]
    cols_full = all_cols
    return cols_bc, cols_full


def _make_training_samples_m5(
    all_data: pd.DataFrame,
    feature_builder,  # callable(history, base_year) -> B+C+D_ratio features
    min_target_year: int = MIN_TARGET_YEAR,
    horizon: int = 3,
) -> Optional[dict[str, tuple]]:
    """
    Generate training samples per commodity group (NO COVID skip for M5).

    Returns dict[group_name → (X_full, cols_bc, cols_full, y)] or None.
    X_full contains all features; cols_bc / cols_full select subsets.
    """
    avail_years = sorted(all_data["year"].unique())
    group_rows_X: dict[str, list] = {g: [] for g in COMM_GROUP_KEYS}
    group_rows_y: dict[str, list] = {g: [] for g in COMM_GROUP_KEYS}

    cols_bc_ref:   list[str] = []
    cols_full_ref: list[str] = []

    for ty in avail_years:
        if ty < min_target_year:
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

        base_pred = medmean_predict(hist, base_year)

        try:
            feat = feature_builder(hist, base_year)
        except Exception:
            continue

        idx = pd.MultiIndex.from_frame(actual[KEY_COMM])
        base_aligned = base_pred.reindex(idx).fillna(0.0)
        X = _build_feature_matrix(feat, idx)

        if not cols_bc_ref:
            cols_bc_ref, cols_full_ref = _split_bc_ext_cols(X)

        log_ratio = np.log1p(actual["tons"].values) - np.log1p(base_aligned.values)
        log_ratio = np.clip(log_ratio, -CORRECTION_CLIP * 5, CORRECTION_CLIP * 5)

        actual_reset = actual.reset_index(drop=True)
        actual_reset["_group"] = actual_reset["commodity"].map(get_commodity_group)

        for grp in COMM_GROUP_KEYS:
            mask = (actual_reset["_group"] == grp).values
            if not mask.any():
                continue
            sub_idx = idx[mask]
            group_rows_X[grp].append(X.loc[sub_idx])
            group_rows_y[grp].append(pd.Series(log_ratio[mask], index=sub_idx))

    result = {}
    for grp in COMM_GROUP_KEYS:
        if not group_rows_y[grp]:
            continue
        X_all = pd.concat(group_rows_X[grp])
        y     = pd.concat(group_rows_y[grp])
        result[grp] = (X_all, y)

    return (result, cols_bc_ref, cols_full_ref) if result else None


class M5Model:
    """Medmean + per-group LightGBM log-ratio correction."""

    def __init__(self):
        self._models_bc:   dict[str, lgb.LGBMRegressor] = {}  # B+C only
        self._models_full: dict[str, lgb.LGBMRegressor] = {}  # B+C+D_ratio
        self._feature_cols_bc:   list[str] = []
        self._feature_cols_full: list[str] = []

    def fit(
        self,
        all_data: pd.DataFrame,
        feature_builder,
        context_year: int,
    ) -> "M5Model":
        """
        Fit M5 models.

        Parameters
        ----------
        all_data        : concatenated train + context
        feature_builder : callable(hist, base_year) -> B+C+D_ratio feature DataFrame
        context_year    : last available year (unused here, kept for interface symmetry)
        """
        out = _make_training_samples_m5(all_data, feature_builder, MIN_TARGET_YEAR)
        if out is None:
            print("  [M5] No training samples.")
            return self

        group_data, cols_bc, cols_full = out
        self._feature_cols_bc   = cols_bc
        self._feature_cols_full = cols_full

        has_ratio_cols = bool(cols_full and any(c.endswith(_RATIO_SUFFIX) for c in cols_full))

        for grp, (X_all, y) in group_data.items():
            if len(y) < 5:
                continue

            # B+C model (no D_ratio columns)
            fc_bc = [c for c in cols_bc if c in X_all.columns]
            if fc_bc:
                m_bc = lgb.LGBMRegressor(**LGB_PARAMS_M5)
                m_bc.fit(X_all[fc_bc].fillna(0.0), y.values)
                self._models_bc[grp] = m_bc

            # B+C+D_ratio model
            if has_ratio_cols:
                fc_full = [c for c in cols_full if c in X_all.columns]
                if fc_full:
                    m_full = lgb.LGBMRegressor(**LGB_PARAMS_M5)
                    m_full.fit(X_all[fc_full].fillna(0.0), y.values)
                    self._models_full[grp] = m_full

        print(f"  [M5] Fitted | groups_bc={len(self._models_bc)} | groups_ext={len(self._models_full)}")
        return self

    def predict(
        self,
        hist_df: pd.DataFrame,
        base_year: int,
        feature_builder,
        target_entities: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """
        Generate M5 predictions.

        Returns
        -------
        dict: {name: Series indexed (origin, destination, commodity)}
        """
        idx = pd.MultiIndex.from_frame(
            target_entities[KEY_COMM].drop_duplicates()
        )

        base_pred = medmean_predict(hist_df, base_year).reindex(idx).fillna(0.0)
        base_vals = base_pred.values

        results = {
            "M5_medmean_base": base_pred.clip(lower=0).rename("M5_medmean_base"),
        }

        if not self._models_bc:
            results["M5_medmean_lgbm"]        = base_pred.clip(lower=0).rename("M5_medmean_lgbm")
            results["M5_medmean_lgbm_extfeat"] = base_pred.clip(lower=0).rename("M5_medmean_lgbm_extfeat")
            return results

        try:
            feat = feature_builder(hist_df, base_year)
            X = _build_feature_matrix(feat, idx)
        except Exception as e:
            print(f"  [M5] Feature build failed: {e}")
            results["M5_medmean_lgbm"]        = base_pred.clip(lower=0).rename("M5_medmean_lgbm")
            results["M5_medmean_lgbm_extfeat"] = base_pred.clip(lower=0).rename("M5_medmean_lgbm_extfeat")
            return results

        comm_arr = np.array(idx.get_level_values("commodity"))
        correction_bc   = np.zeros(len(idx))
        correction_full = np.zeros(len(idx))

        for grp in COMM_GROUP_KEYS:
            mask = np.array([get_commodity_group(c) == grp for c in comm_arr])
            if not mask.any():
                continue

            X_grp = X.iloc[mask]

            if grp in self._models_bc and self._feature_cols_bc:
                fc = [c for c in self._feature_cols_bc if c in X_grp.columns]
                if fc:
                    correction_bc[mask] = np.clip(
                        self._models_bc[grp].predict(X_grp[fc].fillna(0.0)) * SHRINKAGE_FACTOR,
                        -CORRECTION_CLIP, CORRECTION_CLIP,
                    )

            if grp in self._models_full and self._feature_cols_full:
                fc = [c for c in self._feature_cols_full if c in X_grp.columns]
                if fc:
                    correction_full[mask] = np.clip(
                        self._models_full[grp].predict(X_grp[fc].fillna(0.0)) * SHRINKAGE_FACTOR,
                        -CORRECTION_CLIP, CORRECTION_CLIP,
                    )
                else:
                    correction_full[mask] = correction_bc[mask]
            else:
                correction_full[mask] = correction_bc[mask]

        lgbm_vals     = np.expm1(np.log1p(base_vals) + correction_bc).clip(0)
        lgbm_ext_vals = np.expm1(np.log1p(base_vals) + correction_full).clip(0)

        results["M5_medmean_lgbm"]        = pd.Series(lgbm_vals,     index=idx, name="M5_medmean_lgbm")
        results["M5_medmean_lgbm_extfeat"] = pd.Series(lgbm_ext_vals, index=idx, name="M5_medmean_lgbm_extfeat")

        return results
