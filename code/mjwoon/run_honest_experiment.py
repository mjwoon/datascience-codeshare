"""
Truck Route Volume Prediction — MLflow Experiment Runner (3년 후 예측 + 외부 데이터)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모델    : LightGBM (Tweedie p=1.5) — 확정 모델
피처    : context.csv 집계 (8개)
        + train.csv 시계열 trend (5개, 정제된 전체 기간 사용)
        + 외부 데이터 (~39개) = 총 52개
데이터  : data/faf4_faf5_fixed_year_splits/{split_1,2,3}/
            ├── train.csv     학습 피처용 시계열 이력
            ├── context.csv   학습 피처 원본 (commodity 포함 상세)
            ├── val.csv       학습 타겟 (route-level tons, context+2년)
            └── test.csv      평가 타겟 (route-level tons, context+3년)
          data/additional_dataset/   외부 데이터 (15개 CSV)
지표    : wRMSE (1차) / MAE, RMSLE, WMAPE, R² (해석·보고용)

실행 방법:
    python experiment_mlflow.py
"""

import argparse
import json
import os
import random
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import mlflow

from code.mjwoon.modeling_baseline import (
    build_route_features,
    build_train_features,
)


from code.mjwoon.load_external import (
    load_all_external,
    STATE_FEATURE_COLS,
    NATIONAL_FEATURE_COLS,
)


class TwoStageHurdleModel:
    def __init__(self, classifier_params: dict, regressor_cls, regressor_params: dict):
        self.classifier_params = classifier_params
        self.regressor_cls = regressor_cls
        self.regressor_params = regressor_params
        self.classifier = None
        self.regressor = None
        self.__class__.__name__ = "TwoStageHurdleModel"

    def fit(self, X, y, sample_weight=None, categorical_feature=None):
        self.classifier = lgb.LGBMClassifier(**self.classifier_params)
        self.regressor = self.regressor_cls(**self.regressor_params)

        y_bin = (y > 0).astype(int)
        fit_params = {}
        if categorical_feature:
            fit_params["categorical_feature"] = categorical_feature
            
        if sample_weight is not None:
            self.classifier.fit(X, y_bin, sample_weight=sample_weight, **fit_params)
        else:
            self.classifier.fit(X, y_bin, **fit_params)

        # Fit regressor only on positive targets
        pos_mask = y > 0
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]
        
        if sample_weight is not None:
            sw_pos = sample_weight[pos_mask]
        else:
            sw_pos = None

        if len(y_pos) > 0:
            if sw_pos is not None:
                self.regressor.fit(X_pos, y_pos, sample_weight=sw_pos, **fit_params)
            else:
                self.regressor.fit(X_pos, y_pos, **fit_params)
        return self

    def predict(self, X):
        p = self.classifier.predict_proba(X)[:, 1]
        pred_val = self.regressor.predict(X)
        return p * pred_val


# ─────────────────────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────────────────────

CONFIG_PATH   = Path("experiment_config.json")
ARTIFACT_DIR  = Path("mlflow_artifacts")
SPLITS_DIR    = Path("data/faf4_faf5_fixed_year_splits_val192021")
EXTERNAL_DIR  = Path("data/additional_dataset")

# 팀 표준 split 정의 (context year 기준 → +3년 = test)
SPLIT_DEFS = {
    "split_1": {"context_year": 2019, "val_year": 2019, "test_year": 2022, "weight": 0.2},
    "split_2": {"context_year": 2020, "val_year": 2020, "test_year": 2023, "weight": 0.3},
    "split_3": {"context_year": 2021, "val_year": 2021, "test_year": 2024, "weight": 0.5},
}

TARGET_COL = "tons"

# train.csv 시계열 trend 피처 (전체 구간 기반)
TRAIN_FEATURE_COLS = [
    'tons_recent_avg',
    'tons_cagr',
    'tons_volatility',
    'tons_yoy_last',
    'tons_trend_slope',
    'value_recent_avg',
    'value_cagr',
    'value_volatility',
    'value_yoy_last',
    'value_trend_slope',
    # New growth features
    'tons_yoy_lag1',
    'tons_yoy_lag2',
    'tons_cagr_3yr',
    'value_yoy_lag1',
    'value_yoy_lag2',
    'value_cagr_3yr',
    # New unit price features
    'unit_price_recent_avg',
    'unit_price_cagr',
    'unit_price_volatility',
    'unit_price_yoy_last',
    'unit_price_trend_slope',
    'unit_price_yoy_lag1',
    'unit_price_yoy_lag2',
    'unit_price_cagr_3yr'
]
ROUTE_TRAIN_FEATURE_COLS = [f"route_{col}" for col in TRAIN_FEATURE_COLS]
ROUTE_CONTEXT_FEATURE_COLS = ['route_tons_ctx', 'route_value_ctx', 'route_tmiles_ctx']
TARGET_ENCODING_COLS = [
    'comm_te_mean', 'comm_te_std',
    'comm_group_te_mean', 'comm_group_te_std'
]

COMMODITY_GROUPS = {
    # Agricultural & Food
    'Cereal grains': 'agricultural_food',
    'Other ag prods.': 'agricultural_food',
    'Live animals/fish': 'agricultural_food',
    'Meat/seafood': 'agricultural_food',
    'Milled grain prods.': 'agricultural_food',
    'Other foodstuffs': 'agricultural_food',
    'Animal feed': 'agricultural_food',
    'Alcoholic beverages': 'agricultural_food',
    'Tobacco prods.': 'agricultural_food',
    
    # Mineral & Energy
    'Coal': 'mineral_energy',
    'Crude petroleum': 'mineral_energy',
    'Gasoline': 'mineral_energy',
    'Fuel oils': 'mineral_energy',
    'Natural gas/fossil': 'mineral_energy',
    'Metallic ores': 'mineral_energy',
    'Nonmetallic minerals': 'mineral_energy',
    'Building stone': 'mineral_energy',
    'Natural sands': 'mineral_energy',
    'Gravel': 'mineral_energy',
    
    # Chemicals & Plastics
    'Basic chemicals': 'chemical_plastics',
    'Chemical prods.': 'chemical_plastics',
    'Fertilizers': 'chemical_plastics',
    'Plastics/rubber': 'chemical_plastics',
    'Pharmaceuticals': 'chemical_plastics',
    
    # Forest, Wood & Paper
    'Logs': 'forest_paper',
    'Wood prods.': 'forest_paper',
    'Newsprint/paper': 'forest_paper',
    'Paper articles': 'forest_paper',
    'Printed prods.': 'forest_paper',
    
    # Metals & Manufacturing
    'Base metals': 'metals_manufacturing',
    'Articles base metal': 'metals_manufacturing',
    'Nonmetal min. prods.': 'metals_manufacturing',
    'Machinery': 'metals_manufacturing',
    'Electronics': 'metals_manufacturing',
    'Motorized vehicles': 'metals_manufacturing',
    'Transport equip.': 'metals_manufacturing',
    'Precision instruments': 'metals_manufacturing',
    'Furniture': 'metals_manufacturing',
    'Textiles/leather': 'metals_manufacturing',
    'Misc. mfg. prods.': 'metals_manufacturing',
    
    # Other & Mixed
    'Mixed freight': 'other_mixed',
    'Waste/scrap': 'other_mixed',
}

ORIGIN_FEATURE_COLS  = [f"origin_{c}" for c in STATE_FEATURE_COLS]
DEST_FEATURE_COLS    = [f"dest_{c}" for c in STATE_FEATURE_COLS]
NAT_FEATURE_COLS     = NATIONAL_FEATURE_COLS.copy()
DERIVED_FEATURE_COLS = [
    'gdp_ratio', 'pop_ratio', 'income_ratio', 'freight_intensity',
]


# ─────────────────────────────────────────────────────────────
# FAF4-FAF5 다차원 보정 및 변동성 노선 탐지
# ─────────────────────────────────────────────────────────────

def apply_multidimensional_bridging(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. 2016년과 2017년의 [origin, destination, commodity] 별 물동량 합산
    df_2016 = df[df['year'] == 2016].groupby(['origin', 'destination', 'commodity'])['tons'].sum().reset_index()
    df_2017 = df[df['year'] == 2017].groupby(['origin', 'destination', 'commodity'])['tons'].sum().reset_index()
    
    # 2. Route 레벨의 2016-2017 합산 (Fallback 용)
    route_2016 = df[df['year'] == 2016].groupby(['origin', 'destination'])['tons'].sum().reset_index()
    route_2017 = df[df['year'] == 2017].groupby(['origin', 'destination'])['tons'].sum().reset_index()
    route_ratio = pd.merge(route_2017, route_2016, on=['origin', 'destination'], suffixes=('_2017', '_2016'))
    route_ratio['route_m'] = (route_ratio['tons_2017'] + 1e-5) / (route_ratio['tons_2016'] + 1e-5)
    route_ratio['route_m'] = route_ratio['route_m'].clip(0.5, 2.0)
    
    # 3. Commodity-level 합산 (오리진/데스티네이션 매칭 실패 시 Fallback 용)
    comm_2016 = df[df['year'] == 2016].groupby(['commodity'])['tons'].sum().reset_index()
    comm_2017 = df[df['year'] == 2017].groupby(['commodity'])['tons'].sum().reset_index()
    comm_ratio = pd.merge(comm_2017, comm_2016, on=['commodity'], suffixes=('_2017', '_2016'))
    comm_ratio['comm_m'] = (comm_ratio['tons_2017'] + 1e-5) / (comm_ratio['tons_2016'] + 1e-5)
    comm_ratio['comm_m'] = comm_ratio['comm_m'].clip(0.5, 2.0)
    
    # 4. 세그먼트 레벨 머지 및 비율 계산
    merged = pd.merge(df_2017, df_2016, on=['origin', 'destination', 'commodity'], suffixes=('_2017', '_2016'), how='outer').fillna(0)
    
    # 기본 비율 계산
    merged['m'] = (merged['tons_2017'] + 1e-5) / (merged['tons_2016'] + 1e-5)
    
    # Fallback 결합
    merged = pd.merge(merged, route_ratio[['origin', 'destination', 'route_m']], on=['origin', 'destination'], how='left')
    merged = pd.merge(merged, comm_ratio[['commodity', 'comm_m']], on=['commodity'], how='left')
    
    # 2016년이나 2017년 물동량이 둘 다 거의 없거나, 한쪽에만 있어서 비율이 튀는 경우 처리
    is_valid = (merged['tons_2016'] > 1.0) & (merged['tons_2017'] > 1.0)
    merged['final_m'] = np.where(is_valid, merged['m'], merged['route_m'])
    merged['final_m'] = np.where(merged['final_m'].isna(), merged['comm_m'], merged['final_m'])
    merged['final_m'] = merged['final_m'].fillna(1.0).clip(0.5, 2.0)
    
    # 5. 원래 데이터에 보정 계수 곱하기 (2012 ~ 2016년)
    df_aligned = pd.merge(df, merged[['origin', 'destination', 'commodity', 'final_m']], on=['origin', 'destination', 'commodity'], how='left')
    df_aligned['final_m'] = df_aligned['final_m'].fillna(1.0)
    
    # 2016년 이하 데이터에만 적용
    mask = df_aligned['year'] <= 2016
    df_aligned.loc[mask, 'tons'] = df_aligned.loc[mask, 'tons'] * df_aligned.loc[mask, 'final_m']
    
    df_aligned = df_aligned.drop(columns=['final_m'])
    return df_aligned


def detect_shock_routes(df: pd.DataFrame, base_year: int, threshold_pct: float = 0.15) -> set[tuple[str, str]]:
    """
    base_year 기준 과거 4개년 (base_year-3 ~ base_year) 동안의 YoY 변동폭 절대값을 구하고,
    최신 변화율에 가중치를 두는 지수 가중 평균(EWMA)을 통해 상위 threshold_pct에 해당하는 경로를 반환.
    """
    years = [base_year - 3, base_year - 2, base_year - 1, base_year]
    history = df[df['year'].isin(years)].groupby(['origin', 'destination', 'year'])['tons'].sum().unstack(fill_value=0.0)
    
    if history.empty or len(history.columns) < 2:
        return set()
        
    yoy_scores = []
    avail_years = sorted(list(history.columns))
    for i in range(1, len(avail_years)):
        y_prev = history[avail_years[i-1]]
        y_curr = history[avail_years[i]]
        yoy = np.abs(y_curr - y_prev) / (y_prev + 100.0)
        yoy_scores.append(yoy)
        
    if not yoy_scores:
        return set()
        
    # Apply exponential decay weights (giving exponentially higher weight to recent years)
    n_intervals = len(yoy_scores)
    decay_weights = np.exp(np.linspace(-1.0, 0.0, n_intervals))
    decay_weights /= decay_weights.sum()
    
    weighted_yoy_list = []
    for w, yoy in zip(decay_weights, yoy_scores):
        weighted_yoy_list.append(yoy * w)
        
    mean_yoy = pd.concat(weighted_yoy_list, axis=1).sum(axis=1)
    
    cutoff = mean_yoy.quantile(1.0 - threshold_pct)
    shock_routes = mean_yoy[mean_yoy >= cutoff].index.tolist()
    
    return set(shock_routes)


def fit_target_encoding(train_df: pd.DataFrame, col: str, target_col: str, stat: str = "mean", alpha: float = 20.0) -> dict:
    global_stat = train_df[target_col].agg(stat)
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(['count', stat])
    encoded_vals = {}
    for cat, row in stats.iterrows():
        n = row['count']
        val = row[stat]
        if pd.isna(val):
            val = global_mean if stat == "std" else global_stat
        encoded_vals[cat] = (n * val + alpha * global_stat) / (n + alpha)
    return {
        'mapping': encoded_vals,
        'global_stat': global_stat
    }

def transform_target_encoding(df: pd.DataFrame, col: str, te_info: dict) -> pd.Series:
    return df[col].map(te_info['mapping']).fillna(te_info['global_stat'])

def get_feature_cols(granularity: str = "route") -> list[str]:
    # 기존 8개 + 새 피처 1개
    base_cols = [
        "origin_enc", "destination_enc",
        "tons_ctx", "value_ctx", "tmiles_ctx",
        "tons_mean_c", "tons_std_c",
        "n_commodity",
        "is_faf5_era",
    ]
    if granularity == "commodity":
        base_cols.append("commodity_enc")
        base_cols.append("commodity_group_enc")

        return (
            base_cols
            + TRAIN_FEATURE_COLS
            + ROUTE_TRAIN_FEATURE_COLS
            + ROUTE_CONTEXT_FEATURE_COLS
            + TARGET_ENCODING_COLS
            + ORIGIN_FEATURE_COLS
            + DEST_FEATURE_COLS
            + NAT_FEATURE_COLS
            + DERIVED_FEATURE_COLS
        )

    return (
        base_cols
        + TRAIN_FEATURE_COLS
        + ORIGIN_FEATURE_COLS
        + DEST_FEATURE_COLS
        + NAT_FEATURE_COLS
        + DERIVED_FEATURE_COLS
    )

METRIC_NAMES = [
    "wRMSE", "RMSE", "MAE", "WMAPE", "RMSLE", "R2",
    "Skill_Score_Naive", "Skill_Score_YoY_Median", "Skill_Score_Constant_CAGR",
    "Directional_Accuracy", "CAGR_Error", "Growth_WMAPE"
]



# ─────────────────────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """NumPy, Python random, 환경변수 seed 고정."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build_run_name(experimenter: str, model_name: str) -> str:
    return f"{experimenter}_{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"


def write_requirements() -> Path:
    """현재 환경의 pip freeze 결과를 파일로 저장."""
    ARTIFACT_DIR.mkdir(exist_ok=True)
    path = ARTIFACT_DIR / "requirements_freeze.txt"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True
    )
    path.write_text(result.stdout, encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────
# 범주형 인코더 (전체 split 공통)
# ─────────────────────────────────────────────────────────────

def build_encoders(granularity: str = "route") -> dict[str, LabelEncoder]:
    """모든 split의 context.csv에서 등장하는 지역명(및 화물명)을 한 번에 fit."""
    all_locs: set[str] = set()
    all_comms: set[str] = set()
    for sn in SPLIT_DEFS:
        ctx_tmp = pd.read_csv(
            SPLITS_DIR / sn / "context.csv",
            usecols=["origin", "destination", "commodity"],
        )
        all_locs.update(ctx_tmp["origin"].astype(str))
        all_locs.update(ctx_tmp["destination"].astype(str))
        all_comms.update(ctx_tmp["commodity"].astype(str))

    encoders: dict[str, LabelEncoder] = {}
    for col in ["origin", "destination"]:
        le = LabelEncoder()
        le.fit(sorted(all_locs))
        encoders[col] = le
    
    if granularity == "commodity":
        le_c = LabelEncoder()
        le_c.fit(sorted(all_comms))
        encoders["commodity"] = le_c
        
        # Add commodity group encoder
        all_groups = sorted(list(set(COMMODITY_GROUPS.values())))
        le_g = LabelEncoder()
        le_g.fit(all_groups)
        encoders["commodity_group"] = le_g
        
    return encoders


def encode_col(series: pd.Series, le: LabelEncoder) -> pd.Series:
    """unseen 값은 -1로 처리."""
    mapping = {c: i for i, c in enumerate(le.classes_)}
    return series.astype(str).map(mapping).fillna(-1).astype(int)


# ─────────────────────────────────────────────────────────────
# 피처 엔지니어링
# ─────────────────────────────────────────────────────────────

def build_route_features(
    context_df: pd.DataFrame,
    state_feat: pd.DataFrame,
    national_feat: dict[str, float],
    granularity: str = "route",
) -> pd.DataFrame:
    """
    context.csv + 외부 데이터 →
    route(origin × destination) 또는 (origin × destination × commodity) 수준 피처 DataFrame 반환.
    """
    ctx = context_df.copy()
    ctx["year"] = ctx["year"].astype(int)

    # 기존 route-level 또는 commodity-level 집계
    group_cols = ["origin", "destination", "commodity"] if granularity == "commodity" else ["origin", "destination"]
    
    agg = ctx.groupby(group_cols).agg(
        tons_ctx    =("tons",      "sum"),
        value_ctx   =("value",     "sum"),
        tmiles_ctx  =("tmiles",    "sum"),
        tons_mean_c =("tons",      "mean"),
        tons_std_c  =("tons",      "std"),
        n_commodity =("commodity", "nunique"),
        context_year=("year",      "max"),
    ).reset_index()
    agg["tons_std_c"] = agg["tons_std_c"].fillna(0.0)

    if granularity == "commodity":
        # Calculate route-level context sums and merge (Suggestion 2)
        route_agg = ctx.groupby(["origin", "destination"]).agg(
            route_tons_ctx=("tons", "sum"),
            route_value_ctx=("value", "sum"),
            route_tmiles_ctx=("tmiles", "sum"),
        ).reset_index()
        agg = agg.merge(route_agg, on=["origin", "destination"], how="left")

    # Origin-side 외부 피처 join
    origin_ext = state_feat.rename(
        columns={col: f"origin_{col}" for col in STATE_FEATURE_COLS}
    ).rename(columns={"state_name": "origin"})
    agg = agg.merge(origin_ext, on="origin", how="left")

    # Destination-side 외부 피처 join
    dest_ext = state_feat.rename(
        columns={col: f"dest_{col}" for col in STATE_FEATURE_COLS}
    ).rename(columns={"state_name": "destination"})
    agg = agg.merge(dest_ext, on="destination", how="left")

    # National-level 피처
    for col, val in national_feat.items():
        agg[col] = val

    # 파생 피처
    eps = 1e-9
    agg["gdp_ratio"]         = agg["origin_real_gdp"] / (agg["dest_real_gdp"] + eps)
    agg["pop_ratio"]         = agg["origin_population"] / (agg["dest_population"] + eps)
    agg["income_ratio"]      = agg["origin_median_income"] / (agg["dest_median_income"] + eps)
    
    ref_tons = agg["route_tons_ctx"] if granularity == "commodity" else agg["tons_ctx"]
    agg["freight_intensity"] = ref_tons / (agg["origin_real_gdp"] + agg["dest_real_gdp"] + eps)

    agg["is_faf5_era"] = (agg["context_year"] >= 2017).astype(int)

    return agg



def build_train_features(train_df: pd.DataFrame, context_year: int, granularity: str = "route") -> pd.DataFrame:
    """
    train.csv (정제된 전체 기간) → route-level 또는 commodity-level 시계열 trend 피처.
    """
    # 전체 구간 선택
    tr = train_df.copy()

    # 집계 단위 설정
    group_cols = ["origin", "destination", "commodity"] if granularity == "commodity" else ["origin", "destination"]

    # route-level 또는 commodity-level 집계 (distance_band 무시하고 합산)
    tr_agg = (
        tr.groupby(group_cols + ['year'])[['tons', 'value']]
        .sum()
        .reset_index()
    )
    tr_agg['unit_price'] = tr_agg['value'] / (tr_agg['tons'] + 1e-5)

    result = None
    for col in ['tons', 'value', 'unit_price']:
        pivot = tr_agg.pivot_table(
            index=group_cols, columns='year', values=col
        )
        years = sorted(pivot.columns)

        if result is None:
            result = pivot.reset_index()[group_cols].copy()

        # 최근 2년 평균
        recent = years[-2:] if len(years) >= 2 else years
        result[f'{col}_recent_avg'] = pivot[recent].mean(axis=1).values

        # CAGR: (last/first)^(1/(n-1)) - 1  (최소 2년 필요)
        if len(years) >= 2:
            first = pivot[years[0]].replace(0, np.nan)
            last  = pivot[years[-1]].replace(0, np.nan)
            n     = len(years) - 1
            result[f'{col}_cagr'] = ((last / first) ** (1 / n) - 1).values
        else:
            result[f'{col}_cagr'] = 0.0

        # 연도간 표준편차 (경로 안정성)
        result[f'{col}_volatility'] = pivot[years].std(axis=1).values

        # 마지막 YoY 성장률
        if len(years) >= 2:
            prev = pivot[years[-2]].replace(0, np.nan)
            last = pivot[years[-1]]
            result[f'{col}_yoy_last'] = ((last - prev) / (prev.abs() + 1e-9)).values
        else:
            result[f'{col}_yoy_last'] = 0.0

        # 선형 trend slope (scipy 없이 numpy로, vectorized)
        if len(years) >= 2:
            yr_arr = np.array(years, dtype=float)
            yr_arr -= yr_arr.mean()  # centering
            denom  = (yr_arr ** 2).sum()
            y_vals = pivot[years].values.astype(float)
            result[f'{col}_trend_slope'] = (y_vals @ yr_arr) / denom
        else:
            result[f'{col}_trend_slope'] = 0.0

        # New growth features with dynamic context year reference
        def get_year_col(pv, yr):
            if yr in pv.columns:
                return pv[yr]
            else:
                return pd.Series(np.nan, index=pv.index)

        y0 = get_year_col(pivot, context_year)
        y1 = get_year_col(pivot, context_year - 1)
        y2 = get_year_col(pivot, context_year - 2)
        y3 = get_year_col(pivot, context_year - 3)

        result[f'{col}_yoy_lag1'] = ((y0 - y1) / (y1.abs() + 1e-9)).values
        result[f'{col}_yoy_lag2'] = ((y1 - y2) / (y2.abs() + 1e-9)).values
        result[f'{col}_cagr_3yr'] = (((y0 / (y3.replace(0, np.nan))) ** (1/3)) - 1.0).values

    # 결측치 처리 (경로가 특정 연도에 없는 경우)
    cols_to_fill = [c for c in result.columns if c not in group_cols]

    # fallback to commodity group median for growth features
    if granularity == "commodity" and "commodity" in result.columns:
        result["commodity_group"] = result["commodity"].map(COMMODITY_GROUPS)
        for col in cols_to_fill:
            if "yoy" in col or "cagr" in col:
                group_medians = result.groupby("commodity_group")[col].transform("median")
                result[col] = result[col].fillna(group_medians)
        result = result.drop(columns=["commodity_group"])

    for col in cols_to_fill:
        result[col] = result[col].fillna(0.0)
        # CAGR, YoY는 극단값 clip (-1 ~ 5: -100%~+500%)
        if 'cagr' in col or 'yoy' in col:
            result[col] = result[col].clip(-1.0, 5.0)

    return result


# ─────────────────────────────────────────────────────────────
# 지표 계산
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_ctx: np.ndarray = None, y_base: np.ndarray = None, y_cagr_base: np.ndarray = None) -> dict:
    """
    11개 회귀 지표 계산.
    - wRMSE : 모델 비교 1차 기준 (tons 비중 가중)
    - RMSE, MAE, WMAPE, RMSLE, R² : 해석·보고용
    - Skill_Score_Naive, Skill_Score_YoY_Median, Skill_Score_Constant_CAGR, Directional_Accuracy, CAGR_Error, Growth_WMAPE : 성능 지표
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_c = np.clip(y_true, 0, None)
    y_pred_c = np.clip(y_pred, 0, None)

    # Weighted RMSE: 실제 tons 비중을 가중치로 사용
    weights = y_true_c / (y_true_c.sum() + 1e-9)
    wrmse   = float(np.sqrt(np.sum(weights * (y_true - y_pred) ** 2)))
    
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    wmape = float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-9))
    rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_c) - np.log1p(y_true_c)) ** 2)))
    r2    = float(r2_score(y_true, y_pred))

    metrics = {
        "wRMSE": wrmse,
        "RMSE":  rmse,
        "MAE":   mae,
        "WMAPE": wmape,
        "RMSLE": rmsle,
        "R2":    r2,
    }

    # Skill Score (vs. Naive/Ctx)
    if y_ctx is not None:
        y_ctx = np.asarray(y_ctx, dtype=float)
        wrmse_naive = np.sqrt(np.sum(weights * (y_true - y_ctx) ** 2))
        metrics["Skill_Score_Naive"] = 1.0 - wrmse / (wrmse_naive + 1e-9)

        # Directional Accuracy
        act_diff = y_true - y_ctx
        pred_diff = y_pred - y_ctx
        act_sign = np.sign(act_diff)
        pred_sign = np.sign(pred_diff)
        metrics["Directional_Accuracy"] = float(np.mean(act_sign == pred_sign))

        # CAGR Error: 3년 앞선 예측이므로 3개년 CAGR
        act_cagr = (y_true / (y_ctx + 1e-5)) ** (1/3) - 1.0
        pred_cagr = (y_pred / (y_ctx + 1e-5)) ** (1/3) - 1.0
        act_cagr = np.clip(act_cagr, -1.0, 5.0)
        pred_cagr = np.clip(pred_cagr, -1.0, 5.0)
        metrics["CAGR_Error"] = float(np.sqrt(np.mean((act_cagr - pred_cagr) ** 2)))

        # Growth WMAPE: weighted by actual tons
        act_growth = act_diff / (y_ctx + 1e-5)
        pred_growth = pred_diff / (y_ctx + 1e-5)
        growth_err = np.abs(pred_growth - act_growth)
        growth_act = np.abs(act_growth)
        metrics["Growth_WMAPE"] = float(np.sum(y_true_c * growth_err) / (np.sum(y_true_c * growth_act) + 1e-9))
    else:
        metrics["Skill_Score_Naive"] = 0.0
        metrics["Directional_Accuracy"] = 0.0
        metrics["CAGR_Error"] = 0.0
        metrics["Growth_WMAPE"] = 0.0

    # Skill Score (vs. YoY Median Baseline)
    if y_base is not None:
        y_base = np.asarray(y_base, dtype=float)
        wrmse_base = np.sqrt(np.sum(weights * (y_true - y_base) ** 2))
        metrics["Skill_Score_YoY_Median"] = 1.0 - wrmse / (wrmse_base + 1e-9)
    else:
        metrics["Skill_Score_YoY_Median"] = 0.0

    # Skill Score (vs. Constant CAGR Baseline)
    if y_cagr_base is not None:
        y_cagr_base = np.asarray(y_cagr_base, dtype=float)
        wrmse_cagr_base = np.sqrt(np.sum(weights * (y_true - y_cagr_base) ** 2))
        metrics["Skill_Score_Constant_CAGR"] = 1.0 - wrmse / (wrmse_cagr_base + 1e-9)
    else:
        metrics["Skill_Score_Constant_CAGR"] = 0.0

    return metrics


# ─────────────────────────────────────────────────────────────
# 데이터 로드 및 빌드 헬퍼 (엄격한 3-year horizon 적용)
# ─────────────────────────────────────────────────────────────
_RAW_DF = pd.read_csv("data/faf4_faf5_cleaned.csv")
_DATA_CACHE = {}


def build_data_for_target_year(target_year: int, encoders: dict, granularity: str = "route") -> pd.DataFrame:
    cache_key = (target_year, granularity)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()

    base_year = target_year - 3
    
    group_cols = ["origin", "destination", "commodity"] if granularity == "commodity" else ["origin", "destination"]
    
    # Target
    target_df = _RAW_DF[_RAW_DF['year'] == target_year].groupby(group_cols)['tons'].sum().reset_index()
    
    # Context DF for features
    ctx_df = _RAW_DF[_RAW_DF['year'] == base_year].copy()
    
    # External features
    state_feat, national_feat = load_all_external(EXTERNAL_DIR, base_year)
    feat_df = build_route_features(ctx_df, state_feat, national_feat, granularity)
    
    # Time-series features
    train_historical = _RAW_DF[_RAW_DF['year'] <= base_year].copy()
    ts_feat = build_train_features(train_historical, base_year, granularity)
    
    if granularity == "commodity":
        # Build route-level time-series features and merge
        ts_route_feat = build_train_features(train_historical, base_year, "route")
        rename_dict = {col: f"route_{col}" for col in ts_route_feat.columns if col not in ["origin", "destination"]}
        ts_route_feat = ts_route_feat.rename(columns=rename_dict)
        ts_feat = ts_feat.merge(ts_route_feat, on=["origin", "destination"], how="left")
        
    df = feat_df.merge(ts_feat, on=group_cols, how="left")
    df = df.merge(target_df, on=group_cols, how="left")
    df["tons"] = df["tons"].fillna(0.0)
    
    # Baseline prediction (commodity_median_allhistory_none)
    # Median tons over a 3-year window: [base_year - 2, base_year]
    input_years = [base_year - 2, base_year - 1, base_year]
    hist = _RAW_DF[_RAW_DF['year'].isin(input_years)].copy()
    
    # Group commodity-level first to get median
    grouped_hist = hist.groupby(["origin", "destination", "commodity", "year"])["tons"].sum().reset_index()
    med_comm = grouped_hist.groupby(["origin", "destination", "commodity"])["tons"].median().reset_index(name="base_pred")
    
    if granularity == "commodity":
        df = df.merge(med_comm, on=["origin", "destination", "commodity"], how="left")
    else:
        # If route level, baseline prediction is the sum of commodity-level medians
        med_route = med_comm.groupby(["origin", "destination"])["base_pred"].sum().reset_index()
        df = df.merge(med_route, on=["origin", "destination"], how="left")
        
    df["base_pred"] = df["base_pred"].fillna(0.0)
    
    # Constant CAGR Baseline Prediction: tons_ctx * (1 + tons_cagr) ** 3
    df["cagr_pred"] = df["tons_ctx"] * (1.0 + df["tons_cagr"].fillna(0.0)) ** 3
    df["cagr_pred"] = df["cagr_pred"].fillna(0.0).clip(0, None)
    
    # Encode categorical columns
    df["origin_enc"] = encode_col(df["origin"], encoders["origin"])
    df["destination_enc"] = encode_col(df["destination"], encoders["destination"])
    if granularity == "commodity":
        df["commodity_enc"] = encode_col(df["commodity"], encoders["commodity"])
        df["commodity_group_enc"] = encode_col(df["commodity"].map(COMMODITY_GROUPS), encoders["commodity_group"])
        
    _DATA_CACHE[cache_key] = df
    return df.copy()


def evaluate_alpha_on_df_subset(
    vdf: pd.DataFrame,
    alpha: float,
    target_type: str,
    is_hybrid: bool,
    granularity: str,
    group_name: str = None,
) -> float:
    """Computes route-level wRMSE of blended predictions on a validation dataframe for a given alpha and group."""
    df = vdf.copy()
    if group_name is not None and granularity == "commodity":
        if "commodity_group" not in df.columns and "commodity" in df.columns:
            df["commodity_group"] = df["commodity"].map(COMMODITY_GROUPS)
        df = df[df["commodity_group"] == group_name].copy()
        if df.empty:
            return 0.0
            
    if is_hybrid:
        if target_type == "residual":
            blended_diff = np.where(df["is_shock"], alpha * df["raw_val"], 0.0)
            df["val_pred"] = df["base_pred"] + blended_diff
        elif target_type == "ratio":
            ml_val = np.expm1(np.log1p(df["base_pred"]) + df["raw_val"])
            blended_val = alpha * ml_val + (1 - alpha) * df["base_pred"]
            df["val_pred"] = np.where(df["is_shock"], blended_val, df["base_pred"])
        elif target_type == "raw_ratio":
            ml_val = df["base_pred"] * df["raw_val"]
            blended_val = alpha * ml_val + (1 - alpha) * df["base_pred"]
            df["val_pred"] = np.where(df["is_shock"], blended_val, df["base_pred"])
    else:
        if target_type == "residual":
            df["val_pred"] = df["base_pred"] + alpha * df["raw_val"]
        elif target_type == "ratio":
            ml_val = np.expm1(np.log1p(df["base_pred"]) + df["raw_val"])
            df["val_pred"] = alpha * ml_val + (1 - alpha) * df["base_pred"]
        elif target_type == "raw_ratio":
            ml_val = df["base_pred"] * df["raw_val"]
            df["val_pred"] = alpha * ml_val + (1 - alpha) * df["base_pred"]
            
    df["val_pred"] = df["val_pred"].clip(0, None)
    
    if granularity == "commodity":
        val_agg = df.groupby(["origin", "destination"])[["tons", "val_pred"]].sum().reset_index()
        y_val_eval = val_agg["tons"].values
        pred_eval = val_agg["val_pred"].values
    else:
        y_val_eval = df["tons"].values
        pred_eval = df["val_pred"].values
        
    y_val_eval_c = np.clip(y_val_eval, 0, None)
    weights = y_val_eval_c / (y_val_eval_c.sum() + 1e-9)
    wrmse = np.sqrt(np.sum(weights * (y_val_eval - pred_eval) ** 2))
    return wrmse


def get_group_base_wrmse(vdf: pd.DataFrame, granularity: str, group_name: str = None) -> float:
    df = vdf.copy()
    if group_name is not None and granularity == "commodity":
        if "commodity_group" not in df.columns and "commodity" in df.columns:
            df["commodity_group"] = df["commodity"].map(COMMODITY_GROUPS)
        df = df[df["commodity_group"] == group_name].copy()
        if df.empty:
            return 0.0
            
    if granularity == "commodity":
        val_agg = df.groupby(["origin", "destination"])[["tons", "base_pred"]].sum().reset_index()
        y_val_eval = val_agg["tons"].values
        base_eval = val_agg["base_pred"].values
    else:
        y_val_eval = df["tons"].values
        base_eval = df["base_pred"].values
        
    y_val_eval_c = np.clip(y_val_eval, 0, None)
    weights = y_val_eval_c / (y_val_eval_c.sum() + 1e-9)
    wrmse = np.sqrt(np.sum(weights * (y_val_eval - base_eval) ** 2))
    return wrmse


def run_internal_train_backtest(
    start_train_year: int,
    max_train_year: int,
    model_cls,
    model_params: dict,
    encoders: dict,
    target_type: str,
    granularity: str,
    is_hybrid: bool,
    alphas: np.ndarray,
    seed: int,
    scenario: str,
) -> dict[str, dict[float, list[float]]]:
    """Runs walk-forward validation on past target years inside the training set to evaluate alpha candidates by group."""
    groups = sorted(list(set(COMMODITY_GROUPS.values()))) if granularity == "commodity" else ["global"]
    alpha_skill_scores = {g: {alpha: [] for alpha in alphas} for g in groups}
    
    # We need at least 2 years in training to run a backtest fold
    if max_train_year - start_train_year < 1:
        return alpha_skill_scores
        
    feature_cols = get_feature_cols(granularity)
    
    # Run walk-forward validation inside training set
    for y_val_pseudo in range(start_train_year + 1, max_train_year + 1):
        # 1. Build pseudo-train and pseudo-val data
        pseudo_train_years = list(range(start_train_year, y_val_pseudo))
        
        # Build pseudo-train
        pseudo_train_df = pd.concat([
            build_data_for_target_year(y, encoders, granularity)
            for y in pseudo_train_years
        ])
        
        # Build pseudo-val
        pseudo_val_df = build_data_for_target_year(y_val_pseudo, encoders, granularity)
        
        # Target Encoding if commodity
        if granularity == "commodity":
            for df_tmp in [pseudo_train_df, pseudo_val_df]:
                if "commodity_group" not in df_tmp.columns:
                    df_tmp["commodity_group"] = df_tmp["commodity"].map(COMMODITY_GROUPS)
            
            for col_name, prefix in [("commodity", "comm"), ("commodity_group", "comm_group")]:
                te_mean = fit_target_encoding(pseudo_train_df, col_name, "tons", stat="mean", alpha=20.0)
                pseudo_train_df[f"{prefix}_te_mean"] = transform_target_encoding(pseudo_train_df, col_name, te_mean)
                pseudo_val_df[f"{prefix}_te_mean"] = transform_target_encoding(pseudo_val_df, col_name, te_mean)
                
                te_std = fit_target_encoding(pseudo_train_df, col_name, "tons", stat="std", alpha=20.0)
                pseudo_train_df[f"{prefix}_te_std"] = transform_target_encoding(pseudo_train_df, col_name, te_std)
                pseudo_val_df[f"{prefix}_te_std"] = transform_target_encoding(pseudo_val_df, col_name, te_std)
        else:
            for df_tmp in [pseudo_train_df, pseudo_val_df]:
                for c in TARGET_ENCODING_COLS:
                    df_tmp[c] = 0.0

        X_train = pseudo_train_df[feature_cols].copy()
        y_train = pseudo_train_df["tons"].values
        base_train = pseudo_train_df["base_pred"].values
        
        X_val = pseudo_val_df[feature_cols].copy()
        base_val = pseudo_val_df["base_pred"].values
        
        # Impute
        imp = SimpleImputer(strategy="median").set_output(transform="pandas")
        X_train_imp = imp.fit_transform(X_train)
        X_val_imp   = imp.transform(X_val)
        
        is_tree_model = "LGBM" in model_cls.__name__ or "Forest" in model_cls.__name__ or "Hurdle" in model_cls.__name__
        cat_cols = ["origin_enc", "destination_enc"]
        if granularity == "commodity":
            cat_cols.append("commodity_enc")
            cat_cols.append("commodity_group_enc")

        if not is_tree_model:
            X_train_imp = pd.get_dummies(X_train_imp, columns=cat_cols, drop_first=True)
            X_val_imp   = pd.get_dummies(X_val_imp, columns=cat_cols, drop_first=True)
            X_train_imp, X_val_imp = X_train_imp.align(X_val_imp, join='left', axis=1, fill_value=0)
            
        # Target transform
        if target_type == "direct":
            y_fit = y_train
        elif target_type == "residual":
            y_fit = y_train - base_train
        elif target_type == "ratio":
            y_fit = np.log1p(y_train) - np.log1p(base_train)
        elif target_type == "raw_ratio":
            y_fit = np.where(base_train < 0.1, 1.0, y_train / (base_train + 1e-9))
            y_fit = np.clip(y_fit, 0.0, 5.0)
            
        # Train model
        local_params = model_params.copy()
        if target_type == "ratio" and local_params.get("objective") == "tweedie":
            local_params["objective"] = "regression"
            if "tweedie_variance_power" in local_params:
                del local_params["tweedie_variance_power"]
                
        model = model_cls(**local_params)
        fit_params = {}
        if "LGBM" in model.__class__.__name__ or "Hurdle" in model.__class__.__name__:
            fit_params["categorical_feature"] = cat_cols
            
        sw_train = pseudo_train_df["tons"].values
        sw_train = np.clip(sw_train, 1e-3, None)
        model.fit(X_train_imp, y_fit, sample_weight=sw_train, **fit_params)
            
        raw_val = model.predict(X_val_imp)
        pseudo_val_df["raw_val"] = raw_val
        
        # Detect shock routes for pseudo val
        pseudo_val_base_year = y_val_pseudo - 3
        pseudo_val_shock_routes = detect_shock_routes(_RAW_DF, base_year=pseudo_val_base_year, threshold_pct=0.15)
        pseudo_val_df["is_shock"] = pseudo_val_df.apply(
            lambda r: (r["origin"], r["destination"]) in pseudo_val_shock_routes, axis=1
        )
        
        # Evaluate each alpha by group
        for alpha in alphas:
            for g in groups:
                wrmse_model = evaluate_alpha_on_df_subset(pseudo_val_df, alpha, target_type, is_hybrid, granularity, group_name=g if granularity == "commodity" else None)
                wrmse_base = get_group_base_wrmse(pseudo_val_df, granularity, group_name=g if granularity == "commodity" else None)
                
                if wrmse_base <= 1e-5:
                    skill_score = 0.0
                else:
                    skill_score = 1.0 - wrmse_model / (wrmse_base + 1e-9)
                alpha_skill_scores[g][alpha].append(skill_score)
    return alpha_skill_scores


def evaluate_all_splits(
    model_cls,
    model_params: dict,
    encoders: dict[str, LabelEncoder],
    seed: int,
    target_type: str = "direct",
    granularity: str = "route",
    model_name: str = "",
    scenario: str = "A",
) -> tuple[dict, dict, pd.DataFrame]:
    all_metrics: dict[str, float] = {}
    train_year_ranges: dict[str, str] = {}
    test_pred_dfs = []
    past_splits_info = []
    feature_cols = get_feature_cols(granularity)
    is_hybrid = "Hybrid" in model_name

    for split_name, sdef in SPLIT_DEFS.items():
        test_year = sdef["test_year"]
        val_year = test_year - 3
        if scenario == "B" and val_year < 2020:
            print(f"  [Warning] {split_name}: Scenario B has no honest validation year (earliest FAF5 context is 2017). Falling back to val_year = 2020.")
            val_year = 2020
        
        # 엄격한 3-year horizon 적용 및 시나리오별 학습 시작 연도 제어
        start_train_year = 2015 if scenario == "A" else 2020
        
        # Purging & Embargo 적용
        val_feature_year = val_year - 3
        embargo_buffer = 1
        max_train_year = val_feature_year - 1 - embargo_buffer
        
        if max_train_year < start_train_year:
            print(f"  [Warning] {split_name}: Lack of historical data. Purging/Embargo relaxed to train on target year {start_train_year}.")
            max_train_year = start_train_year
            
        train_year_ranges[split_name] = f"{start_train_year}-{max_train_year}"

        train_df = pd.concat([
            build_data_for_target_year(y, encoders, granularity) 
            for y in range(start_train_year, max_train_year + 1)
        ])
        val_df = build_data_for_target_year(val_year, encoders, granularity)
        test_df = build_data_for_target_year(test_year, encoders, granularity)
        
        # Target Encoding (Suggestion 5)
        if granularity == "commodity":
            for df_tmp in [train_df, val_df, test_df]:
                if "commodity_group" not in df_tmp.columns:
                    df_tmp["commodity_group"] = df_tmp["commodity"].map(COMMODITY_GROUPS)
            
            for col_name, prefix in [("commodity", "comm"), ("commodity_group", "comm_group")]:
                # Mean Target Encoding
                te_mean = fit_target_encoding(train_df, col_name, "tons", stat="mean", alpha=20.0)
                train_df[f"{prefix}_te_mean"] = transform_target_encoding(train_df, col_name, te_mean)
                val_df[f"{prefix}_te_mean"] = transform_target_encoding(val_df, col_name, te_mean)
                test_df[f"{prefix}_te_mean"] = transform_target_encoding(test_df, col_name, te_mean)
                
                # Std Target Encoding
                te_std = fit_target_encoding(train_df, col_name, "tons", stat="std", alpha=20.0)
                train_df[f"{prefix}_te_std"] = transform_target_encoding(train_df, col_name, te_std)
                val_df[f"{prefix}_te_std"] = transform_target_encoding(val_df, col_name, te_std)
                test_df[f"{prefix}_te_std"] = transform_target_encoding(test_df, col_name, te_std)
        else:
            for df_tmp in [train_df, val_df, test_df]:
                for c in TARGET_ENCODING_COLS:
                    df_tmp[c] = 0.0

        X_train = train_df[feature_cols].copy()
        y_train = train_df["tons"].values
        base_train = train_df["base_pred"].values
        
        X_val = val_df[feature_cols].copy()
        y_val = val_df["tons"].values
        base_val = val_df["base_pred"].values
        
        X_test  = test_df[feature_cols].copy()
        y_test  = test_df["tons"].values
        base_test = test_df["base_pred"].values

        # 결측치 보정
        imp = SimpleImputer(strategy="median").set_output(transform="pandas")
        X_train_imp = imp.fit_transform(X_train)
        X_val_imp   = imp.transform(X_val)
        X_test_imp  = imp.transform(X_test)

        # 타 모델 예외 처리
        is_tree_model = "LGBM" in model_cls.__name__ or "Forest" in model_cls.__name__ or "Hurdle" in model_cls.__name__
        cat_cols = ["origin_enc", "destination_enc"]
        if granularity == "commodity":
            cat_cols.append("commodity_enc")
            cat_cols.append("commodity_group_enc")

        if not is_tree_model:
            X_train_imp = pd.get_dummies(X_train_imp, columns=cat_cols, drop_first=True)
            X_val_imp   = pd.get_dummies(X_val_imp, columns=cat_cols, drop_first=True)
            X_test_imp  = pd.get_dummies(X_test_imp, columns=cat_cols, drop_first=True)
            X_train_imp, X_val_imp = X_train_imp.align(X_val_imp, join='left', axis=1, fill_value=0)
            X_train_imp, X_test_imp = X_train_imp.align(X_test_imp, join='left', axis=1, fill_value=0)

        # Target 설정 (direct, residual, ratio, raw_ratio)
        if target_type == "direct":
            y_fit = y_train
        elif target_type == "residual":
            y_fit = y_train - base_train
        elif target_type == "ratio":
            y_fit = np.log1p(y_train) - np.log1p(base_train)
        elif target_type == "raw_ratio":
            y_fit = np.where(base_train < 0.1, 1.0, y_train / (base_train + 1e-9))
            y_fit = np.clip(y_fit, 0.0, 5.0)
        else:
            raise ValueError("Invalid target_type")
            
        # 모델 학습 (Tons 가중치 적용)
        local_params = model_params.copy()
        if target_type == "ratio" and local_params.get("objective") == "tweedie":
            local_params["objective"] = "regression"
            if "tweedie_variance_power" in local_params:
                del local_params["tweedie_variance_power"]
                
        model = model_cls(**local_params)
        
        fit_params = {}
        if "LGBM" in model.__class__.__name__ or "Hurdle" in model.__class__.__name__:
            fit_params["categorical_feature"] = cat_cols
            
        sw_train = train_df["tons"].values
        sw_train = np.clip(sw_train, 1e-3, None)
        model.fit(X_train_imp, y_fit, sample_weight=sw_train, **fit_params)

        # 예측 및 검증(Val) 세트를 통한 Alpha 최적화 (direct 방식은 alpha 없이 그대로 사용)
        raw_val = model.predict(X_val_imp)
        raw_test = model.predict(X_test_imp)
        
        val_df["raw_val"] = raw_val
        test_df["raw_test"] = raw_test
        
        # 변동성 노선 탐지 (Base Year 시점의 누적된 과거 데이터를 기준으로 구함)
        val_base_year = val_year - 3
        test_base_year = test_year - 3
        val_shock_routes = detect_shock_routes(_RAW_DF, base_year=val_base_year, threshold_pct=0.15)
        test_shock_routes = detect_shock_routes(_RAW_DF, base_year=test_base_year, threshold_pct=0.15)

        val_df["is_shock"] = val_df.apply(lambda r: (r["origin"], r["destination"]) in val_shock_routes, axis=1)
        test_df["is_shock"] = test_df.apply(lambda r: (r["origin"], r["destination"]) in test_shock_routes, axis=1)

        if target_type == "direct":
            # direct인 경우에는 alpha 최적화 과정이 없으므로, 하이브리드가 지정되어 있어도 개별 노선별 라우팅을 수행함
            if is_hybrid:
                # shock 노선은 ML 예측값, stable 노선은 Baseline 사용
                val_df["val_pred"] = np.where(val_df["is_shock"], raw_val, val_df["base_pred"])
                test_df["final_pred"] = np.where(test_df["is_shock"], raw_test, test_df["base_pred"])
            else:
                val_df["val_pred"] = raw_val
                test_df["final_pred"] = raw_test
            
            test_df["final_pred"] = test_df["final_pred"].clip(0, None)
            if granularity == "commodity":
                test_agg = test_df.groupby(["origin", "destination"])[["tons", "final_pred", "tons_ctx", "base_pred"]].sum().reset_index()
                y_test_eval = test_agg["tons"].values
                final_pred = test_agg["final_pred"].values
                y_ctx_eval = test_agg["tons_ctx"].values
                y_base_eval = test_agg["base_pred"].values
            else:
                final_pred = test_df["final_pred"].values
                y_test_eval = y_test
                y_ctx_eval = test_df["tons_ctx"].values
                y_base_eval = test_df["base_pred"].values
        else:
            # residual or ratio 블렌딩 모델
            alphas = np.arange(0.0, 1.51, 0.05) if target_type == "residual" else np.arange(0.0, 2.01, 0.05)
            
            # Setup list of groups
            groups = sorted(list(set(COMMODITY_GROUPS.values()))) if granularity == "commodity" else ["global"]
            best_alphas_dict = {}
            
            # 1. Run internal train rolling backtests to get fold-level skill scores for each alpha by group
            train_alpha_skill_scores = {}
            if scenario == "A" and (max_train_year - start_train_year >= 1):
                train_alpha_skill_scores = run_internal_train_backtest(
                    start_train_year=start_train_year,
                    max_train_year=max_train_year,
                    model_cls=model_cls,
                    model_params=model_params,
                    encoders=encoders,
                    target_type=target_type,
                    granularity=granularity,
                    is_hybrid=is_hybrid,
                    alphas=alphas,
                    seed=seed,
                    scenario=scenario,
                )
            else:
                train_alpha_skill_scores = {g: {alpha: [] for alpha in alphas} for g in groups}
                
            # 2. For each alpha and each group, collect all historical folds (train backtests + past val splits)
            for g in groups:
                candidate_metrics = {}
                for alpha in alphas:
                    fold_skill_scores = list(train_alpha_skill_scores.get(g, {}).get(alpha, []))
                    
                    # Add past splits validation skill scores for this group
                    for past_sn, past_vdf in past_splits_info:
                        wrmse_model = evaluate_alpha_on_df_subset(past_vdf, alpha, target_type, is_hybrid, granularity, group_name=g if granularity == "commodity" else None)
                        wrmse_base = get_group_base_wrmse(past_vdf, granularity, group_name=g if granularity == "commodity" else None)
                        
                        if wrmse_base <= 1e-5:
                            skill_score = 0.0
                        else:
                            skill_score = 1.0 - wrmse_model / (wrmse_base + 1e-9)
                        fold_skill_scores.append(skill_score)
                        
                    candidate_metrics[alpha] = fold_skill_scores
                    
                # 3. Filter candidates based on historical performance (consistency check) for this group
                has_history = any(len(scores) > 0 for scores in candidate_metrics.values())
                
                filtered_alphas = []
                if has_history:
                    print(f"\n    [Group Robust Selection] Split={split_name} | Group={g} | Historical Folds:")
                    for alpha in alphas:
                        scores = candidate_metrics[alpha]
                        avg_skill = np.mean(scores)
                        min_skill = np.min(scores)
                        
                        # Print stats for a subset of alphas to keep logs clean
                        if abs(alpha * 10 % 2) < 1e-5 or alpha == 0.0 or alpha == 1.0:
                            print(f"      alpha={alpha:.2f} | Folds={len(scores)} | Avg Skill={avg_skill:+.4f} | Min Skill={min_skill:+.4f}")
                            
                        if avg_skill > 0.0 and min_skill >= -0.05:
                            filtered_alphas.append(alpha)
                            
                    print(f"    [Group Robust Selection] Filtered Alphas for {g}: {[f'{a:.2f}' for a in filtered_alphas]}")
                else:
                    filtered_alphas = list(alphas)
                    print(f"    [Group Robust Selection] Split={split_name} | Group={g} | No history, using all candidates.")
                    
                # 4. Final selection: from filtered candidates, find the one that minimizes current split's val wRMSE for this group.
                if not filtered_alphas:
                    print(f"    [Group Robust Selection] Group={g} | No candidate passed. Falling back to alpha = 0.0.")
                    best_alpha_g = 0.0
                else:
                    best_val_wrmse = float('inf')
                    best_alpha_g = 0.0
                    
                    # Build val_df with raw_val and is_shock
                    val_df["raw_val"] = raw_val
                    val_df["is_shock"] = val_df.apply(lambda r: (r["origin"], r["destination"]) in val_shock_routes, axis=1)
                    
                    for alpha in filtered_alphas:
                        wrmse_val = evaluate_alpha_on_df_subset(val_df, alpha, target_type, is_hybrid, granularity, group_name=g if granularity == "commodity" else None)
                        if wrmse_val < best_val_wrmse:
                            best_val_wrmse = wrmse_val
                            best_alpha_g = alpha
                            
                    print(f"    [Group Robust Selection] Group={g} | Best Alpha: {best_alpha_g:.2f} (Val wRMSE = {best_val_wrmse:,.2f})")
                
                best_alphas_dict[g] = best_alpha_g

            # Build prediction row weights using mapped alphas
            val_df["raw_val"] = raw_val
            val_df["is_shock"] = val_df.apply(lambda r: (r["origin"], r["destination"]) in val_shock_routes, axis=1)
            
            test_df["raw_test"] = raw_test
            test_df["is_shock"] = test_df.apply(lambda r: (r["origin"], r["destination"]) in test_shock_routes, axis=1)
            
            if granularity == "commodity":
                val_row_alphas = val_df["commodity_group"].map(best_alphas_dict).fillna(0.0).values
                test_row_alphas = test_df["commodity_group"].map(best_alphas_dict).fillna(0.0).values
            else:
                val_row_alphas = np.full(len(val_df), best_alphas_dict["global"])
                test_row_alphas = np.full(len(test_df), best_alphas_dict["global"])

            # Recompute val_df["val_pred"] using selected group alphas
            if is_hybrid:
                if target_type == "residual":
                    blended_diff = np.where(val_df["is_shock"], val_row_alphas * val_df["raw_val"], 0.0)
                    val_df["val_pred"] = val_df["base_pred"] + blended_diff
                elif target_type == "ratio":
                    ml_val = np.expm1(np.log1p(val_df["base_pred"]) + val_df["raw_val"])
                    blended_val = val_row_alphas * ml_val + (1 - val_row_alphas) * val_df["base_pred"]
                    val_df["val_pred"] = np.where(val_df["is_shock"], blended_val, val_df["base_pred"])
                elif target_type == "raw_ratio":
                    ml_val = val_df["base_pred"] * val_df["raw_val"]
                    blended_val = val_row_alphas * ml_val + (1 - val_row_alphas) * val_df["base_pred"]
                    val_df["val_pred"] = np.where(val_df["is_shock"], blended_val, val_df["base_pred"])
            else:
                if target_type == "residual":
                    val_df["val_pred"] = val_df["base_pred"] + val_row_alphas * val_df["raw_val"]
                elif target_type == "ratio":
                    ml_val = np.expm1(np.log1p(val_df["base_pred"]) + val_df["raw_val"])
                    val_df["val_pred"] = val_row_alphas * ml_val + (1 - val_row_alphas) * val_df["base_pred"]
                elif target_type == "raw_ratio":
                    ml_val = val_df["base_pred"] * val_df["raw_val"]
                    val_df["val_pred"] = val_row_alphas * ml_val + (1 - val_row_alphas) * val_df["base_pred"]
            val_df["val_pred"] = val_df["val_pred"].clip(0, None)
            
            # Predict test_df["final_pred"] using selected group alphas
            if is_hybrid:
                if target_type == "residual":
                    blended_diff_test = np.where(test_df["is_shock"], test_row_alphas * test_df["raw_test"], 0.0)
                    test_df["final_pred"] = test_df["base_pred"] + blended_diff_test
                elif target_type == "ratio":
                    ml_test = np.expm1(np.log1p(test_df["base_pred"]) + test_df["raw_test"])
                    blended_test = test_row_alphas * ml_test + (1 - test_row_alphas) * test_df["base_pred"]
                    test_df["final_pred"] = np.where(test_df["is_shock"], blended_test, test_df["base_pred"])
                elif target_type == "raw_ratio":
                    ml_test = test_df["base_pred"] * test_df["raw_test"]
                    blended_test = test_row_alphas * ml_test + (1 - test_row_alphas) * test_df["base_pred"]
                    test_df["final_pred"] = np.where(test_df["is_shock"], blended_test, test_df["base_pred"])
            else:
                if target_type == "residual":
                    test_df["final_pred"] = test_df["base_pred"] + test_row_alphas * test_df["raw_test"]
                elif target_type == "ratio":
                    ml_test = np.expm1(np.log1p(test_df["base_pred"]) + test_df["raw_test"])
                    test_df["final_pred"] = test_row_alphas * ml_test + (1 - test_row_alphas) * test_df["base_pred"]
                elif target_type == "raw_ratio":
                    ml_test = test_df["base_pred"] * test_df["raw_test"]
                    test_df["final_pred"] = test_row_alphas * ml_test + (1 - test_row_alphas) * test_df["base_pred"]
            test_df["final_pred"] = test_df["final_pred"].clip(0, None)
            
            # Setup evaluations
            if granularity == "commodity":
                test_agg = test_df.groupby(["origin", "destination"])[["tons", "final_pred", "tons_ctx", "base_pred", "cagr_pred"]].sum().reset_index()
                y_test_eval = test_agg["tons"].values
                final_pred = test_agg["final_pred"].values
                y_ctx_eval = test_agg["tons_ctx"].values
                y_base_eval = test_agg["base_pred"].values
                y_cagr_base_eval = test_agg["cagr_pred"].values
            else:
                y_test_eval = y_test
                final_pred = test_df["final_pred"].values
                y_ctx_eval = test_df["tons_ctx"].values
                y_base_eval = test_df["base_pred"].values
                y_cagr_base_eval = test_df["cagr_pred"].values

        final_pred = np.clip(final_pred, 0, None)

        # ────────────── Validation Evaluation ──────────────
        if granularity == "commodity":
            val_agg = val_df.groupby(["origin", "destination"])[["tons", "val_pred", "tons_ctx", "base_pred", "cagr_pred"]].sum().reset_index()
            y_val_eval = val_agg["tons"].values
            val_pred_eval = val_agg["val_pred"].values
            y_val_ctx_eval = val_agg["tons_ctx"].values
            y_val_base_eval = val_agg["base_pred"].values
            y_val_cagr_base_eval = val_agg["cagr_pred"].values
        else:
            y_val_eval = y_val
            val_pred_eval = val_df["val_pred"].values
            y_val_ctx_eval = val_df["tons_ctx"].values
            y_val_base_eval = val_df["base_pred"].values
            y_val_cagr_base_eval = val_df["cagr_pred"].values

        # ────────────── 세그먼트별 평가 지표 계산 ──────────────
        # 경로 레벨에서 shock_mask 획득
        if granularity == "commodity":
            test_agg["is_shock"] = test_agg.apply(lambda r: (r["origin"], r["destination"]) in test_shock_routes, axis=1)
            shock_mask = test_agg["is_shock"].values
        else:
            test_df["is_shock"] = test_df.apply(lambda r: (r["origin"], r["destination"]) in test_shock_routes, axis=1)
            shock_mask = test_df["is_shock"].values
            
        stable_mask = ~shock_mask
        
        # 1. 전체 경로 (Test & Val)
        m_val = compute_metrics(y_val_eval, val_pred_eval, y_val_ctx_eval, y_val_base_eval, y_val_cagr_base_eval)
        m_all = compute_metrics(y_test_eval, final_pred, y_ctx_eval, y_base_eval, y_cagr_base_eval)

        # Map to precise keys as required by guide (wRMSE mapped to RMSE, R2 mapped to R2_Score)
        s_num = split_name.replace("_", "")  # "split1", "split2", "split3"
        
        # Test split metrics
        all_metrics[f"{s_num}_RMSE"] = m_all["RMSE"]
        all_metrics[f"{s_num}_MAE"] = m_all["MAE"]
        all_metrics[f"{s_num}_WMAPE"] = m_all["WMAPE"]
        all_metrics[f"{s_num}_RMSLE"] = m_all["RMSLE"]
        all_metrics[f"{s_num}_R2_Score"] = m_all["R2"]
        all_metrics[f"{s_num}_Skill_Score_Naive"] = m_all["Skill_Score_Naive"]
        all_metrics[f"{s_num}_Skill_Score_YoY_Median"] = m_all["Skill_Score_YoY_Median"]
        all_metrics[f"{s_num}_Skill_Score_Constant_CAGR"] = m_all["Skill_Score_Constant_CAGR"]
        all_metrics[f"{s_num}_Directional_Accuracy"] = m_all["Directional_Accuracy"]
        all_metrics[f"{s_num}_CAGR_Error"] = m_all["CAGR_Error"]
        all_metrics[f"{s_num}_Growth_WMAPE"] = m_all["Growth_WMAPE"]
        
        # Validation split metrics
        all_metrics[f"val_{s_num}_RMSE"] = m_val["RMSE"]
        all_metrics[f"val_{s_num}_MAE"] = m_val["MAE"]
        all_metrics[f"val_{s_num}_WMAPE"] = m_val["WMAPE"]
        all_metrics[f"val_{s_num}_RMSLE"] = m_val["RMSLE"]
        all_metrics[f"val_{s_num}_R2_Score"] = m_val["R2"]
        all_metrics[f"val_{s_num}_Skill_Score_Naive"] = m_val["Skill_Score_Naive"]
        all_metrics[f"val_{s_num}_Skill_Score_YoY_Median"] = m_val["Skill_Score_YoY_Median"]
        all_metrics[f"val_{s_num}_Skill_Score_Constant_CAGR"] = m_val["Skill_Score_Constant_CAGR"]
        all_metrics[f"val_{s_num}_Directional_Accuracy"] = m_val["Directional_Accuracy"]
        all_metrics[f"val_{s_num}_CAGR_Error"] = m_val["CAGR_Error"]
        all_metrics[f"val_{s_num}_Growth_WMAPE"] = m_val["Growth_WMAPE"]

        # Collect predictions for CSV artifact
        if granularity == "commodity":
            pred_df = test_agg[["origin", "destination", "tons", "final_pred"]].copy()
        else:
            pred_df = test_df[["origin", "destination", "tons", "final_pred"]].copy()
        pred_df["split"] = s_num
        pred_df["year"] = test_year
        test_pred_dfs.append(pred_df)
            
        # 2. Stable 경로 (Test - console display only)
        if np.sum(stable_mask) > 0:
            m_stable = compute_metrics(y_test_eval[stable_mask], final_pred[stable_mask], y_ctx_eval[stable_mask], y_base_eval[stable_mask], y_cagr_base_eval[stable_mask])
        else:
            m_stable = {k: 0.0 for k in METRIC_NAMES}
            
        # 3. Shock 경로 (Test - console display only)
        if np.sum(shock_mask) > 0:
            m_shock = compute_metrics(y_test_eval[shock_mask], final_pred[shock_mask], y_ctx_eval[shock_mask], y_base_eval[shock_mask], y_cagr_base_eval[shock_mask])
        else:
            m_shock = {k: 0.0 for k in METRIC_NAMES}

        # 4. 성장률 세그먼트별 평가 (아이디어 3)
        if granularity == "commodity":
            past_cagr_col = "route_tons_cagr"
            if past_cagr_col not in test_agg.columns:
                cagr_map = test_df.groupby(["origin", "destination"])[past_cagr_col].first().reset_index()
                test_agg = test_agg.merge(cagr_map, on=["origin", "destination"], how="left")
            past_cagr_vals = test_agg[past_cagr_col].fillna(0.0).values
        else:
            past_cagr_vals = test_df["tons_cagr"].fillna(0.0).values

        high_growth_mask = past_cagr_vals > 0.05
        stable_growth_mask = (past_cagr_vals >= -0.03) & (past_cagr_vals <= 0.05)
        decline_mask = past_cagr_vals < -0.03

        growth_segments = {
            "high_growth": high_growth_mask,
            "stable_growth": stable_growth_mask,
            "decline": decline_mask
        }

        for seg_name, mask in growth_segments.items():
            if np.sum(mask) > 0:
                m_seg = compute_metrics(
                    y_test_eval[mask],
                    final_pred[mask],
                    y_ctx_eval[mask],
                    y_base_eval[mask],
                    y_cagr_base_eval[mask]
                )
                all_metrics[f"{s_num}_{seg_name}_wRMSE"] = m_seg["wRMSE"]
                all_metrics[f"{s_num}_{seg_name}_WMAPE"] = m_seg["WMAPE"]
            else:
                all_metrics[f"{s_num}_{seg_name}_wRMSE"] = 0.0
                all_metrics[f"{s_num}_{seg_name}_WMAPE"] = 0.0

        alpha_str = "N/A"
        if "best_alphas_dict" in locals():
            if len(best_alphas_dict) == 1:
                alpha_str = f"{list(best_alphas_dict.values())[0]:.2f}"
            else:
                mean_alpha = np.mean(list(best_alphas_dict.values()))
                alpha_str = f"mean={mean_alpha:.2f}"

        print(
            f"  {split_name} (test={test_year}, base={test_year-3}) | "
            f"Alpha={alpha_str} | "
            f"wRMSE={m_all['wRMSE']:,.1f}  "
            f"Stable_wRMSE={m_stable['wRMSE']:,.1f}  "
            f"Shock_wRMSE={m_shock['wRMSE']:,.1f} | "
            f"WMAPE={m_all['WMAPE']:.4f}  R2={m_all['R2']:.4f}  Growth_WMAPE={m_all['Growth_WMAPE']:.4f}"
        )
        print(
            f"    Growth Segments: "
            f"High={all_metrics[f'{s_num}_high_growth_wRMSE']:,.1f} wRMSE, {all_metrics[f'{s_num}_high_growth_WMAPE']:.4f} WMAPE | "
            f"Stable={all_metrics[f'{s_num}_stable_growth_wRMSE']:,.1f} wRMSE, {all_metrics[f'{s_num}_stable_growth_WMAPE']:.4f} WMAPE | "
            f"Decline={all_metrics[f'{s_num}_decline_wRMSE']:,.1f} wRMSE, {all_metrics[f'{s_num}_decline_WMAPE']:.4f} WMAPE"
        )
        # Store validation dataframe for future splits' walk-forward evaluation
        save_cols = ["origin", "destination", "tons", "raw_val", "base_pred", "is_shock"]
        if granularity == "commodity":
            save_cols.append("commodity")
            save_cols.append("commodity_group")
        past_splits_info.append((split_name, val_df[save_cols].copy()))
    # 가중 평균 (팀 표준 핵심 지표)
    target_metrics = [
        "RMSE", "MAE", "WMAPE", "RMSLE", "R2_Score",
        "Skill_Score_Naive", "Skill_Score_YoY_Median", "Skill_Score_Constant_CAGR",
        "Directional_Accuracy", "CAGR_Error", "Growth_WMAPE"
    ]
    for metric in target_metrics:
        # Test weighted average
        all_metrics[f"weighted_{metric}"] = sum(
            all_metrics[f"split{i}_{metric}"] * SPLIT_DEFS[f"split_{i}"]["weight"]
            for i in range(1, 4)
        )
        # Validation weighted average
        all_metrics[f"val_weighted_{metric}"] = sum(
            all_metrics[f"val_split{i}_{metric}"] * SPLIT_DEFS[f"split_{i}"]["weight"]
            for i in range(1, 4)
        )

    # 성장률 세그먼트 가중 평균 (high_growth, stable_growth, decline)
    for seg in ["high_growth", "stable_growth", "decline"]:
        for metric in ["wRMSE", "WMAPE"]:
            all_metrics[f"weighted_{seg}_{metric}"] = sum(
                all_metrics[f"split{i}_{seg}_{metric}"] * SPLIT_DEFS[f"split_{i}"]["weight"]
                for i in range(1, 4)
            )

    combined_preds_df = pd.concat(test_pred_dfs, ignore_index=True)
    return all_metrics, train_year_ranges, combined_preds_df


# ─────────────────────────────────────────────────────────────
# MLflow 실험 실행
# ─────────────────────────────────────────────────────────────

def run_experiment(
    model_name: str,
    model_cls,
    model_params: dict,
    encoders: dict[str, LabelEncoder],
    config: dict,
    seed: int,
    target_type: str = "direct",
    granularity: str = "route",
    scenario: str = "A",
) -> None:
    """단일 모델에 대한 전체 MLflow run 실행."""
    defaults     = config["run_defaults"]
    mlflow_cfg   = config["mlflow"]
    run_name     = f"{defaults['experimenter']}_{model_name}_Scenario{scenario}_HonestVal_{datetime.now().strftime('%m%d_%H%M%S')}"
    requirements = write_requirements()
    feature_cols = get_feature_cols(granularity)

    print(f"\n{'=' * 70}")
    print(f"[{model_name}] 실험 시작  (Scenario {scenario} | 3년 후 {granularity.capitalize()}-Level 예측)")
    print(f"Run Name: {run_name}")
    print("=" * 70)

    # 3-split 평가
    metrics, train_year_ranges, combined_preds_df = evaluate_all_splits(model_cls, model_params, encoders, seed, target_type, granularity, model_name, scenario=scenario)

    # MLflow 로깅
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    scenario_desc = (
        "Scenario A: Multidimensional Bridging + Full History"
        if scenario == "A"
        else "Scenario B: FAF5 Data Only (2017+)"
    )
    with mlflow.start_run(run_name=run_name, description=f"{defaults['description']} ({scenario_desc})") as run:

        # Tags
        mlflow.set_tag("dataset_version",   defaults["dataset_version"])
        mlflow.set_tag("description",       f"{defaults['description']} ({scenario_desc})")
        mlflow.set_tag("hostname",          socket.gethostname())
        mlflow.set_tag("primary_metric",    "weighted_RMSE")
        mlflow.set_tag("metric_schema",     "mlflow_sample_compatible_v2")
        mlflow.set_tag("result_dir",        str(ARTIFACT_DIR.resolve()))
        mlflow.set_tag("model_name",        model_name)
        mlflow.set_tag("selection_status",  "selected")
        mlflow.set_tag("prediction_target", f"3-year-ahead {granularity}-level tons")
        mlflow.set_tag("scenario",          scenario)
        mlflow.set_tag("validation_type",   "honest_3yr_lag")
        for sn, sv in SPLIT_DEFS.items():
            s_num = sn.replace("_", "")
            mlflow.set_tag(f"{s_num}_train",    train_year_ranges[sn])
            mlflow.set_tag(f"{s_num}_context",  str(sv["context_year"]))
            mlflow.set_tag(f"{s_num}_val",      str(sv["val_year"]))
            mlflow.set_tag(f"{s_num}_test",     str(sv["test_year"]))
            mlflow.set_tag(f"{s_num}_weight",   str(sv["weight"]))

        # Params
        mlflow.log_param("random_seed",       seed)
        mlflow.log_param("experimenter",      defaults["experimenter"])
        mlflow.log_param("model_name",        model_name)
        mlflow.log_param("dataset_version",   defaults["dataset_version"])
        mlflow.log_param("n_features",        len(feature_cols))
        mlflow.log_param("feature_cols",      ",".join(feature_cols))
        mlflow.log_param("target_type",       target_type)
        mlflow.log_param("granularity",       granularity)
        mlflow.log_param("prediction_gap",    "3 years")
        mlflow.log_param("scenario",          scenario)
        mlflow.log_param("target_level",      f"{granularity} (origin × destination" + (" × commodity)" if granularity == "commodity" else ")"))
        if model_params:
            mlflow.log_params({
                k: str(v) for k, v in model_params.items()
                if k not in ("random_state", "n_jobs")
            })

        # Metrics (split별 25개 + weighted 10개 = 35개)
        mlflow.log_metrics(metrics)

        # Log train ranges
        for sn, yr_range in train_year_ranges.items():
            mlflow.log_param(f"{sn}_train_range", yr_range)

        # Artifacts
        mlflow.log_artifact(str(requirements))
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source_code")
        mlflow.log_artifact(str(CONFIG_PATH.resolve()),    artifact_path="source_code")
        
        # Save predictions CSV
        pred_csv_path = ARTIFACT_DIR / f"predictions_{run_name}.csv"
        pred_csv_path.parent.mkdir(exist_ok=True, parents=True)
        combined_preds_df.to_csv(pred_csv_path, index=False)
        mlflow.log_artifact(str(pred_csv_path))

        print(f"\n  Run ID          : {run.info.run_id}")
        print(f"  ★ weighted_RMSE : {metrics['weighted_RMSE']:,.1f}")
        print(f"    weighted_MAE  : {metrics['weighted_MAE']:,.1f}")
        print(f"    weighted_WMAPE: {metrics['weighted_WMAPE']:.4f}")
        print(f"    weighted_RMSLE: {metrics['weighted_RMSLE']:.4f}")
        print(f"    weighted_R2_Score: {metrics['weighted_R2_Score']:.4f}")
        print(f"    weighted_Skill_Naive: {metrics['weighted_Skill_Score_Naive']:.4f}")
        print(f"    weighted_Skill_Median: {metrics['weighted_Skill_Score_YoY_Median']:.4f}")
        print(f"    weighted_Skill_CAGR: {metrics['weighted_Skill_Score_Constant_CAGR']:.4f}")
        print(f"    weighted_DirAcc: {metrics['weighted_Directional_Accuracy']:.4f}")
        print(f"    weighted_CAGR_Err: {metrics['weighted_CAGR_Error']:.4f}")
        print(f"    weighted_Growth_WMAPE: {metrics['weighted_Growth_WMAPE']:.4f}")
        print(
            f"    Growth Segments: \n"
            f"      High-Growth (CAGR > 5%) : {metrics['weighted_high_growth_wRMSE']:,.1f} wRMSE | {metrics['weighted_high_growth_WMAPE']:.4f} WMAPE\n"
            f"      Stable (-3% ~ 5%)        : {metrics['weighted_stable_growth_wRMSE']:,.1f} wRMSE | {metrics['weighted_stable_growth_WMAPE']:.4f} WMAPE\n"
            f"      Decline (CAGR < -3%)     : {metrics['weighted_decline_wRMSE']:,.1f} wRMSE | {metrics['weighted_decline_WMAPE']:.4f} WMAPE"
        )
        print(f"  MLflow 확인     : {mlflow_cfg['tracking_uri']}")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FAF Hybrid Modeling Scenario Runner")
    parser.add_argument("--scenario", type=str, required=True, choices=["A", "B"], help="Scenario to run: A (multidimensional bridging) or B (FAF5 only)")
    parser.add_argument("--model", type=str, default="LightGBM_Hybrid_Tweedie", help="Model name to run")
    args = parser.parse_args()

    global _RAW_DF
    config = load_config()
    defaults = config["run_defaults"]
    seed   = defaults["random_seed"]
    granularity = defaults.get("granularity", "route")
    set_seed(seed)

    # 시나리오에 따른 데이터 전처리
    if args.scenario == "A":
        print(f"\n[Scenario A] Applying multidimensional bridging to align FAF4 (2012-2016) with FAF5 (2017+)...")
        _RAW_DF = apply_multidimensional_bridging(_RAW_DF)
    elif args.scenario == "B":
        print(f"\n[Scenario B] Filtering dataset to FAF5 only (year >= 2017) to avoid historical distortion...")
        _RAW_DF = _RAW_DF[_RAW_DF['year'] >= 2017].copy()
    
    # 캐시 초기화
    _DATA_CACHE.clear()

    # 범주형 인코더 빌드 (전체 split 공통)
    encoders = build_encoders(granularity)

    experiments = [
        (
            "LightGBM_Hybrid_Tweedie",
            lgb.LGBMRegressor,
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "objective": "tweedie",
                "tweedie_variance_power": 1.5,
                "random_state": seed,
                "n_jobs": -1,
                "verbose": -1,
            },
            "ratio",
        ),
        (
            "LightGBM_Direct_Tweedie",
            lgb.LGBMRegressor,
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "objective": "tweedie",
                "tweedie_variance_power": 1.5,
                "random_state": seed,
                "n_jobs": -1,
                "verbose": -1,
            },
            "direct",
        ),
        (
            "LightGBM_Ratio_Tweedie",
            lgb.LGBMRegressor,
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "objective": "tweedie",
                "tweedie_variance_power": 1.5,
                "random_state": seed,
                "n_jobs": -1,
                "verbose": -1,
            },
            "ratio",
        ),
        (
            "LightGBM_Hurdle_Tweedie",
            TwoStageHurdleModel,
            {
                "classifier_params": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "random_state": seed,
                    "n_jobs": -1,
                    "verbose": -1,
                },
                "regressor_cls": lgb.LGBMRegressor,
                "regressor_params": {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "objective": "tweedie",
                    "tweedie_variance_power": 1.5,
                    "random_state": seed,
                    "n_jobs": -1,
                    "verbose": -1,
                }
            },
            "direct",
        ),
        (
            "LightGBM_Hurdle_MSE",
            TwoStageHurdleModel,
            {
                "classifier_params": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "random_state": seed,
                    "n_jobs": -1,
                    "verbose": -1,
                },
                "regressor_cls": lgb.LGBMRegressor,
                "regressor_params": {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "objective": "regression",
                    "random_state": seed,
                    "n_jobs": -1,
                    "verbose": -1,
                }
            },
            "direct",
        ),
        (
            "LightGBM_RawRatio_Tweedie",
            lgb.LGBMRegressor,
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "objective": "tweedie",
                "tweedie_variance_power": 1.5,
                "random_state": seed,
                "n_jobs": -1,
                "verbose": -1,
            },
            "raw_ratio",
        ),
        (
            "LightGBM_Hybrid_RawRatio_Tweedie",
            lgb.LGBMRegressor,
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "objective": "tweedie",
                "tweedie_variance_power": 1.5,
                "random_state": seed,
                "n_jobs": -1,
                "verbose": -1,
            },
            "raw_ratio",
        ),
    ]

    # 모델 필터링
    if args.model:
        experiments = [exp for exp in experiments if exp[0] == args.model]
        if not experiments:
            print(f"Error: Model '{args.model}' not found in defined experiments.")
            sys.exit(1)

    for model_name, model_cls, params, target_type in experiments:
        run_experiment(
            model_name, model_cls, params,
            encoders, config, seed, target_type, granularity,
            scenario=args.scenario
        )

    print(f"\n{'=' * 70}")
    print(f"시나리오 {args.scenario} 실험 완료!")
    print(f"결과 확인: {config['mlflow']['tracking_uri']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
