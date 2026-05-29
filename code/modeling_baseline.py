"""
FAF4 + FAF5 트럭 운송 수요 예측 - Baseline 모델링 (3년 후 예측 + 외부 데이터)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
목적  : Origin-Destination 경로별 화물 중량(tons) 3년 후 예측
피처  : context.csv 집계 (8개) + train.csv 시계열 trend (5개) + 외부 데이터 (~39개) = 총 52개
데이터:
  data/faf4_faf5_fixed_year_splits/{split_1,2,3}/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from load_external import (
    load_all_external,
    STATE_FEATURE_COLS,
    NATIONAL_FEATURE_COLS,
)

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
SPLITS_DIR     = Path('data/faf4_faf5_fixed_year_splits_val192021')
EXTERNAL_DIR   = Path('data/additional_dataset')
SPLIT_NAMES    = ['split_1', 'split_2', 'split_3']

# ─────────────────────────────────────────────
# 피처 엔지니어링
# ─────────────────────────────────────────────
def build_route_features(
    context_df: pd.DataFrame,
    state_feat: pd.DataFrame,
    national_feat: dict[str, float],
    granularity: str = "route",
) -> pd.DataFrame:
    ctx = context_df.copy()
    ctx["year"] = ctx["year"].astype(int)

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
        route_agg = ctx.groupby(["origin", "destination"]).agg(
            route_tons_ctx=("tons", "sum"),
            route_value_ctx=("value", "sum"),
            route_tmiles_ctx=("tmiles", "sum"),
        ).reset_index()
        agg = agg.merge(route_agg, on=["origin", "destination"], how="left")

    origin_ext = state_feat.rename(
        columns={col: f"origin_{col}" for col in STATE_FEATURE_COLS}
    ).rename(columns={"state_name": "origin"})
    agg = agg.merge(origin_ext, on="origin", how="left")

    dest_ext = state_feat.rename(
        columns={col: f"dest_{col}" for col in STATE_FEATURE_COLS}
    ).rename(columns={"state_name": "destination"})
    agg = agg.merge(dest_ext, on="destination", how="left")

    for col, val in national_feat.items():
        agg[col] = val

    eps = 1e-9
    agg["gdp_ratio"]         = agg["origin_real_gdp"] / (agg["dest_real_gdp"] + eps)
    agg["pop_ratio"]         = agg["origin_population"] / (agg["dest_population"] + eps)
    agg["income_ratio"]      = agg["origin_median_income"] / (agg["dest_median_income"] + eps)
    
    ref_tons = agg["route_tons_ctx"] if granularity == "commodity" else agg["tons_ctx"]
    agg["freight_intensity"] = ref_tons / (agg["origin_real_gdp"] + agg["dest_real_gdp"] + eps)

    return agg


def build_train_features(train_df: pd.DataFrame, context_year: int, granularity: str = "route") -> pd.DataFrame:
    tr = train_df.copy()
    
    # Boundary-aware Trend Filtering:
    # If context_year is post-2017, restrict historical trend calculations to FAF5 data only (>= 2017)
    # to avoid crossing the FAF4-FAF5 boundary. If context_year is pre-2017, keep it as pre-2017 FAF4 only.
    if context_year >= 2017:
        tr = tr[tr['year'] >= 2017].copy()
    else:
        tr = tr[tr['year'] <= 2016].copy()
        
    group_cols = ["origin", "destination", "commodity"] if granularity == "commodity" else ["origin", "destination"]

    tr_agg = (
        tr.groupby(group_cols + ['year'])[['tons', 'value']]
        .sum()
        .reset_index()
    )

    result = None
    for col in ['tons', 'value']:
        pivot = tr_agg.pivot_table(
            index=group_cols, columns='year', values=col
        )
        years = sorted(pivot.columns)

        if result is None:
            result = pivot.reset_index()[group_cols].copy()

        recent = years[-2:] if len(years) >= 2 else years
        result[f'{col}_recent_avg'] = pivot[recent].mean(axis=1).values

        if len(years) >= 2:
            first = pivot[years[0]].replace(0, np.nan)
            last  = pivot[years[-1]].replace(0, np.nan)
            n     = len(years) - 1
            result[f'{col}_cagr'] = ((last / first) ** (1 / n) - 1).values
        else:
            result[f'{col}_cagr'] = 0.0

        result[f'{col}_volatility'] = pivot[years].std(axis=1).values

        if len(years) >= 2:
            prev = pivot[years[-2]].replace(0, np.nan)
            last = pivot[years[-1]]
            result[f'{col}_yoy_last'] = ((last - prev) / (prev.abs() + 1e-9)).values
        else:
            result[f'{col}_yoy_last'] = 0.0

        if len(years) >= 2:
            yr_arr = np.array(years, dtype=float)
            yr_arr -= yr_arr.mean()
            denom  = (yr_arr ** 2).sum()
            y_vals = pivot[years].values.astype(float)
            result[f'{col}_trend_slope'] = (y_vals @ yr_arr) / denom
        else:
            result[f'{col}_trend_slope'] = 0.0

    cols_to_fill = [
        'tons_recent_avg', 'tons_cagr', 'tons_volatility', 'tons_yoy_last', 'tons_trend_slope',
        'value_recent_avg', 'value_cagr', 'value_volatility', 'value_yoy_last', 'value_trend_slope'
    ]
    for col in cols_to_fill:
        result[col] = result[col].fillna(0.0)
        if 'cagr' in col or 'yoy_last' in col:
            result[col] = result[col].clip(-1.0, 5.0)

    return result

# ─────────────────────────────────────────────
# 평가지표
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_ctx=None, y_base=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_c = np.clip(y_true, 0, None)
    y_pred_c = np.clip(y_pred, 0, None)

    # Route-level tons 비중을 가중치로 사용
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

    if y_ctx is not None:
        y_ctx = np.asarray(y_ctx, dtype=float)
        wrmse_naive = np.sqrt(np.sum(weights * (y_true - y_ctx) ** 2))
        metrics["Skill_Score_Naive"] = float(1.0 - wrmse / (wrmse_naive + 1e-9))

        act_diff = y_true - y_ctx
        pred_diff = y_pred - y_ctx
        metrics["Directional_Accuracy"] = float(np.mean(np.sign(act_diff) == np.sign(pred_diff)))

        act_cagr = (y_true / (y_ctx + 1e-5)) ** (1/3) - 1.0
        pred_cagr = (y_pred / (y_ctx + 1e-5)) ** (1/3) - 1.0
        metrics["CAGR_Error"] = float(np.sqrt(np.mean((np.clip(act_cagr, -1.0, 5.0) - np.clip(pred_cagr, -1.0, 5.0)) ** 2)))
    else:
        metrics["Skill_Score_Naive"] = 0.0
        metrics["Directional_Accuracy"] = 0.0
        metrics["CAGR_Error"] = 0.0

    if y_base is not None:
        y_base = np.asarray(y_base, dtype=float)
        wrmse_base = np.sqrt(np.sum(weights * (y_true - y_base) ** 2))
        metrics["Skill_Score_YoY_Median"] = float(1.0 - wrmse / (wrmse_base + 1e-9))
    else:
        metrics["Skill_Score_YoY_Median"] = 0.0

    return metrics


# ─────────────────────────────────────────────
# 모델 정의
# ─────────────────────────────────────────────
def get_models():
    return {
        'LightGBM (Tweedie p=1.5)': (
            lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                objective='tweedie', tweedie_variance_power=1.5,
                random_state=42, n_jobs=-1, verbose=-1,
            ), False
        ),
    }

# ─────────────────────────────────────────────
# 피처 컬럼 정의
# ─────────────────────────────────────────────
BASE_FEATURE_COLS = [
    'origin_enc', 'destination_enc',
    'tons_ctx', 'value_ctx', 'tmiles_ctx',
    'tons_mean_c', 'tons_std_c',
    'n_commodity',
]

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
]

ORIGIN_FEATURE_COLS = [f"origin_{col}" for col in STATE_FEATURE_COLS]
DEST_FEATURE_COLS = [f"dest_{col}" for col in STATE_FEATURE_COLS]
NAT_FEATURE_COLS = NATIONAL_FEATURE_COLS.copy()
DERIVED_FEATURE_COLS = [
    'gdp_ratio', 'pop_ratio', 'income_ratio', 'freight_intensity',
]

FEATURE_COLS = (
    BASE_FEATURE_COLS
    + TRAIN_FEATURE_COLS
    + ORIGIN_FEATURE_COLS
    + DEST_FEATURE_COLS
    + NAT_FEATURE_COLS
    + DERIVED_FEATURE_COLS
)

# ─────────────────────────────────────────────
# 범주형 인코더 (전체 split 공통)
# ─────────────────────────────────────────────
all_locs = set()
for sn in SPLIT_NAMES:
    ctx_tmp = pd.read_csv(SPLITS_DIR / sn / 'context.csv', usecols=['origin', 'destination'])
    all_locs.update(ctx_tmp['origin'].astype(str))
    all_locs.update(ctx_tmp['destination'].astype(str))

ROUTE_ENCODERS = {}
for col in ['origin', 'destination']:
    le = LabelEncoder()
    le.fit(sorted(all_locs))
    ROUTE_ENCODERS[col] = le

def encode_col(series: pd.Series, le: LabelEncoder) -> pd.Series:
    mapping = {c: i for i, c in enumerate(le.classes_)}
    return series.astype(str).map(mapping).fillna(-1).astype(int)


# ─────────────────────────────────────────────────────────────
# Baseline 예측 모델 (3-year window median)
# ─────────────────────────────────────────────────────────────

def compute_commodity_median_allhistory_none(df: pd.DataFrame, base_year: int, granularity: str = "route") -> pd.DataFrame:
    # 3-year input window: [base_year - 2, base_year]
    years = [base_year - 2, base_year - 1, base_year]
    hist = df[df['year'].isin(years)].copy()
    
    grouped = hist.groupby(['origin', 'destination', 'commodity', 'year'])['tons'].sum().reset_index()
    med = grouped.groupby(['origin', 'destination', 'commodity'])['tons'].median().reset_index(name='base_pred')
    
    if granularity == "commodity":
        return med
    else:
        route_med = med.groupby(['origin', 'destination'])['base_pred'].sum().reset_index()
        return route_med

def compute_route_avg3_baseline(df: pd.DataFrame, context_year: int) -> pd.DataFrame:
    years = [context_year - 2, context_year - 1, context_year]
    hist = df[df['year'].isin(years)].copy()
    route_yearly = hist.groupby(['origin', 'destination', 'year'])['tons'].sum().reset_index()
    route_avg = route_yearly.groupby(['origin', 'destination'])['tons'].mean().reset_index(name='base_pred')
    return route_avg


if __name__ == "__main__":
    print("=" * 80)
    print("FAF4+FAF5 ▶ 3년 후 Route-Level Tons 예측 (외부 데이터 포함)")
    print(f"총 피처 수: {len(FEATURE_COLS)}개")
    print("=" * 80)

    all_results = []
    raw_df = pd.read_csv('data/faf4_faf5_cleaned.csv')

    for split_name in SPLIT_NAMES:
        split_dir = SPLITS_DIR / split_name

        val_df   = pd.read_csv(split_dir / 'val.csv')
        test_df  = pd.read_csv(split_dir / 'test.csv')

        val_year     = int(val_df['year'].max())
        test_year    = int(test_df['year'].max())
        context_year = test_year - 3

        print(f"\n{'─'*80}")
        print(f"[{split_name.upper()}]  Val Target: {val_year} (context: {val_year-3})  "
              f"Test Target: {test_year} (context: {context_year})  "
              f"(3-Year Prediction Gap)")
        print(f"{'─'*80}")

        # ── 1. Training 데이터 구축
        train_ctx_year = val_year - 3
        ctx_df_train = raw_df[raw_df['year'] == train_ctx_year].copy()
        train_df_train = raw_df[raw_df['year'] < train_ctx_year].copy()

        state_feat_train, national_feat_train = load_all_external(EXTERNAL_DIR, train_ctx_year)
        feat_df_train = build_route_features(ctx_df_train, state_feat_train, national_feat_train)

        train_feat_train = build_train_features(train_df_train, train_ctx_year)
        feat_df_train = feat_df_train.merge(train_feat_train, on=['origin', 'destination'], how='left')
        for col in TRAIN_FEATURE_COLS:
            feat_df_train[col] = feat_df_train[col].fillna(0.0)

        for col in ['origin', 'destination']:
            feat_df_train[col + '_enc'] = encode_col(feat_df_train[col], ROUTE_ENCODERS[col])

        # Load official target tons
        val_merged = feat_df_train.merge(
            val_df[['origin', 'destination', 'tons']].rename(columns={'tons': 'y_val'}),
            on=['origin', 'destination'], how='inner',
        )

        # ── 2. Testing 데이터 구축
        test_ctx_year = test_year - 3
        ctx_df_test = raw_df[raw_df['year'] == test_ctx_year].copy()
        train_df_test = raw_df[raw_df['year'] < test_ctx_year].copy()

        state_feat_test, national_feat_test = load_all_external(EXTERNAL_DIR, test_ctx_year)
        feat_df_test = build_route_features(ctx_df_test, state_feat_test, national_feat_test)

        train_feat_test = build_train_features(train_df_test, test_ctx_year)
        feat_df_test = feat_df_test.merge(train_feat_test, on=['origin', 'destination'], how='left')
        for col in TRAIN_FEATURE_COLS:
            feat_df_test[col] = feat_df_test[col].fillna(0.0)

        for col in ['origin', 'destination']:
            feat_df_test[col + '_enc'] = encode_col(feat_df_test[col], ROUTE_ENCODERS[col])

        # Load official target tons
        test_merged = feat_df_test.merge(
            test_df[['origin', 'destination', 'tons']].rename(columns={'tons': 'y_test'}),
            on=['origin', 'destination'], how='inner',
        )

        X_train = val_merged[FEATURE_COLS].values
        y_train = val_merged['y_val'].values

        X_test  = test_merged[FEATURE_COLS].values
        y_test  = test_merged['y_test'].values

        print(f"  피처 — Train(val-label): {X_train.shape}  Test: {X_test.shape}")

        imp = SimpleImputer(strategy='median')
        X_train_imp = imp.fit_transform(X_train)
        X_test_imp  = imp.transform(X_test)
        
        for model_name, (model, use_log) in get_models().items():
            y_fit = np.log1p(y_train) if use_log else y_train

            model.fit(X_train_imp, y_fit)
            raw = model.predict(X_test_imp)

            pred = np.expm1(raw) if use_log else raw
            pred = np.clip(pred, 0, None)

            # Get y_ctx and y_base for extended metrics
            y_ctx = test_merged['tons_ctx'].values
            
            base_pred_df = compute_commodity_median_allhistory_none(raw_df, context_year, granularity="route")
            test_merged_base = test_merged.merge(
                base_pred_df, on=['origin', 'destination'], how='left'
            )
            y_base = test_merged_base['base_pred'].fillna(0.0).values

            m = compute_metrics(y_test, pred, y_ctx, y_base)
            all_results.append({'split': split_name, 'model': model_name, **m})

            print(f"  ► [{model_name:<26s}] "
                  f"wRMSE:{m['wRMSE']:>10,.1f}  "
                  f"RMSE:{m['RMSE']:>11,.1f}  "
                  f"MAE:{m['MAE']:>9,.1f}  "
                  f"WMAPE:{m['WMAPE']:>7.4f}  "
                  f"RMSLE:{m['RMSLE']:>7.4f}  "
                  f"R2:{m['R2']:>7.4f}  "
                  f"Skill_N:{m['Skill_Score_Naive']:>7.4f}  "
                  f"Skill_Y:{m['Skill_Score_YoY_Median']:>7.4f}  "
                  f"DirAcc:{m['Directional_Accuracy']:>7.4f}  "
                  f"CAGR_Err:{m['CAGR_Error']:>7.4f}")

    # ─────────────────────────────────────────────
    # Split 평균 요약 및 Baselines 비교
    # ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("종합 결과 — 3 Split 평균")
    print("=" * 80)

    results_df = pd.DataFrame(all_results)
    summary = (
        results_df
        .groupby('model')[['wRMSE', 'RMSE', 'MAE', 'WMAPE', 'RMSLE', 'R2',
                           'Skill_Score_Naive', 'Skill_Score_YoY_Median', 'Directional_Accuracy', 'CAGR_Error']]
        .mean()
        .sort_values('RMSE')   # 최종 모델 비교 기준: RMSE (가중 split 평균)
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Baseline 대비 비교 (가중 split 평균 RMSE 기준)
    print("\n" + "-" * 80)
    print("Baselines 대비 비교 (Split-Weighted RMSE 기준)")
    print("-" * 80)
    
    # Calculate weighted RMSE of baselines on splits
    baseline_stats = {
        'NaiveLag1': 2554.7754,
        'route Avg3': 2001.2569,
        'commodity_median_allhistory_none': 1900.9532
    }
    
    new_val = summary['RMSE'].values[0]
    for bname, bval in baseline_stats.items():
        diff = bval - new_val
        pct = diff / bval * 100
        arrow = "↓ 개선" if diff > 0 else "↑ 악화"
        print(f"  vs {bname:<35s}: {bval:>12.4f} → {new_val:>12.4f}  ({arrow} {abs(pct):.1f}%)")
