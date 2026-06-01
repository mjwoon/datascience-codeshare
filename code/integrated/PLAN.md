# 통합 실험 구현 계획

**작성자**: mjwoon  
**최초 작성**: 2026-06-01 / **업데이트**: 2026-06-01 (M5 yulim 추가)  
**브랜치**: lightGBM-tweedie  
**디렉토리**: `code/integrated/`

---

## 1. 목표

5개 팀원(bayasgalan, dongbin, minseok, mjwoon, yulim) 실험의 최선 요소를 통합한 단일 파이프라인 구축.  
공식 splits 기준 tons-weighted RMSE(wRMSE) 최소화 및 MLflow 실험 추적.

### 각 실험 성능 (val wRMSE, 참고용)

| 실험 | val wRMSE | 비고 |
|---|---|---|
| **yulim** | **14,797** | 현재 최고 |
| bayasgalan CM3 baseline | 16,379 | — |
| bayasgalan CM3+LGBM | 16,744 | baseline보다 낮음 |
| minseok | ~16,379 | CM3와 동등 |
| mjwoon / dongbin | 미공개 | — |

---

## 2. 확정 설계 결정 (12개)

### 2-1. 5개 차원 결정

| 항목 | 결정 | 출처 |
|---|---|---|
| 예측 단위 | OD×commodity 예측 → route 집계 | dongbin, yulim |
| 외부 데이터 | 14개 CSV — 절대값(D_abs, M1~M4) + ratio3(D_ratio, M5) 두 형태 병행 | mjwoon + yulim |
| 경제 지표 | M$ = error_reduction_kton × 1000 × $10 / 1e6 | minseok |
| Phase 3 선택 | 전 모델 통일: commodity SSE greedy + 15% margin guard | dongbin + minseok |
| FAF4/FAF5 경계 — **피처 계산** | 전 모델 통일: **전체 history 사용** (context_year branching 제거) | yulim |
| FAF4/FAF5 경계 — **ML 학습 샘플** | M2~M5 전 모델: `min_fy=2017` (FAF5-only 학습) | bayasgalan |

> **FAF4/FAF5 경계 분리 근거**: 피처 계산에서는 전체 history를 쓸수록 lag/trend 피처의 품질이 높아진다 (yulim 성능 근거). ML 학습 샘플에서의 min_fy=2017은 FAF4↔FAF5 분포 이동(distribution shift) 문제를 피하기 위한 별도 결정이다 (bayasgalan 실험 근거).

### 2-2. 7개 추가 결정

| # | 항목 | 결정 |
|---|---|---|
| 1 | COVID 필터 | M2/M3/M4: 2020을 **학습 target**에서 제외. M5: 미채택 (yulim 원본이 이 방식을 쓰지 않음) |
| 2 | 샘플 가중 | M4(dongbin)만 적용: `tons^0.35 × (1 + 4×large_route + 2×intrastate)` |
| 3 | 테스트 예측 | val 선택 후 train+context 재학습 → test 예측 |
| 4 | 2020 외부 데이터 결측 | 선형 보간 (`_interpolate_year_gaps`, 이미 구현됨) |
| 5 | Phase 3 세부 | §2-1 Phase 3 선택과 동일 — route RMSE 체크 없음, commodity SSE 단독 기준 |
| 6 | LightGBM 최소 샘플 | threshold 없음, LightGBM `min_child_samples`로만 제어 |
| 7 | 예측 상한 | **M2~M5** (ML 모델): correction clip ±0.10 (log-ratio 보정 제한). M1: 적용 안 함 (median은 자연적으로 bounded) |

---
    
## 3. 데이터 구조

### 3-1. Split 정의

```
data/faf4_faf5_fixed_year_splits/split_{n}/
  train.csv          — detail rows (origin, destination, commodity, year, tons, value, tmiles, distance_band)
  context.csv        — detail rows (동일 schema, 1년치)
  val_commodity.csv  — route_commodity (origin, destination, commodity, year, tons)
  val.csv            — route (origin, destination, year, tons)
  test_commodity.csv — route_commodity (동일)
  test.csv           — route (동일)
```

| Split | train years | context | val | test | weight |
|---|---|---|---|---|---|
| split_1 | 2012–2018 | 2019 | 2021 | 2022 | 0.2 |
| split_2 | 2012–2019 | 2020 | 2022 | 2023 | 0.3 |
| split_3 | 2012–2020 | 2021 | 2023 | 2024 | 0.5 |

### 3-2. 외부 데이터

```
data/additional_data/
  state_unify_gdp.csv                    → real_gdp (절대) / gdp_ratio3 (ratio)
  state_unify_ue-rate.csv                → unemployment_rate (절대)
  population_by_states.csv               → population, pop_growth_rate (절대) / pop_ratio3 (ratio)
  income_annual.csv                      → median_income (절대, 2020 보간)
  pce_by_state_real_2017.csv             → pce_total, pce_goods (절대) / pce_goods/energy/food/motor_ratio3 (ratio)
  pce_by_state_real_2017_changes.csv     → pce_growth_rate (절대)
  eia_coal_production_by_state.csv       → coal_production_ktons (절대, 26개 주) / coal_prod_ratio3 (ratio)
  eia_coal_electricity_gen_by_state.csv  → coal_elec_gen_gwh (절대) / coal_gen_ratio3 (ratio)
  eia_heating_degree_days_by_state.csv   → hdd_annual (절대, 48개 주) / hdd_ratio3 (ratio)
  noaa_pdsi_by_state.csv                 → pdsi_annual_avg (절대, 48개 주)
  usda_crop_production_by_state.csv      → corn/wheat/soy_prod_bu (절대) / crop_prod_ratio3 (ratio)
  migration_od_annual.csv                → net_migration (절대, 2020 보간)
  eia_natgas_electricity_gen_national.csv → natgas_elec_gen_gwh (절대) / natgas_ratio3 (ratio)
  fao_fertilizer_price_index.csv         → dap/urea_price_usd_mt (절대) / fert_price_ratio3 (ratio)
```

> ⚠️ `building_permits_by_state.csv`는 dongbin 코드에서 참조되나 repo에 없음 → 제외

---

## 4. 파일 구조

```
code/integrated/
├── PLAN.md                        ← 이 문서
├── config.py                      경로·상수·split 정의
├── data_loader.py                 split CSV 로드 + commodity pivot
├── features/
│   ├── __init__.py
│   ├── context_features.py        A: context.csv → OD×commodity 6개
│   ├── lag_trend_features.py      B: commodity-level lag/trend ~15개
│   ├── structure_features.py      C: n_commodities, cv, dist_band 등 5개
│   ├── external_features.py       D_abs: 절대값 외부 피처 ~35개 (M1~M4용)
│   └── external_ratio_features.py D_ratio: ratio3 외부 피처 14개 (M5용)
├── candidates/
│   ├── __init__.py
│   ├── m0_baseline.py             M0: 6개 후보 (순수 통계, 외부 데이터 없음)
│   ├── m1_statistical.py          M1: 12개 후보 (통계 + 외부 ratio factor)
│   ├── m2_bayasgalan.py           M2: CM3 + LightGBM residual
│   ├── m3_mjwoon.py               M3: TwoStage Hurdle Tweedie
│   ├── m4_dongbin.py              M4: per-commodity LightGBM
│   └── m5_yulim.py                M5: medmean + log-ratio LGBM + soft gate
├── selector.py                    Phase 3: greedy + margin guard (M1~M5 통합)
├── allocator.py                   Phase 4: log-space route residual 재배분
├── metrics.py                     wRMSE, M$ 계산
└── run_experiment.py              MLflow 실험 진입점
```

---

## 5. 피처 그룹

### M2~M4 공통: A+B+C+D_abs

**인덱스**: `(origin, destination, commodity)`

#### A. Context 집계 피처 (~6개)
`context_features.py` → `build_context_features(context_df)`

| 피처 | 설명 |
|---|---|
| `ctx_tons_last` | context year tons |
| `ctx_tons_pct_route` | commodity의 route 내 비중 |
| `ctx_tons_log1p` | log1p(ctx_tons_last) |
| `ctx_value_per_ton` | value / tons |
| `ctx_zero_flag` | tons == 0 여부 |
| `ctx_route_total` | route-level context tons 합 |

#### B. Lag/Trend 피처 (~15개) — M2~M4 공통
`lag_trend_features.py` → `build_lag_trend_features(train_df, context_year)`

**전체 history 사용** (context_year branching 없음, 설계 결정 §2-1 참조):

| 피처 | 설명 |
|---|---|
| `tons_lag1/2/3` | context 기준 t-1, t-2, t-3 |
| `tons_mean_3yr`, `tons_std_3yr` | 최근 3년 통계 |
| `tons_mean_all`, `tons_cv_all` | 전체 history 통계 |
| `tons_yoy_last` | (lag1 − lag2) / (lag2 + 1) |
| `tons_trend`, `tons_trend_norm` | OLS slope, slope/mean |
| `n_years_hist` | 가용 year 수 |
| `tons_lag1_route` | route-level lag1 |
| `tons_ratio_lag1` | commodity/route lag1 비중 |
| `trend5_d05`, `trend5_d10` | damped trend (minseok 방식) |

#### C. 구조 피처 (~5개)
`structure_features.py` → `build_structure_features(train_df)`

| 피처 | 설명 |
|---|---|
| `n_commodities` | route 내 활성 commodity 수 |
| `modal_dist_band` | distance_band 최빈값 (label encode) |
| `tons_cv_route` | route-level CV (std/mean) |
| `large_route_flag` | route가 상위 20% tons 여부 |
| `intrastate_flag` | origin == destination |

#### D_abs. 외부 절대값 피처 (~35개) — M1~M4
`external_features.py` → `build_external_features(context_year, data_dir)`

```python
# origin state에 'orig_' 접두사 → 16개 (GDP, UE, pop, income, PCE total/goods/growth, coal prod/elec, HDD, PDSI, crop, net_migration)
# dest state에 'dest_' 접두사 → 16개
# national (전 rows 동일) → 3개 (natgas_elec_gen_gwh, dap_price, urea_price)
# 합계 35개
```

---

### M5 전용: B + C + D_ratio

M5는 B (전체 history, 피처 계산 통일 원칙과 동일)와 C를 공유하고, 외부 피처만 D_ratio를 사용한다.

M5 내부 베이스라인 계산용으로 `build_comm_feature_frame()`에서 추가로 사용하는 피처:

| 피처 | 설명 |
|---|---|
| `base_median` | 최근 3년 commodity median |
| `base_mean` | 최근 3년 commodity mean |
| `base_wmean` | 최근 3년 가중 평균 (0.2/0.3/0.5) |
| `prior_share` | commodity의 route 내 예측 비중 |
| `route_gap_ratio` | (route_ref − route_prior) / route_prior |

#### D_ratio. 외부 ratio3 피처 (14개) — M5 전용
`external_ratio_features.py` → `build_external_ratio_features(base_year, data_dir)`

```python
# ratio3 = val_t / val_{t-3}, clip [0.5, 1.8], fillna 1.0
# base_year = context_year − 3 (= tgt_year − 6 → context가 tgt_year-3이므로 base_year = context_year)
```

| 피처 | 출처 |
|---|---|
| `orig_gdp_ratio3` | GDP 원주 3년 변화율 |
| `dest_gdp_ratio3` | GDP 목적지 3년 변화율 |
| `orig_pop_ratio3` | 인구 원주 |
| `dest_pop_ratio3` | 인구 목적지 |
| `dest_hdd_ratio3` | 난방도일 목적지 |
| `orig_coal_prod_ratio3` | 석탄생산 원주 |
| `dest_coal_gen_ratio3` | 석탄발전 목적지 |
| `dest_pce_goods_ratio3` | PCE 소비재 목적지 |
| `dest_pce_energy_ratio3` | PCE 에너지 목적지 |
| `dest_pce_food_ratio3` | PCE 식품 목적지 |
| `dest_pce_motor_ratio3` | PCE 자동차 목적지 |
| `orig_crop_prod_ratio3` | 작물생산 원주 |
| `natgas_ratio3` | 천연가스 발전 전국 |
| `fert_price_ratio3` | 비료가격 전국 |

---

## 6. 5개 모델 패밀리 (후보 생성)

### 공통 인터페이스

```python
class CandidateModel:
    def fit(self, train_df, context_df, feature_df, context_year): ...
    def predict(self, feature_df) -> dict[str, pd.Series]:
        # key: candidate name
        # value: Series with index (origin, destination, commodity), values: tons

def apply_correction_clip(correction: pd.Series, clip: float = 0.10) -> pd.Series:
    """log-ratio 보정값 상한: M2~M5 공통. ±clip 범위로 제한."""
    return correction.clip(lower=-clip, upper=clip)
```

---

### M0: 통계 기준선 (`m0_baseline.py`)

**기반**: M1과 동일한 통계 로직, 외부 데이터 없음

**6개 후보** (외부 조정 없음):

```
M0_median_all    — 전체 history median
M0_median_3      — 최근 3년 median  ← Phase 3 baseline
M0_median_4      — 최근 4년 median
M0_trend5_d05    — 최근 5년 trend, damp=0.5
M0_trend5_d10    — 최근 5년 trend, damp=1.0
M0_trend3_d05    — 최근 3년 trend, damp=0.5
```

피처: 없음 (통계 계산만 사용). M1과의 비교를 통해 외부 데이터 기여도 측정.

---

### M1: 통계 + 외부 factor (`m1_statistical.py`)

**기반**: `minseok/main.py`의 COMM pivot + trend/median 함수

**6개 후보** (M0 기본 후보에 ratio3 multiplicative factor 적용):

```
M1_median_3_ext, M1_median_4_ext
M1_trend5_d05_ext, M1_trend5_d10_ext
M1_trend3_d05_ext, M1_median_all_ext
```

**commodity 그룹별 외부 factor** (ratio3 형태 재사용):

| 그룹 | FAF commodity | factor |
|---|---|---|
| FUEL | 2 (Fuel oils) | `dest_hdd_ratio3` |
| AG | 4, 6 (Animal feed, Cereal grains) | `orig_crop_prod_ratio3` |
| BULK | 1, 3 (Live animals, Logs) | `mean(orig_gdp_ratio3, dest_gdp_ratio3)` |
| others | 나머지 | `mean(orig_gdp_ratio3, dest_gdp_ratio3)` |

> ⚠️ `context_year − 3` 데이터 범위 밖이면 factor = 1.0으로 fallback

---

### M2: CM3 + LightGBM Residual (`m2_bayasgalan.py`)

**기반**: `bayasgalan/bayasgalan_model.py`의 `CM3LGBMModel`

**설계**:
- 피처: A+B+C+D_abs
- `min_fy=2017`: target_fy < 2017인 학습 샘플 제외
- COVID filter: `skip_target_years={2020}`
- 잔차 타겟: `log1p(actual_tons) − log1p(cm3_pred)` (log-ratio)
- 샘플 가중: 없음

**학습 샘플 생성**:
```python
for fy in range(min_fy, max_train_year + 1):
    if fy in skip_target_years: continue
    X = feature_df(data[year <= fy])
    y = log1p(actual_tons(fy + 3)) - log1p(cm3_pred(fy))   # log-ratio residual
# final: expm1(log1p(cm3_pred) + correction).clip(lower=0)
```

**2개 후보**: `M2_cm3only`, `M2_cm3_lgbm`

---

### M3: TwoStage Hurdle Tweedie (`m3_mjwoon.py`)

**기반**: `mjwoon/run_honest_experiment.py`의 `TwoStageHurdleModel`

**설계**:
- 피처: A+B+C+D_abs
- 예측 단위: commodity-level
- COVID filter: 2020 target 제외
- 샘플 가중: 없음

**구조**:
```python
# Stage 1: LGBMClassifier → P(tons > 0)
# Stage 2: LGBMRegressor → log1p(actual) − log1p(cm3_pred) (log-ratio residual)
# predict = P(>0) × expm1(log1p(cm3_pred) + correction).clip(lower=0)
```

**2개 후보**: `M3_tweedie`, `M3_hurdle`

---

### M4: per-Commodity LightGBM (`m4_dongbin.py`)

**기반**: `dongbin/r2030_standalone.py`의 `fit_lgb()` + `allocate_route_residual()`

**설계**:
- 피처: A+B+C+D_abs
- 잔차 타겟: `log1p(actual_tons) − log1p(cm3_pred)` (log-ratio)
- COVID filter: 2020 target 제외
- 샘플 가중: `tons^0.35 × (1 + 4×large_route_flag + 2×intrastate_flag)`
- min samples threshold: 제거 (`min_child_samples`로만 제어)
- final: `expm1(log1p(cm3_pred) + correction).clip(lower=0)` → route allocation 입력

**Route residual allocation** (M4 내부):
```python
def allocate_route_residual(commodity_preds, route_model_pred, alpha):
    route_error = route_model_pred - commodity_preds.groupby(["origin","destination"]).sum()
    w = log1p(commodity_preds) / log1p(commodity_preds).groupby(route).transform("sum")
    return commodity_preds + alpha × route_error × w
```

**5개 후보**: `M4_lgbm`, `M4_resid_alpha003`, `M4_resid_alpha005`, `M4_resid_alpha008`, `M4_resid_alpha010`

---

### M5: Yulim — medmean + log-ratio LGBM (`m5_yulim.py`)

**기반**: `yulim/commodity_stacking_experiment.py` + `yulim/high_impact_commodity_experiment.py`

**설계**:
- 피처: B + C + D_ratio
- 잔차 타겟: `log1p(actual_tons) − log1p(base_pred)` (log-ratio, M2~M4와 동일 방식)
- `min_fy=2017`: M2~M4와 동일하게 FAF5-only 학습 샘플 사용
- COVID filter: 적용 안 함
- 샘플 가중: 없음

**Step 1 — 베이스라인 생성**:
```python
# commodity_medmean_a0.20 = 0.80 × median(최근3년) + 0.20 × mean(최근3년)
# per OD×commodity, window=3
base_pred = 0.80 * median_3yr + 0.20 * mean_3yr
```

**Step 2 — log-ratio 잔차 학습 (commodity-group 단위 일률 적용)**:
```python
y_train = log1p(actual_tons) - log1p(base_pred)   # log-ratio

# 모든 OD×commodity에 동일하게 commodity-group별 LGBM 적용
# 피처: B + C + D_ratio
# correction = model.predict(X); clip to [-0.10, 0.10]
final_pred = expm1(log1p(base_pred) + correction).clip(lower=0)
```

**3개 후보**:

| 후보 | 설명 |
|---|---|
| `M5_medmean_base` | 베이스라인만 (보정 없음) |
| `M5_medmean_lgbm` | log-ratio LGBM 보정 (외부 피처 없음) |
| `M5_medmean_lgbm_extfeat` | log-ratio LGBM + D_ratio 외부 피처 ← **yulim 최고 성능** |

---

### 후보 수 요약

| 모델 | 후보 수 | 피처 그룹 |
|---|---|---|
| M0 기준선 | 6 | 없음 (통계 계산만) |
| M1 통계+외부 | 6 | D_ratio factor 적용 |
| M2 CM3+LGBM | 2 | A+B+C+D_abs |
| M3 Hurdle | 2 | A+B+C+D_abs |
| M4 LGBM | 5 | A+B+C+D_abs |
| M5 Yulim | 3 | B+C+D_ratio |
| **합계** | **24** | — |

---

## 7. Phase 3: Greedy 선택 (`selector.py`)

M5 후보(`M5_*`)는 이미 soft gate blending이 내부에서 완료된 상태로 출력되므로, Phase 3에서는 다른 후보들과 동일한 SSE 비교 대상으로 포함한다.

```python
def greedy_select(
    candidates: dict[str, pd.Series],   # {name: (OD×comm) → tons}
    val_commodity_df: pd.DataFrame,      # (origin, destination, commodity, tons)
    baseline_name: str = "M0_median_3",
    margin: float = 0.15,
) -> dict[str, str]:                     # {commodity → best_candidate_name}
```

**알고리즘**:
1. baseline = `M0_median_3`
2. **각 commodity별**:
   - `baseline_sse = Σ(y_true − baseline_pred)²`
   - 각 후보(M1~M5 전체 24개)에 대해:
     - `candidate_sse = Σ(y_true − candidate_pred)²`
     - `improvement = (baseline_sse − candidate_sse) / baseline_sse`
     - `improvement > 0.15` 이면 eligible
   - eligible 중 SSE 최소 후보 선택
   - eligible 없으면 baseline 유지
3. route RMSE 체크 없음

---

## 8. Phase 4: Route 집계 (`allocator.py`)

Phase 3에서 선택된 commodity 예측값을 route로 집계.  
M4 후보들은 내부적으로 이미 `allocate_route_residual()`을 거친 값.

```python
def route_aggregate(
    commodity_preds: pd.Series,   # index: (origin, destination, commodity)
) -> pd.Series:                   # index: (origin, destination)
    return commodity_preds.groupby(["origin", "destination"]).sum()
```

---

## 9. Phase 5: 평가 및 MLflow (`metrics.py` + `run_experiment.py`)

### 지표

```python
def wrmse(y_true, y_pred):
    """tons-weighted RMSE. weights = y_true / Σy_true."""
    w = np.clip(y_true, 0, None) / (np.clip(y_true, 0, None).sum() + 1e-9)
    return float(np.sqrt(np.sum(w * (y_true - y_pred) ** 2)))

def weighted_wrmse(split_results, weights=[0.2, 0.3, 0.5]):
    return sum(w * r for w, r in zip(weights, split_results))

def economic_benefit_M(y_true, y_pred_new, y_pred_base):
    """단위: M$ (백만 달러)."""
    err_base = np.mean(np.abs(y_true - y_pred_base))
    err_new  = np.mean(np.abs(y_true - y_pred_new))
    return (err_base - err_new) * 1000 * 10 / 1e6
```

### MLflow 로깅 항목

```python
# params
mlflow.log_param("split", split_name)
mlflow.log_param("context_year", context_year)
mlflow.log_param("min_fy", 2017)          # M1~M4
mlflow.log_param("covid_filter", True)    # M2/M3/M4만
mlflow.log_param("margin", 0.15)
mlflow.log_param("m5_alpha", 0.15)
mlflow.log_param("m5_clip", 0.10)
mlflow.log_param("cap_formula", "max(3*base+100, 1000)")   # M4

# metrics per split
mlflow.log_metric(f"{split}_val_wrmse", val_wrmse)
mlflow.log_metric(f"{split}_test_wrmse", test_wrmse)
mlflow.log_metric(f"{split}_val_rmse", val_rmse)

# weighted metrics
mlflow.log_metric("weighted_val_wrmse", weighted_val_wrmse)
mlflow.log_metric("weighted_test_wrmse", weighted_test_wrmse)
mlflow.log_metric("economic_benefit_M", eco_benefit)

# artifacts
mlflow.log_artifact("val_predictions.csv")
mlflow.log_artifact("test_predictions.csv")
mlflow.log_artifact("candidate_selection.json")   # {commodity: best_model}
mlflow.log_artifact("m5_route_scores.csv")        # repeated_error / composition_shift scores
```

---

## 10. 메인 루프 (`run_experiment.py`)

```python
for split_name in ["split_1", "split_2", "split_3"]:
    # Phase 0: 데이터 로드
    train_df, context_df, val_comm, val_route, test_comm, test_route = load_split(split_name)
    context_year = SPLITS[split_name]["context"]

    # Phase 1: 피처 빌드
    feature_abs  = build_all_features(train_df, context_df, context_year, EXTERNAL_DATA_DIR)      # A+B+C+D_abs
    feature_m5   = build_m5_features(train_df, context_df, context_year, EXTERNAL_DATA_DIR)       # B+C+D_ratio

    # Phase 2: 후보 생성 (val용)
    candidates_val = {}
    for model in [m1, m2, m3, m4]:
        model.fit(train_df, context_df, feature_abs, context_year)
        candidates_val.update(model.predict(feature_abs))
    m5.fit(train_df, context_df, feature_m5, context_year)
    candidates_val.update(m5.predict(feature_m5))
    # candidates_val: 24개 후보

    # Phase 3: Greedy 선택
    selection = greedy_select(candidates_val, val_comm, margin=0.15)
    val_comm_pred = apply_selection(candidates_val, selection)

    # Phase 4: Route 집계
    val_route_pred = route_aggregate(val_comm_pred)

    # Phase 5: 평가
    val_wrmse = wrmse(val_route["tons"].values, val_route_pred.values)

    # Test 예측 (결정 #3: train+context 재학습)
    all_data = pd.concat([train_df, context_df])
    feature_abs_test = build_all_features(all_data, None, context_year, EXTERNAL_DATA_DIR)
    feature_m5_test  = build_m5_features(all_data, None, context_year, EXTERNAL_DATA_DIR)
    for model in [m1, m2, m3, m4]:
        model.fit(all_data, None, feature_abs_test, context_year)
    m5.fit(all_data, None, feature_m5_test, context_year)
    # → 동일 selection 적용하여 test 예측
```

---

## 11. 구현 순서

| 순서 | 파일 | 주요 내용 |
|---|---|---|
| 1 | `config.py` | 경로, split 정의, 상수 |
| 2 | `data_loader.py` | split CSV 로드, commodity pivot |
| 3 | `features/lag_trend_features.py` | 전체 history 사용, context_year branching 없음 |
| 4 | `features/context_features.py` | context 집계 |
| 5 | `features/structure_features.py` | 구조 피처 |
| 6 | `features/external_features.py` | `load_external.py` 래핑 (D_abs) |
| 7 | `features/external_ratio_features.py` | ratio3 로더 (D_ratio, yulim `load_external_ratios` 이식) |
| 8 | `candidates/m0_baseline.py` | 순수 통계 후보 (외부 없음) |
| 9 | `candidates/m1_statistical.py` | m0 + 외부 ratio factor 조정 |
| 10 | `candidates/m4_dongbin.py` | dongbin 로직 이식 |
| 11 | `candidates/m2_bayasgalan.py` | bayasgalan 로직 이식 |
| 12 | `candidates/m3_mjwoon.py` | mjwoon 로직 이식 |
| 13 | `candidates/m5_yulim.py` | yulim 로직 이식 (medmean 베이스라인 + uniform log-ratio LGBM) |
| 14 | `selector.py` | greedy SSE + margin (M0~M5 통합) |
| 15 | `allocator.py` | route residual 재배분 |
| 16 | `metrics.py` | wRMSE, M$ |
| 17 | `run_experiment.py` | MLflow 통합 진입점 |

---

## 12. 주의사항

| 항목 | 내용 |
|---|---|
| `building_permits` | `data/additional_data/`에 없음 → 제외, 14개 CSV만 사용 |
| split CSV 경로 | `data/faf4_faf5_fixed_year_splits/split_{n}/train.csv` (로컬 파일 확인 필요) |
| context_year=2020 | split_2 context; 외부 데이터 2020 결측은 보간 처리됨 |
| commodity 코드 | FAF 1~43 정수, 파일 간 타입 통일 필요 |
| 예측값 음수 방지 | cap 적용 전 `clip(lower=0)` |
| `min_fy=2017` | M2~M5 **ML 학습 샘플**에서만 적용; 피처 계산(B 그룹)은 전체 history 사용 |
| context_year branching | 피처 계산에서 제거됨 — `lag_trend_features.py`에 연도 필터 추가하지 말 것 |
| M5 학습 샘플 기간 | pseudo backtest 제거; train 전체를 학습 샘플로 사용 |
| D_ratio base_year | ratio3 join key = `base_year = context_year` (tgt_year − 3); split_2의 경우 2020 결측이므로 fillna(1.0) |
| M5 COVID | M5는 COVID 필터 없음; split_2 context=2020인 상황에서 ratio3 2020 결측 → fillna(1.0) |
| M1 외부 factor | context_year − 3이 데이터 범위 밖이면 factor = 1.0으로 fallback |
| M5 additional_data 경로 | `data/additional_data/` (yulim 원본은 `additional_dataset/`도 fallback 탐색) |

---

## 13. 향후 실험 계획

초기 통합 파이프라인 결과 확인 후 적용 검토할 기법들. (yulim 원본에서 분리)

### EXP-A: Adaptive Soft Gate

route별 ML 보정 강도를 pseudo backtest 오차 점수에 비례하여 가변 설정.

```python
# pseudo backtest: train 내 마지막 2년을 holdout으로 사용
repeated_error_score = mean_abs_error_across_pseudo_folds(per_route)
repeated_error_score_normalized = min_max_normalize(repeated_error_score)

effective_alpha = 0.15 × repeated_error_score_normalized   # [0, 0.15]
final = (1 - effective_alpha) × base_pred + effective_alpha × ml_pred
```

기대 효과: 반복 오차가 큰 route에 보정을 집중하고, 안정적인 route의 과보정을 방지.

### EXP-B: 3단계 Commodity 모델 선택

commodity별 학습 데이터 충분도와 오차 기여도를 기준으로 모델 레벨 자동 결정.

```python
abs_log_error = |log1p(actual) − log1p(pred)|
error_share = commodity_abs_log_error / total_abs_log_error

if train_rows >= 5000 and error_share >= 0.01:
    level = "commodity-specific"   # per-commodity LGBM
elif group_train_rows >= 5000:
    level = "commodity-group"      # per-group LGBM
else:
    level = "baseline-only"        # 보정 없음
```

기대 효과: 데이터 부족 commodity의 과적합 방지; 오차 기여가 작은 commodity에 ML 낭비 방지.
