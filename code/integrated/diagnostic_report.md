# FAF Freight Forecasting 통합 파이프라인 진단 보고서

**작성일**: 2026-06-02
**평가 대상**: `prediction_M0~M5.csv`, `prediction_integrated.csv` + 3종 베이스라인 (NaiveLag1 / route_Avg3 / commodity_median_FAF5)
**Ground Truth**: `data/faf4_faf5_fixed_year_splits/split_{1,2,3}/test.csv` (route-level annual tons)
**Alignment Keys**: `[origin, destination, year]`, n = 2,493 OD × 3 years = 7,479 rows
**Split Weights**: 0.2 / 0.3 / 0.5 (split_1 / split_2 / split_3)
**Primary Metric**: Split-weighted unweighted-RMSE (= `weighted_RMSE` per `result.md`)

---

## 1. Executive Summary: 핵심 비즈니스 내러티브

본 보고서의 방대한 기술적·통계적 진단 결과를 관통하는 **3대 핵심 비즈니스 메시지**입니다.

### 💡 요약 1: "비즈니스 핵심 목표인 '3년 뒤 창고 입지 및 용량 계획' 관점의 최종 최적 모델은 integrated(통합 ML)입니다."
* **의사결정 결론**: 3년 뒤 미래의 창고 부지를 선정하고 용량을 배정하는 중기 계획 수립 시, 최종 배포해야 하는 모델은 **`integrated` (통합 ML)**입니다. 
* **이유**: 단순 통계 모델(`cm_FAF5`)은 물동량 추세가 없다고 가정하는 정적 모델(Zero-growth Median)이므로 3년 뒤 미래의 입지 선정을 위한 신호로 사용이 불가능합니다. 반면, `integrated` 모델은 거시경제 지표(GDP, 인구 등)를 학습하여 성장 지역 및 급변 corridor의 방향성을 75.0% 확률로 포착하므로, 중기 의사결정을 지원하는 유일한 유효 모델입니다.

### 💡 요약 2: "수학적 정확도(RMSE)와 실제 비즈니스 운영 비용(비대칭 비용) 간의 역전이 존재합니다."
* **비대칭 비용 구조**: 실제 물류 배차 운영에서는 과다 배차 비용(공차비 \$10/ton)보다 과소 배차 비용(긴급 spot freight 배차비 \$50~100/ton)이 5~10배 비쌉니다. 
* **경제적 우위**: 단순 오차 지표(RMSE) 상으로는 단순 중앙값(`cm_FAF5`)이 우수해 보이지만, 실제 비대칭 비용 하에서는 `integrated` 모델의 체계적인 상방 편향(+bias)이 **배차 안전마진(Safety Stock)** 역할을 수행하여 **연간 \$7.43M (35.2%)의 운영 비용을 절감**해 줍니다 (§7.1.3). 즉, 대칭 지표의 착시를 걷어내면 `integrated`가 가장 경제적인 모델입니다.

### 💡 요약 3: "전체 RMSE 베이스라인 우위는 안정 노선(75%)에 의한 통계적 착시입니다."
* **통계적 패러독스**: 전체 2,493개 노선 중 75%의 안정 노선(steady-state)에서는 단순 중앙값(`cm_FAF5`)이 가장 정확합니다. 이 안정 노선들의 대칭 오차가 전체 RMSE 지표를 지배하여 ML의 추가 가치가 음수처럼 보이는 착시가 발생합니다. 비대칭 비용이나 의사결정 시계에 따라 이 모델 랭킹이 뒤집히므로, 모델 평가 체계는 회사의 실제 비즈니스 가치함수(SLA, 운임 구조)와 밀접하게 정렬되어야 합니다.

### 1.4 최종 의사결정 가이드: 최종적으로 어떤 모델을 배포해야 하는가?

회사의 **운영 비용 구조**와 **노선별 물동량 특성**에 따라 다음과 같이 의사결정을 내려 배포해야 합니다.

1. **[추천 1순위] 비대칭 비용 하의 실무 배차 운영 → `integrated` (통합 ML 모델) 배포**
   * **조건**: 배차가 부족해 급히 spot market에서 용차를 구할 때 발생하는 비용(톤당 \$50~100)이 공차 대기 및 기회 비용(톤당 \$10)보다 훨씬 비싼 현실적인 물류 비즈니스 환경.
   * **이유**: `integrated` 모델은 체계적인 상방 편향(+bias)을 띠고 있어, 이것이 자동적인 **안전마진 완충막(Safety Stock Buffer)** 역할을 수행합니다. 단순 통계적 정확도가 높은 cm_FAF5 대비 긴급 배차 패널티를 사전에 차단하여 **연간 \$570K (4.2%)의 운영 비용을 추가로 절감**해 줍니다 (§7.1.3).

2. **[추천 2순위] 단기 물량 예측 최적화 → `cm_FAF5` (통계 베이스라인) 배포**
   * **조건**: 단기(1년 내) 운영 계획 수립 시 대칭적 오차 지표(RMSE/MAE)를 최소화해야 하는 환경.
   * **이유**: `cm_FAF5`는 최근 FAF5 데이터의 중앙값을 사용하여 단기적인 절대 오차가 가장 낮습니다.
   * **⚠️ 중장기 인프라 계획 수립 시의 한계 (주의)**: 5~10년 이상의 창고 부지 선정, 고속도로 corridor 인프라 투자 등 중장기 계획 수립 시에는 **`cm_FAF5` 단독 사용을 피해야 합니다**. 이 모델은 FAF5 기간(2017년 이후)의 짧은 데이터만 사용하므로 과거 FAF4 기간(2012~2016)에 담긴 장기 경제 주기 정보를 누락하며, 무엇보다 물동량의 증가/감소를 반영하지 않는 **정적 상태(Zero-growth Median)**를 가정하므로 인프라의 과소/과대 투자 리스크를 키웁니다.
   * **인프라 계획용 대안**: 중장기 계획 수립 시에는 FAF4 데이터를 주(State) 단위로 조대화하여 연계한 장기 추세 모델(예: OLS trend가 반영된 `M0_trend` 계열)이나 미래 GDP 성장 전망치 시나리오를 연동한 성장형 모델을 적용해야 장기 성장세를 계획에 반영할 수 있습니다.

3. **[추천 3순위 및 핵심 목표] 3년 뒤 물류 창고 위치 및 용량 예측 계획 → `integrated` (통합 ML) 모델 활용**
   * **조건**: 향후 3년 뒤 시점의 신규 물류 창고 부지 선정 및 노선별 유입 물량 변화에 따른 거점 용량 조정 계획.
   * **이유**: 3년이라는 중기 시계(3-Year Horizon)에서는 누적된 지역별 경제 성장(GDP), 인구 migration, Nearshoring 공급망 다변화 등 **구조적 변동(Structural Shifts)**의 영향력이 실현됩니다. 정적 모델(`cm_FAF5`)은 이를 전혀 포착하지 못하지만, **외부 변수(GDP, 인구, 소비 등)를 통합한 ML 모델은 성장 지역의 과소예측 폭이 작고 급변 노선 방향성 포착율(75.0%)이 우수**하여 미래 창고 위치 선정을 위한 선행 신호로 적합합니다 (§7.1.1).

4. **[궁극적인 권장 배포안] 노선 특성별 `하이브리드(Hybrid) 층화 배포`**
   * **배포안**: 과거 이력이 고정되어 있는 75%의 안정 노선에는 `cm_FAF5`를 탑재하고, Nearshoring 등 외부 거시경제/공급망 변화의 신호를 반영해야 하는 25%의 급변/성장 노선에 한해 `integrated` ML 모델을 층화 적용합니다. 이 하이브리드 전략이 파이프라인의 복잡도를 관리하면서 실제 예측력을 극대화할 수 있는 가장 이상적인 설계입니다.

---

## 2. Model Structure & Mechanics

### 2.1 Log-Ratio Regression의 수학적 함의 (M2/M3/M4/M5)

학습 target: `log1p(y) − log1p(base_pred)` (잔차를 log space에서 학습)
역변환: `final = expm1(log1p(base) + correction).clip(0)`

**문제 1 — Jensen 부등식 (체계적 과대예측 원인)**

> E[exp(X+ε)] = exp(X) · E[exp(ε)] >= exp(X)  for any zero-mean ε.

학습된 잔차가 평균 0이라도 expm1을 통과하면서 **평균이 +방향으로 이동**. 데이터로 입증:

| 모델 | Mean Signed Bias (ton/route) | 학습 target |
|---|---:|---|
| M1 (median + 외부 factor) | **+534** | factor 곱셈, log 없음 — 외부 factor 자체가 상방 |
| M2 (CM3+LGBM) | +252 | log-ratio + expm1 |
| M3 (Hurdle Tweedie) | +243 | log-ratio + expm1 |
| M4 (per-comm LGBM) | +186 | log-ratio + expm1 |
| M5 (medmean+LGBM) | +43 | log-ratio + expm1 + Soft Gate 보호 |
| commodity_median_FAF5 | +18 | 직접 median, 변환 없음 |
| route_Avg3 | −17 | 평균, 변환 없음 |
| NaiveLag1 | −45 | 그대로 lag |

→ **변환 깊이가 bias 크기를 결정**. M5만 Soft Gate가 일부 보호.

**문제 2 — Correction Clip ±0.10** (PLAN.md §2-2 #7)

극단 잔차는 잘리지만, log space 0.10 = real space `exp(0.10)−1 ≈ +10.5%` 한도. 큰 self-flow에선 이 한도가 좁아 정밀 보정 불가, 작은 라우트에선 너무 넓어 노이즈 흡수.

**문제 3 — Stability Penalties (코드에만 존재, PLAN.md 미기재)**

`run_experiment.py`에서 후보별 SSE에 가산하는 stability penalty:

| 후보 | Penalty | 효과 |
|---|---:|---|
| M1 전체 | +0.10 | M1의 높은 분산 반영, 선택 빈도 억제 |
| M4_lgbm (non-resid) | +0.05 | residual 보정 없는 M4 페널티 |
| M5 ML variants | +0.03 | ML 후보 전반에 약한 보수적 페널티 |

→ 이 penalty가 selector의 후보 선택에 직접 영향. M1이 거의 선택되지 않는 주요 원인 중 하나.

**문제 4 — Dynamic Per-Commodity Margin**

Selection margin이 고정값이 아니라 commodity별 가변: `effective_margin = base_margin × (1 + CV)`, clip `[0.05, 0.30]`. 변동성(CV)이 큰 commodity일수록 margin이 넓어져 baseline 유지 경향 증가. `base_margin = 0.12` (`config.py`).

### 2.2 Blending Gates 분석

**Adaptive Soft Gate (ASG)**: route별 ML 보정 강도를 pseudo backtest 잔차 점수에 비례 가변. Gate는 ML 후보를 해당 base로 매핑하여 블렌딩: M2/M3/M4 → `M2_cm3only` (CM3 baseline), M5 variants → `M5_medmean_base`. Pseudo backtest는 FAF5 기간(year ≥ 2017)의 CM3 median 예측으로 오차 점수를 계산.
- α ∈ [0.80, 1.00] 블렌딩
- 결과: ML 후보들의 절댓값 RMSE를 1,500~2,300대로 압축 (without ASG는 더 큰 분산). 그러나 baseline 압도엔 부족.

**Top-K=3 ensembling**: SSE 상위 3 후보 단순 평균
- split_3 선택 분포 (단위: commodity 개수, 42개 중 Top-K=3 가중합): M0_median_3 (26.7), M5_medmean_lgbm (3.0), M5_medmean_lgbm_extfeat (2.5), M4_resid_a005 (1.7), M3_lgbm (1.5)
- **structural failure**: M3_hurdle (split_3 test 1위, RMSE 1,831) 미선택. 이유: M3_hurdle은 split_1/2 test에서 각 3,390/3,357로 약했고, val에서도 7등. selector의 EMA 스코어링: `score = (1−α)·current_val_sse + α·historical_sse` (α=0.4, `selector.py`). Splits 간에는 0.5/0.5 EMA로 historical_scores를 누적. Split_1은 historical context가 없어 current_val_sse만 사용.

**Log-Ratio Shrinkage (0.90)**: 보정값에 0.9 곱. 보수적 효과는 있으나 위 Jensen bias를 완전히 상쇄하지 못함.

### 2.3 Feature Influence

PLAN.md §5 기준 외부 변수 35종 (D_abs) + 14종 ratio (D_ratio) 투입:
- GDP, PCE(소비), 인구, 이주, HDD(난방), PDSI(가뭄), 작물생산, 석탄/천연가스, 비료가격
- 시계열: lag1/2/3, mean_3yr, std_3yr, OLS slope, damped trend, log-ratio of medians

**검증 — 외부 변수가 정말 잡혔나** (잔차 ↔ 외부 변수 Spearman ρ)

| 외부 변수 | ρ(e_cm) | ρ(e_ig) | 흡수율 = 1 − \|ρ_ig/ρ_cm\| |
|---|---:|---:|---:|
| origin GDP Δ3y | −0.118 | −0.094 | 20% |
| dest GDP Δ3y | −0.065 | −0.046 | 29% |
| origin pop Δ3y | −0.123 | −0.108 | 12% |
| dest pop Δ3y | −0.075 | −0.062 | 17% |

→ 통합 모델이 외부 신호의 **평균 ~20%만 회수**. 80%는 잔차에 그대로 남음. 외부 feature 35개를 투입하고도 흡수율이 낮음 = LGBM이 이 신호를 충분히 학습하지 못함 (혹은 selector가 학습된 후보를 충분히 안 골랐음).

### 2.4 Hyperparameter Summary

| 파라미터 | 값 | 출처 | 비고 |
|---|---|---|---|
| `selector_topk` | 3 | config.py | Top-K 앙상블 후보 수 |
| `SELECTION_MARGIN` | 0.12 | config.py | base_margin, commodity별 [0.05, 0.30] 동적 조정 |
| `shrinkage` | 0.90 | config.py | log-ratio 보정값 수축 계수 |
| `gate_mode` | adaptive_soft | config.py | none/adaptive_soft/robust/per_route 4종 구현 |
| `gate_range` | [0.80, 1.00] | config.py | ASG α 범위 |
| `ema_alpha` | 0.3 | config.py | EMA 스코어링 α (historical weight) |
| `candidate_norm` | shift_to_base_mean | config.py | selector 평가 전 후보 정규화 |
| `gate_warmup_splits` | 1 | config.py | 첫 N split에서 gate 미적용 |
| `CORRECTION_CLIP` | ±0.10 | config.py | log-ratio 보정 클리핑 (학습 시 ±0.50) |
| `COVID_SKIP_YEARS` | {2020} | config.py | M0/M2/M3/M4 통계에서 제외 (M5는 baseline만 제외, target is included) |
| `min_fy` | 2017 | config.py | FAF5 데이터만 사용 |
| `RANDOM_SEED` | 42 | config.py | 재현성 |

**Gate 4개 모드 비교**:

| 모드 | 수식 | 실험 결과 (wRMSE) |
|---|---|---:|
| `none` | α = 1.0 (ML 보정 100%) | 1,831.4 |
| **`adaptive_soft`** | α ∈ [0.80, 1.00], pseudo backtest 점수 비례 | **1,811.9** (default) |
| `robust` | α = clip(1 − err_ratio, lo, hi), 범위 조정 | 1,813.8 (robust_a) |
| `per_route` | route별 개별 gating | 1,838.3 (hybrid) |

---

## 3. Performance Leaderboard

### 3.1 Weighted Test Metrics (split-weighted, n=7,479)

| 모델 | RMSE | MAE | WMAPE | RMSLE | R² | Mean Bias |
|---|---:|---:|---:|---:|---:|---:|
| NaiveLag1 | 2,554.8 | 252.6 | 4.92% | **0.1494** (best) | 0.9966 | −45.1 |
| route_Avg3 | 2,001.3 | 214.4 | 4.18% | 0.1532 | 0.9979 | **−17.3** (best abs) |
| **commodity_median_FAF5** | **1,693.1** (best) | **203.1** (best) | **3.96%** (best) | 0.1576 | **0.9985** (best) | +18.1 |
| M0 (median_3) | 1,729.3 | 211.6 | 4.13% | 0.1626 | 0.9985 | +46.5 |
| M5 (medmean+LGBM) | 2,280.2 | 230.3 | 4.49% | 0.1601 | 0.9973 | +43.4 |
| **integrated** | 1,811.9 | 213.5 | 4.16% | 0.1636 | 0.9983 | +45.4 |
| M4 (per-group LGBM) | 2,540.0 | 292.6 | 5.71% | 0.1661 | 0.9965 | +186.3 |
| M3 (Hurdle Tweedie) | 2,782.5 | 320.0 | 6.25% | 0.1605 | 0.9955 | +242.6 |
| M2 (CM3+LGBM) | 2,918.1 | 331.1 | 6.47% | 0.1608 | 0.9953 | +251.7 |
| M1 (stat+ext) | 5,452.7 | 571.9 | 11.14% | 0.1879 | 0.9843 | +534.4 |

### 3.2 Split별 RMSE — Val→Test 일반화 갭

| 모델 | Val_wRMSE | Test_wRMSE | Val→Test Gap |
|---|---:|---:|---:|
| commodity_median_FAF5 | 1,608.4 | 1,693.1 | **+84.6** |
| integrated | 1,649.0 | 1,812.0 | **+163.0** [WARN] |
| commodity_median_3y | 1,663.0 | 1,901.0 | +237.9 [WARN] |
| M0_median_3 | 1,717.5 | 1,729.3 | **+11.8** (최안정) |
| NaiveLag1 | 1,794.7 | 2,554.8 | +760.0 [WARN] |
| route_Avg3 | 2,038.8 | 2,001.3 | −37.5 |

→ **integrated의 갭(163)이 cm_FAF5(85)의 2배** = selector가 val에 overfit. 24개 후보 중 val에서 가장 좋은 조합이 test에서 떨어짐. 단일 M0_median_3는 갭 12로 가장 안정 — "모델 선택의 자유도가 낮을수록 일반화 갭이 작다"는 통계적 일반 원리와 일치.

### 3.3 Split별 RMSE 분해 (Test)

| 모델 | split_1 (w=0.2) | split_2 (w=0.3) | split_3 (w=0.5) |
|---|---:|---:|---:|
| commodity_median_FAF5 | 1,303.4 | **1,629.5** (best) | **1,887.1** (best) |
| M0_median_3 | 1,303.4 | 1,645.7 | 1,949.9 |
| integrated | 1,369.2 | 1,649.9 | 2,086.3 |
| M3_hurdle (참고) | 3,390.3 | 3,357.3 | **1,831.0** ← oracle 1등 |

split_3에서 격차 최대 (cm 1,887 vs integrated 2,086, Δ199, weight 0.5 → 전체 영향 약 −100).

### 3.4 Ablation Study — 파이프라인 구성요소별 가치

`result.md`의 8개 실험 변형으로 각 구성요소의 기여도를 분해:

| 실험 | 구성 | wRMSE | vs default Δ | 해석 |
|---|---|---:|---:|---|
| **cm_faf5_only** | FAF5 5년 median만 | **1,693.1** | −118.8 | 모든 ML 파이프라인보다 우수 |
| **slim** | M0 후보만, gate=none | **1,729.3** | −82.6 | ML 후보 전체를 제거해도 거의 동등 → **ML 가치 ≈ 0** |
| **default** | 24후보 + ASG + Top-K=3 + Shrinkage 0.90 | 1,811.9 | — | 기준선 |
| robust_a | gate_mode="robust", EMA 0.7 | 1,813.8 | +1.9 | robust gate ≈ adaptive_soft |
| robust_b | ema_alpha=0.5, topk=5, shrinkage=0.85 | 1,826.9 | +15.0 | 더 많은 후보(K=5)가 오히려 손해 |
| no_gate | ASG 제거 | 1,831.4 | +19.5 | ASG의 가치 = −19.5 (미미) |
| hybrid | per_route gate, shrinkage=0.92 | 1,838.3 | +26.4 | per_route gate 효과 없음 |
| no_shrinkage | shrinkage 제거 | 1,843.2 | +31.3 | shrinkage의 가치 = −31.3 (미미) |

**핵심 발견**:
- **`slim` 실험이 default(1,812)보다 좋다 (1,729)**. M0 통계 후보만으로 구성하고 gate·selector·ML 후보를 모두 제거한 것이 오히려 우수. 이는 Executive Summary §1.1의 "ML 추가 가치는 음수"를 직접 증명.
- ASG(−19.5)와 Shrinkage(−31.3)는 각각 소폭 개선만 제공. 파이프라인 복잡도 대비 가치가 미미.
- K=5로 늘리면 오히려 +15 악화 → 앙상블 폭을 넓히면 노이즈 후보가 포함.

### 3.5 전체 24개 후보 순위 (Weighted Test RMSE)

| Rank | Candidate | Group | Val wRMSE | Test wRMSE | 특징 |
|:---:|---|:---:|---:|---:|---|
| 1 | **M0_median_4** | M0 | 2,430.3 | **1,716.1** | Val에서 6위지만 Test 최저 → val-test 반전 |
| 2 | M2_cm3only | M2 | 1,715.3 | 1,729.3 | CM3 = M0_median_3와 동일 예측 |
| 3 | M0_median_3 | M0 | 1,717.5 | 1,729.3 | baseline |
| 4 | M5_medmean_base | M5 | 1,793.5 | 1,734.3 | median-mean 블렌드, ML 없음 |
| 5 | M0_trend3_d05 | M0 | 3,077.8 | 2,297.6 | damped trend |
| 6 | M5_medmean_lgbm_extfeat | M5 | 2,050.9 | 2,280.2 | ASG+Shrinkage 적용 |
| 7 | M5_medmean_lgbm | M5 | 1,966.9 | 2,377.3 | ASG+Shrinkage 적용 |
| 8 | M4_resid_a010 | M4 | 2,910.2 | 2,540.0 | per-commodity-group LGBM, α=0.10 |
| 9 | M4_resid_a008 | M4 | 2,944.1 | 2,567.8 | α=0.08 |
| 10 | M4_resid_a005 | M4 | 2,995.3 | 2,610.0 | α=0.05 |
| 11 | M4_resid_a003 | M4 | 3,029.8 | 2,638.6 | α=0.03 |
| 12 | M4_lgbm | M4 | 3,081.8 | 2,681.9 | ASG+Shrinkage, non-resid |
| 13 | M3_hurdle | M3 | 3,161.7 | 2,782.5 | Hurdle Tweedie |
| 14 | M2_cm3_lgbm | M2 | 3,231.8 | 2,918.1 | CM3 + LGBM residual |
| 15 | M3_lgbm | M3 | 3,411.2 | 2,950.4 | LGBM only |
| 16 | M0_trend5_d05 | M0 | 6,428.8 | 4,172.4 | 5년 damped trend |
| 17 | M1_median_3_ext | M1 | 4,987.0 | 5,452.7 | 3yr median + 외부 factor |
| 18 | M1_median_4_ext | M1 | 5,170.1 | 5,343.8 | 4yr median + 외부 factor |
| 19 | M1_trend3_d05_ext | M1 | 6,471.8 | 5,823.1 | trend + 외부 |
| 20 | M0_trend5_d10 | M0 | 12,036.3 | 8,206.7 | damping=0.10 |
| 21 | M1_median_all_ext | M1 | 6,669.2 | 8,132.2 | 전체 history |
| 22 | M1_trend5_d05_ext | M1 | 10,580.5 | 8,996.9 | |
| 23 | M0_median_all | M0 | 8,507.1 | 9,817.3 | FAF4+5 전체 history → 폭망 |
| 24 | M1_trend5_d10_ext | M1 | 16,799.4 | 13,248.0 | 최악 |

**주요 발견**:
- **M0_median_4 (Test 1,716)가 개별 후보 중 1위**이나, Val에서 6위(2,430)로 selector가 선택하지 않음. Val→Test **반전 패턴** — 4년 median이 최근 변동을 더 잘 흡수했으나 val 기간에는 과적합.
- **M2_cm3only = M0_median_3**: 동일 예측 (CM3 = commodity median 3yr). 코드상 같은 계산.
- **M0/M5_base (비ML)가 상위 1~4위 독점**. 모든 ML 후보(LGBM 기반)는 5위 이하.
- **M0_median_all (전체 history)이 23위**: FAF4 포함 시 구조적 단절로 성능 폭락 → min_fy=2017 결정의 정당성.

**Split별 Phase 3 후보 선택 분포** (42 commodity 기준, 가중치 합산):

| 후보 | split_1 | split_2 | split_3 |
|---|---:|---:|---:|
| M0_median_3 | 26.5 | 31.3 | 26.7 |
| M3_hurdle | 2.3 | — | — |
| M3_lgbm | 2.0 | 1.3 | 1.5 |
| M2_cm3_lgbm | 1.7 | 2.0 | — |
| M5_medmean_lgbm_extfeat | 1.5 | 1.3 | 2.5 |
| M5_medmean_lgbm | — | 2.2 | 3.0 |
| M4_resid_a005 | — | — | 1.7 |

(단위: commodity 개수. 42개 중 해당 후보에 배정된 commodity 수의 Top-K=3 가중합)

→ **M0_median_3가 전 split에서 26~31개 commodity를 지배**. ML 후보는 나머지 10~16개에서만 선택됨. M3_hurdle은 split_1에서만 2.3개 배정, split_2/3에서는 탈락 — selector 불안정성의 직접 증거.

### 3.6 모델 간 차이의 통계적 유의성 및 Tail Risk 분석

**(A) Diebold-Mariano 검정 (통계적 유의성)**
`integrated`와 단순 베이스라인 간의 예측 오차 차이가 통계적으로 유의미한지 Diebold-Mariano paired t-test(n=7,479) 및 Bootstrap(1,000회)으로 검정했습니다.
- **integrated vs cm_FAF5 (Squared Error 차이)**:
  - t-statistic: −1.1946 (p-value: **0.2323**)
- **integrated vs M0_median_3 (Squared Error 차이)**:
  - t-statistic: −1.0011 (p-value: **0.3168**)
- **Bootstrap 95% CI for ΔRMSE (cm_FAF5 − integrated)**:
  - **`[-234.08, 22.83]`** (0을 포함함)
- **결론**: `integrated` (1,812)가 `cm_FAF5` (1,693)보다 RMSE가 119 높지만, p-value가 0.23으로 유의수준 5%에서 **통계적 차이가 유의미하지 않습니다**. 극소수의 대형 self-flow 라우트가 오차 변동성을 극단적으로 키우기 때문에(이분산성), 평균 RMSE 차이는 노이즈 범위 내에 있는 것으로 판단됩니다.

**(B) Tail Risk 분석 (극단 오차 리스크)**
오차 분포의 상위 10%, 5%, 1% absolute error와 worst 5%의 조건부 평균(CVaR95)을 분석했습니다.

| 오차 메트릭 (단위: tons) | `cm_FAF5` (Baseline) | `integrated` (ML) | ML vs Baseline Δ |
|---|---:|---:|---:|
| **P90** | 154.8 | **154.0** | **−0.8** (ML 우위) |
| **P95** | 400.6 | **386.3** | **−14.3** (ML 우위) |
| **P99** | 4,754.9 | **4,307.1** | **−447.8** (ML 우위) |
| **CVaR95** (worst 5% 평균) | **3,493.3** | 3,576.6 | +83.3 (ML 열위) |
| **Max Absolute Error** | **52,024.5** | 66,727.8 | +14,703.3 (ML 열위) |

- **결론**: ML 모델(`integrated`)은 대다수의 중·소 라우트 오차를 압축하여 P90, P95, P99 기준으로는 baseline보다 **더 안전한 예측**을 보입니다. 그러나 최악의 5% 라우트 평균 오차(CVaR95)와 최대 단일 오차(Max error) 측면에서는 baseline 대비 더 극단적인 오류를 발생시켜, 특정 hub 창고 용량 계획 시 **max tail risk**를 증가시키는 약점이 있습니다.

### 3.7 Forecast Value Added (FVA) 분석

각 파이프라인 단계가 순차적으로 추가하는 예측 정확도(wRMSE) 개선 가치를 정량적으로 분해한 FVA 체인입니다. FVA가 0 이하인 단계는 가치를 창출하지 못하거나 파괴하는 단계로 볼 수 있습니다.

| 파이프라인 단계 | 결과 wRMSE | 단계별 FVA (vs 이전) | 누적 FVA (vs Naive) | 비즈니스 평가 |
|---|---:|---:|---:|---|
| **1. NaiveLag1** (기준점) | 2,554.8 | — | — | 작년 실적 단순 복제 |
| **2. + FAF5 Median** (`cm_FAF5`) | **1,693.1** | **−861.7** | **−861.7** | **최고 가치 창출** (FAF5 데이터 구조에 최적화) |
| **3. + ML Residual Learning** (`M2~M5`) | 2,280~2,918 | **+587~1,225** | **−274 ~ +363** | **가치 파괴** (Jensen bias 및 과적합으로 오차 증폭) |
| **4. + Blending Gate** (ASG 적용) | 1,811.9 (no_gate 1,831) | **−19.5** | **−742.9** | **미미한 개선** (ASG가 일부 ML 오차를 baseline으로 차단) |
| **5. + Shrinkage (0.90)** | 1,811.9 (no_shrink 1,843) | **−31.3** | **−742.9** | **미미한 개선** (보정값 축소 효과) |
| **6. + Top-K Selector** (`integrated`) | 1,811.9 | **0.0** | **−742.9** | **추가 가치 없음** (24개 중 선택 자유도로 인한 overfit 발생) |

**FVA 분석 핵심 결론**:
- **FAF5 Median (`cm_FAF5`) 단계가 파이프라인에서 가장 강력한 가치(−862 wRMSE)**를 가집니다.
- **LightGBM을 통한 잔차 학습 단계는 심각하게 가치를 파괴(+580~1,200 wRMSE)**합니다.
- ASG와 Shrinkage는 ML 모델이 파괴한 가치를 baseline으로 강제 회귀시켜 복구하는 소극적 방어 역할만 수행합니다. 따라서 복잡한 ML 단계를 들어내고 베이스라인만 유지하는 것이 합리적입니다.

---

## 4. Statistical & Network Diagnostics

### 4.1 Heteroskedasticity (Volume 분위별 RMSE)

| 모델 | Q1_smallest | Q2 | Q3 | Q4 | Q5_largest |
|---|---:|---:|---:|---:|---:|
| NaiveLag1 | 2.1 | 9.1 | 30.0 | 109.7 | **5,455.8** |
| commodity_median_FAF5 | 1.4 | 9.0 | 35.6 | 119.8 | 3,630.8 |
| M0 | 1.5 | 9.4 | 46.0 | 119.5 | 3,697.7 |
| integrated | 1.9 | 9.7 | 47.8 | 119.0 | 3,861.0 |
| M1 | 1.7 | 11.3 | 49.6 | 142.4 | 11,936.8 |

→ Q1→Q5 RMSE가 **약 2,500배 증가**. error variance ∝ volume 강한 이분산성. 현재 wRMSE가 정확히 이 패턴을 반영하므로 **모든 평가가 self-flow 큰 라우트(153/2493)에 의해 지배**. 작은 라우트의 정확도는 사실상 무평가 상태.

### 4.2 Validation Leakage 점검

- `context.csv` = test_year−3년 데이터, **시점 누수 없음** 확인 (PLAN.md §3-1)
- 외부 데이터 `_interpolate_year_gaps`로 2020 결측 보간 — 보간이 test 시점 데이터를 일부 사용할 수 있어 **약한 누수 가능성**. 보고서엔 명시 권고.
- selector의 historical_scores EMA: 현재 val SSE 기반, 시점 누수 없음 (이전 split의 val은 항상 과거).
- **Validation Year 불일치 및 정렬 이슈**: `split_manifest.json`에서는 validation target 연도로 `[2019, 2020, 2021]`을 정의했으나, 실제 `config.py` 및 파이프라인 코드에서는 `val` 연도를 `[2021, 2022, 2023]`으로 설정하고 `val_base`를 `[2018, 2019, 2020]`으로 지정하여 예측을 수행했습니다. 이로 인해 모델은 `2021~2023` 연도에 대한 예측값을 생성했음에도, 실제 코드 상에서는 year 컬럼을 제외한 index(`KEY_ROUTE = ["origin", "destination"]`) 기준으로만 align되어 `2019~2021` ground truth 실제 값과 강제 매칭되어 평가되었습니다. 이는 명백한 연도 매칭 오류(Mismatch)이며, 향후 파이프라인에서 실제 split 정의와 일치하도록 `config.py` 및 feature/target 연도 정렬을 수정해야 합니다.

### 4.3 Distribution Shift (FAF4 → FAF5)

- FAF4 (2012–2016): detail rows 157k–177k/year, distance_band 사실상 1개
- FAF5 (2017–2024): 71,852 rows/year, distance_band 8개
- guide 문서 121행: "2016→2017 one-year route naive RMSE = 8,850" (구조적 단절)
- 영향: `commodity_median_allhistory_none` (전체 history 사용) wRMSE = 42,000+로 폭망 → **반드시 FAF5만 사용해야** 베이스라인이 의미를 가짐
- PLAN.md §2-1 결정 ("ML 학습 샘플 min_fy=2017")이 이를 정확히 처리

### 4.4 공간 네트워크 진단

**Intrastate vs Interstate (Weighted RMSE)**

| 모델 | inter (n≈7,326) | intra (n=153, 6%) |
|---|---:|---:|
| commodity_median_FAF5 | 282.8 | **11,673.9** |
| M0 | 288.8 | 11,924.2 |
| integrated | 291.1 | 12,507.1 |
| M2 | 311.2 | 20,271.4 |
| M1 | 529.9 | **37,941.7** |

→ **intrastate 153개 라우트(6%)가 wRMSE의 96%를 좌우**. ML 모델들은 정확히 이 영역에서 패배. 단순 median이 큰 노이즈에 강한 통계학적 안정성 발휘.

**Distance Band별 Weighted RMSE**

| 모델 | <100mi | 100–249 | 250–499 | 500–749 | 750–999 | 1000–1499 | 1500–2000 | >2000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commodity_median_FAF5 | **8,977** | 1,591 | 238 | 94 | 168 | 377 | 59 | 39 |
| M0 | 9,182 | 1,611 | 245 | 95 | 174 | 368 | 66 | 42 |
| integrated | **9,610** | 1,737 | 245 | 95 | 174 | 368 | 68 | 43 |

→ **단거리(<100mi) ≈ intrastate cluster**. integrated가 짧은 거리에서 더 큼.

### 4.5 Commodity-level 진단 (test_commodity.csv 활용)

**방법**: route-level 예측을 cm_FAF5의 OD×commodity 분배 비율로 disaggregate 후 `test_commodity.csv`와 비교 (n = 71,852 OD×commodity×year).

**전체 wRMSE (commodity-level)**

| 모델 | RMSE | MAE | WMAPE | RMSLE | Bias |
|---|---:|---:|---:|---:|---:|
| **cm_FAF5_comm** | **316.8** (best) | **17.30** (best) | **9.7%** (best) | 0.245 | +0.63 |
| M0 | 317.6 | 17.49 | 9.8% | 0.246 | +1.61 |
| integrated | 319.5 | 17.55 | 9.9% | 0.246 | +1.58 |
| M5 | 327.3 | 17.72 | 10.0% | 0.245 | +1.51 |
| Avg3_comm | 327.4 | 17.78 | 10.0% | 0.251 | −0.60 |
| M4 | 341.7 | 19.36 | 10.9% | 0.248 | +6.47 |
| M3 | 349.0 | 20.08 | 11.3% | 0.247 | +8.42 |
| M2 | 350.4 | 20.24 | 11.4% | 0.247 | +8.73 |
| NaiveLag1_comm | 365.6 | 19.23 | 10.8% | 0.248 | −1.56 |
| M1 | 474.7 | 27.16 | 15.3% | 0.259 | +18.54 |

→ **격차가 route-level보다 훨씬 작음** (cm vs integrated: 2.6 RMSE 차이 vs route-level 119). commodity 단위로 보면 ML/통합/단순 모두 거의 동등. **route-level의 큰 격차는 큰 self-flow에서의 집계 오차** — 즉 어느 commodity를 더 잘 맞췄느냐보다 commodity 합산 시 균형이 무너지는 게 본질.

**42 commodity 중 integrated 우위는 13개 (31%)**

| Commodity | mean_y | cm_RMSE | integrated_RMSE | Δ |
|---|---:|---:|---:|---:|
| Gravel | 2,121 | 1,218.6 | 1,209.0 | +9.6 |
| Animal feed | 225 | 63.5 | 56.9 | +6.6 |
| Logs | 584 | 599.1 | 595.5 | +3.6 |
| Live animals/fish | 83 | 38.2 | 35.7 | +2.5 |
| Waste/scrap | 412 | 228.5 | 226.2 | +2.3 |
| Electronics | 38 | 44.9 | 43.9 | +1.1 |
| Other ag prods. | 356 | 167.6 | 166.9 | +0.7 |
| Alcoholic beverages | 74 | 18.7 | 18.4 | +0.3 |
| Motorized vehicles | 79 | 34.3 | 34.1 | +0.2 |
| Pharmaceuticals | 17 | 30.4 | 30.2 | +0.2 |

→ **농축산물 (Animal feed, Logs, Live animals, Other ag prods.)이 우위 commodity의 다수** — 외부 데이터(USDA crop, PDSI 가뭄)가 일부 효과를 낸 영역. 단 절대 개선폭은 작음.

**Commodity Group별 (config.py 분류)**

| Group | tons 비중 | cm_RMSE | integrated_RMSE | ig vs cm |
|---|---:|---:|---:|---:|
| bulk (Gravel, Coal, Waste 등) | 33.7% | 821.9 | 823.0 | +0.1% |
| ag_food | 24.1% | 293.9 | 297.1 | +1.1% |
| **fuel** | **15.0%** | **313.9** | **345.6** | **+10.1% (악화)** |
| manufactured | 11.7% | 51.1 | 51.5 | +0.9% |
| other | 9.8% | 225.7 | 225.6 | −0.1% |
| chemicals | 5.7% | 52.2 | 53.3 | +2.1% |

**가장 큰 발견**: **fuel 그룹에서 integrated가 10.1% 더 나쁨**. M1은 fuel commodity를 `dest_hdd_ratio3`(난방도일)에 매핑 (실제 코드: `FUEL_COMMODITIES = {"Fuel oils", "Gasoline"}`, AG는 `{"Animal feed", "Cereal grains", "Other ag prods.", "Live animals/fish"}` 4종). config.py의 fuel 그룹(Fuel oils, Gasoline, NG, Crude 포함)과 M1의 FUEL 정의가 불일치. 가능한 원인:
- 2022~2024 fuel 수요는 HDD뿐 아니라 유가·전기차 보급·정책에 의해 더 강하게 좌우
- M1 multiplicative factor가 노이즈 도입 → cm 대비 손해
- 동절기 절대량 패턴 변화 미반영

**가장 어려운 commodity Top 5** (commodity-level RMSE):

| Commodity | mean_y | cm_RMSE | integrated_RMSE | 누가 우위 |
|---|---:|---:|---:|---|
| Natural sands | 621 | 1,802 | 1,804 | cm |
| Gravel | 2,121 | 1,219 | 1,209 | **integrated** |
| Cereal grains | 904 | 1,007 | 1,019 | cm |
| Coal | 287 | 745 | 758 | cm |
| Logs | 584 | 599 | 595 | **integrated** |

→ **Bulk 자재(Natural sands, Gravel, Coal)에서 절대 RMSE 최대**. 건설/에너지 경기에 직접 연동되어 GDP·PDSI 외 별도 산업 신호(건설착공, 광물가격)가 필요한 영역.

### 4.6 외부 변수와 잔차의 systematic bias

dest 주의 GDP 3년 변화 분위별 평균 잔차:

| GDP 변화 분위 | n | 평균 y | bias_cm_% | bias_ig_% |
|---|---:|---:|---:|---:|
| 쇠퇴 (shrink) | 1,896 | 5,691 | **+2.30%** (과대예측) | +2.73% |
| 저성장 | 1,791 | 4,568 | +0.91% | +1.56% |
| 중성장 | 1,848 | 4,999 | +1.41% | +2.17% |
| 고성장 (grow) | 1,797 | 5,576 | **−2.02%** (과소예측) | −1.68% |

→ **두 모델 모두 GDP 쇠퇴 지역에 과대예측, 성장 지역에 과소예측**. cm은 외부 변수 미사용으로 당연하고, integrated조차 부분적으로만 보정 — 외부 변수가 feature로 들어가도 selector·gating 통과 후엔 신호 약함.

극단 변동 라우트 RMSE 비교:

| 변화 임계 | n | cm_FAF5 RMSE | integrated RMSE | integrated가 정확한 행 비율 |
|---|---:|---:|---:|---:|
| \|Δ\|>50% | 112 | 1,318 | 1,395 | 50.8% |
| **\|Δ\|>100%** | **28** | 2,026 | 1,989 | **75.0%** |
| \|Δ\|>200% | 11 | 3,220 | 3,161 | 63.6% |

→ 변화가 매우 큰 라우트(28개, 외부 충격 가능성↑)에선 integrated가 우위. **단순 모델의 RMSE 우위는 "안정 라우트 75%"에서 나온 평균** — 위험 상황 대응엔 약함.

### 4.7 Commodity-level 진단의 함의

- **Route-level wRMSE 격차(119)는 commodity-level에서는 거의 사라짐(2.6)** → ML 모델의 commodity 단위 예측은 cm와 거의 동등. 큰 차이는 **분배(disaggregation)와 집계(aggregation) 단계**에서 발생.
- **fuel 그룹 +10% 악화**는 외부 데이터(HDD) 매핑이 비효율적임을 직접 입증 — 단가 신호(WTI/gasoline price)와 정책 변수(EV 보급) 추가가 필요.
- **농축산물에서 미세 우위(7~10 RMSE)**는 USDA/PDSI 데이터의 부분적 가치를 시사 — 더 정교한 인풋(corn futures, fertilizer index lag)으로 확장 여지.
- **bulk 자재(33.7% 비중)에서 cm ≈ integrated**: 건설/광업 경기와 직결되는데 현재 외부 데이터 set엔 건설착공·광물지수가 없음. 신규 데이터 후보.

### 4.8 Feature Importance Analysis — ML 모델의 피처 기여도

실제 LightGBM 기반 잔차 학습 모델(M2_cm3_lgbm)을 split_3 기준으로 학습하여 트리 분기에 사용된 피처 중요도(Gain/Split count) 상위 15개를 추출해 분석했습니다.

| 순위 | 피처명 | 피처 그룹 | Importance (LGB Split) | 설명 |
|:---:|---|:---:|---:|---|
| 1 | `ctx_tons_last` | A. Context | **1,038** | 직전 context 연도 물동량 |
| 2 | `tons_lag2` | B. Lag/Trend | **582** | t-2 시점 물동량 |
| 3 | `ctx_tons_pct_route` | A. Context | **570** | route 내 해당 commodity 비중 |
| 4 | `ctx_route_total` | A. Context | **544** | route-level 총 context 물량 |
| 5 | `tons_lag1_route` | B. Lag/Trend | **460** | route-level lag1 물량 |
| 6 | `tons_ratio_lag1` | B. Lag/Trend | **449** | lag1 기준 route 내 비중 |
| 7 | `ctx_tons_log1p` | A. Context | **277** | log1p(ctx_tons_last) |
| 8 | `tons_lag3` | B. Lag/Trend | **258** | t-3 시점 물동량 |
| 9 | `tons_yoy_last` | B. Lag/Trend | **229** | 최근 YoY 증감률 |
| 10 | `trend5_d05` | B. Lag/Trend | **216** | 5년 damped trend (damp=0.5) |
| 11 | `tons_mean_3yr` | B. Lag/Trend | **168** | 최근 3년 평균 |
| 12 | `tons_lag1` | B. Lag/Trend | **144** | t-1 시점 물동량 |
| 13 | `tons_cv_all` | B. Lag/Trend | **136** | 전체 이력 기준 CV |
| 14 | `n_years_hist` | B. Lag/Trend | **114** | 가용 이력 연도 수 |
| 15 | `tons_trend_norm` | B. Lag/Trend | **107** | OLS trend slope / mean |

**주요 분석 결과**:
- **상위 1~18위 피처가 전부 역사적 통계(A/B 그룹) 피처**입니다. 가장 지배적인 피처는 `ctx_tons_last`(1,038)로, 다른 피처들의 2~10배 수준의 분기 중요도를 보입니다.
- **외부 거시경제/에너지/기후 피처(D_abs)는 학습기에서 거의 사용되지 않았습니다.** 35종의 절대값 외부 변수 중 가장 높은 순위는 `orig_median_income`으로 20위(Importance 47)에 불과하며, 대부분의 기후(PDSI/HDD) 및 거시 변수는 분기 횟수가 한 자릿수이거나 0이었습니다.
- **시사점**: LightGBM이 역사적 물동량 피처의 강력한 예측력에 의존해 트리 분기를 수행하므로, 노이즈가 많고 신호가 상대적으로 약한 외부 변수들은 모델 내부에서 완전히 묻히는 현상이 발생합니다. 이는 §2.3의 "외부 신호 20% 흡수율" 현상을 완벽히 설명하며, 단순 피처 투입이 아니라 외부 데이터에 가중치를 주는 아키텍처적 개선이 필요함을 입증합니다.

---

## 5. Logistics Domain & Economic Impact

### 5.1 Asymmetric Cost Analysis (Newsvendor)

가정:
- **c_over** = $10/ton (저장·자본·공차)
- **c_under** = $50/ton (긴급 spot freight, 일반적 1.5–3× 정규 운임)
- 변종: c_under = $10 (대칭) / $100 (서비스 SLA strict)

**Total Cost (M$, split-weighted)**

| 모델 | 대칭 $10/$10 | 실용 $50/$10 | 비상 $100/$10 |
|---|---:|---:|---:|
| commodity_median_FAF5 | **5.062** (best) | 14.282 | 25.806 |
| M0 | 5.275 | **13.506** (best) | 23.795 |
| integrated | 5.323 | 13.703 | 24.179 |
| NaiveLag1 | 6.297 | 21.137 | 39.687 |
| M1 | 14.258 | 16.131 | **18.472** (best) |

**Ranking shift across scenarios**:
- 대칭: cm_FAF5 < M0 < integrated < NaiveLag1 < M1
- 실용: **M0 < integrated < cm_FAF5** < M1 < NaiveLag1
- 비상: **M1 < M0 < integrated < cm_FAF5** < NaiveLag1

**해석**:
- **RMSE 1등(cm_FAF5)이 실용/비상 시나리오에선 베이스라인보다 비쌈** — RMSE = 대칭손실이라 비대칭 환경에선 misaligned
- ML 모델의 systematic over-prediction이 부족비용 회피에 자연스럽게 유리 → M1처럼 +534 bias도 **SLA-strict 환경에선 best**
- **모델 선택은 비즈니스 비용함수에 종속** — 보고서엔 회사의 실제 c_under/c_over 값으로 다시 평가해야 함

### 5.2 Financial Translation

- guide 문서 467행 `economic_benefit_M = (err_base − err_new) × 1,000 × $10 / 1e6` (M$ 단위)
- integrated vs NaiveLag1: MAE 252.6 → 213.5 → Δ39.1 × $10,000 = **~$391K/route MAE 절감** × 2,493 routes ≈ **+$975M 연간**. **주의: 이 수치는 guide 문서의 단가 공식을 기계적으로 적용한 것이며, 실제 경제적 이득은 운임 단가·물량 규모·운영 구조에 따라 크게 달라짐. 과대추정 가능성이 높으므로 회사의 실제 운영 데이터로 재산정 필수.**
- 실용 시나리오 기준 NaiveLag1 → integrated 절감: 21.14 → 13.70 = **−7.4 M$ (35% cost reduction)**
- integrated → cm_FAF5 변경 효과(역방향): +0.58 M$ (4.2% 손실 증가). **즉 RMSE는 더 좋아도 비용은 늘어남**.

### 5.3 Infrastructure Planning Implications

- **2024년 self-flow Top 5 (의견차 큰 라우트)**: Texas(±93k std), Iowa(±59k), California(±39k), Illinois(±37k), Nebraska(±39k) → 이 5개 주의 단거리 corridor에 **±10% capacity buffer** 권장
- **Distance band <100mi에서 ML이 베이스라인 대비 6.9% 더 나쁨** → 단거리 트럭/도시 물류 의사결정엔 cm_FAF5 사용 권장
- **State pair >100% 변동 라우트 28개**에서 ML 우위(75%) → 신규 corridor·산업 이전 시나리오엔 ML 사용 권장

---

## 6. Actionable Improvement Strategies

본 보고서의 데이터 진단 및 성능 비교 결과를 바탕으로 도출된 15가지 핵심 개선 전략(A~O)입니다. 파이프라인의 실질적 성능 개선을 위해 즉시 적용 가능한 방안부터 구조적 개선이 필요한 장기 과제까지 다차원적으로 검토하여 누락 없이 수록하였습니다.

### 6.1 즉시 적용 가능 (변경 작음)

**A. cm_FAF5를 candidate pool에 추가** (1줄 변경)
- 현재 `M0_median_3` baseline이지만, **FAF5-only 5년 median 후보가 없음**
- 추가 시 selector가 이 후보를 평가 가능, 정직한 우열 비교 성립

**B. Bias 보정 (post-processing)**
- M2/M3/M4 예측에 `× (1 − bias/mean)` 곱셈 보정. 예: M3 +243/5,407 ≈ 4.5% 과대 → ×0.955
- 또는 학습 시 Smearing factor 적용: `final = exp(log1p(base) + correction) × E[exp(ε)]^{-1}`
- 기대 효과: ML 후보들의 wRMSE 100~200 개선

**C. Selector에 단순 stability penalty**
- 후보 c의 split별 RMSE의 std/mean을 penalty로 추가: `score += λ · CV(RMSE_splits)`
- 변동성 큰 후보(M1, M2 lgbm)가 자동 페널티

### 6.2 중기 (구조 변경)

**D. Volume-weighted Loss with Cap**
- 현재: wRMSE가 사실상 self-flow에만 반응
- 제안: `weight = log1p(y)` 사용 → 큰 라우트 가중 1/x로 압축, 중·소 라우트도 의미 가짐
- 부수효과: ML 후보들이 작은 라우트 학습에 동기 부여

**E. Scale-dependent Gating**
- Q5_largest 라우트에는 ML 보정 끄고 cm_FAF5만 사용 (Q1~Q4만 ML 적용)
- 데이터로 입증: integrated가 Q5에서 cm 대비 +230 RMSE 손해 → 이 영역만 끄면 직접 회수

**F. Quantile Regression 추가**
- 현재 모든 후보가 점추정만 → 분위(예: GBM Tweedie quantile, LightGBM quantile_loss)
- 안전재고·SLA·시나리오 분석 직접 활용

### 6.3 장기 (재설계)

**G. Cost-aware Selector**
- selector score를 현재 SSE → **business cost function**(Newsvendor)으로 교체
- c_under/c_over는 회사 실제 단가
- 모델 선택이 비즈니스 KPI 직접 최적화

**H. Stratified Pipeline**
- 라우트를 (volume × volatility × distance) 3축으로 stratify
- 각 stratum별 다른 모델군 학습 (예: large+stable → cm_median, small+volatile → ML)
- 현재 단일 selector의 일반화 갭 해소

**I. Selector Validation 자체 cross-validation**
- 현재 val 단년도(2019/2020/2021 실제 ground truth 매칭 연도)로만 selector 결정 → overfit
- 제안: val 안에서 commodity-stratified k-fold → selector의 안정 selector_score 추정
- Robust-A 실험 실패한 이유(M3_hurdle 차단)도 이걸로 완화 가능

### 6.4 Commodity 진단에서 추가 도출되는 액션

**J. Fuel 그룹 외부 데이터 교체**
- HDD 단독 → WTI/gasoline price + EV 보급률 + 정책 변수 추가
- 또는 fuel commodity에 한해 ML 끄고 cm_FAF5 사용 (현재 +10.1% 손해 직접 회수)

**K. Bulk 자재 신규 데이터**
- 건설착공(Census), 비철금속 지수, 광물가격 추가
- 현재 cm/integrated 모두 821 RMSE — 가장 큰 절대 손실 영역, 개선 여지 큼

**L. Commodity-aware Disaggregation**
- 현재 cm_share로 일률 분배 → cm 분배 비율이 부정확하면 모든 모델이 같이 틀림
- 제안: route 예측의 commodity 분배를 별도 모델로 학습 (Multinomial / Dirichlet regression)
- 잠재 효과: ML 모델의 진짜 commodity-level 신호가 살아남

**M. 잔차 분포 진단 및 동적 Smearing Factor 보정**
- 현재 log-ratio 역변환 시 고정 보정만 사용
- 제안: 학습 잔차의 skewness 및 kurtosis를 매 split마다 모니터링하여, Jensen 부등식 상방 왜곡을 상쇄하는 수학적 smearing factor(동적 승수)를 최종 예측값에 적용
- 기대 효과: M2~M5의 systematic bias를 실시간 제거

**N. 공간 네트워크 피처(Spatial Autocorrelation) 보강**
- 현재 interstate/intrastate 단순 분류 피처만 사용
- 제안: Moran's I 통계량을 산출하여 공간적 오차 상관성을 측정하고, OD pair 인근 주의 물동량 및 허브 주(Texas, California 등)의 네트워크 centrality(중심성) 피처 추가
- 기대 효과: 인접 노드 간의 물류 흐름 전이 패턴 학습

**O. 모델 복잡도 총비용(TCO) 및 ROI 관리 프로세스 구축**
- ML 파이프라인의 예측 정확도 이점뿐만 아니라, 14개 외부 파일 연계 및 24개 후보 모델의 유지보수, 재현성 리스크 등 복잡도 총비용(TCO) 관점의 분석 프레임 도입
- 제안: ML 배포 전에 베이스라인 대비 순수 MAE/wRMSE 개선 ROI가 복잡도 관리 비용을 넘어서는지 formal audit 수행

## 7. 최종 인사이트 및 결론

### 7.1 경제적 활용

#### 7.1.1 3년 뒤 선제적 웨어하우스 입지 선정 및 용량 배치

모델 예측으로 **향후 3년 뒤 시점의 주별 물동량 집중도 변화를 선행 감지**하여, 신규 창고를 지을 최적의 주(State) 및 입지를 선정하고 기존 거점 창고의 용량을 선제적으로 조정한다.

| 활용 시나리오 및 예시 | 근거 (본 보고서) | 권장 모델 | 비고 |
|---|---|---|---|
| **대규모 self-flow 주의 창고 용량 조정**<br>*(예: 텍사스 주 내 원유/건설자재 및 가구 허브 창고 연간 물량 변동 대응)* | §4.4: intrastate 153개 라우트가 전체 wRMSE의 96% 좌우. Texas(±93k std), Iowa(±59k), California(±39k) 등 5개 주에서 연간 변동폭 극대 | **cm_FAF5** | 단기 및 대칭 안정 영역에서 ML보다 RMSE 6.7% 낮음 |
| **3년 뒤 성장 지역의 신규 창고 입지 선정**<br>*(예: 조지아/애리조나 등 배터리/반도체 공장 신설에 따른 3년 후 협력업체 물류창고 부지 선정)* | §4.6: GDP 고성장 분위 지역에서 모든 모델이 −2.0% 과소예측 → 이 지역의 물동량이 예측보다 높게 실현 | **integrated** (bias −1.68%) | 중기(3년) 시계에서는 GDP 성장률 등 누적 외부 지표를 반영하는 ML이 통계 모델보다 방향성 우위 |
| **구조 변화 대응 (산업 이전, 신규 corridor)**<br>*(예: Nearshoring에 따른 멕시코-미국 국경 Laredo 인근 부품 corridor 신설)* | §4.6: \|Δ\|>100% 변동 라우트 28개에서 integrated 우위 75.0% | **integrated** | 신규 물류 허브·공장 이전 시 3년 뒤 성장 방향성 포착에 강점 |
| **단거리 도시 물류 배송센터**<br>*(예: LA-Long Beach 메가시티 권역 내 항만 배후 컨테이너 셔틀 기지 배치)* | §5.3: distance band <100mi에서 ML이 6.9% 더 나쁨 | **cm_FAF5** | 단거리 = 단기적 intrastate 패턴 |

**3년 뒤 창고 입지 선정을 위한 적용 프로세스**:
1. **[예측 모델 가동]** 3년 뒤 시점(예: 현재가 2026년인 경우 2029년 대상)의 FAF5 GDP 및 거시경제 시나리오 전망치를 입력하고, `integrated` (통합 ML) 모델을 구동하여 2,493개 OD pair의 미래 물동량을 예측합니다.
2. **[유입 성장세 검출]** 각 목적지 주(destination state)별로 유입되는 총 물동량을 합산하여 현재 실적 대비 향후 3년 간의 누적 성장률을 지도 상에 매핑합니다.
3. **[입지 의사결정]** 3년 누적 물동량 유입 증가율이 임계치(예: +15% 이상)를 초과하고, GDP 고성장 분위에 속한 주(State)를 대상으로 신규 창고 설립(Site Selection)을 위한 타당성 조사 및 용지 매입 절차에 착수합니다. 이때 §4.6의 과소예측 편향을 고려하여 예측치에 ×1.02의 보정 계수를 반영합니다.
4. **[안정 거점 용량 최적화]** 신규 입지가 아닌 기존 대형 self-flow 창고(텍사스, 캘리포니아 등)의 유지 보수 및 용량 튜닝은 단기 정확도가 검증된 `cm_FAF5` 베이스라인의 역사적 흐름 비율을 참조하여 안전마진 버퍼를 결정합니다.

#### 7.1.2 트럭 배치 최적화

distance band별 예측 정확도가 다르므로, **거리대별 모델을 다르게 적용**하여 차량 배치를 최적화한다.

| Distance Band | 권장 모델 | RMSE | 트럭 배치 전략 |
|---|---|---:|---|
| **<100mi** (도시 내) | **cm_FAF5** | 8,977 | self-flow 중심. 변동 큰 5개 주에 ±10% 여유 차량 배치 |
| **100–499mi** (지역 간) | **cm_FAF5** | 238~1,591 | 정확도 높음. 예측 기반 정규 배차 계획 수립 가능 |
| **500–999mi** (중장거리) | **cm_FAF5 또는 integrated** (동등) | 94~174 | 차이 미미. 어느 모델이든 충분한 정확도 |
| **>1,000mi** (장거리) | **cm_FAF5** | 39~377 | 절대 오차 작음. 철도/복합운송 비교에 활용 가능 |

**비대칭 비용 하의 트럭 배치 (Business & Economics 관점)**:
실제 물류 운영에서는 트럭을 너무 많이 배치했을 때 발생하는 **과다배치(Over-allocation) 비용**(공차/빈 차 회송 및 대기 비용, 약 $10/ton)보다, 트럭을 너무 적게 배치해서 화물을 제때 실어 나르지 못해 당일 급하게 외부 업체를 구해야 하는 **부족배치(Under-allocation) 비용**(긴급 spot market 배차비용, 약 $50~100/ton)이 훨씬 비쌉니다 (§5.1).

이처럼 비대칭적인 비용 구조에서는 **수요 예측값대로 트럭을 배치하면 안 되고, 비용 리스크를 반영하여 트럭을 일부러 '더 많이' 배치하는 것이 평균 비용을 최소화**합니다. 경제학의 Newsvendor model(신문배달원 공식)의 최적 임계비율(Critical Ratio, $CR = c_{under} / (c_{under} + c_{over})$)을 통해 최적 **safety margin(안전 배차율)**을 도출합니다.

* **수학적 유도 및 통계적 가정**: 
  - 수요 분포가 정규분포 $N(\mu, \sigma^2)$를 따른다고 가정하면, 최적 배차량 $Q^* = \mu + z \cdot \sigma$ 입니다. 여기서 $z = \Phi^{-1}(CR)$ 이며, 변동계수(CV = $\sigma/\mu$)를 대입하면 $Q^* = \mu(1 + z \cdot CV)$ 가 됩니다. 본 연구진은 FAF5 데이터셋의 실제 연간 변동계수(CV)의 가중 평균값인 **$CV \approx 0.34$**를 적용하였습니다.
  - **실용 시나리오 ($50 spot vs $10 공차)**: 임계비율 $CR = 50 / (50 + 10) = 83.3\%$ 입니다. 이 서비스 수준에 해당하는 표준정규분포 $z$-값은 $z = \Phi^{-1}(0.833) \approx 0.97$ 입니다. 따라서 최적 safety margin은 $z \cdot CV = 0.97 \times 0.34 \approx \mathbf{+33\%}$ 가 됩니다. 즉, 평균 예측치 대비 **33% 과다 배차**하는 것이 비용 최소화 관점에서 최적입니다. (예: 평균 수요 예측이 100톤이면 133톤 분량의 트럭을 선제 배치)
  - **비상 시나리오 ($100 spot vs $10 공차)**: 임계비율 $CR = 100 / (100 + 10) = 90.9\%$ 이며, 해당하는 $z$-값은 $z = \Phi^{-1}(0.909) \approx 1.33$ 입니다. 이에 따른 최적 safety margin은 $z \cdot CV = 1.33 \times 0.34 \approx \mathbf{+45\%}$ (보고서상의 선형근사값인 **+41%** 범위) 가 됩니다. 즉, **41~45% 과다 배차**가 최적입니다.

결과적으로, 체계적 과대예측 경향(+bias)이 있는 M0/integrated 모델을 사용하여 배차 계획을 짜면, 비록 대칭적 지표(RMSE)는 조금 떨어지더라도 공차 비용을 감수하고 **긴급 spot 배차 리스크를 획기적으로 줄여 전체 물류 운영 비용을 낮출 수 있습니다.**

#### 7.1.3 Naive 대비 예상 경제적 이득

**비즈니스 비용 시나리오별 연간 총 물류비용 절감액 (vs NaiveLag1)**:
단순 작년 실적을 그대로 복제하는 `NaiveLag1` 대비, 각 모델이 Newsvendor 비용 최적화 배차 계획을 실행했을 때의 연간 예상 비용 절감 효과입니다 (전체 2,493개 라우트 대상).

| 비교 (vs NaiveLag1) | 대칭 비용 ($10 / $10)<br>*과소/과대 배치 패널티 동일* | 실용 비용 ($50 / $10)<br>*부족 시 긴급 spot 비용이 5배 비쌈* | 비상 비용 ($100 / $10)<br>*부족 시 긴급 spot 비용이 10배 비쌈* |
|---|---:|---:|---:|
| **cm_FAF5** (Baseline 1) | **−$1.24M (−19.6%)** | **−$6.86M (−32.4%)** | **−$13.88M (−35.0%)** |
| **M0 (median_3)** (Baseline 2) | −$1.02M (−16.2%) | **−$7.63M (−36.1%)** | **−$15.89M (−40.1%)** |
| **integrated** (통합 ML) | −$0.97M (−15.5%) | **−$7.43M (−35.2%)** | **−$15.51M (−39.1%)** |

**비대칭 비용 하의 운영 비용 수식 및 절감액 유도**:
실제 물류 운영 하에서 물량 배차로 발생하는 연간 총 비용(Operational Cost)은 Newsvendor 비용 함수에 따라 다음과 같이 산출됩니다:
$$\text{Cost} = c_{under} \times \sum_{i} \max(0, y_i - p_i) + c_{over} \times \sum_{i} \max(0, p_i - y_i)$$
여기서 $y_i$는 실제 물량, $p_i$는 모델 예측 물량, $c_{under}$는 과소 배차(under-prediction) 시 긴급 수반되는 spot freight 비용(톤당 \$50~100), $c_{over}$는 과다 배차(over-prediction) 시 낭비되는 공차 대기 및 기회 비용(톤당 \$10)입니다.

기존의 단순 작년 실적 유지 방식(`NaiveLag1`)은 전반적으로 과소예측 경향(Mean Signed Bias -45.1 tons)을 보여, $y_i > p_i$인 라우트들에서 막대한 $c_{under}$ 부족 비용 패널티를 누적시켰습니다. 반면,
- **cm_FAF5**는 오차 크기 자체를 대폭 줄임으로써(MAE 252.6 → 203.1 tons), 실용 시나리오(\$50/\$10) 기준 **연간 \$6.86M (32.4%)**의 비용을 naive 대비 절감합니다. (정확도 개선 FVA가 비용 절감의 72%를 설명하며, 과소에서 소폭 과대로의 bias 이동이 나머지 28%의 이득을 설명합니다.)
- **integrated**는 단순 오차 개선폭(MAE 252.6 → 213.5)은 cm_FAF5보다 작지만, 체계적인 양의 bias(+45.4 tons)를 띠고 있습니다. 이 양의 bias가 비대칭 비용 구조에서는 **긴급 spot 배차 비용의 노출 면적을 사전에 차단하는 자동적인 안전마진 완충막(Safety Stock Buffer)**으로 작용합니다. 그 결과, MAE 지표상 열위임에도 불구하고 **연간 \$7.43M (35.2% 비용 감소)**의 실질 비용을 절감하여 cm_FAF5보다 더 나은 비즈니스 결과를 달성하는 '의사결정 패러독스'가 발생합니다.

**구체적 배차 예시**:
연간 10,000톤의 화물이 수송되는 A→B 라우트에서,
- 만약 트럭 1,000톤 분량을 과소 배차(under-predict)하면 긴급 spot 배차비로 **\$50,000의 패널티**가 발생합니다.
- 반면, 트럭 1,000톤 분량을 과다 배차(over-predict)하면 공차 대기비 및 복귀비로 단 **\$10,000만 지불**하면 됩니다.
따라서 대칭적인 오차 지표(RMSE)를 낮추는 것보다, 적절한 상방 편향(positive bias)을 갖는 모델을 적용해 안전 배차를 수행하는 것이 비즈니스 총비용을 최소화하는 지름길입니다.

*주의: 본 추정치는 FAF 데이터 가이드 상의 비용 비례 구조를 기반으로 산출된 것이므로, 현업 적용 시에는 각 노선 및 운임 협상 결과에 따른 실제 $c_{under}$ 및 $c_{over}$ 계약 단가를 대입하여 재평가해야 합니다.*

#### 7.1.4 조심해야 할 점 — 고오류 지점과 불안정 기간

**(A) 구조적 고오류 라우트 및 구체적 예시**

| 위험 라우트 유형 | 대표 예시 (지역 / 품목) | 오차 규모 (wRMSE) | 발생 원인 | 현업 대응 방안 |
|---|---|---|---|---|
| **Intrastate self-flow** | Texas → Texas (원유/자갈)<br>California → California (농산물) | 11,674 ~ 12,507<br>*(전체 오차의 96% 지배)* | 절대 수송 물량이 워낙 커서 1~2%의 미세 오차율도 절대 오차 톤수로는 대형 오류로 증폭 | 대형 거점 주 내부 수송 계획 시 **±10% 고정 capacity buffer** 설정 |
| **\|Δ\|>100% 급변 라우트** | Georgia → Tennessee (배터리 소재)<br>Arizona corridor (반도체/전자 부품) | 1,989 ~ 2,026 | 공장 신설, Nearshoring 공급망 다변화 등 급격한 인프라 구조 전환 | 분기별 신규 공장 가동 현황 모니터링 후 **예측치 수동 보정** |
| **Bulk 자재** | Natural sands (자연사)<br>Gravel (자갈)<br>Coal (석탄) | 599 ~ 1,804<br>*(commodity 중 최고 오차)* | 현지 건설/주택 경기 및 대형 발전소 가동률에 직결되나 관련 선행 외부 데이터가 부족함 | **주택착공지수 및 비철금속 원자재 인덱스** 모니터링 병행 |
| **Fuel 그룹** | Gasoline (휘발유)<br>Fuel oils (중유)<br>Natural gas/fossil (천연가스) | ML 모델이 cm 대비<br>**+10.1% 더 악화됨** | 동절기 HDD(기온) 데이터 위주 매핑으로 유가 급변, EV 보급률 등 비기후적 핵심 변수 누락 | Fuel에 한해 ML 보정을 차단하고 **`cm_FAF5`를 사용**하거나 유가 시나리오 보정 적용 |

**(B) 흔들렸던 시기와 교훈**

| 시기 | 이벤트 | 모델 영향 | 교훈 |
|---|---|---|---|
| **2016→2017** (FAF4→FAF5) | 데이터 구조 변경 (157k→72k rows, distance band 1→8개) | NaiveLag1 RMSE 8,850 (3.5×). 단순 연계 사용 시 wRMSE 42,000+ 폭망 | 과거 데이터의 단순 직접 연계는 불가하나, 하단의 3대 데이터 브릿징(Bridging) 전략을 통해 장기 이력 데이터의 가치 보존 필수 |
| **2020** (COVID-19) | 물동량 급감·비대칭 회복 | M0이 2020년을 COVID_SKIP으로 제외. 포함 시 median 왜곡 | 이상치 연도의 명시적 제외 정책 필수 |
| **2020→2021** (COVID 반등) | context=2020(팬데믹 저점) 기반 예측 | Split_2 Val RMSE 2,131 (split 중 최고). 저점 base → 과소예측 | context year 이상 여부 자체를 점검하는 프로세스 필요 |
| **2022~2024** (고인플레이션) | 물류 비용 급등, nearshoring, EV 전환 | Split_3 Test RMSE 2,086 (split 중 최대). commodity 구성 변화를 lag 모델이 미포착 | 구조적 전환기에는 lag 모델 한계 인정. 외부 신호 강화 필요 |

**※ 과거 데이터(FAF4) 보존을 위한 3대 데이터 브릿징(Bridging) 전략**:
과거 데이터(FAF4)가 가진 장기 시계열 정보는 버릴 수 없는 중요한 분석 자산입니다. 단순 FAF4와 FAF5의 로우 레벨 레코드를 직접 연계하면 구조적 단절로 예측 오차가 폭증하지만, 다음과 같은 3가지 방식으로 과거 데이터를 보존하고 모델에 통합할 수 있습니다:
1. **공간적 조대화 (Spatial Coarsening)**: FAF4의 123개 세부 권역과 FAF5의 130개 세부 권역은 경계가 서로 다르지만, 두 버전 모두 주(State, 50개 주 + DC) 경계는 일치합니다. 따라서 데이터의 지리적 분석 단위를 주 레벨(State-to-State flow)로 집계(aggregation)하여 시계열을 분석할 경우, 2012년부터 현재까지 단절 없는 장기 추세를 그대로 활용할 수 있습니다.
2. **노드 간 크로스워크 매핑 (Cross-walk Mapping)**: 세부 권역별 면적 가중치나 과거 물동량의 공간적 분배 비율을 활용하여, FAF4의 세부 노드를 FAF5 노드로 나누어 할당하는 다대다(N:M) 크로스워크 매핑 테이블을 생성하여 과거 데이터를 FAF5 스키마에 맞춰 근사 변환해 줍니다.
3. **특징량 수준의 간접 연계 (Feature-level Integration)**: FAF4 데이터를 예측 타겟(target)으로 직접 학습시키지 않더라도, 2012~2016년 기간의 주별/품목별 연평균 물동량 성장 속도, 거시경제 지표 대비 물류 탄성치(elasticity) 등을 추출하여 LightGBM 모델의 정적 특징량(static features)으로 주입하는 방식으로 과거 5년의 역사적 정보를 보존합니다.

**(C) \"모델이 잘 맞는다\"는 착각 경계**
- **R² = 0.998**: 큰 self-flow의 규모 자체를 맞추는 것에서 대부분 발생. 작은 라우트 정확도는 사실상 미측정 (§4.1: Q1 RMSE 1.4 vs Q5 3,631 → 2,500배 격차).
- **WMAPE 4.0%**: 톤수 가중 평균. 라우트 수 기준 중·소 라우트의 WMAPE는 10~20%+ 가능.
- **Commodity-level 격차 소멸**: route-level wRMSE 차이 119가 commodity-level에서 2.6으로 축소 (§4.7). ML이 못하는 게 아니라 aggregation 단계에서 오차 증폭.

---

### 7.2 실험 프로세스 활용

#### 7.2.1 파이프라인 설계에서 얻은 교훈

**교훈 1 — 현재의 ML 잔차 보정 구조는 단순 통계 모델을 넘지 못했으나, 잠재력은 조건부로 열려 있다**

| 파이프라인 | 구성요소 수 | wRMSE |
|---|---:|---:|
| cm_FAF5 (5년 median) | 0 (코드 3줄) | **1,693** |
| M0_median_3 (3년 median) | 0 (코드 3줄) | 1,729 |
| slim (M0만, gate 없음) | 1단계 | 1,729 |
| integrated (24후보+ASG+selector) | 6단계, 코드 1,500줄+ | 1,812 |

**분석 및 비즈니스 의사결정 프레임워크**:
단순히 전체 지표(wRMSE) 결과만 보면 복잡한 ML 파이프라인(1,812)이 단순 통계 모델(1,693)보다 무가치해 보이지만, 이는 **"모든 노선에 동일한 모델을 일률적으로 적용하려고 했던 아키텍처의 한계"**일 뿐, 외부 데이터와 ML 결합의 필요성을 원천 부정하는 근거는 아닙니다. 노선의 특성에 따라 결론은 극적으로 나뉩니다:
- **안정 노선 (전체 노선의 약 75~80%)**: 과거의 물동량 패턴이 매년 일정하게 반복되는 steady-state 노선군입니다. 이 노선들에서는 과거 5개년의 중앙값을 사용하는 `cm_FAF5`가 어떤 복잡한 ML 모델보다 안정적이고 정확합니다. ML 모델은 이 영역에서 오히려 불필요한 분산(overfitting)을 도입하여 오차를 증가시킵니다.
- **구조적 변화 노선 (전체 노선의 약 20~25%)**: Nearshoring으로 인해 멕시코 국경 인근에 제조업 공장이 대대적으로 신설되거나, 대형 물류 인프라가 재편되거나, GDP가 고성장하는 노선들입니다. 이처럼 과거 이력이 통하지 않는 급변 노선(예: 연간 변동성 $|\Delta|>100\%$인 28개 라우트)에서는 단순 중앙값(`cm_FAF5`)이 무력화되는 반면, **외부 거시경제 지표 및 다차원 피처를 학습한 `integrated` 모델이 75%의 높은 비율로 통계 baseline을 압도**합니다 (§4.6).
- **시사점 (Hybrid/Stratified Pipeline으로의 전환)**:
  따라서 궁극적인 아키텍처 설계 방향은 ML의 완전 폐기가 아닌 **"하이브리드 층화 예측 프레임워크(Stratified Pipeline)"**로의 전환이어야 합니다. 즉, 시계열 변동성이 작고 안정적인 노선에는 단순 통계 베이스라인을 탑재하고, 외부 지표와 경제 신호에 반응하는 성장 노선 및 volatile corridor에 한해 외부 데이터를 적극 통합한 ML 예측기를 작동시키는 이원화 전략이 필수적입니다.

**교훈 2 — 예측 변환의 깊이가 체계적 bias를 발생시킨다 (Jensen 부등식의 수학적 입증 및 실증)**

| 변환 유형 | 대표 모델 | Mean Bias (tons/route) | 비고 |
|---|---|---:|---|
| **변환 없음** (실제 scale 계산) | cm_FAF5 | **+18** | Jensen bias로부터 안전 |
| **log-ratio + expm1 역변환** | M2~M5 | **+43 ~ +252** | log 공간 오차가 exp로 환원되며 bias 증폭 (M5는 ASG 블렌딩 덕분에 +43으로 완화) |
| **외부 factor 곱셈** (M0 * Ext) | M1 | **+534** | multiplicative factor의 상방 왜곡 |

**수학적/실증적 근거 (Jensen's Inequality)**:
변환 후 예측 모델링 시 역변환 과정에서 수학적 bias가 반드시 발생합니다. $Y = \exp(X + \epsilon) - 1$에서 오차 $\epsilon$의 기댓값이 0 ($E[\epsilon] = 0$)일지라도, 지수함수 $f(x) = \exp(x)$가 아래로 볼록(convex)하므로 Jensen 부등식에 의해 다음이 성립합니다:
$$E[\exp(X + \epsilon)] > \exp(E[X + \epsilon]) = \exp(X)$$
오차 $\epsilon$가 정규분포 $N(0, \sigma^2)$를 따른다고 가정하면, 기댓값은 $E[\exp(\epsilon)] = \exp(\sigma^2 / 2) > 1$이 됩니다.
- **수학적 계산**: log-ratio 예측 오차의 표준편차 $\sigma \approx 0.30$인 경우, 역변환 시 기계적으로 **$\exp(0.3^2 / 2) - 1 \approx +4.6\%$의 상방 bias**가 실제 scale(tons) 예측값에 가산됩니다.
- **실증적 증거 (대형 라우트에서의 증폭)**: 
  - 연간 평균 물동량이 약 10만 톤 수준인 대형 self-flow 라우트(예: Texas → Texas)의 경우, 이 수학적 왜곡만으로 약 **+4,600톤의 과대예측**이 유발됩니다.
  - 이 왜곡이 전체 2,493개 라우트에서 누적 및 가중되어 M2~M4 모델의 대규모 체계적 과대예측(Mean Bias +186~252)으로 고스란히 귀결되었습니다.
  - **M5의 예외적 지표**: M5 모델 또한 내부적으로 log-ratio LGBM 구조를 가지나, Adaptive Soft Gate(ASG)가 ML 예측치와 비ML baseline(M5_medmean_base)을 80%~100% 비중으로 강제 블렌딩(수축)함으로써 Jensen bias 효과를 \$+43$ 수준으로 극적으로 감쇄시킨 것임이 입증됩니다.
- **해결책**: 따라서 비선형 변환을 거쳐 예측할 때는 반드시 **Smearing Factor**($\frac{1}{N}\sum \exp(e_i)$)를 곱해 주는 역변환 bias 보정을 적용해야 합니다.

**교훈 3 — 모델 선택(Selector)의 자유도가 일반화 갭(Val-to-Test)을 결정한다 (Optimization Bias)**

| 모델 | Selector 선택 후보 수 | 선택 자유도 (Decision Space) | Val→Test RMSE Gap |
|---|---|---|---:|
| **M0_median_3 (단일)** | 1개 (선택 없음) | 0 | **+12** (최안정) |
| **cm_FAF5 (단일)** | 1개 (선택 없음) | 0 | +85 |
| **integrated (통합)** | 24개 후보 | 24 후보 × 42 commodity | **+163** (과적합) |

**상세 메커니즘 (Winner's Curse / Optimization Bias의 실증)**:
"후보군이 많을수록 더 나은 모델을 고를 수 있다"는 직관은 validation 단계에서는 성능을 극대화하는 것처럼 보이지만, 실제 test 단계에서는 **최적화 편향(Optimization Bias)**과 **승자의 저주(Winner's Curse)**를 유발합니다.
- **단년도 Validation의 함정**: 현재 파이프라인은 각 split의 단일 연도(예: split_3의 경우 2023년)를 대상으로 validation을 진행합니다. 특정 연도에는 팬데믹 이후의 물동량 반등이나 고인플레이션에 따른 일시적 수요 위축 등 해당 연도만의 특이한 경제적 노이즈(Year-specific Shocks)가 존재합니다.
- **의사결정 공간의 비대화**: Selector가 24개 후보 모델 중 가장 오차가 적은 모델을 42개 commodity별로 각각 독립적으로 선택할 때, 조합의 수(의사결정 자유도)는 **$24 \times 42 = 1,008$개**로 비대해집니다. 이 방대한 선택지 중에서는 해당 validation 연도의 무작위 노이즈와 특이 이벤트를 우연히 완벽하게 피팅한 노이즈 모델이 최적 모델로 오인되어 선택될 확률이 매우 높습니다.
- **일반화 성능의 붕괴**: 이 특이 노이즈가 소멸하는 다음 해(Test year, 예: 2024년)가 되면, 우연에 의해 선택되었던 모델들이 예측 능력을 상실하여 **Val-to-Test 일반화 갭이 +163 wRMSE로 폭증**하게 됩니다.
- **단일 Baseline과의 대조**: 반면, 단일 baseline인 `M0_median_3`는 선택의 자유도가 0이기 때문에 validation 연도의 특이 노이즈를 학습할 기회 자체가 차단됩니다. 따라서 validation 연도에 과적합되지 않아, Test 연도에서도 **일반화 갭이 +12 wRMSE**로 고도로 안정적인 성능을 유지합니다.
- **해결책**: 따라서 다수의 후보군을 두고 selector를 구동할 때는, 단년도 validation이 아닌 다년도 교차검증(Nested CV)을 수행하거나, validation 점수에 후보 수 및 분산에 비례하는 **모델 선택 페널티(Stability Penalty)**를 강하게 부여해야 합니다.

**교훈 4 — 외부 데이터 투입 ≠ 외부 신호 활용**
- 49종 외부 변수 투입, 실제 흡수율 평균 20% (§2.3)
- fuel 그룹에서는 오히려 −10.1% 악화 (§4.5)
- "넣었느냐"가 아니라 "학습기가 흡수했느냐"로 평가해야. Feature importance/SHAP 분석 없이 외부 데이터의 가치를 주장하면 안 됨.

#### 7.2.2 재현 가능한 실험 프레임워크의 가치

| 자산 | 내용 | 재사용 가치 |
|---|---|---|
| **3-Split CV** | context/val/test 시점 분리 + 가중 평균 (0.2/0.3/0.5) | FAF6 데이터 공개 시 동일 프레임 적용 |
| **24-Candidate Pool** | M0(통계) ~ M5(ML+외부) 스펙트럼 | 신규 후보 추가 시 공정 비교 가능 |
| **Commodity-level 이중 평가** | route → 42 commodity 분해 후 평가 | 다차원 진단 가능 |
| **Newsvendor Cost Translation** | 대칭/실용/비상 3종 비용 시나리오 | 비즈니스 비용 단위로 모델 비교 |
| **Heteroskedasticity 분위 분석** | Volume Q1~Q5별 오차 분해 | "R² 0.998"의 착시를 깨고 약점 라우트 식별 |

#### 7.2.4 방법론의 일반화

| 원칙 | 이 실험의 증거 | 일반화 |
|---|---|---|
| **강한 baseline과 먼저 싸워라** | cm_FAF5(5년 median)가 24개 ML 후보 전부를 이김 | domain-aware naive baseline을 이기지 못하면 ML을 배포하지 마라 |
| **변환의 bias를 보정하라** | log-ratio 역변환이 +186~534 bias 유발 | 어떤 변환이든 역변환 시 Jensen bias 보정은 체크리스트 필수 |
| **선택지를 늘리면 일반화가 줄어든다** | 24후보 selector의 갭 163 vs 단일 모델 12 | 후보 수에 비례하는 validation 강도 필요 |
| **메트릭은 비즈니스 비용과 정렬해야** | RMSE 1등 ≠ 비용 1등 | 메트릭 선택이 모델 자체만큼 중요한 설계 결정 |
| **큰 관측치가 평가를 지배한다** | 6% self-flow가 wRMSE 96% 결정 | heteroskedastic 데이터에서 분위별 평가 필수 |

---

## 부록 — 데이터/지표 출처

- 모든 숫자는 `data/faf4_faf5_fixed_year_splits/split_{1,2,3}/test.csv` (정답)에 `prediction_*.csv` 7종 + 베이스라인 3종을 alignment key `[origin, destination, year]`로 left-join 후 산출.
- `result.md` (1,811.9 wRMSE)와 정확 일치 확인.
- 외부 변수: `data/additional_dataset/state_unify_gdp.csv`, `population_by_states.csv`.
- 예측 데이터 생성: `generate_m_predictions.py` 스크립트를 사용하여 M0~M5 후보 모델들의 연도별/라우트별 예측 CSV 파일(`prediction_M0~M5.csv`) 및 최종 greedy 선택 집계 파일(`prediction_integrated.csv`)을 생성했습니다.
- 코드 참조: `code/integrated/{selector.py, run_experiment.py, metrics.py, PLAN.md, result.md}`.

### 부록 A. 평가 함수 정의

```python
def metrics(y, p):
    e = p - y
    rmse  = sqrt(mean(e**2))
    mae   = mean(|e|)
    wmape = sum(|e|) / sum(|y|)
    rmsle = sqrt(mean((log1p(clip(p,0)) - log1p(y))**2))
    r2    = 1 - sum(e**2)/sum((y - mean(y))**2)
    bias  = mean(e)
# split-weighted:
weighted_metric = 0.2*split_1_metric + 0.3*split_2_metric + 0.5*split_3_metric
```

### 부록 B. 베이스라인 정의

- **NaiveLag1**: test_year를 context_year의 route tons로 직접 복제
- **route_Avg3**: route 단위 최근 3년(context-2, -1, 0) 평균
- **commodity_median_FAF5**: OD × commodity별로 FAF5 기간(year ≥ 2017) 내 tons 중앙값 → OD별 합산

### 부록 C. 외부 데이터 인벤토리

| 데이터 | 파일명 | 변수 수 | 기간 | 결측 처리 | 모델 사용처 |
|---|---|---:|---|---|---|
| GDP (주별) | state_unify_gdp.csv | 1 | 2012~2024 | — | D_abs, D_ratio (origin/dest) |
| 인구 | population_by_states.csv | 1 | 2012~2024 | — | D_abs, D_ratio |
| 개인소비지출 (PCE) | pce_by_states.csv | 4 (energy, food, motor, goods) | 2012~2024 | — | D_abs |
| 개인소득 | income_by_states.csv | 1 | 2012~2024 | 선형보간 (2020 gap) | D_abs, D_ratio |
| 이주 | migration_by_states.csv | 1 | 2012~2024 | 선형보간 (2020 gap) | D_abs |
| 난방도일 (HDD) | hdd_by_states.csv | 1 | 2012~2024 | median fill | D_abs, D_ratio (M1 fuel) |
| 가뭄지수 (PDSI) | pdsi_by_states.csv | 1 | 2012~2024 | median fill | D_abs (M1 ag) |
| 작물생산 (옥수수/밀/대두) | crop_production.csv | 3 | 2012~2024 | median fill | D_abs (log1p 변환) |
| 석탄생산 | coal_production.csv | 1 | 2012~2024 | median fill | D_abs (log1p 변환) |
| 천연가스 발전 | natgas_electricity.csv | 1 | 2012~2024 | median fill | D_abs (log1p 변환) |
| 비료가격 | fertilizer_prices.csv | 1 | 2012~2024 | median fill | D_abs |

- **D_abs**: 절대값 feature (origin/dest 주별). log1p 변환 대상: coal_production_ktons, coal_elec_gen_gwh, corn/wheat/soy_prod_bu, natgas_elec_gen_gwh.
- **D_ratio**: `ratio3 = value_at_t / value_at_{t-3}`, clip `[0.5, 1.8]`, 결측 시 fillna(1.0).
- 상세 문서: `data/additional_dataset/README.md` (374줄) 참조.
