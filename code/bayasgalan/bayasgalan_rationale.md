# 모델 선택 근거 — CommodityMedian3 + LightGBM Residual

**작성자**: Bayasgalan  
**날짜**: 2026-05-29  
**코드**: `code/bayasgalan_model.py`

---

## 1. 모델 구조

```
최종 예측 = CommodityMedian3(baseline) + LightGBM(residual 보정)
```

두 단계로 동작한다:

1. **CommodityMedian3**: 최근 3년간 각 OD×상품 조합의 중앙값을 구하고 상품별로 합산 → 경로별 기본 예측값
2. **LightGBM**: 베이스라인이 틀린 만큼(residual)을 경로 특성(추세, lag, 변동성 등)으로 보정

---

## 2. 왜 CommodityMedian3가 강력한 베이스라인인가

### 2-1. 최근 3년이 가장 중요하다

미국 화물 수요는 장기 평균보다 **최근 3년 추세**를 더 잘 따른다.  
전체 역사(2012~)의 중앙값을 쓰면 FAF4(2012–2016) 시대의 낮은 수치가 예측을 끌어내린다.

| Baseline | val wRMSE |
|---|---|
| CommodityMedian (전체 역사) | 140,000+ |
| **CommodityMedian3 (최근 3년)** | **16,379** |

### 2-2. 상품별로 나눠서 중앙값 → 합산하는 이유

경로 직접 중앙값보다 정확하다. 경로마다 상품 구성이 다르고, 상품별로 수요 사이클이 다르기 때문에 **상품 단위에서 안정적인 추세를 잡은 뒤 합산**하는 것이 더 robust하다.

---

## 3. 왜 LightGBM을 추가하는가

CM3는 "최근 3년 평균"만 보기 때문에 다음을 잡지 못한다:

- **성장/감소 추세가 있는 경로**: CM3는 중앙값이라 추세를 반영 못함
- **변동성이 큰 경로**: CV(변동계수)가 높은 경로는 단순 중앙값이 부정확
- **연도별 yoy 변화**: 직전 년도 대비 급등/급락 패턴

LightGBM은 이 residual을 14개 feature로 보정한다:

| Feature 그룹 | 변수 |
|---|---|
| Lag | `tons_lag1`, `tons_lag2`, `tons_lag3` |
| 최근 추세 | `tons_mean_3yr`, `tons_std_3yr`, `tons_yoy_last` |
| 전체 특성 | `tons_mean_all`, `tons_cv_all`, `tons_trend`, `tons_trend_norm` |
| 구조적 | `n_years_hist`, `n_commodities`, `modal_dist_band` |
| 베이스라인 | `base_pred_log` (CM3 예측값 자체도 feature로 활용) |

---

## 4. 두 가지 핵심 설계 결정

### 4-1. FAF5-only 학습 샘플 (`min_fy=2017`)

LightGBM 학습 샘플을 FAF4 시대(2012–2016)로 확장하면 **성능이 오히려 떨어진다**.

| min_fy | val wRMSE |
|---|---|
| 2012 (전체) | 22,771 ❌ |
| 2015 | 19,451 ❌ |
| 2016 | 17,862 ❌ |
| **2017 (FAF5-only)** | **16,744** |
| CM3 baseline | 16,379 |

**이유**: FAF4(2012–2016)와 FAF5(2017–2024)는 데이터 수집 방식, 물동량 스케일, 경제 구조가 다르다(distribution shift). 과거 데이터를 더 쓸수록 오히려 모델이 현재 패턴을 잘못 학습한다.

### 4-2. COVID filter (`covid_filter=True`)

2020년을 학습 샘플의 **target year**에서 제외한다.

- 2020년은 COVID로 인한 외부 충격으로 화물 수요가 15~20% 급감
- 이 샘플을 포함하면 모델이 "정상 → 충격" 패턴을 학습해 정상 연도 예측을 왜곡
- 제거하면 split_3 val RMSE 소폭 개선 확인 (1,629 → 1,627, FAF5-only simple RMSE 기준)

> ⚠️ 주의: 2020 데이터 자체는 유지됨. 2020을 **예측 타겟**으로 쓰는 학습 샘플만 제거.

---

## 5. 실험 결과 요약

### tons-weighted RMSE (팀 공식 metric, FAF4+FAF5 splits)

| Model | val wRMSE |
|---|---|
| CommodityMedian (전체 역사) | 140,000+ |
| NaiveLag1 | 28,440 |
| RouteAvg3 | 21,823 |
| **CommodityMedian3** | **16,379** ← 베이스라인 |
| CM3 + LightGBM (min_fy=2017, COVID filter) | 16,744 |

### simple RMSE (FAF5-only splits)

| Model | val RMSE | test RMSE |
|---|---|---|
| CommodityMedian3 | 1,575 | 1,693 |
| **CM3 + LightGBM + COVID filter** | **1,563** | 1,702 |

---

## 6. 한계

- LightGBM이 tons-weighted RMSE val 기준으로는 CM3를 아직 이기지 못한다 (16,744 vs 16,379)
- **이유**: 3yr-ahead 예측에서 FAF5 데이터만 쓰면 split당 1~2년치 학습 샘플밖에 없어 모델이 충분히 학습하지 못함
- 데이터가 더 쌓이거나(2025~) commodity-level 예측 후 집계하는 방식으로 개선 가능

---

## 7. 실행 방법

```bash
# 팀 공식 splits 기준 평가
python code/bayasgalan_model.py
```

**필요한 파일**:
- `data/faf4_faf5_fixed_year_splits/` — 팀 공식 splits (repo에 포함)
- `bayasgalan/data/faf4_faf5_fixed.csv` — LightGBM 학습용 원본 데이터 (없으면 CM3 baseline만 실행)

**의존성**:
```
pandas, numpy, lightgbm
```
