# Experiment Context Guide

이 문서는 앞으로 실험을 이어갈 때 항상 같이 읽히는 최소 컨텍스트다.

## 1. 프로젝트 목적

미국 내 origin-destination 간 트럭 화물 수요를 예측해 물류 허브와 트럭 배치를 선제적으로 판단하는 것이 목적이다.

주 예측 대상:

```text
origin-destination route-level annual tons
```

`value`는 비즈니스 인사이트와 보조 feature로 쓸 수 있지만, 주 목표는 운송 물량이다. 트럭/창고 수요는 화물 가치보다 중량과 물동량에 더 직접적으로 연결되기 때문이다.

## 2. 데이터

현재 레포 기준 주 데이터:

- source: `data/additional_data/faf4_faf5_fixed.csv`
- split root: `data/faf4_faf5_fixed_year_splits/`
- 주요 컬럼: `origin`, `destination`, `commodity`, `distance_band`, `year`, `tons`, `value`, `tmiles`

평가용 validation/test target은 route-level로 사전 집계되어 있으며 다음 컬럼만 가진다.

```text
origin, destination, year, tons
```

이렇게 고정한 이유는 평가 시점에 target year의 commodity/value/tmiles 정보를 feature처럼 쓰는 leakage를 막기 위해서다.

Commodity-level 보조 target도 같은 split root에 추가로 생성한다.

```text
val_commodity.csv
test_commodity.csv

origin, destination, commodity, year, tons
```

이 파일은 commodity 단위 모델 학습/진단을 위한 보조 답지다. 공식 성능 비교는 기존 route-level `val.csv`/`test.csv`를 유지하고, commodity-level 예측을 route 단위로 합산한 뒤 평가한다. `val_commodity.csv`/`test_commodity.csv`를 `origin, destination, year`로 합산하면 기존 route-level 답지와 일치해야 한다.

보조 데이터 후보:

- FRED: GDP, 고용/실업률, 물가 등
- EIA: diesel fuel price
- Census: 인구
- 산업/항만/건설/농업/에너지 관련 지표

외부 변수는 반드시 예측 시점 이전에 알 수 있는 값만 사용한다. 예를 들어 2024년 target을 3-year horizon으로 예측할 때 2024년에 확정되는 GDP는 사용할 수 없다.

## 3. 예측 설정

현재 실험의 표준 설정:

- input window: 3년
- forecast horizon: 3년
- split 방식: expanding time-style split
- primary metric: split-weighted RMSE

| split | train | context | validation | test | weight |
| --- | --- | --- | --- | --- | ---: |
| split_1 | 2012-2018 | 2019 | 2021 | 2022 | 0.2 |
| split_2 | 2012-2019 | 2020 | 2022 | 2023 | 0.3 |
| split_3 | 2012-2020 | 2021 | 2023 | 2024 | 0.5 |

`context.csv`는 test 예측에 필요한 최신 과거연도를 제공하기 위한 파일이다. 예를 들어 split 3의 2024년 예측은 2019-2021 input window를 쓰고, 2021은 `context.csv`에서 온다.

보조 metric:

- MAE: 원본 tons 단위의 평균 절대 오차
- WMAPE: 비즈니스 커뮤니케이션용 비율 오차
- RMSLE: 대형 route가 지표를 지배하는 상황에서 중소형 route의 비율 오차를 함께 확인하기 위한 지표
- R2: 설명력 참고 지표

성능 비교는 validation weighted RMSE를 우선한다. test 결과는 최종 확인과 해석에 사용하고, test만 좋아진 후보를 selected model로 올리지 않는다.

## 4. 현재 선택 모델

현재 가장 방어 가능한 모델은 다음이다.

```text
commodity_median_allhistory_none
```

정의:

```text
각 origin-destination-commodity 단위에서
input window 안의 available tons 중앙값을 예측값으로 사용한다.

최종 route 예측값은 같은 origin-destination에 속한
commodity 예측값을 합산한다.
```

핵심 성능:

| model | validation weighted RMSE | test weighted RMSE | test weighted WMAPE |
| --- | ---: | ---: | ---: |
| NaiveLag1 | 2156.3440 | 2554.7754 | 0.0492 |
| route Avg3 | 1831.5168 | 2001.2569 | 0.0418 |
| commodity_median_allhistory_none | 1556.8987 | 1900.9532 | 0.0409 |
| controlled YoY-drop commodity median | 1774.1988 | 1899.3259 | 0.0406 |

해석:

- naive lag-1 대비 test weighted RMSE를 약 25.6% 줄인다.
- route 전체를 직접 예측하는 것보다 commodity 단위로 예측 후 합산하는 방식이 안정적이다.
- mean보다 median이 commodity bottom-up 구조에서 더 안정적이다.
- YoY shock drop은 test를 아주 조금 개선하지만 validation을 크게 악화시키므로 sensitivity result로만 둔다.

## 5. 데이터 해석상 주의

FAF4와 FAF5는 완전히 같은 기준의 연속 시계열로 보기 어렵다.

중요한 구조 단절:

- 2012-2016: FAF4 기반 구간, detail row가 연도별 약 157k-177k, distance band는 사실상 unknown 1개
- 2017-2021: FAF5 기반 구간, 연도별 71,852 rows, route 2,493개, distance band 8개
- 2016 -> 2017 one-year route naive RMSE는 corrected data 기준 `8850.12`로 여전히 크다

모델링 원칙:

- FAF4/FAF5를 동일 기준의 매끈한 시계열로 가정하지 않는다.
- 2017년 이전 distance_band feature는 post-2017 distance_band와 같은 의미로 해석하지 않는다.
- 오래된 history를 많이 넣는 것보다 최근 3년의 robust summary가 더 안정적이었다.
- pre-2017 보정은 현재 best median baseline을 이기지 못했다.

## 6. 실험 규칙

새 실험은 다음을 지킨다.

- validation/test 답지는 절대 바꾸지 않는다. 평가 target 파일, target 집계 로직, validation/test route-year 정답은 고정한다.
- commodity-level 보조 답지(`val_commodity.csv`, `test_commodity.csv`)는 세분화 학습/진단용으로만 사용한다. selected model 성능은 기존 route-level 답지로 산출한다.
- train/context 생성이나 feature engineering은 바꿀 수 있지만, validation/test 정답을 다시 만들거나 후처리해서 성능을 맞추지 않는다.
- split 파일과 target 집계 방식을 바꾸는 일은 원칙적으로 금지한다. 데이터 오류 수정처럼 불가피한 경우에는 새 버전으로 명시하고 기존 결과와 섞지 않는다.
- validation/test target year의 commodity, value, tmiles를 feature로 쓰지 않는다.
- test score로 모델을 고른 뒤 selected model이라고 말하지 않는다.
- 새 모델은 최소한 `NaiveLag1`, `route Avg3`, `commodity_median_allhistory_none`과 비교한다.
- weighted RMSE와 함께 MAE, WMAPE, RMSLE, R2를 남긴다.
- 외부 변수는 availability cutoff를 명시한다.

## 7. 모델링 방향

이 프로젝트의 기본 모델링 방향은 raw `tons` 자체를 처음부터 supervised model로 직접 학습하는 것이 아니다.

먼저 통계 기반 baseline으로 안정적인 물동량 추정치를 만든다.

```text
base_pred = commodity_median_allhistory_none
```

그 다음 모델은 baseline이 설명하지 못한 오차만 학습한다.

```text
residual = actual_tons - base_pred
model_target = residual
final_pred = base_pred + predicted_residual
```

또는 scale 차이가 큰 route를 안정화하기 위해 log-ratio correction을 사용할 수 있다.

```text
log_ratio = log((actual_tons + 1) / (base_pred + 1))
final_pred = base_pred * exp(predicted_log_ratio)
```

즉, `commodity_median_allhistory_none`은 버릴 baseline이 아니라 모델의 prior/base prediction이다. ML은 이 baseline 위에서 residual 또는 log-ratio를 보정하는 역할을 맡는다.

현재 이후 실험의 기준 방향은 `validation_selected_sum` 구조를 참고하되, 이를 직접 route-total 보정으로 개선하지 않는 것이다. 앞으로의 개선 대상은 다음 구조의 commodity별 residual 모델이다.

```text
route_pred = sum_commodity(base_prediction + per_commodity_residual_prediction)
```

따라서 새 실험은 commodity별 residual 후보를 더 잘 만들고 선택하는 데 집중한다. 대형 route를 고려하더라도 top route는 test error가 아니라 train-only rolling/backtest 또는 validation에서 반복적으로 확인된 commodity-route residual 패턴으로 선택해야 한다. `validation_selected_sum`은 강한 legacy benchmark로 사용하되, 최종 selected model로 쓰려면 각 commodity의 route/gate/alpha 선택이 train-only 또는 validation-only 절차로 재현되어야 한다.

## 8. MLflow 규칙

MLflow는 팀 단위 실험 추적과 재현성 관리를 위한 표준 기록 도구로 사용한다.

접속 정보:

- tracking server URL: `https://mlflow.hyu.life`
- experiment name: `Truck_Route_Volume_Prediction`
- run name 권장 형식: `[실험자이름]_[모델명/알고리즘]_[자동생성시간]`
- run name 예시: `Dongbin_XGBoost_0325_153022`

필수 tag:

- `dataset_version`: 사용한 데이터셋 또는 split 버전
- `primary_metric`: `weighted_RMSE`
- `description`: 실험 목적 요약
- split별 train/val/test/context 범위
- 외부 변수 사용 시 availability cutoff

필수 param:

- 모델명
- 주요 하이퍼파라미터
- random seed
- feature set 이름
- baseline/base prediction 이름
- residual target 종류: `residual`, `log_ratio`, `direct_tons` 등

필수 metric:

- split별 원본 지표:
  - `split1_RMSE`, `split2_RMSE`, `split3_RMSE`
  - `split1_MAE`, `split2_MAE`, `split3_MAE`
  - `split1_WMAPE`, `split2_WMAPE`, `split3_WMAPE`
  - `split1_RMSLE`, `split2_RMSLE`, `split3_RMSLE`
  - `split1_R2`, `split2_R2`, `split3_R2`
- 최종 weighted 지표:
  - `weighted_RMSE`
  - `weighted_MAE`
  - `weighted_WMAPE`
  - `weighted_RMSLE`
  - `weighted_R2`

필수 artifact:

- 실행한 코드 원본 또는 이 레포 안의 script/notebook path 기록
- config 파일
- feature column list
- split별 prediction 파일 또는 prediction summary
- leaderboard CSV

재현성 규칙:

- random seed를 고정한다.
- 데이터셋 원본은 artifact로 올리지 않는다.
- validation/test 답지를 artifact로 새로 생성해 덮어쓰지 않는다.
- MLflow에 기록된 metric은 고정된 validation/test 답지로 산출된 값이어야 한다.
