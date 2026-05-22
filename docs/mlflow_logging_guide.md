# MLflow Logging Guide

현재 코드 공유 레포의 MLflow 표준 기록 대상은 로컬 `mlruns`가 아니라 팀 서버다.

- tracking URI: `https://mlflow.hyu.life`
- experiment: `Truck_Route_Volume_Prediction`
- primary metric: `weighted_RMSE`
- metric schema tag: `mlflow_sample_compatible_v2`
- dataset version: `faf4_faf5_fixed_state_route_v1`

## Run Name Rules

run name은 아래 형식을 쓴다.

```text
dongbin_{run_alias}_{MMDD_HHMMSS}
```

예시:

```text
dongbin_route_stack_validation_selected_raw_sum_0520_233025
```

규칙:

- `run_alias`: 모델/실험을 구분할 수 있는 짧은 snake_case 이름
- `MMDD_HHMMSS`: 실행 시각
- 전체 길이는 MLflow 제한을 피하기 위해 250자 이하로 자른다.
- 같은 결과를 재기록할 때도 timestamp를 새로 붙이고, 기존 run을 덮어쓰지 않는다.

## Metric Rules

기존 스크립트 기준 metric 이름은 아래 key를 그대로 사용한다.

test weighted metrics:

- `weighted_RMSE`
- `weighted_MAE`
- `weighted_WMAPE`
- `weighted_RMSLE`
- `weighted_R2_Score`

validation weighted metrics:

- `val_weighted_RMSE`
- `val_weighted_MAE`
- `val_weighted_WMAPE`
- `val_weighted_RMSLE`
- `val_weighted_R2_Score`

test split metrics:

- `split1_RMSE`
- `split1_MAE`
- `split1_WMAPE`
- `split1_RMSLE`
- `split1_R2_Score`
- `split2_RMSE`
- `split2_MAE`
- `split2_WMAPE`
- `split2_RMSLE`
- `split2_R2_Score`
- `split3_RMSE`
- `split3_MAE`
- `split3_WMAPE`
- `split3_RMSLE`
- `split3_R2_Score`

validation split metrics:

- `val_split1_RMSE`
- `val_split1_MAE`
- `val_split1_WMAPE`
- `val_split1_RMSLE`
- `val_split1_R2_Score`
- `val_split2_RMSE`
- `val_split2_MAE`
- `val_split2_WMAPE`
- `val_split2_RMSLE`
- `val_split2_R2_Score`
- `val_split3_RMSE`
- `val_split3_MAE`
- `val_split3_WMAPE`
- `val_split3_RMSLE`
- `val_split3_R2_Score`

중요: test weighted RMSE는 반드시 `weighted_RMSE`로 남긴다. `test_weighted_RMSE`나 `test_weighted_rmse`를 primary metric으로 쓰지 않는다.

## Required Metadata

최소 tag:

- `dataset_version`
- `primary_metric = weighted_RMSE`
- `metric_schema = mlflow_sample_compatible_v2`
- `result_dir`
- `model_name`
- `selection_status`
- split별 train/context/val/test/weight

최소 artifact:

- 결과 CSV
- 실행한 source script 또는 notebook
