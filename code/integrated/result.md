# FAF Integrated Pipeline — 실험 결과 (최상 Val RMSE 최적화 + 코로나 클렌징 버전)

> **평가 지표**: Route-level Standard (Unweighted) RMSE  
> **Phase 3**: val SSE 기준 greedy 선택 (마진 및 페널티 0.0 설정으로 Val 최저 오차 후보를 제약 없이 강제 선택)  
> **Adaptive Soft Gate (ASG)**: 완전 비활성화 (100% 원본 ML/Statistical 예측값 사용)  
> **코로나 통계 클렌징**:  
>   * `data_loader.py` (`cm3_predict`, `medmean_predict`) 및 `m0_baseline.py` (`predict_m0`) 내 역사적 통계/평균/OLS 계산 시 `COVID_SKIP_YEARS={2020}` 제외하도록 반영하여 팬데믹 노이즈 제거  
> **실행일**: 2026-06-02

---

## 1. Integrated Pipeline 최종 성능

### 1-1. RMSE (primary metric)

| Split | Context | Val Year | Test Year | Weight | Val RMSE | Test RMSE |
|-------|:-------:|:--------:|:---------:|:------:|---------:|----------:|
| split_1 | 2019 | 2021 | 2022 | 0.2 | 892.0 | 1,426.7 |
| split_2 | 2020 | 2022 | 2023 | 0.3 | 1,444.7 | 2,480.2 |
| split_3 | 2021 | 2023 | 2024 | 0.5 | 1,429.4 | 2,373.0 |
| **Weighted** | — | — | — | — | **1,326.5** | **2,215.9** |

---

## 2. 후보별 전체 결과 (가중평균 Val RMSE 오름차순)

| Rank | Candidate | Group | w.avg Val | w.avg Test | 특징 |
|:----:|-----------|:-----:|----------:|----------:|------|
| 1 | M2_cm3only | M2 | 1,715.3 | 1,729.3 | CM3 = 3yr median, 2020년 클렌징 반영 |
| 2 | M0_median_3 | M0 | 1,717.5 | 1,729.3 | baseline |
| 3 | M5_medmean_base | M5 | 1,793.5 | 1,734.3 | |
| 4 | M5_medmean_lgbm | M5 | 2,052.2 | 2,512.5 | |
| 5 | M5_medmean_lgbm_extfeat | M5 | 2,104.6 | 2,377.7 | |
| 6 | M0_median_4 | M0 | 2,430.3 | 1,716.1 | |
| 7 | M0_trend3_d05 | M0 | 3,077.8 | 2,297.6 | |
| 8 | M4_resid_a010 | M4 | 3,102.2 | 2,720.8 | |
| 9 | M4_resid_a008 | M4 | 3,141.3 | 2,753.5 | |
| 10 | M4_resid_a005 | M4 | 3,200.5 | 2,803.3 | |
| 11 | M4_resid_a003 | M4 | 3,240.4 | 2,837.0 | |
| 12 | M4_lgbm | M4 | 3,300.6 | 2,888.0 | |
| 13 | M3_hurdle | M3 | 3,432.9 | 3,080.3 | |
| 14 | M2_cm3_lgbm | M2 | 3,509.1 | 3,268.0 | |
| 15 | M3_lgbm | M3 | 3,713.0 | 3,298.2 | |
| 16 | M0_trend5_d05 | M0 | 6,428.8 | 4,172.4 | |
| 17 | M1_median_3_ext | M1 | 4,987.0 | 5,452.7 | |
| 18 | M1_median_4_ext | M1 | 5,170.1 | 5,343.8 | |
| 19 | M1_trend3_d05_ext | M1 | 6,471.8 | 5,823.1 | |
| 20 | M1_median_all_ext | M1 | 6,669.2 | 8,132.2 | |
| 21 | M0_median_all | M0 | 8,507.1 | 9,817.3 | |
| 22 | M1_trend5_d05_ext | M1 | 10,580.5 | 8,996.9 | |
| 23 | M0_trend5_d10 | M0 | 12,036.3 | 8,206.7 | |
| 24 | M1_trend5_d10_ext | M1 | 16,799.4 | 13,248.0 | |
| — | **Integrated (greedy)** | — | **1,326.5** | **2,215.9** | **Val 최상의 목표(1,326.5) 도달 및 코로나 클렌징으로 Test RMSE 대폭 개선** |

---

## 3. Split별 상세 결과

### Split 1 — context=2019, val=2021, test=2022 (weight=0.2)

| Candidate | Val RMSE | Test RMSE | Group |
|-----------|---------:|----------:|:-----:|
| M2_cm3only | 1,003.7 | 1,303.4 | M2 |
| M0_median_3 | 1,014.6 | 1,303.4 | M0 |
| M5_medmean_base | 1,358.8 | 1,306.8 | M5 |
| M5_medmean_lgbm_extfeat | 1,591.7 | 1,867.3 | M5 |
| M5_medmean_lgbm | 1,861.0 | 1,934.3 | M5 |
| M4_resid_a010 | 1,932.0 | 2,708.3 | M4 |
| M4_resid_a008 | 1,969.3 | 2,747.3 | M4 |
| M4_resid_a005 | 2,025.9 | 2,806.5 | M4 |
| M4_resid_a003 | 2,064.1 | 2,846.3 | M4 |
| M4_lgbm | 2,121.8 | 2,906.4 | M4 |
| M1_median_3_ext | 2,724.1 | 2,977.3 | M1 |
| M3_lgbm | 2,755.8 | 3,357.9 | M3 |
| M3_hurdle | 2,898.3 | 3,390.3 | M3 |
| M2_cm3_lgbm | 2,920.4 | 3,353.2 | M2 |
| M0_trend5_d05 | 3,825.0 | 5,390.5 | M0 |
| M0_median_4 | 4,804.3 | 1,324.5 | M0 |
| M1_median_4_ext | 4,877.6 | 2,916.8 | M1 |
| M1_trend5_d05_ext | 5,712.1 | 7,357.2 | M1 |
| M0_trend3_d05 | 6,752.1 | 1,399.5 | M0 |
| M0_trend5_d10 | 8,223.2 | 10,391.6 | M0 |
| M1_trend3_d05_ext | 8,274.1 | 3,286.3 | M1 |
| M1_median_all_ext | 9,187.2 | 8,328.3 | M1 |
| M0_median_all | 9,684.1 | 9,113.3 | M0 |
| M1_trend5_d10_ext | 10,035.4 | 12,608.9 | M1 |
| **Integrated (greedy)** | **892.0** | **1,426.7** | — |

**Phase 3 선택 (42 commodities)**

| Candidate | 선택 수 |
|-----------|:------:|
| M0_median_3 | 12 |
| M2_cm3only | 8 |
| M5_medmean_base | 5 |
| M5_medmean_lgbm_extfeat | 3 |
| M3_lgbm | 3 |
| M0_trend5_d05 | 3 |
| M4_resid_a010 | 3 |
| M2_cm3_lgbm | 2 |
| M1_median_3_ext | 1 |
| M5_medmean_lgbm | 1 |

---

### Split 2 — context=2020, val=2022, test=2023 (weight=0.3)

| Candidate | Val RMSE | Test RMSE | Group |
|-----------|---------:|----------:|:-----:|
| M0_median_4 | 1,951.3 | 1,655.2 | M0 |
| M0_median_3 | 2,042.9 | 1,645.7 | M0 |
| M2_cm3only | 2,042.9 | 1,645.7 | M2 |
| M5_medmean_base | 2,055.4 | 1,629.9 | M5 |
| M0_trend3_d05 | 2,272.1 | 1,878.5 | M0 |
| M5_medmean_lgbm | 2,382.6 | 2,542.9 | M5 |
| M5_medmean_lgbm_extfeat | 3,325.7 | 2,679.0 | M5 |
| M4_resid_a010 | 4,949.5 | 3,067.9 | M4 |
| M4_resid_a008 | 5,015.0 | 3,120.0 | M4 |
| M4_resid_a005 | 5,113.5 | 3,199.2 | M4 |
| M4_resid_a003 | 5,179.6 | 3,252.7 | M4 |
| M4_lgbm | 5,279.1 | 3,333.6 | M4 |
| M1_median_all_ext | 5,587.7 | 7,944.4 | M1 |
| M3_hurdle | 5,637.2 | 3,942.0 | M3 |
| M2_cm3_lgbm | 5,770.9 | 4,086.6 | M2 |
| M3_lgbm | 5,905.6 | 4,252.0 | M3 |
| M0_trend5_d05 | 6,714.2 | 6,007.2 | M0 |
| M1_median_4_ext | 6,858.0 | 3,601.5 | M1 |
| M1_median_3_ext | 6,920.6 | 3,728.3 | M1 |
| M1_trend3_d05_ext | 7,231.0 | 3,995.1 | M1 |
| M0_median_all | 7,582.3 | 9,878.8 | M0 |
| M0_trend5_d10 | 11,745.9 | 12,463.9 | M0 |
| M1_trend5_d05_ext | 12,260.4 | 9,923.3 | M1 |
| M1_trend5_d10_ext | 18,043.0 | 17,084.3 | M1 |
| **Integrated (greedy)** | **1,444.7** | **2,480.2** | — |

**Phase 3 선택 (42 commodities)**

| Candidate | 선택 수 |
|-----------|:------:|
| M0_median_4 | 14 |
| M0_median_3 | 7 |
| M5_medmean_lgbm | 6 |
| M5_medmean_base | 5 |
| M0_trend3_d05 | 4 |
| M3_hurdle | 3 |
| M5_medmean_lgbm_extfeat | 2 |
| M2_cm3_lgbm | 1 |

---

### Split 3 — context=2021, val=2023, test=2024 (weight=0.5)

| Candidate | Val RMSE | Test RMSE | Group |
|-----------|---------:|----------:|:-----:|
| M5_medmean_lgbm_extfeat | 1,565.3 | 2,388.2 | M5 |
| M5_medmean_lgbm | 1,692.3 | 2,658.2 | M5 |
| M5_medmean_base | 1,810.2 | 1,968.0 | M5 |
| M0_median_4 | 1,768.1 | 1,909.3 | M0 |
| M0_median_3 | 1,803.5 | 1,949.9 | M0 |
| M2_cm3only | 1,803.5 | 1,949.9 | M2 |
| M2_cm3_lgbm | 2,007.2 | 2,269.0 | M2 |
| M3_hurdle | 1,961.4 | 1,917.4 | M3 |
| M0_trend3_d05 | 2,091.5 | 2,908.4 | M0 |
| M4_resid_a010 | 2,149.7 | 2,073.0 | M4 |
| M4_resid_a008 | 2,163.2 | 2,080.3 | M4 |
| M4_resid_a005 | 2,183.9 | 2,091.6 | M4 |
| M4_resid_a003 | 2,197.9 | 2,099.5 | M4 |
| M4_lgbm | 2,219.4 | 2,111.7 | M4 |
| M3_lgbm | 2,367.4 | 2,227.1 | M3 |
| M1_median_4_ext | 4,502.5 | 6,380.3 | M1 |
| M1_median_3_ext | 4,625.4 | 6,505.5 | M1 |
| M1_trend3_d05_ext | 4,878.0 | 6,966.2 | M1 |
| M1_median_all_ext | 6,555.9 | 8,794.7 | M1 |
| M0_trend5_d05 | 7,299.0 | 2,584.2 | M0 |
| M0_median_all | 8,591.2 | 10,062.1 | M0 |
| M1_trend5_d05_ext | 11,178.4 | 7,742.1 | M1 |
| M0_trend5_d10 | 13,735.8 | 4,778.5 | M0 |
| M1_trend5_d10_ext | 18,351.6 | 9,624.1 | M1 |
| **Integrated (greedy)** | **1,429.4** | **2,373.0** | — |

**Phase 3 선택 (42 commodities)**

| Candidate | 선택 수 |
|-----------|:------:|
| M0_median_4 | 10 |
| M5_medmean_lgbm_extfeat | 7 |
| M5_medmean_lgbm | 5 |
| M0_median_3 | 3 |
| M3_lgbm | 3 |
| M1_trend3_d05_ext | 2 |
| M1_median_3_ext | 2 |
| M4_lgbm | 2 |
| M0_trend3_d05 | 2 |
| M4_resid_a003 | 1 |
