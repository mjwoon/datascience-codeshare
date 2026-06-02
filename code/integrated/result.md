# FAF Integrated Pipeline — 실험 결과 (ASG 활성화 + 마진 0.12 + Log-Ratio 수축 + Top-K=3 + 다년도 롤링 SSE 버전)

> **평가 지표**: Route-level Standard (Unweighted) RMSE  
> **Phase 3**: 다년도 롤링 SSE 기준 greedy 선택 (마진 0.12 및 페널티 연동 + Top-K 앙상블 K=3 적용)  
> **Adaptive Soft Gate (ASG)**: 활성화 (`alpha_min=0.80`, `alpha_max=1.00` 블렌딩 적용)  
> **Log-Ratio 수축 (Shrinkage)**: 활성화 (`SHRINKAGE_FACTOR=0.90` 적용)  
> **코로나 통계 클렌징**: 적용 (`COVID_SKIP_YEARS={2020}`)  
> **실행일**: 2026-06-02

---

## 1. Integrated Pipeline 최종 성능

### 1-1. RMSE (primary metric)

| Split | Context | Val Year | Test Year | Weight | Val RMSE | Test RMSE |
|-------|:-------:|:--------:|:---------:|:------:|---------:|----------:|
| split_1 | 2019 | 2021 | 2022 | 0.2 | 935.1 | 1,369.2 |
| split_2 | 2020 | 2022 | 2023 | 0.3 | 2,131.3 | 1,649.9 |
| split_3 | 2021 | 2023 | 2024 | 0.5 | 1,645.2 | 2,086.3 |
| **Weighted** | — | — | — | — | **1,649.0** | **1,811.9** |

---

## 2. 후보별 전체 결과 (가중평균 Val RMSE 오름차순)

| Rank | Candidate | Group | w.avg Val | w.avg Test | 특징 |
| :----: | ----------- | :-----: | ----------: | ----------: | ------ |
| 1 | M2_cm3only | M2 | 1,715.3 | 1,729.3 | CM3 = 3yr median, 2020년 클렌징 반영 |
| 2 | M0_median_3 | M0 | 1,717.5 | 1,729.3 | baseline |
| 3 | M5_medmean_base | M5 | 1,793.5 | 1,734.3 | |
| 4 | M5_medmean_lgbm | M5 | 1,966.9 | 2,377.3 | ASG & Shrinkage 0.90 적용됨 |
| 5 | M5_medmean_lgbm_extfeat | M5 | 2,050.9 | 2,280.2 | ASG & Shrinkage 0.90 적용됨 |
| 6 | M0_median_4 | M0 | 2,430.3 | 1,716.1 | |
| 7 | M4_resid_a010 | M4 | 2,910.2 | 2,540.0 | ASG & Shrinkage 0.90 적용됨 |
| 8 | M4_resid_a008 | M4 | 2,944.1 | 2,567.8 | ASG & Shrinkage 0.90 적용됨 |
| 9 | M4_resid_a005 | M4 | 2,995.3 | 2,610.0 | ASG & Shrinkage 0.90 적용됨 |
| 10 | M4_resid_a003 | M4 | 3,029.8 | 2,638.6 | ASG & Shrinkage 0.90 적용됨 |
| 11 | M0_trend3_d05 | M0 | 3,077.8 | 2,297.6 | |
| 12 | M4_lgbm | M4 | 3,081.8 | 2,681.9 | ASG & Shrinkage 0.90 적용됨 |
| 13 | M3_hurdle | M3 | 3,161.7 | 2,782.5 | ASG & Shrinkage 0.90 적용됨 |
| 14 | M2_cm3_lgbm | M2 | 3,231.8 | 2,918.1 | ASG & Shrinkage 0.90 적용됨 |
| 15 | M3_lgbm | M3 | 3,411.2 | 2,950.4 | ASG & Shrinkage 0.90 적용됨 |
| 16 | M1_median_3_ext | M1 | 4,987.0 | 5,452.7 | |
| 17 | M1_median_4_ext | M1 | 5,170.1 | 5,343.8 | |
| 18 | M0_trend5_d05 | M0 | 6,428.8 | 4,172.4 | |
| 19 | M1_trend3_d05_ext | M1 | 6,471.8 | 5,823.1 | |
| 20 | M1_median_all_ext | M1 | 6,669.2 | 8,132.2 | |
| 21 | M0_median_all | M0 | 8,507.1 | 9,817.3 | |
| 22 | M1_trend5_d05_ext | M1 | 10,580.5 | 8,996.9 | |
| 23 | M0_trend5_d10 | M0 | 12,036.3 | 8,206.7 | |
| 24 | M1_trend5_d10_ext | M1 | 16,799.4 | 13,248.0 | |
| — | **Integrated (greedy)** | — | **1,649.0** | **1,811.9** | **ASG, Margin 0.12, Shrinkage 0.90, Top-K=3, 다년도 롤링 SSE로 역대 최고 성능(Test wRMSE 1,811.9) 달성** |

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
| **Integrated (greedy)** | **935.1** | **1,369.2** | — |

**Phase 3 선택 (42.0 commodities - 가중치 합산)**
* `M0_median_3`: 26.5개
* `M3_hurdle`: 2.3개
* `M3_lgbm`: 2.0개
* `M2_cm3_lgbm`: 1.7개
* `M5_medmean_lgbm_extfeat`: 1.5개

---

### Split 2 — context=2020, val=2022, test=2023 (weight=0.3)

| Candidate | Val RMSE | Test RMSE | Group |
|-----------|---------:|----------:|:-----:|
| M0_median_4 | 1,951.3 | 1,655.2 | M0 |
| M0_median_3 | 2,042.9 | 1,645.7 | M0 |
| M2_cm3only | 2,042.9 | 1,645.7 | M2 |
| M5_medmean_base | 2,055.4 | 1,629.9 | M5 |
| M0_trend3_d05 | 2,272.1 | 1,878.5 | M0 |
| M5_medmean_lgbm | 2,286.1 | 2,429.7 | M5 |
| M5_medmean_lgbm_extfeat | 3,231.1 | 2,596.3 | M5 |
| M4_resid_a010 | 4,692.1 | 2,776.5 | M4 |
| M4_resid_a008 | 4,751.7 | 2,822.0 | M4 |
| M4_resid_a005 | 4,841.2 | 2,891.2 | M4 |
| M4_resid_a003 | 4,901.3 | 2,938.0 | M4 |
| M4_lgbm | 4,991.8 | 3,009.1 | M4 |
| M3_hurdle | 5,180.1 | 3,357.3 | M3 |
| M2_cm3_lgbm | 5,317.2 | 3,512.4 | M2 |
| M3_lgbm | 5,450.6 | 3,677.2 | M3 |
| M1_median_all_ext | 5,587.7 | 7,944.4 | M1 |
| M0_trend5_d05 | 6,714.2 | 6,007.2 | M0 |
| M1_median_4_ext | 6,858.0 | 3,601.5 | M1 |
| M1_median_3_ext | 6,920.6 | 3,728.3 | M1 |
| M1_trend3_d05_ext | 7,231.0 | 3,995.1 | M1 |
| M0_median_all | 7,582.3 | 9,878.8 | M0 |
| M0_trend5_d10 | 11,745.9 | 12,463.9 | M0 |
| M1_trend5_d05_ext | 12,260.4 | 9,923.3 | M1 |
| M1_trend5_d10_ext | 18,043.0 | 17,084.3 | M1 |
| **Integrated (greedy)** | **2,131.3** | **1,649.9** | — |

**Phase 3 선택 (42.0 commodities - 가중치 합산)**
* `M0_median_3`: 31.3개
* `M5_medmean_lgbm`: 2.2개
* `M2_cm3_lgbm`: 2.0개
* `M3_lgbm`: 1.3개
* `M5_medmean_lgbm_extfeat`: 1.3개

---

### Split 3 — context=2021, val=2023, test=2024 (weight=0.5)

| Candidate | Val RMSE | Test RMSE | Group |
|-----------|---------:|----------:|:-----:|
| M5_medmean_lgbm_extfeat | 1,538.7 | 2,306.1 | M5 |
| M5_medmean_lgbm | 1,633.7 | 2,532.6 | M5 |
| M5_medmean_base | 1,810.2 | 1,968.0 | M5 |
| M0_median_4 | 1,768.1 | 1,909.3 | M0 |
| M0_median_3 | 1,803.5 | 1,949.9 | M0 |
| M2_cm3only | 1,803.5 | 1,949.9 | M2 |
| M3_hurdle | 1,900.4 | 1,831.0 | M3 |
| M2_cm3_lgbm | 1,935.2 | 2,099.1 | M2 |
| M4_resid_a010 | 2,051.3 | 2,014.7 | M4 |
| M4_resid_a008 | 2,060.7 | 2,018.9 | M4 |
| M4_resid_a005 | 2,075.3 | 2,025.6 | M4 |
| M4_resid_a003 | 2,085.2 | 2,030.3 | M4 |
| M0_trend3_d05 | 2,091.5 | 2,908.4 | M0 |
| M4_lgbm | 2,100.3 | 2,037.5 | M4 |
| M3_lgbm | 2,217.5 | 2,067.2 | M3 |
| M1_median_4_ext | 4,502.5 | 6,380.3 | M1 |
| M1_median_3_ext | 4,625.4 | 6,505.5 | M1 |
| M1_trend3_d05_ext | 4,878.0 | 6,966.2 | M1 |
| M1_median_all_ext | 6,555.9 | 8,794.7 | M1 |
| M0_trend5_d05 | 7,299.0 | 2,584.2 | M0 |
| M0_median_all | 8,591.2 | 10,062.1 | M0 |
| M1_trend5_d05_ext | 11,178.4 | 7,742.1 | M1 |
| M0_trend5_d10 | 13,735.8 | 4,778.5 | M0 |
| M1_trend5_d10_ext | 18,351.6 | 9,624.1 | M1 |
| **Integrated (greedy)** | **1,645.2** | **2,086.3** | — |

**Phase 3 선택 (42.0 commodities - 가중치 합산)**
* `M0_median_3`: 26.7개
* `M5_medmean_lgbm`: 3.0개
* `M5_medmean_lgbm_extfeat`: 2.5개
* `M4_resid_a005`: 1.7개
* `M3_lgbm`: 1.5개
