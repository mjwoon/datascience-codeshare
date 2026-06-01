# Additional Dataset 명세서

**목적**: FAF truck freight residual model 개선을 위한 commodity별, 경제·기후 external feature  
**수집일**: 2026-05-06  
**기준 범위**: 주로 2005–2024, 미국 state 단위 (예외는 각 파일 설명 참조)

---

## ⚠️ 병합 전 주의사항

| 이슈 | 내용 |
| --- | --- |
| **비료 가격 — 암모니아 누락** | World Bank 데이터에 DAP·Urea만 포함, 암모니아 없음 |
| **PDSI / HDD: AK·HI 제외** | NOAA CONUS 기준, 알래스카·하와이 미포함 (48개 주) |
| **석탄 생산: 26개 주만 존재** | 광산이 없는 24개 주는 행 자체가 없음 (0이 아닌 NaN 처리 주의) |
| **Migration: 2020년 결측** | IRS SOI는 COVID로 인해 2020년 데이터 미발표 |
| **Income: 2020년 결측** | ACS도 동일한 이유로 2020년 결측 |

---

## 파일 목록 요약

| # | 파일명 | 대상 Commodity | 단위 | Year Range | State 수 |
| --- | --- | --- | --- | --- | --- |
| 1 | `eia_coal_production_by_state.csv` | Coal | short tons | 2005–2024 | 26 |
| 2 | `eia_coal_electricity_gen_by_state.csv` | Coal | GWh | 2005–2024 | 49 |
| 3 | `eia_natgas_electricity_gen_national.csv` | Coal (displacement) | GWh | 2005–2024 | 전국 |
| 4 | `eia_heating_degree_days_by_state.csv` | Fuel oils | degree-days | 2005–2024 | 48 |
| 5 | `noaa_pdsi_by_state.csv` | Cereal grains, Logs | 무차원 | 2005–2024 | 48 |
| 6 | `usda_crop_production_by_state.csv` | Cereal grains, Fertilizers | acres / bushels | 2005–2024 | 49 |
| 7 | `fao_fertilizer_price_index.csv` | Fertilizers | USD/metric ton | 2005–2024 | 전국 |
| 8 | `income_annual.csv` | All (수요 proxy) | USD | 2017–2024 | ~50 |
| 9 | `migration_od_annual.csv` | All (수요 이동) | 명 | 2012–2024 | 51→51 OD |
| 10 | `pce_by_state_real_2017.csv` | All (소비 패턴) | 백만 USD | 1997–2023 | 50 |
| 11 | `pce_by_state_real_2017_changes.csv` | All (소비 변화율) | USD / % | 1997–2023 | 50 |
| 12 | `population_by_states.csv` | All (수요 정규화) | 명 | 2012–2024 | ~51 |
| 13 | `state_unify_gdp.csv` | All (경제 활동) | 백만 USD | 2005–2025 | ~51 |
| 14 | `state_unify_ue-rate.csv` | All (경기 사이클) | % | 1976–2025 | ~51 |

---

## 파일별 상세 설명

---

### 1. `eia_coal_production_by_state.csv`

**Source**: EIA — Coal Aggregate Production (`/v2/coal/aggregate-production/`)  
**대상 Commodity**: Coal  
**Shape**: 481 rows × 3 columns · 26개 주 (광산 있는 주만)

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `state_name` | str | — | 주 이름 |
| `year` | int | — | 연도 |
| `coal_production_ktons` | int | short tons | 전체 mine type 합산 생산량 |

**모델 활용**: 석탄 freight의 supply-side feature. 생산량 증감이 출발지 freight tonnage와 직접 연관됨. 생산량의 구조적 감소 trend가 residual에 반영됨.

---

### 2. `eia_coal_electricity_gen_by_state.csv`

**Source**: EIA — Electric Power Operational Data (fuel `COW`)  
**대상 Commodity**: Coal  
**Shape**: 965 rows × 3 columns · 49개 주

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `state_name` | str | — | 주 이름 |
| `year` | int | — | 연도 |
| `coal_elec_gen_gwh` | float | GWh | 석탄 연소 발전량 |

**모델 활용**: 석탄 소비의 demand-side proxy. 발전소 소비 = freight destination 수요. 2012년 이후 천연가스·재생에너지에 의해 급격히 감소하는 trend.

---

### 3. `eia_natgas_electricity_gen_national.csv`

**Source**: EIA — Electric Power Operational Data (연료 `NG`, 위치 `US`)  
**대상 Commodity**: Coal (displacement indicator)  
**Shape**: 20 rows × 2 columns · 전국 단위

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `year` | int | — | 연도 |
| `natgas_elec_gen_gwh` | float | GWh | 천연가스 발전량 (전국 합계) |

**모델 활용**: NG 발전 ↑ → coal 수요 ↓ displacement 관계. Coal freight residual의 national-level trend 설명. `year`만으로 모든 state 행에 join.

---

### 4. `eia_heating_degree_days_by_state.csv`

**Source**: NOAA Climate at a Glance (statewide HDD)  
**대상 Commodity**: Fuel oils  
**Shape**: 960 rows × 3 columns · 48개 주 (AK·HI 제외)

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `state_name` | str | — | 주 이름 |
| `year` | int | — | 연도 |
| `hdd_annual` | int | degree-days | 연간 Heating Degree Days 합계 (기준 65°F) |

**값 범위**: 429 (Florida 온난년) ~ 10,201 (Minnesota 혹한년)

**모델 활용**: HDD = max(0, 65°F − 일평균기온)의 연간 합. HDD ↑ → 난방 수요 ↑ → fuel oil freight 증가.

---

### 5. `noaa_pdsi_by_state.csv`

**Source**: NOAA Climate at a Glance (Palmer Drought Severity Index)  
**대상 Commodity**: Cereal grains, Logs  
**Shape**: 960 rows × 3 columns · 48개 주 (AK·HI 제외)

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `state_name` | str | — | 주 이름 |
| `year` | int | — | 연도 |
| `pdsi_annual_avg` | float | 무차원 | 연간 평균 PDSI (12개월 평균) |

| PDSI 범위 | 상태 |
| --- | --- |
| ≥ 4.0 | Extremely wet |
| −0.4 ~ 0.4 | Near normal |
| −2.9 ~ −2.0 | Moderate drought |
| ≤ −4.0 | Extreme drought |

**모델 활용**: 가뭄 → 곡물 수확 감소 → cereal grain freight 감소. 중서부 농업 주의 residual 설명에 핵심 feature.

---

### 6. `usda_crop_production_by_state.csv`

**Source**: USDA NASS QuickStats API  
**대상 Commodity**: Cereal grains, Fertilizers  
**Shape**: 2,672 rows × 7 columns · 49개 주  
**Commodities**: CORN, WHEAT, SOYBEANS

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `state_name` | str | — | 주 이름 |
| `year` | int | — | 연도 |
| `commodity` | str | — | `CORN`, `WHEAT`, `SOYBEANS` |
| `harvested_acres` | float | acres | 실제 수확 면적 |
| `planted_acres` | float | acres | 파종 면적 |
| `production_bu` | float | bushels | 총 생산량 |
| `yield_bu_per_acre` | float | bu/acre | 단위 면적당 생산성 |

**모델 활용**: 작물 생산량이 cereal freight의 직접 demand driver. `yield_bu_per_acre`는 가뭄·기술 개선 효과를 내포하므로 PDSI와 조합 시 설명력 향상.

---

### 7. `fao_fertilizer_price_index.csv`

**Source**: World Bank Commodity Price Data ("Pink Sheet")  
**대상 Commodity**: Fertilizers  
**Shape**: 20 rows × 3 columns · 전국 단위 (연 1행)

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `year` | int | — | 연도 |
| `dap_price_usd_mt` | float | USD/metric ton | DAP (인산이암모늄) 국제 spot 가격 |
| `urea_price_usd_mt` | float | USD/metric ton | Urea (요소비료) 국제 spot 가격 |

**가격 동향**: 2022년 Russia-Ukraine 충격 → DAP $862, Urea $700 peak → 이후 하락.

**모델 활용**: 비료 가격 spike 시기의 freight demand shock (구매 연기·대체)를 포착. `year`만으로 모든 state 행에 join.

---

### 8. `income_annual.csv`

**Source**: U.S. Census Bureau — American Community Survey (ACS)  
**대상 Commodity**: All (소비 수요 proxy)  
**Shape**: 350 rows × 3 columns · ~50개 주  
**Year Range**: 2017–2024 (2020년 결측)

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `state_name` | str | — | 주 이름 |
| `year` | int | — | 연도 |
| `median_income` | float | USD | 가구 중위소득 |

**값 범위**: ~$45K (Mississippi) ~ ~$114K (Massachusetts), 2024 기준.

**모델 활용**: 소득 수준 ↑ → 소비재 freight 수요 ↑. 소득 성장률을 feature로 쓰면 demand-side secular trend를 residual에서 분리 가능. 2020년 결측은 interpolation 또는 전년도 값으로 보완.

---

### 9. `migration_od_annual.csv`

**Source**: IRS Statistics of Income (SOI) — Migration Data  
**대상 Commodity**: All (수요의 공간적 이동)  
**Shape**: 30,345 rows × 4 columns  
**Year Range**: 2012–2024 (2020년 결측)

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `origin` | str | — | 출발 주 |
| `destination` | str | — | 도착 주 |
| `year` | int | — | 연도 |
| `migrants` | int | 명 | 납세자 기준 이동 인원 |

**모델 활용**: 인구 이동이 freight의 OD 패턴을 변화시킴. Net migration (inflow − outflow)을 집계해 state-level feature로 변환 후 사용.

```python
inflow  = df.groupby(["destination", "year"])["migrants"].sum()
outflow = df.groupby(["origin", "year"])["migrants"].sum()
net = inflow.sub(outflow, fill_value=0).rename("net_migration")
```

---

### 10. `pce_by_state_real_2017.csv`

**Source**: BEA (Bureau of Economic Analysis) — State Personal Consumption Expenditures  
**대상 Commodity**: All (소비 패턴 breakdown)  
**Shape**: 34,272 rows × 4 columns · 50개 주  
**Year Range**: 1997–2023

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `state_name` | str | — | 주 이름 |
| `category` | str | — | 소비 범주 (20개 이상) |
| `year` | int | — | 연도 |
| `pce_value` | float | 백만 USD | Real PCE (2017 chained dollars) |

**주요 Category**:
- `Personal consumption expenditures` — 전체 합계
- `Goods` → `Durable goods` / `Nondurable goods`
- `Gasoline and other energy goods`
- `Food and beverages purchased for off-premises consumption`

**모델 활용**: Commodity 그룹에 해당하는 category를 필터링 → demand proxy로 사용. 예: fuel freight → `Gasoline and other energy goods`, 소비재 freight → `Goods` 전체.

---

### 11. `pce_by_state_real_2017_changes.csv`

**Source**: 파일 #10에서 파생 (연도 간 차분)  
**대상 Commodity**: All  
**Shape**: 33,048 rows × 8 columns

| Column | Type | 설명 |
| --- | --- | --- |
| `state_name` | str | 주 이름 |
| `category` | str | 소비 범주 |
| `prev_year` / `current_year` | int | 이전/현재 연도 |
| `pce_value_prev` / `pce_value_current` | float | 이전/현재 PCE 값 |
| `pce_diff` | float | 절대 변화량 (current − prev) |
| `pce_growth_rate` | float | YoY growth rate (소수점, 예: 0.043 = 4.3%) |

**모델 활용**: freight 변화율을 target으로 쓸 때 같은 scale의 feature를 바로 사용 가능. Recession 구간(음수 growth)을 binary flag feature로도 활용.

---

### 12. `population_by_states.csv`

**Source**: U.S. Census Bureau — Population Estimates Program  
**대상 Commodity**: All (수요 정규화 / 성장 driver)  
**Shape**: 676 rows × 3 columns  
**Year Range**: 2012–2024

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `state_name` | str | — | 주 이름 |
| `year` | int | — | 연도 |
| `population` | int | 명 | 연말 추정 인구 |

**모델 활용**: freight tonnage를 per-capita로 정규화 → 주 크기 bias 제거. 인구 성장률이 높은 주(Texas, Florida)의 구조적 freight 증가 설명.

---

### 13. `state_unify_gdp.csv`

**Source**: BEA — Regional GDP (Real, chained 2017 dollars)  
**대상 Commodity**: All (경제 활동 수준)  
**Shape**: 1,071 rows × 3 columns  
**Year Range**: 2005–2025 (2025는 잠정치)

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `year` | int | — | 연도 |
| `real_GDP` | float | 백만 USD | 주 실질 GDP |
| `state_name` | str | — | 주 이름 |

**모델 활용**: GDP는 freight 수요의 가장 강력한 macro-level driver. 경기 침체 구간(2008–09, 2020)의 residual spike 설명에 효과적. Freight intensity (tonnage / GDP) 분석에도 사용.

---

### 14. `state_unify_ue-rate.csv`

**Source**: BLS (Bureau of Labor Statistics) — State Unemployment  
**대상 Commodity**: All (경기 사이클 indicator)  
**Shape**: 2,550 rows × 3 columns  
**Year Range**: 1976–2025

| Column | Type | Unit | 설명 |
| --- | --- | --- | --- |
| `year` | int | — | 연도 |
| `unemployment_rate` | float | % | 연간 실업률 |
| `state_name` | str | — | 주 이름 |

**모델 활용**: 실업률 ↑ → 소비 위축 → freight 감소. 제조업 중심 주(Michigan, Ohio)에서 industrial goods freight와 correlation이 강함. 1976년부터의 긴 history는 경기 사이클 lag feature 생성에 유용.

---

## Merge Reference (Python)

```python
import pandas as pd

BASE = "for_model/additional_data/"

# Coal
coal_prod = pd.read_csv(BASE + "eia_coal_production_by_state.csv")
coal_elec = pd.read_csv(BASE + "eia_coal_electricity_gen_by_state.csv")
natgas    = pd.read_csv(BASE + "eia_natgas_electricity_gen_national.csv")

# Climate
hdd  = pd.read_csv(BASE + "eia_heating_degree_days_by_state.csv")
pdsi = pd.read_csv(BASE + "noaa_pdsi_by_state.csv")

# Agriculture
crops = pd.read_csv(BASE + "usda_crop_production_by_state.csv")
fert  = pd.read_csv(BASE + "fao_fertilizer_price_index.csv")

# Economic & demographic
income = pd.read_csv(BASE + "income_annual.csv")
gdp    = pd.read_csv(BASE + "state_unify_gdp.csv")
ue     = pd.read_csv(BASE + "state_unify_ue-rate.csv")
pop    = pd.read_csv(BASE + "population_by_states.csv")

# PCE
pce         = pd.read_csv(BASE + "pce_by_state_real_2017.csv")
pce_changes = pd.read_csv(BASE + "pce_by_state_real_2017_changes.csv")

# Migration → net migration 집계 후 join
migration = pd.read_csv(BASE + "migration_od_annual.csv")
net_mig = (
    migration.groupby(["destination", "year"])["migrants"].sum()
    .sub(migration.groupby(["origin", "year"])["migrants"].sum(), fill_value=0)
    .reset_index()
    .rename(columns={"destination": "state_name", "migrants": "net_migration"})
)

# Join 방법
# state-level    → merge on ["state_name", "year"]
# national-level (natgas, fert) → merge on ["year"] only
```

---

## Merge Key 요약

| 파일 | Key | 비고 |
| --- | --- | --- |
| `eia_coal_production_by_state.csv` | `[state_name, year]` | 26개 주만 존재 |
| `eia_coal_electricity_gen_by_state.csv` | `[state_name, year]` | |
| `eia_natgas_electricity_gen_national.csv` | `[year]` | 전국 단위 |
| `eia_heating_degree_days_by_state.csv` | `[state_name, year]` | AK·HI 없음 |
| `noaa_pdsi_by_state.csv` | `[state_name, year]` | AK·HI 없음 |
| `usda_crop_production_by_state.csv` | `[state_name, year]` | |
| `fao_fertilizer_price_index.csv` | `[year]` | 전국 단위 |
| `income_annual.csv` | `[state_name, year]` | |
| `migration_od_annual.csv` | net migration으로 집계 후 `[state_name, year]` | |
| `pce_by_state_real_2017.csv` | `[state_name, category, year]` | |
| `pce_by_state_real_2017_changes.csv` | `[state_name, category, current_year]` | |
| `population_by_states.csv` | `[state_name, year]` | |
| `state_unify_gdp.csv` | `[state_name, year]` | |
| `state_unify_ue-rate.csv` | `[state_name, year]` | |
