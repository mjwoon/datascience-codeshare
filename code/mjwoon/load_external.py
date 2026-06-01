"""
External Dataset Loader — FAF Freight Forecasting Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15개 외부 CSV를 로드·정제·병합하여 state-level / national-level 피처를 생성.

FAF 데이터의 state_name 기준 (51개: 50 states + "Washington DC")에 맞춰
외부 데이터의 state_name 변형을 통일한다.

사용법:
    from load_external import load_all_external
    state_feat, national_feat = load_all_external(data_dir, context_year)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# State name 통일 매핑
# FAF 기준: "Washington DC" (Washington state와 구분)
# ─────────────────────────────────────────────────────────────
_STATE_RENAME = {
    "District of Columbia": "Washington DC",
    # UE 파일에서는 이미 "Washington DC"이므로 추가 매핑 불필요
}


def _normalize_state(df: pd.DataFrame, col: str = "state_name") -> pd.DataFrame:
    """FAF state_name 기준으로 통일."""
    df = df.copy()
    df[col] = df[col].replace(_STATE_RENAME)
    return df


# ─────────────────────────────────────────────────────────────
# 개별 로더 함수
# ─────────────────────────────────────────────────────────────

def _load_gdp(path: Path) -> pd.DataFrame:
    """state_unify_gdp.csv → (state_name, year, real_gdp)"""
    df = pd.read_csv(path)
    df = df.rename(columns={"real_GDP": "real_gdp"})
    return _normalize_state(df)[["state_name", "year", "real_gdp"]]


def _load_unemployment(path: Path) -> pd.DataFrame:
    """state_unify_ue-rate.csv → (state_name, year, unemployment_rate)"""
    df = pd.read_csv(path)
    return _normalize_state(df)[["state_name", "year", "unemployment_rate"]]


def _load_population(path: Path) -> pd.DataFrame:
    """population_by_states.csv → (state_name, year, population)"""
    df = pd.read_csv(path)
    return _normalize_state(df)[["state_name", "year", "population"]]


def _load_income(path: Path) -> pd.DataFrame:
    """income_annual.csv → (state_name, year, median_income)
    주의: 2020년 결측 → interpolation 필요
    """
    df = pd.read_csv(path)
    return _normalize_state(df)[["state_name", "year", "median_income"]]


def _load_pce(path: Path) -> pd.DataFrame:
    """pce_by_state_real_2017.csv → (state_name, year, pce_total, pce_goods)
    category 필터: 'Personal consumption expenditures', 'Goods'
    """
    df = pd.read_csv(path)
    df = _normalize_state(df)

    # Total PCE
    total = (
        df[df["category"] == "Personal consumption expenditures"]
        .rename(columns={"pce_value": "pce_total"})
        [["state_name", "year", "pce_total"]]
    )

    # Goods PCE
    goods = (
        df[df["category"] == "Goods"]
        .rename(columns={"pce_value": "pce_goods"})
        [["state_name", "year", "pce_goods"]]
    )

    return total.merge(goods, on=["state_name", "year"], how="outer")


def _load_pce_changes(path: Path) -> pd.DataFrame:
    """pce_by_state_real_2017_changes.csv → (state_name, year, pce_growth_rate)
    Total PCE의 YoY growth rate만 추출.
    """
    df = pd.read_csv(path)
    df = _normalize_state(df)
    total = df[df["category"] == "Personal consumption expenditures"].copy()
    total = total.rename(columns={"current_year": "year"})
    return total[["state_name", "year", "pce_growth_rate"]]


def _load_coal_production(path: Path) -> pd.DataFrame:
    """eia_coal_production_by_state.csv → (state_name, year, coal_production_ktons)
    26개 주만 존재 → merge 후 fillna(0)
    """
    df = pd.read_csv(path)
    return _normalize_state(df)[["state_name", "year", "coal_production_ktons"]]


def _load_coal_electricity(path: Path) -> pd.DataFrame:
    """eia_coal_electricity_gen_by_state.csv → (state_name, year, coal_elec_gen_gwh)"""
    df = pd.read_csv(path)
    return _normalize_state(df)[["state_name", "year", "coal_elec_gen_gwh"]]


def _load_hdd(path: Path) -> pd.DataFrame:
    """eia_heating_degree_days_by_state.csv → (state_name, year, hdd_annual)
    48개 주 (AK·HI 제외)
    """
    df = pd.read_csv(path)
    return _normalize_state(df)[["state_name", "year", "hdd_annual"]]


def _load_pdsi(path: Path) -> pd.DataFrame:
    """noaa_pdsi_by_state.csv → (state_name, year, pdsi_annual_avg)
    48개 주 (AK·HI 제외)
    """
    df = pd.read_csv(path)
    return _normalize_state(df)[["state_name", "year", "pdsi_annual_avg"]]


def _load_crop(path: Path) -> pd.DataFrame:
    """usda_crop_production_by_state.csv → pivot to (state_name, year, corn_prod_bu, wheat_prod_bu, soy_prod_bu)"""
    df = pd.read_csv(path)
    df = _normalize_state(df)

    # pivot commodity → columns
    pivot = df.pivot_table(
        index=["state_name", "year"],
        columns="commodity",
        values="production_bu",
        aggfunc="sum",
    ).reset_index()

    rename_map = {
        "CORN": "corn_prod_bu",
        "WHEAT": "wheat_prod_bu",
        "SOYBEANS": "soy_prod_bu",
    }
    pivot = pivot.rename(columns=rename_map)

    keep_cols = ["state_name", "year"] + [v for v in rename_map.values() if v in pivot.columns]
    return pivot[keep_cols]


def _load_migration(path: Path) -> pd.DataFrame:
    """migration_od_annual.csv → net_migration per state per year.
    OD 형태 → inflow - outflow = net_migration.
    """
    df = pd.read_csv(path)
    # Normalize state names in both origin and destination
    df["origin"] = df["origin"].replace(_STATE_RENAME)
    df["destination"] = df["destination"].replace(_STATE_RENAME)

    inflow = df.groupby(["destination", "year"])["migrants"].sum().reset_index()
    inflow = inflow.rename(columns={"destination": "state_name", "migrants": "inflow"})

    outflow = df.groupby(["origin", "year"])["migrants"].sum().reset_index()
    outflow = outflow.rename(columns={"origin": "state_name", "migrants": "outflow"})

    net = inflow.merge(outflow, on=["state_name", "year"], how="outer").fillna(0)
    net["net_migration"] = net["inflow"] - net["outflow"]

    return net[["state_name", "year", "net_migration"]]


def _load_natgas(path: Path) -> pd.DataFrame:
    """eia_natgas_electricity_gen_national.csv → (year, natgas_elec_gen_gwh)"""
    return pd.read_csv(path)[["year", "natgas_elec_gen_gwh"]]


def _load_fertilizer(path: Path) -> pd.DataFrame:
    """fao_fertilizer_price_index.csv → (year, dap_price_usd_mt, urea_price_usd_mt)"""
    return pd.read_csv(path)[["year", "dap_price_usd_mt", "urea_price_usd_mt"]]


# ─────────────────────────────────────────────────────────────
# 결측치 보간
# ─────────────────────────────────────────────────────────────

def _interpolate_year_gaps(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """state별 year 기준 선형 보간 (2020년 ACS/IRS 결측 대응)."""
    df = df.copy()
    df = df.sort_values(["state_name", "year"])
    for col in value_cols:
        df[col] = df.groupby("state_name")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
    return df


# ─────────────────────────────────────────────────────────────
# 메인 로더
# ─────────────────────────────────────────────────────────────

def load_all_external(
    data_dir: Path,
    context_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    15개 외부 CSV를 로드·정제·병합.

    Parameters
    ----------
    data_dir : Path
        additional_dataset 디렉토리 경로
    context_year : int
        context year (해당 연도의 외부 피처만 추출)

    Returns
    -------
    state_features : pd.DataFrame
        (state_name,) 키 — context_year 기준 state-level 피처
    national_features : dict[str, float]
        context_year 기준 national-level 피처 (단일 행이므로 dict)
    """
    d = Path(data_dir)

    # ── State-level 로드
    gdp      = _load_gdp(d / "state_unify_gdp.csv")
    ue       = _load_unemployment(d / "state_unify_ue-rate.csv")
    pop      = _load_population(d / "population_by_states.csv")
    income   = _load_income(d / "income_annual.csv")
    pce      = _load_pce(d / "pce_by_state_real_2017.csv")
    pce_chg  = _load_pce_changes(d / "pce_by_state_real_2017_changes.csv")
    coal_p   = _load_coal_production(d / "eia_coal_production_by_state.csv")
    coal_e   = _load_coal_electricity(d / "eia_coal_electricity_gen_by_state.csv")
    hdd      = _load_hdd(d / "eia_heating_degree_days_by_state.csv")
    pdsi     = _load_pdsi(d / "noaa_pdsi_by_state.csv")
    crop     = _load_crop(d / "usda_crop_production_by_state.csv")
    mig      = _load_migration(d / "migration_od_annual.csv")

    # ── 2020 결측 보간 (income, migration)
    income = _interpolate_year_gaps(income, ["median_income"])
    mig    = _interpolate_year_gaps(mig, ["net_migration"])

    # ── 인구 성장률 계산
    pop = pop.sort_values(["state_name", "year"])
    pop["pop_growth_rate"] = pop.groupby("state_name")["population"].pct_change()

    # ── 전부 state-level merge
    state_all = gdp.copy()
    for df in [ue, pop, income, pce, pce_chg, coal_p, coal_e, hdd, pdsi, crop, mig]:
        state_all = state_all.merge(df, on=["state_name", "year"], how="left")

    # ── 결측치 처리
    # coal 생산: 광산 없는 주 → 0
    state_all["coal_production_ktons"] = state_all["coal_production_ktons"].fillna(0)
    # coal 발전: 발전 없는 주 → 0
    state_all["coal_elec_gen_gwh"] = state_all["coal_elec_gen_gwh"].fillna(0)
    # crop: 재배하지 않는 주 → 0
    for col in ["corn_prod_bu", "wheat_prod_bu", "soy_prod_bu"]:
        if col in state_all.columns:
            state_all[col] = state_all[col].fillna(0)
    # HDD/PDSI: AK·HI 결측 → 해당 연도 median
    for col in ["hdd_annual", "pdsi_annual_avg"]:
        median_by_year = state_all.groupby("year")[col].transform("median")
        state_all[col] = state_all[col].fillna(median_by_year)
    # migration: 없는 경우 0
    state_all["net_migration"] = state_all["net_migration"].fillna(0)
    # pop_growth_rate: 첫 연도 NaN → 0
    state_all["pop_growth_rate"] = state_all["pop_growth_rate"].fillna(0)

    # ── 거대 지표 Log1p 변환 (Skewness 완화)
    skew_cols = [
        "coal_production_ktons",
        "coal_elec_gen_gwh",
        "corn_prod_bu",
        "wheat_prod_bu",
        "soy_prod_bu"
    ]
    for col in skew_cols:
        if col in state_all.columns:
            state_all[col] = np.log1p(state_all[col].clip(lower=0))

    # ── context_year 필터
    state_ctx = state_all[state_all["year"] == context_year].drop(columns=["year"]).copy()

    # ── National-level 로드
    natgas = _load_natgas(d / "eia_natgas_electricity_gen_national.csv")
    fert   = _load_fertilizer(d / "fao_fertilizer_price_index.csv")

    national = natgas.merge(fert, on="year", how="outer")
    nat_row = national[national["year"] == context_year]

    if len(nat_row) == 0:
        national_dict = {}
    else:
        national_dict = nat_row.drop(columns=["year"]).iloc[0].to_dict()
        if "natgas_elec_gen_gwh" in national_dict:
            national_dict["natgas_elec_gen_gwh"] = np.log1p(national_dict["natgas_elec_gen_gwh"])

    return state_ctx, national_dict


# ─────────────────────────────────────────────────────────────
# State-level 피처 컬럼 목록 (origin_ / dest_ 프리픽스 전)
# ─────────────────────────────────────────────────────────────
STATE_FEATURE_COLS = [
    "real_gdp",
    "unemployment_rate",
    "population",
    "pop_growth_rate",
    "median_income",
    "pce_total",
    "pce_goods",
    "pce_growth_rate",
    "coal_production_ktons",
    "coal_elec_gen_gwh",
    "hdd_annual",
    "pdsi_annual_avg",
    "corn_prod_bu",
    "wheat_prod_bu",
    "soy_prod_bu",
    "net_migration",
]

NATIONAL_FEATURE_COLS = [
    "natgas_elec_gen_gwh",
    "dap_price_usd_mt",
    "urea_price_usd_mt",
]
