"""
Route-level 3-year-ahead tons forecast — 최종 모델 (FAF5 only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행:  python code/main.py
입력:  data/faf5_cleaned.csv  (FAF5 only, year ≥ 2017)
출력:  콘솔 — 단계별 진단 + 최종 weighted RMSE / wRMSE / 경제적 이득

[모델 한줄 요약]
  median_all (FAF5 전체기간 median)  +  상위 오차 라우트만 추세(trend) 교체
  - 라우트별 candidate (median 류 vs 추세 류) 중 val abs error 최소를 고름
  - 단, val 에서 baseline 을 margin(15%) 이상 명확히 이길 때만 교체
  - 교체 대상은 오차 상위 K 개 라우트로 한정 (소량 라우트 노이즈 회피)
  - K 는 val weighted RMSE 로 선택

[지표]
  RMSE       = split 별 일반 RMSE                                  (1차로 사용)
  wRMSE      = utils 표준 — weights = y_true/Σy_true, tons-가중 RMSE  (참고)
  weighted_*  = 3개 split 결과를 가중 0.2/0.3/0.5 로 합한 값

[데이터 split — 팀 합의]
  split_1:  test=2022, val=2021, base_year=2019  (가중 0.2)
  split_2:  test=2023, val=2022, base_year=2020  (가중 0.3)
  split_3:  test=2024, val=2023, base_year=2021  (가중 0.5)
  3-year horizon: 예측 시 history 는 year ≤ target_year-3 만 사용.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# 0. Configuration
# ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "faf5_cleaned.csv"

SPLITS = {
    "split_1": {"val": 2021, "test": 2022, "weight": 0.2},
    "split_2": {"val": 2022, "test": 2023, "weight": 0.3},
    "split_3": {"val": 2023, "test": 2024, "weight": 0.5},
}
TOP_K_GRID = [0, 5, 10, 20, 30, 50, 100, 200]   # K 후보 (val 기준 best 선택)
MARGIN = 0.15                                    # per-route 교체 마진 (val 에서 baseline 대비 15% 이상 이겨야 채택)
NET_PROFIT_PER_TON_USD = 10.0                    # 트럭 1회 25~30톤 × ~$250-300 순이익 ≈ $10/ton 가정
FAF_TONS_UNIT_FACTOR = 1000                      # FAF tons 컬럼은 천 톤(kton) 단위 → 실제 톤 환산 ×1000
# (검증: 라우트 value/tons ≈ 0.8 → 단위가 M$/kton 이면 $800/ton 으로 freight 의 합리적 가격대)

# ────────────────────────────────────────────────────────────────────
# 1. Metrics
# ────────────────────────────────────────────────────────────────────
def rmse(yt, yp):
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def wrmse(yt, yp):
    """utils 표준 tons-가중 wRMSE. weights = y_true/Σy_true."""
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    w = np.clip(yt, 0, None) / (np.clip(yt, 0, None).sum() + 1e-9)
    return float(np.sqrt(np.sum(w * (yt - yp) ** 2)))

# ────────────────────────────────────────────────────────────────────
# 2. Load + pivot
# ────────────────────────────────────────────────────────────────────
print("=" * 78)
print("STEP 1) 데이터 로드 (FAF5 only)")
print("=" * 78)
RAW = pd.read_csv(DATA_PATH)
print(f"  raw 행수: {len(RAW):,}   컬럼: {list(RAW.columns)}")
print(f"  연도 범위: {RAW['year'].min()} ~ {RAW['year'].max()}")

ROUTE = (RAW.groupby(["origin", "destination", "year"])["tons"].sum()
         .reset_index().pivot_table(index=["origin", "destination"], columns="year", values="tons"))
COMM = (RAW.groupby(["origin", "destination", "commodity", "year"])["tons"].sum()
        .reset_index().pivot_table(index=["origin", "destination", "commodity"], columns="year", values="tons"))
VALUE = (RAW.groupby(["origin", "destination", "year"])["value"].sum()
         .reset_index().pivot_table(index=["origin", "destination"], columns="year", values="value"))
YEARS = sorted(ROUTE.columns)
print(f"  라우트(o,d): {len(ROUTE):,}   commodity-라우트: {len(COMM):,}   사용 연도: {YEARS}")

# ────────────────────────────────────────────────────────────────────
# 3. Predictors
# ────────────────────────────────────────────────────────────────────
def _hist(base, k=None):
    ys = [y for y in YEARS if y <= base]
    return ys[-k:] if k else ys

def median_all(piv, base):
    """가용 history 전체기간 median."""
    ys = _hist(base); return piv[ys].median(axis=1)

def median_k(piv, base, k):
    """최근 K년 median."""
    ys = _hist(base, k); return piv[ys].median(axis=1)

def trend(piv, base, k, target, damp=1.0):
    """최근 K년 선형회귀로 target 외삽, median 쪽으로 damp 만큼 댐핑."""
    ys = _hist(base, k); med = piv[ys].median(axis=1)
    if len(ys) < 2: return med
    x = np.array(ys, float); xm = x - x.mean()
    Y = piv[ys].values.astype(float)
    Y = np.where(np.isnan(Y), np.nanmean(Y, axis=1, keepdims=True), Y)
    slope = (Y @ xm) / (xm ** 2).sum(); inter = Y.mean(axis=1) - slope * x.mean()
    full = inter + slope * target
    return (med + damp * (pd.Series(full, index=piv.index) - med)).clip(lower=0)

# per-route 라우트별 후보 (모두 route-level)
CANDIDATES = {
    "median_all": lambda piv, b, t: median_all(piv, b),
    "median_3":   lambda piv, b, t: median_k(piv, b, 3),
    "median_4":   lambda piv, b, t: median_k(piv, b, 4),
    "trend5_d5":  lambda piv, b, t: trend(piv, b, 5, t, 0.5),   # 5년 추세 50% 댐핑
    "trend5_d10": lambda piv, b, t: trend(piv, b, 5, t, 1.0),   # 5년 추세 풀
    "trend3_d5":  lambda piv, b, t: trend(piv, b, 3, t, 0.5),   # 3년 추세 50% 댐핑
}

# ────────────────────────────────────────────────────────────────────
# 4. Baseline 평가
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("STEP 2) Baseline 평가")
print("=" * 78)

def eval_weighted(predfn, which):
    """which='val'|'test'. predfn(piv, base, target)→Series. (w_RMSE, w_wRMSE) 반환."""
    tr = tw = 0.0
    for sn, s in SPLITS.items():
        tgt_y = s[which]; w = s["weight"]
        yt = ROUTE[tgt_y].dropna()
        pred = predfn(ROUTE, tgt_y - 3, tgt_y).reindex(yt.index).fillna(0.0)
        tr += w * rmse(yt.values, pred.values)
        tw += w * wrmse(yt.values, pred.values)
    return tr, tw

# 공식 baseline = commodity 단위 median(3년) → route 합산
def official_baseline(piv_route_unused, base, target):
    ys = _hist(base, 3)
    return COMM[ys].median(axis=1).clip(lower=0).groupby(level=[0, 1]).sum()

ref_r, ref_w = eval_weighted(official_baseline, "test")
print(f"  공식 baseline (comm_median_3)        test  RMSE = {ref_r:>8,.1f}   wRMSE = {ref_w:>10,.1f}")

mv_r, mv_w = eval_weighted(lambda p, b, t: median_all(p, b), "val")
mt_r, mt_w = eval_weighted(lambda p, b, t: median_all(p, b), "test")
print(f"  median_all (FAF5 전체 history)        val   RMSE = {mv_r:>8,.1f}   wRMSE = {mv_w:>10,.1f}")
print(f"  median_all                             test  RMSE = {mt_r:>8,.1f}   wRMSE = {mt_w:>10,.1f}"
      f"   (vs 공식 RMSE {(mt_r/ref_r-1)*100:+.1f}%)")

# ────────────────────────────────────────────────────────────────────
# 5. 라우트별 오차 분해 (어디서 오차가 나는지)
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("STEP 3) 오차 분해 — 어떤 라우트가 weighted_RMSE 를 지배하는가")
print("=" * 78)

def route_sqerr_contribution():
    """median_all 기준 라우트별 squared error 의 split-가중 합."""
    rows = {}
    for sn, s in SPLITS.items():
        ty = s["test"]; w = s["weight"]
        yt = ROUTE[ty].dropna()
        pr = median_all(ROUTE, ty - 3).reindex(yt.index).fillna(0.0)
        for r in yt.index:
            rows[r] = rows.get(r, 0.0) + w * (yt[r] - pr[r]) ** 2
    return pd.Series(rows).sort_values(ascending=False)

err_contrib = route_sqerr_contribution()
total_sq = err_contrib.sum()
print(f"  총 squared error 합 = {total_sq:,.0f}   (n_route = {len(err_contrib)})")
print(f"  {'route':30s} {'share%':>8s}  {'누적%':>8s}")
cum = 0.0
for (o, d), v in err_contrib.head(10).items():
    cum += 100 * v / total_sq
    print(f"  {o + ' -> ' + d:30s} {100*v/total_sq:>7.1f}%  {cum:>7.1f}%")
top_routes = list(err_contrib.index)

# ────────────────────────────────────────────────────────────────────
# 6. 라우트별 candidate 비교 (val) + 마진 가드
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"STEP 4) 라우트별 candidate 평가 (val abs error) — 마진 {int(MARGIN*100)}% 이상 이길 때만 교체")
print("=" * 78)

def per_route_choice(margin=MARGIN):
    """모든 라우트에 대해 후보별 split-가중 val abs error 계산 → argmin → 마진 통과만 채택."""
    base_ve = pd.Series(0.0, index=ROUTE.index)
    cand_ve = {c: pd.Series(0.0, index=ROUTE.index) for c in CANDIDATES}
    for sn, s in SPLITS.items():
        vy = s["val"]; w = s["weight"]
        yt = ROUTE[vy]
        bp = median_all(ROUTE, vy - 3).reindex(yt.index)
        base_ve = base_ve.add(w * (yt - bp).abs().reindex(ROUTE.index), fill_value=0.0)
        for c, fn in CANDIDATES.items():
            cp = fn(ROUTE, vy - 3, vy).reindex(yt.index)
            cand_ve[c] = cand_ve[c].add(w * (yt - cp).abs().reindex(ROUTE.index), fill_value=0.0)
    M = pd.DataFrame(cand_ve)
    best_c = M.idxmin(axis=1); best_v = M.min(axis=1)
    return {r: best_c[r] for r in ROUTE.index
            if best_v.get(r, np.inf) < base_ve.get(r, np.inf) * (1 - margin)}

choice = per_route_choice(MARGIN)
print(f"  마진 통과 라우트 수: {len(choice)} / {len(ROUTE)}")

# ────────────────────────────────────────────────────────────────────
# 7. 상위 K 라우트 교체 + K 선택 (val)
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"STEP 5) 상위 오차 K 개 라우트 교체 — K 는 val 로 선택")
print("=" * 78)

def assemble(piv, base, target, override_set):
    """median_all 으로 시작 → override_set 의 라우트는 choice[r] 후보로 교체."""
    base_pred = median_all(piv, base)
    for r in override_set:
        if r in base_pred.index:
            base_pred.loc[r] = CANDIDATES[choice[r]](piv, base, target).get(r, base_pred.loc[r])
    return base_pred

print(f"  {'K':>4s} {'n_override':>11s} {'val RMSE':>10s} {'test RMSE':>11s} {'test wRMSE':>12s}  vs 공식")
best = None
for K in TOP_K_GRID:
    ov = set(top_routes[:K]) & set(choice.keys())
    pf = lambda p, b, t, ov=ov: assemble(p, b, t, ov)
    vr, vw = eval_weighted(pf, "val")
    tr, tw = eval_weighted(pf, "test")
    if best is None or vr < best["val_rmse"]:
        best = {"K": K, "n": len(ov), "val_rmse": vr, "test_rmse": tr, "test_wrmse": tw}
    print(f"  {K:>4d} {len(ov):>11d} {vr:>10,.1f} {tr:>11,.1f} {tw:>12,.1f}   {(tr/ref_r-1)*100:+5.1f}%")

print(f"\n  → val-best  K = {best['K']}  (실제 교체 = {best['n']} 라우트)")
print(f"  → 최종 test RMSE  = {best['test_rmse']:,.1f}   (공식 {ref_r:,.1f} 대비 {(best['test_rmse']/ref_r-1)*100:+.1f}%)")
print(f"  → 최종 test wRMSE = {best['test_wrmse']:,.1f}")

# ────────────────────────────────────────────────────────────────────
# 8. 경제적 이득 (트럭 1회 순이익 기준)
# ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"STEP 6) 경제적 이득 — 트럭 1회 순이익 ≈ ${NET_PROFIT_PER_TON_USD}/ton (25-30톤 × ~$250-300)")
print("=" * 78)

K = best["K"]; final_set = set(top_routes[:K]) & set(choice.keys())
rows = []
total_reduction = 0.0
for r in top_routes[:20]:
    be = sum(s["weight"] * abs(ROUTE[s["test"]].get(r, 0) - median_all(ROUTE, s["test"] - 3).get(r, 0))
             for s in SPLITS.values())
    if r in final_set:
        c = choice[r]
        ne = sum(s["weight"] * abs(ROUTE[s["test"]].get(r, 0)
                                   - CANDIDATES[c](ROUTE, s["test"] - 3, s["test"]).get(r, 0))
                 for s in SPLITS.values())
    else:
        c = "(median_all)"; ne = be
    red = be - ne
    if red > 0: total_reduction += red
    rows.append((r, c, be, ne, red))

print(f"  (FAF tons 컬럼은 kton 단위 → 실제 톤 = 값 × {FAF_TONS_UNIT_FACTOR})")
print(f"  {'route':28s} {'predictor':14s} {'base_err(kton)':>15s} {'new(kton)':>11s} {'red(kton)':>11s} {'$ benefit (M$)':>16s}")
for r, c, be, ne, rd in rows[:12]:
    benefit_M = max(0.0, rd) * FAF_TONS_UNIT_FACTOR * NET_PROFIT_PER_TON_USD / 1e6  # M$
    print(f"  {r[0]+' -> '+r[1]:28s} {c:14s} {be:>15,.0f} {ne:>11,.0f} {rd:>11,.0f} {benefit_M:>16,.1f}")

total_benefit_M = total_reduction * FAF_TONS_UNIT_FACTOR * NET_PROFIT_PER_TON_USD / 1e6
print(f"\n  상위 20 라우트 합 — 톤 오차 감소: {total_reduction:,.0f} kton "
      f"(= {total_reduction*FAF_TONS_UNIT_FACTOR:,.0f} ton)")
print(f"  → 트럭 순이익 기준 경제 이득: ≈ ${total_benefit_M:,.1f} M$ (≈ ${total_benefit_M:,.1f} million)")
print(f"  (가정: 트럭 1회 25~30톤, 순이익 $250~300 → ≈ $10/ton 순이익.")
print(f"   해석: '예측이 개선되어 더 잘 배차/적재할 수 있게 된 화물의 순이익 규모'.)")

print("\n" + "=" * 78)
print("완료. 위 STEP 5 의 'val-best' 가 최종 보고 숫자.")
print("=" * 78)
