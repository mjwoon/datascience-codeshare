"""
analyze_experiments.py — Per-model performance analysis across all splits.

Usage:
    python code/integrated/analyze_experiments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from collections import Counter

from config import SPLITS, EXTERNAL_DIR, KEY_COMM, KEY_ROUTE, RANDOM_SEED, SELECTION_MARGIN
from data_loader import load_split
from features.lag_trend_features import build_lag_trend_features
from features.context_features import build_context_features
from features.structure_features import build_structure_features
from features.external_features import attach_external_features
from features.external_ratio_features import attach_ratio_features
from candidates.m0_baseline import predict_m0
from candidates.m1_statistical import predict_m1
from candidates.m2_bayasgalan import M2Model
from candidates.m3_mjwoon import M3Model
from candidates.m4_dongbin import M4Model
from candidates.m5_yulim import M5Model
from selector import greedy_select, apply_selection
from allocator import route_aggregate
from metrics import wrmse, weighted_wrmse

import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_abs_features(hist_df, base_year, ctx_df, ext_dir):
    lag  = build_lag_trend_features(hist_df, base_year)
    ctx_feat = build_context_features(ctx_df) if ctx_df is not None and len(ctx_df) > 0 else None
    struct = build_structure_features(hist_df)
    feat = lag.copy()
    if ctx_feat is not None:
        feat = feat.join(ctx_feat, how="left")
    feat = feat.join(struct, how="left")
    feat = attach_external_features(feat.reset_index(), base_year, ext_dir).set_index(KEY_COMM)
    return feat


def build_ratio_features(hist_df, base_year, ext_dir):
    lag    = build_lag_trend_features(hist_df, base_year)
    struct = build_structure_features(hist_df)
    feat   = lag.join(struct, how="left")
    feat   = attach_ratio_features(feat.reset_index(), base_year, ext_dir).set_index(KEY_COMM)
    return feat


def candidate_route_wrmse(cand_series, gt_route):
    """Aggregate commodity-level series to route, compute wRMSE vs ground truth."""
    route_pred = route_aggregate(cand_series).reindex(gt_route.index).fillna(0.0)
    return wrmse(gt_route.values, route_pred.values)


def analyze_split(split_name, ext_dir):
    cfg = SPLITS[split_name]
    context_year = cfg["context"]
    val_year     = cfg["val"]
    test_year    = cfg["test"]

    train_df, ctx_df, val_comm, val_route, test_comm, test_route = load_split(split_name)
    all_data = pd.concat([train_df, ctx_df], ignore_index=True)

    val_base  = val_year  - 3
    test_base = test_year - 3

    hist_val  = all_data[all_data["year"] <= val_base].copy()
    hist_test = all_data[all_data["year"] <= test_base].copy()
    ctx_for_val  = all_data[all_data["year"] == val_base].copy()
    ctx_for_test = ctx_df.copy()

    def feat_builder_val(h, b):  return build_abs_features(h, b, ctx_for_val,  ext_dir)
    def feat_builder_test(h, b): return build_abs_features(h, b, ctx_for_test, ext_dir)
    def ratio_builder_val(h, b):  return build_ratio_features(h, b, ext_dir)
    def ratio_builder_test(h, b): return build_ratio_features(h, b, ext_dir)

    val_gt  = val_route.set_index(KEY_ROUTE)["tons"]
    test_gt = test_route.set_index(KEY_ROUTE)["tons"]

    # ── Fit models ────────────────────────────────────────────────
    m2 = M2Model(); m2.fit(all_data, feat_builder_test, context_year)
    m3 = M3Model(); m3.fit(all_data, feat_builder_test, context_year)
    m4 = M4Model(); m4.fit(all_data, feat_builder_test, context_year)
    m5 = M5Model(); m5.fit(all_data, ratio_builder_test, context_year)

    # ── Generate candidates ────────────────────────────────────────
    all_val  = {
        **predict_m0(hist_val,  None, val_base,  val_year),
        **predict_m1(hist_val,  None, val_base,  val_year,  ext_dir),
        **m2.predict(hist_val,  val_base,  feat_builder_val,  val_comm),
        **m3.predict(hist_val,  val_base,  feat_builder_val,  val_comm),
        **m4.predict(hist_val,  val_base,  feat_builder_val,  val_comm),
        **m5.predict(hist_val,  val_base,  ratio_builder_val, val_comm),
    }
    all_test = {
        **predict_m0(hist_test, None, test_base, test_year),
        **predict_m1(hist_test, None, test_base, test_year, ext_dir),
        **m2.predict(hist_test, test_base, feat_builder_test, test_comm),
        **m3.predict(hist_test, test_base, feat_builder_test, test_comm),
        **m4.predict(hist_test, test_base, feat_builder_test, test_comm),
        **m5.predict(hist_test, test_base, ratio_builder_test, test_comm),
    }
    # ── Apply Adaptive Soft Gate ────────────────────────────────────
    from gate import compute_repeated_error_score, apply_adaptive_soft_gate
    gate_norm = compute_repeated_error_score(train_df)
    all_val = apply_adaptive_soft_gate(all_val, gate_norm)
    all_test = apply_adaptive_soft_gate(all_test, gate_norm)

    # ── Per-candidate wRMSE ────────────────────────────────────────
    rows = []
    for name in all_val:
        v = candidate_route_wrmse(all_val[name],  val_gt)
        t = candidate_route_wrmse(all_test[name], test_gt)
        rows.append({"candidate": name, "val_wrmse": v, "test_wrmse": t})
    cand_df = pd.DataFrame(rows).sort_values("val_wrmse")

    # Compute commodity-specific dynamic margins based on historical CV
    yearly_comm = train_df.groupby(["commodity", "year"])["tons"].sum().reset_index()
    comm_stats = yearly_comm.groupby("commodity")["tons"].agg(["mean", "std"]).fillna(0.0)
    comm_cv = np.where(comm_stats["mean"] != 0, comm_stats["std"] / comm_stats["mean"], 0.0)
    comm_cv_series = pd.Series(comm_cv, index=comm_stats.index)
    
    margins_dict = {}
    base_margin = SELECTION_MARGIN
    for comm in val_comm["commodity"].unique():
        cv_val = comm_cv_series.get(comm, 0.0)
        margins_dict[comm] = float(np.clip(base_margin * (1.0 + cv_val), 0.05, 0.30))

    # Stability penalties based on candidate reliability
    stability_penalties = {}
    for name in all_val.keys():
        penalty = 0.0
        if name.startswith("M1_"):
            penalty = 0.10
        elif name.startswith("M4_") and "resid" not in name:
            penalty = 0.05
        elif name.startswith("M5_") and name != "M5_medmean_base":
            penalty = 0.03
        stability_penalties[name] = penalty

    # ── Phase 3 greedy selection ───────────────────────────────────
    selection = greedy_select(
        all_val, 
        val_comm, 
        baseline_name="M0_median_3", 
        margins=0.0, 
        stability_penalties=None
    )
    sel_counts = Counter(selection.values())

    val_pred_aligned  = apply_selection(all_val,  selection).pipe(route_aggregate).reindex(val_gt.index).fillna(0.0)
    test_pred_aligned = apply_selection(all_test, selection).pipe(route_aggregate).reindex(test_gt.index).fillna(0.0)

    integrated_val  = wrmse(val_gt.values,  val_pred_aligned.values)
    integrated_test = wrmse(test_gt.values, test_pred_aligned.values)

    return {
        "split": split_name,
        "context": context_year,
        "val_year": val_year,
        "test_year": test_year,
        "weight": cfg["weight"],
        "cand_df": cand_df,
        "selection": sel_counts,
        "integrated_val":  integrated_val,
        "integrated_test": integrated_test,
    }


def print_split_report(r):
    s = r["split"]
    print(f"\n{'='*68}")
    print(f"  {s}  context={r['context']}  val={r['val_year']}  test={r['test_year']}  (w={r['weight']})")
    print(f"{'='*68}")

    df = r["cand_df"].copy()
    df["group"] = df["candidate"].str.split("_").str[0]  # M0, M1, ... M5

    # Group-level best (min val_wrmse per group)
    group_best = df.groupby("group")[["val_wrmse", "test_wrmse"]].min().reset_index()
    group_best.columns = ["group", "best_val", "best_test"]

    # Per-candidate table
    print(f"\n{'Candidate':<30} {'Val wRMSE':>12} {'Test wRMSE':>12}  Group")
    print("-" * 66)
    prev_group = None
    for _, row in df.iterrows():
        grp = row["candidate"].split("_")[0]
        if prev_group and grp != prev_group:
            print()
        print(f"  {row['candidate']:<28} {row['val_wrmse']:>12,.1f} {row['test_wrmse']:>12,.1f}  {grp}")
        prev_group = grp

    print()
    print(f"  {'[Integrated (greedy)]':<28} {r['integrated_val']:>12,.1f} {r['integrated_test']:>12,.1f}")

    # Group summary
    print(f"\n{'Group':>10} {'Best Val':>12} {'Best Test':>12}")
    print("-" * 36)
    for _, row in group_best.sort_values("best_val").iterrows():
        print(f"  {row['group']:>8} {row['best_val']:>12,.1f} {row['best_test']:>12,.1f}")

    # Selection summary
    sel = r["selection"]
    total = sum(sel.values())
    print(f"\n  Phase 3 selection  (total {total} commodities):")
    for name, cnt in sel.most_common(10):
        bar = "█" * cnt
        print(f"    {name:<30} {cnt:>3}  {bar}")


def main():
    ext_dir = EXTERNAL_DIR
    split_results = []

    for split_name in ["split_1", "split_2", "split_3"]:
        print(f"\n[Running {split_name}...]", flush=True)
        r = analyze_split(split_name, ext_dir)
        split_results.append(r)
        print_split_report(r)

    # Weighted summary
    weights = [r["weight"] for r in split_results]
    w_val  = weighted_wrmse([r["integrated_val"]  for r in split_results], weights)
    w_test = weighted_wrmse([r["integrated_test"] for r in split_results], weights)

    print(f"\n{'='*68}")
    print(f"  WEIGHTED SUMMARY (val 기준 greedy 선택)")
    print(f"{'='*68}")

    print(f"\n  {'Split':<10} {'Weight':>8} {'Val wRMSE':>12} {'Test wRMSE':>12}")
    print(f"  {'-'*44}")
    for r in split_results:
        print(f"  {r['split']:<10} {r['weight']:>8.1f} {r['integrated_val']:>12,.1f} {r['integrated_test']:>12,.1f}")
    print(f"  {'─'*44}")
    print(f"  {'Weighted':<10} {'':>8} {w_val:>12,.1f} {w_test:>12,.1f}")

    # Cross-split candidate ranking (avg val_wrmse rank)
    print(f"\n  Cross-split: 각 그룹 best val wRMSE (가중평균)")
    print(f"  {'Candidate':<30} {'w.avg Val':>12} {'w.avg Test':>12}")
    print(f"  {'-'*56}")
    all_cands = split_results[0]["cand_df"]["candidate"].tolist()
    rows = []
    for name in all_cands:
        wv = sum(r["weight"] * r["cand_df"].set_index("candidate").loc[name, "val_wrmse"]  for r in split_results)
        wt = sum(r["weight"] * r["cand_df"].set_index("candidate").loc[name, "test_wrmse"] for r in split_results)
        rows.append((name, wv, wt))
    rows.sort(key=lambda x: x[1])
    prev_group = None
    for name, wv, wt in rows:
        grp = name.split("_")[0]
        if prev_group and grp != prev_group:
            print()
        print(f"  {name:<30} {wv:>12,.1f} {wt:>12,.1f}")
        prev_group = grp

    print(f"\n  Integrated (greedy)            {w_val:>12,.1f} {w_test:>12,.1f}")
    print()


if __name__ == "__main__":
    main()
