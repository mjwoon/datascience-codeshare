"""
run_experiment.py — Integrated FAF freight forecasting pipeline.

Usage:
    python code/integrated/run_experiment.py
    python code/integrated/run_experiment.py --no-mlflow
    python code/integrated/run_experiment.py --splits split_1 split_2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

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
from metrics import wrmse, weighted_wrmse, economic_benefit_M, all_metrics, weighted_metric

import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAF integrated experiment")
    p.add_argument("--splits", nargs="+", default=["split_1", "split_2", "split_3"])
    p.add_argument("--external-dir", default=str(EXTERNAL_DIR))
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def build_abs_features(hist_df: pd.DataFrame, base_year: int, ctx_df: pd.DataFrame, ext_dir: Path) -> pd.DataFrame:
    """Build A+B+C+D_abs feature frame for a given history snapshot."""
    lag = build_lag_trend_features(hist_df, base_year)
    ctx_feat = build_context_features(ctx_df) if ctx_df is not None and len(ctx_df) > 0 else None
    struct = build_structure_features(hist_df)

    feat = lag.copy()
    if ctx_feat is not None:
        feat = feat.join(ctx_feat, how="left")
    feat = feat.join(struct, how="left")

    feat = attach_external_features(feat.reset_index(), base_year, ext_dir).set_index(KEY_COMM)
    return feat


def build_ratio_features(hist_df: pd.DataFrame, base_year: int, ext_dir: Path) -> pd.DataFrame:
    """Build B+C+D_ratio feature frame for M5."""
    lag = build_lag_trend_features(hist_df, base_year)
    struct = build_structure_features(hist_df)
    feat = lag.join(struct, how="left")
    feat = attach_ratio_features(feat.reset_index(), base_year, ext_dir).set_index(KEY_COMM)
    return feat


def run_split(
    split_name: str,
    ext_dir: Path,
    use_mlflow: bool,
) -> dict:
    cfg = SPLITS[split_name]
    context_year = cfg["context"]
    val_year = cfg["val"]
    test_year = cfg["test"]

    train_df, ctx_df, val_comm, val_route, test_comm, test_route = load_split(split_name)
    all_data = pd.concat([train_df, ctx_df], ignore_index=True)

    print(f"\n{'='*60}")
    print(f"Split: {split_name}  context={context_year}  val={val_year}  test={test_year}")
    print(f"{'='*60}")

    # Feature builders
    val_base = val_year - 3    # = context_year - 1 for split defs
    test_base = test_year - 3  # = context_year

    hist_val = all_data[all_data["year"] <= val_base].copy()
    hist_test = all_data[all_data["year"] <= test_base].copy()
    ctx_for_val = all_data[all_data["year"] == val_base].copy()
    ctx_for_test = ctx_df.copy()

    def feat_builder_val(h, b):
        return build_abs_features(h, b, ctx_for_val, ext_dir)

    def feat_builder_test(h, b):
        return build_abs_features(h, b, ctx_for_test, ext_dir)

    def ratio_builder_val(h, b):
        return build_ratio_features(h, b, ext_dir)

    def ratio_builder_test(h, b):
        return build_ratio_features(h, b, ext_dir)

    # ── M0: pure statistical ──────────────────────────────────────
    print("\n[M0] Generating statistical candidates...")
    m0_val  = predict_m0(hist_val,  None, val_base,  val_year)
    m0_test = predict_m0(hist_test, None, test_base, test_year)

    # ── M1: statistical + external factor ─────────────────────────
    print("[M1] Generating external-factor candidates...")
    m1_val  = predict_m1(hist_val,  None, val_base,  val_year,  ext_dir)
    m1_test = predict_m1(hist_test, None, test_base, test_year, ext_dir)

    # ── M2: CM3 + LGBM ────────────────────────────────────────────
    print("[M2] Fitting CM3+LGBM model...")
    m2_model = M2Model()
    m2_model.fit(all_data, feat_builder_test, context_year)

    m2_val  = m2_model.predict(hist_val,  val_base,  feat_builder_val,  val_comm)
    m2_test = m2_model.predict(hist_test, test_base, feat_builder_test, test_comm)

    # ── M3: TwoStage Hurdle ────────────────────────────────────────
    print("[M3] Fitting TwoStage Hurdle model...")
    m3_model = M3Model()
    m3_model.fit(all_data, feat_builder_test, context_year)

    m3_val  = m3_model.predict(hist_val,  val_base,  feat_builder_val,  val_comm)
    m3_test = m3_model.predict(hist_test, test_base, feat_builder_test, test_comm)

    # ── M4: per-group LGBM with sample weights ─────────────────────
    print("[M4] Fitting per-group LGBM model...")
    m4_model = M4Model()
    m4_model.fit(all_data, feat_builder_test, context_year)

    m4_val  = m4_model.predict(hist_val,  val_base,  feat_builder_val,  val_comm)
    m4_test = m4_model.predict(hist_test, test_base, feat_builder_test, test_comm)

    # ── M5: medmean + log-ratio LGBM ───────────────────────────────
    print("[M5] Fitting medmean+LGBM model...")
    m5_model = M5Model()
    m5_model.fit(all_data, ratio_builder_test, context_year)

    m5_val  = m5_model.predict(hist_val,  val_base,  ratio_builder_val,  val_comm)
    m5_test = m5_model.predict(hist_test, test_base, ratio_builder_test, test_comm)

    # ── Merge all candidates ───────────────────────────────────────
    all_val  = {**m0_val,  **m1_val,  **m2_val,  **m3_val,  **m4_val,  **m5_val}
    all_test = {**m0_test, **m1_test, **m2_test, **m3_test, **m4_test, **m5_test}

    # ── Apply Adaptive Soft Gate ────────────────────────────────────
    from gate import compute_repeated_error_score, apply_adaptive_soft_gate
    print("  [Gate] Computing repeated error score from train data...")
    gate_norm = compute_repeated_error_score(train_df)
    print("  [Gate] Applying adaptive soft gate to ML candidates...")
    all_val = apply_adaptive_soft_gate(all_val, gate_norm)
    all_test = apply_adaptive_soft_gate(all_test, gate_norm)

    print(f"\n[Phase3] {len(all_val)} candidates → greedy selection on val...")

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

    # ── Phase 3: greedy selection ──────────────────────────────────
    selection = greedy_select(
        all_val, 
        val_comm, 
        baseline_name="M0_median_3", 
        margins=0.0, 
        stability_penalties=None
    )

    val_comm_pred  = apply_selection(all_val,  selection)
    test_comm_pred = apply_selection(all_test, selection)

    # ── Phase 4: route aggregation ─────────────────────────────────
    val_route_pred  = route_aggregate(val_comm_pred)
    test_route_pred = route_aggregate(test_comm_pred)

    # align with ground truth
    val_gt  = val_route.set_index(KEY_ROUTE)["tons"]
    test_gt = test_route.set_index(KEY_ROUTE)["tons"]

    val_pred_aligned  = val_route_pred.reindex(val_gt.index).fillna(0.0)
    test_pred_aligned = test_route_pred.reindex(test_gt.index).fillna(0.0)

    # M0_median_3 baseline for economic benefit
    m0_val_route  = route_aggregate(all_val["M0_median_3"]).reindex(val_gt.index).fillna(0.0)
    m0_test_route = route_aggregate(all_test["M0_median_3"]).reindex(test_gt.index).fillna(0.0)

    val_wrmse  = wrmse(val_gt.values,  val_pred_aligned.values)
    test_wrmse = wrmse(test_gt.values, test_pred_aligned.values)
    eco_val    = economic_benefit_M(val_gt.values,  val_pred_aligned.values,  m0_val_route.values)
    eco_test   = economic_benefit_M(test_gt.values, test_pred_aligned.values, m0_test_route.values)

    val_m  = all_metrics(val_gt.values,  val_pred_aligned.values)
    test_m = all_metrics(test_gt.values, test_pred_aligned.values)

    print(f"  val  wRMSE={val_wrmse:>10,.1f}  MAE={val_m['MAE']:>10,.1f}  WMAPE={val_m['WMAPE']:.4f}  RMSLE={val_m['RMSLE']:.4f}  R2={val_m['R2']:.4f}")
    print(f"  test wRMSE={test_wrmse:>10,.1f}  MAE={test_m['MAE']:>10,.1f}  WMAPE={test_m['WMAPE']:.4f}  RMSLE={test_m['RMSLE']:.4f}  R2={test_m['R2']:.4f}")
    print(f"  eco_benefit  val={eco_val:+.3f} M$  test={eco_test:+.3f} M$")

    from collections import Counter
    sel_counts = Counter(selection.values())
    top5 = sel_counts.most_common(5)
    print(f"  Top selected models: {top5}")

    return {
        "split": split_name,
        "val_wrmse": val_wrmse,
        "test_wrmse": test_wrmse,
        "val_mae":   val_m["MAE"],
        "test_mae":  test_m["MAE"],
        "val_wmape":  val_m["WMAPE"],
        "test_wmape": test_m["WMAPE"],
        "val_rmsle":  val_m["RMSLE"],
        "test_rmsle": test_m["RMSLE"],
        "val_r2":   val_m["R2"],
        "test_r2":  test_m["R2"],
        "eco_val": eco_val,
        "eco_test": eco_test,
        "weight": cfg["weight"],
        "selection": selection,
        "val_pred": val_pred_aligned,
        "test_pred": test_pred_aligned,
    }


def main() -> None:
    args = parse_args()
    ext_dir = Path(args.external_dir)

    use_mlflow = not args.no_mlflow
    mlflow_run = None

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("faf_integrated")
            mlflow_run = mlflow.start_run()
        except Exception as e:
            print(f"MLflow unavailable ({e}), running without logging.")
            use_mlflow = False

    split_results = []
    for split_name in args.splits:
        if split_name not in SPLITS:
            print(f"Unknown split: {split_name}, skipping.")
            continue
        result = run_split(split_name, ext_dir, use_mlflow)
        split_results.append(result)

    # Weighted summary
    weights = [r["weight"] for r in split_results]
    val_scores  = [r["val_wrmse"]  for r in split_results]
    test_scores = [r["test_wrmse"] for r in split_results]

    w_val  = weighted_wrmse(val_scores,  weights)
    w_test = weighted_wrmse(test_scores, weights)

    def wavg(key): return weighted_metric([r[key] for r in split_results], weights)

    w_val_mae   = wavg("val_mae");   w_test_mae   = wavg("test_mae")
    w_val_wmape = wavg("val_wmape"); w_test_wmape = wavg("test_wmape")
    w_val_rmsle = wavg("val_rmsle"); w_test_rmsle = wavg("test_rmsle")
    w_val_r2    = wavg("val_r2");    w_test_r2    = wavg("test_r2")

    print(f"\n{'='*60}")
    print(f"{'Metric':<12}  {'Val (weighted)':>18}  {'Test (weighted)':>18}")
    print(f"{'-'*50}")
    print(f"{'wRMSE':<12}  {w_val:>18,.1f}  {w_test:>18,.1f}")
    print(f"{'MAE':<12}  {w_val_mae:>18,.1f}  {w_test_mae:>18,.1f}")
    print(f"{'WMAPE':<12}  {w_val_wmape:>18.4f}  {w_test_wmape:>18.4f}")
    print(f"{'RMSLE':<12}  {w_val_rmsle:>18.4f}  {w_test_rmsle:>18.4f}")
    print(f"{'R2':<12}  {w_val_r2:>18.4f}  {w_test_r2:>18.4f}")
    print(f"{'='*60}")

    if use_mlflow and mlflow_run:
        try:
            import mlflow
            mlflow.log_metric("weighted_RMSE",       w_test)
            mlflow.log_metric("val_weighted_RMSE",   w_val)
            mlflow.log_metric("weighted_MAE",        w_test_mae)
            mlflow.log_metric("val_weighted_MAE",    w_val_mae)
            mlflow.log_metric("weighted_WMAPE",      w_test_wmape)
            mlflow.log_metric("val_weighted_WMAPE",  w_val_wmape)
            mlflow.log_metric("weighted_RMSLE",      w_test_rmsle)
            mlflow.log_metric("val_weighted_RMSLE",  w_val_rmsle)
            mlflow.log_metric("weighted_R2_Score",   w_test_r2)
            mlflow.log_metric("val_weighted_R2_Score", w_val_r2)
            for r in split_results:
                sn = r["split"].replace("split_", "split")
                mlflow.log_metric(f"{sn}_RMSE",    r["test_wrmse"])
                mlflow.log_metric(f"{sn}_MAE",     r["test_mae"])
                mlflow.log_metric(f"{sn}_WMAPE",   r["test_wmape"])
                mlflow.log_metric(f"{sn}_RMSLE",   r["test_rmsle"])
                mlflow.log_metric(f"{sn}_R2_Score",r["test_r2"])
                mlflow.log_metric(f"val_{sn}_RMSE",    r["val_wrmse"])
                mlflow.log_metric(f"val_{sn}_MAE",     r["val_mae"])
                mlflow.log_metric(f"val_{sn}_WMAPE",   r["val_wmape"])
                mlflow.log_metric(f"val_{sn}_RMSLE",   r["val_rmsle"])
                mlflow.log_metric(f"val_{sn}_R2_Score",r["val_r2"])
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
