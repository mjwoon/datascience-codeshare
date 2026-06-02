"""
generate_m_predictions.py — Generates test predictions for M0~M5 families and final integrated model aligned to example.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Set path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config import SPLITS, EXTERNAL_DIR, KEY_COMM, KEY_ROUTE, RANDOM_SEED
from data_loader import load_split
from analyze_experiments import build_abs_features, build_ratio_features
from candidates.m0_baseline import predict_m0
from candidates.m1_statistical import predict_m1
from candidates.m2_bayasgalan import M2Model
from candidates.m3_mjwoon import M3Model
from candidates.m4_dongbin import M4Model
from candidates.m5_yulim import M5Model
from gate import compute_repeated_error_score, apply_adaptive_soft_gate
from allocator import route_aggregate
from selector import greedy_select, apply_selection

def main():
    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "example.csv"
    if not template_path.exists():
        print(f"Error: {template_path} not found.")
        sys.exit(1)
        
    template_df = pd.read_csv(template_path)
    print(f"Loaded template {template_path.name} with {len(template_df)} rows.")

    # Best/representative candidates for M0~M5
    candidates_to_export = {
        "M0": "M0_median_3",
        "M1": "M1_median_3_ext",
        "M2": "M2_cm3_lgbm",
        "M3": "M3_hurdle",
        "M4": "M4_resid_a010",
        "M5": "M5_medmean_lgbm_extfeat"
    }

    # Structure to hold predictions: {family: {year: Series}}
    predictions_by_family = {family: {} for family in candidates_to_export}
    predictions_by_family["integrated"] = {}
    
    ext_dir = EXTERNAL_DIR
    historical_scores = None

    for split_name in ["split_1", "split_2", "split_3"]:
        cfg = SPLITS[split_name]
        context_year = cfg["context"]
        val_year     = cfg["val"]
        test_year    = cfg["test"]
        
        print(f"\nProcessing {split_name} (test_year={test_year})...")
        train_df, ctx_df, val_comm, val_route, test_comm, test_route = load_split(split_name)
        all_data = pd.concat([train_df, ctx_df], ignore_index=True)
        
        val_base = val_year - 3
        test_base = test_year - 3
        hist_val = all_data[all_data["year"] <= val_base].copy()
        hist_test = all_data[all_data["year"] <= test_base].copy()
        ctx_for_val = all_data[all_data["year"] == val_base].copy()
        ctx_for_test = ctx_df.copy()
        
        def feat_builder_val(h, b): return build_abs_features(h, b, ctx_for_val, ext_dir)
        def feat_builder_test(h, b): return build_abs_features(h, b, ctx_for_test, ext_dir)
        def ratio_builder_val(h, b): return build_ratio_features(h, b, ext_dir)
        def ratio_builder_test(h, b): return build_ratio_features(h, b, ext_dir)
        
        print("  Fitting candidates...")
        m2 = M2Model(); m2.fit(all_data, feat_builder_test, context_year)
        m3 = M3Model(); m3.fit(all_data, feat_builder_test, context_year)
        m4 = M4Model(); m4.fit(all_data, feat_builder_test, context_year)
        m5 = M5Model(); m5.fit(all_data, ratio_builder_test, context_year)
        
        print("  Generating predictions...")
        all_val = {
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
        
        print("  Applying Adaptive Soft Gate...")
        gate_norm = compute_repeated_error_score(train_df)
        all_val = apply_adaptive_soft_gate(all_val, gate_norm)
        all_test = apply_adaptive_soft_gate(all_test, gate_norm)
        
        # Calculate dynamic margins based on CV
        yearly_comm = train_df.groupby(["commodity", "year"])["tons"].sum().reset_index()
        comm_stats = yearly_comm.groupby("commodity")["tons"].agg(["mean", "std"]).fillna(0.0)
        comm_cv = np.where(comm_stats["mean"] != 0, comm_stats["std"] / comm_stats["mean"], 0.0)
        comm_cv_series = pd.Series(comm_cv, index=comm_stats.index)
        
        from config import SELECTION_MARGIN
        margins_dict = {}
        base_margin = SELECTION_MARGIN
        for comm in val_comm["commodity"].unique():
            cv_val = comm_cv_series.get(comm, 0.0)
            margins_dict[comm] = float(np.clip(base_margin * (1.0 + cv_val), 0.05, 0.30))

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

        selection = greedy_select(
            all_val, 
            val_comm, 
            baseline_name="M0_median_3", 
            margins=margins_dict, 
            stability_penalties=stability_penalties,
            historical_scores=historical_scores
        )

        # Update historical scores with current raw SSE
        raw_sse = {}
        val_aligned = val_comm.set_index(KEY_COMM)["tons"] if "tons" in val_comm.columns else val_comm.set_index(KEY_COMM).iloc[:, 0]
        val_aligned = val_aligned.clip(lower=0)
        for comm in val_aligned.index.get_level_values("commodity").unique():
            comm_mask = val_aligned.index.get_level_values("commodity") == comm
            y_true = val_aligned[comm_mask]
            if len(y_true) == 0:
                continue
            raw_sse[comm] = {}
            for cand_name, cand_series in all_val.items():
                cp = cand_series.reindex(y_true.index).fillna(0.0)
                raw_sse[comm][cand_name] = float(((y_true.values - cp.values) ** 2).sum())

        if historical_scores is None:
            historical_scores = raw_sse
        else:
            for comm, cand_dict in raw_sse.items():
                if comm not in historical_scores:
                    historical_scores[comm] = cand_dict.copy()
                else:
                    for cand_name, val_sse in cand_dict.items():
                        if cand_name in historical_scores[comm]:
                            historical_scores[comm][cand_name] = 0.5 * historical_scores[comm][cand_name] + 0.5 * val_sse
                        else:
                            historical_scores[comm][cand_name] = val_sse

        # 1. Store integrated test predictions
        integrated_test_comm = apply_selection(all_test, selection)
        predictions_by_family["integrated"][test_year] = route_aggregate(integrated_test_comm)

        # 2. Store individual model test predictions
        for family, cand_name in candidates_to_export.items():
            if cand_name in all_test:
                comm_pred = all_test[cand_name]
                route_pred = route_aggregate(comm_pred)  # Index: (origin, destination)
                predictions_by_family[family][test_year] = route_pred
            else:
                print(f"  Warning: {cand_name} not found in split candidates.")

    # Align with template and write to CSV for all target outputs
    print("\nExporting aligned predictions...")
    for target in list(candidates_to_export.keys()) + ["integrated"]:
        all_preds_list = []
        for yr, yr_preds in predictions_by_family[target].items():
            df_yr = yr_preds.reset_index()
            df_yr.columns = ["origin", "destination", "prediction"]
            df_yr["year"] = int(yr)
            all_preds_list.append(df_yr)
            
        if all_preds_list:
            merged_preds = pd.concat(all_preds_list, ignore_index=True)
            # Re-align to template structure using merge
            aligned_df = template_df.drop(columns=["prediction"]).merge(
                merged_preds, on=["origin", "destination", "year"], how="left"
            )
            aligned_df["prediction"] = aligned_df["prediction"].fillna(0.0)
        else:
            aligned_df = template_df.copy()
            aligned_df["prediction"] = 0.0
            
        output_path = repo_root / f"prediction_{target}.csv"
        aligned_df.to_csv(output_path, index=False)
        print(f"  Saved {output_path.name} ({len(aligned_df)} rows)")

if __name__ == "__main__":
    main()
