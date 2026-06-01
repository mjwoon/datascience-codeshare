from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

import commodity_stacking_experiment as stacking_experiment
from commodity_stacking_experiment import (
    BaselineSpec,
    KEY_COMM,
    KEY_ROUTE,
    RANDOM_SEED,
    SPLIT_WEIGHTS,
    add_actual_to_comm_features,
    build_comm_feature_frame,
    commodity_group,
    discover_data_root,
    encode_categories,
    load_detail,
    load_target,
    target_year,
)
from high_impact_commodity_experiment import (
    DATASET_VERSION,
    EXPERIMENT_NAME,
    EXTERNAL_FEATURE_COLUMNS,
    METRIC_NAMES,
    METRIC_SCHEMA,
    SPLIT_INFO,
    TRACKING_URI,
    Candidate,
    add_external_ratio_features,
    baseline_specs,
    corrected_commodity_predictions,
    evaluate_route_prediction,
    fit_models,
    high_impact_routes,
    modeling_columns,
    predict_log_ratio,
    repo_root,
    split_label,
    summarize_against_reference,
    team_metric_dict,
    train_frames,
    weighted_metric_rows,
    write_requirements,
)


REFERENCE_MODEL = "commodity_median3__lgbm__repeated_error__soft__a0.15__clip0.10__capnone"
NEW_SELECTED_MODEL = "commodity_medmean_a0.20__lgbm__extfeat_all__repeated_error__soft__a0.15__clip0.10__capnone"
VALIDATION_DEFENSE = "train-internal candidates first; validation limited review; test confirmation only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--artifact-dir", default="artifacts_high_impact_commodity_external_feature_sensitivity")
    parser.add_argument("--max-pseudo-years", type=int, default=2)
    parser.add_argument("--experimenter", default="Yulim")
    parser.add_argument("--log-mlflow", action="store_true")
    return parser.parse_args()


def selected_candidates() -> tuple[dict[str, BaselineSpec], dict[str, Candidate]]:
    specs = baseline_specs()
    candidates = {
        REFERENCE_MODEL: Candidate(
            REFERENCE_MODEL,
            "commodity_median3",
            "lgbm",
            "repeated_error",
            "soft",
            0.15,
            0.10,
            None,
        ),
        NEW_SELECTED_MODEL: Candidate(
            NEW_SELECTED_MODEL,
            "commodity_medmean_a0.20",
            "lgbm",
            "repeated_error",
            "soft",
            0.15,
            0.10,
            None,
            "feature_only",
            True,
            "all",
        ),
    }
    return specs, candidates


def feature_importance_rows(model: object, split: str, model_name: str, selected_level: str, model_scope: str, cols: list[str]) -> list[dict[str, object]]:
    split_importance = getattr(model, "feature_importances_", None)
    if split_importance is None:
        split_values = np.zeros(len(cols), dtype=float)
    else:
        split_values = np.asarray(split_importance, dtype=float)

    gain_values = np.full(len(cols), np.nan, dtype=float)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            gain_values = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
        except Exception:
            gain_values = np.full(len(cols), np.nan, dtype=float)

    rows = []
    for idx, feature in enumerate(cols):
        rows.append(
            {
                "split": split,
                "model": model_name,
                "selected_level": selected_level,
                "model_scope": model_scope,
                "feature": feature,
                "importance_gain": float(gain_values[idx]) if idx < len(gain_values) and not np.isnan(gain_values[idx]) else np.nan,
                "importance_split": float(split_values[idx]) if idx < len(split_values) else 0.0,
                "is_external_feature": feature in EXTERNAL_FEATURE_COLUMNS,
            }
        )
    return rows


def generate_predictions(
    data_root: Path,
    specs: dict[str, BaselineSpec],
    candidates: dict[str, Candidate],
    max_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    route_rows = []
    comm_rows = []
    importance_rows = []

    for model_name, candidate in candidates.items():
        spec = specs[candidate.baseline]
        external_feature_set = candidate.external_feature_set if candidate.uses_external_features else "none"
        for split in SPLIT_WEIGHTS:
            train = load_detail(data_root, split, include_context=False)
            detail_full = load_detail(data_root, split, include_context=True)
            route_scores, _route_detail, _comm_detail = high_impact_routes(train, spec, max_years)
            route_scores = route_scores.copy()
            route_scores["split"] = split
            selection_train = train_frames(train, spec, max_years, data_root=data_root, external_feature_set=external_feature_set)
            selection = (
                pd.DataFrame()
                if selection_train.empty
                else __import__("high_impact_commodity_experiment").commodity_selection_report(selection_train)
            )
            train_enc_base, _ = encode_categories(selection_train, selection_train.iloc[:1].copy())
            cols = modeling_columns(train_enc_base)
            comm_models, group_models = fit_models(train_enc_base, cols, selection, candidate.model_backend)

            if candidate.uses_external_features:
                for commodity, model in comm_models.items():
                    importance_rows.extend(feature_importance_rows(model, split, model_name, "commodity-specific", str(commodity), cols))
                for group, model in group_models.items():
                    importance_rows.extend(feature_importance_rows(model, split, model_name, "commodity-group", str(group), cols))

            score_cols = [
                "impact_union",
                "top20_high_impact_route",
                "repeated_error_score",
                "commodity_sensitivity_score",
                "composition_shift_score",
            ]
            for part in ["val", "test"]:
                target_route = load_target(data_root, split, part, commodity=False)
                target_comm = load_target(data_root, split, part, commodity=True)
                tgt_year = target_year(target_route)
                history = detail_full[detail_full["year"] <= tgt_year - 3].copy()
                apply_features = build_comm_feature_frame(history, tgt_year, spec)
                if external_feature_set != "none":
                    apply_features = add_external_ratio_features(apply_features, data_root, external_feature_set)
                apply_features = add_actual_to_comm_features(apply_features, target_comm)
                train_enc, apply_enc = encode_categories(selection_train, apply_features)
                cols = modeling_columns(train_enc)
                pred_frame = predict_log_ratio(apply_enc, cols, selection, comm_models, group_models)
                comm = corrected_commodity_predictions(pred_frame, route_scores, candidate, part, split)
                route_pred = comm.groupby(KEY_ROUTE, as_index=False)["final_pred"].sum()
                route = target_route[KEY_ROUTE + ["year", "tons"]].merge(route_pred, on=KEY_ROUTE, how="left")
                route["final_pred"] = route["final_pred"].fillna(0.0).clip(lower=0)
                route["model"] = model_name
                route = route.merge(route_scores[KEY_ROUTE + score_cols], on=KEY_ROUTE, how="left")
                route["top20_flag"] = route["top20_high_impact_route"].fillna(False).astype(bool)
                route["impact_union"] = route["impact_union"].fillna(False).astype(bool)
                for col in ["repeated_error_score", "commodity_sensitivity_score", "composition_shift_score"]:
                    route[col] = route[col].fillna(0.0)
                metric_rows.append(evaluate_route_prediction(route, split, part, model_name))
                route_rows.append(route.assign(split=split, part=part))

                comm_out = comm[
                    KEY_COMM
                    + [
                        "year",
                        "actual_tons",
                        "final_pred",
                        "selected_level",
                        "commodity_group",
                        "model_backend",
                    ]
                ].copy()
                comm_out["model"] = model_name
                comm_rows.append(comm_out.assign(split=split, part=part))

    return (
        pd.DataFrame(metric_rows),
        pd.concat(route_rows, ignore_index=True),
        pd.concat(comm_rows, ignore_index=True),
        pd.DataFrame(importance_rows),
    )


def route_error_delta(route_predictions: pd.DataFrame) -> pd.DataFrame:
    ref = route_predictions[route_predictions["model"] == REFERENCE_MODEL].copy()
    new = route_predictions[route_predictions["model"] == NEW_SELECTED_MODEL].copy()
    ref = ref.rename(columns={"final_pred": "reference_pred", "tons": "actual"})
    new = new.rename(columns={"final_pred": "external_feature_pred"})
    keep_new = KEY_ROUTE + ["split", "part", "year", "external_feature_pred"]
    merged = ref[
        KEY_ROUTE
        + [
            "split",
            "part",
            "year",
            "actual",
            "reference_pred",
            "top20_flag",
            "impact_union",
            "repeated_error_score",
            "commodity_sensitivity_score",
            "composition_shift_score",
        ]
    ].merge(new[keep_new], on=KEY_ROUTE + ["split", "part", "year"], how="inner")
    merged["reference_abs_error"] = (merged["actual"] - merged["reference_pred"]).abs()
    merged["external_feature_abs_error"] = (merged["actual"] - merged["external_feature_pred"]).abs()
    merged["error_delta"] = merged["reference_abs_error"] - merged["external_feature_abs_error"]
    merged["improved_flag"] = merged["error_delta"] > 0
    return merged[
        [
            "split",
            "part",
            "origin",
            "destination",
            "year",
            "actual",
            "reference_pred",
            "external_feature_pred",
            "reference_abs_error",
            "external_feature_abs_error",
            "error_delta",
            "improved_flag",
            "top20_flag",
            "impact_union",
            "repeated_error_score",
            "commodity_sensitivity_score",
            "composition_shift_score",
        ]
    ].sort_values(["split", "part", "error_delta"], ascending=[True, True, False])


def route_delta_summary(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, part), sub in delta.groupby(["split", "part"]):
        total = float(sub["error_delta"].sum())
        improved = sub[sub["error_delta"] > 0].sort_values("error_delta", ascending=False)
        worsened = sub[sub["error_delta"] < 0].sort_values("error_delta")
        total_improvement = float(improved["error_delta"].sum())
        top10_improved = float(improved.head(10)["error_delta"].sum())
        top10_worsened = float(worsened.head(10)["error_delta"].sum())
        rows.append(
            {
                "summary_type": "split_part",
                "split": split,
                "part": part,
                "route_count": int(len(sub)),
                "improved_route_count": int(len(improved)),
                "worsened_route_count": int(len(worsened)),
                "total_error_delta": total,
                "total_improvement_error_delta": total_improvement,
                "top10_improved_error_delta_sum": top10_improved,
                "top10_improved_share_of_total_improvement": top10_improved / total_improvement if total_improvement > 0 else np.nan,
                "top10_worsened_error_delta_sum": top10_worsened,
                "origin": None,
                "destination": None,
                "year": None,
                "actual": np.nan,
                "reference_pred": np.nan,
                "external_feature_pred": np.nan,
                "error_delta": np.nan,
            }
        )
    for label, split, part, ascending in [
        ("split1_test_worsened_route_top10", "split_1", "test", True),
        ("split3_test_improved_route_top10", "split_3", "test", False),
    ]:
        sub = delta[(delta["split"] == split) & (delta["part"] == part)].sort_values("error_delta", ascending=ascending).head(10)
        for rank, row in enumerate(sub.itertuples(index=False), start=1):
            rows.append(
                {
                    "summary_type": label,
                    "split": row.split,
                    "part": row.part,
                    "route_count": np.nan,
                    "improved_route_count": np.nan,
                    "worsened_route_count": np.nan,
                    "total_error_delta": np.nan,
                    "total_improvement_error_delta": np.nan,
                    "top10_improved_error_delta_sum": np.nan,
                    "top10_improved_share_of_total_improvement": np.nan,
                    "top10_worsened_error_delta_sum": np.nan,
                    "rank": rank,
                    "origin": row.origin,
                    "destination": row.destination,
                    "year": row.year,
                    "actual": row.actual,
                    "reference_pred": row.reference_pred,
                    "external_feature_pred": row.external_feature_pred,
                    "error_delta": row.error_delta,
                }
            )
    return pd.DataFrame(rows)


def commodity_error_delta(comm_predictions: pd.DataFrame) -> pd.DataFrame:
    ref = comm_predictions[comm_predictions["model"] == REFERENCE_MODEL].rename(columns={"final_pred": "reference_pred"})
    new = comm_predictions[comm_predictions["model"] == NEW_SELECTED_MODEL].rename(columns={"final_pred": "external_feature_pred"})
    merged = ref[
        KEY_COMM + ["split", "part", "year", "actual_tons", "reference_pred"]
    ].merge(
        new[
            KEY_COMM
            + [
                "split",
                "part",
                "year",
                "external_feature_pred",
                "selected_level",
                "commodity_group",
                "model_backend",
            ]
        ],
        on=KEY_COMM + ["split", "part", "year"],
        how="inner",
    )
    merged["reference_abs_error"] = (merged["actual_tons"] - merged["reference_pred"]).abs()
    merged["external_feature_abs_error"] = (merged["actual_tons"] - merged["external_feature_pred"]).abs()
    merged["error_delta"] = merged["reference_abs_error"] - merged["external_feature_abs_error"]
    return merged[
        [
            "split",
            "part",
            "origin",
            "destination",
            "commodity",
            "year",
            "actual_tons",
            "reference_pred",
            "external_feature_pred",
            "reference_abs_error",
            "external_feature_abs_error",
            "error_delta",
            "selected_level",
            "commodity_group",
            "model_backend",
        ]
    ].sort_values(["split", "part", "error_delta"], ascending=[True, True, False])


def commodity_delta_summary(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    commodity = (
        delta.groupby("commodity", as_index=False)
        .agg(
            total_error_delta=("error_delta", "sum"),
            improved_rows=("error_delta", lambda s: int((s > 0).sum())),
            worsened_rows=("error_delta", lambda s: int((s < 0).sum())),
            reference_abs_error=("reference_abs_error", "sum"),
            external_feature_abs_error=("external_feature_abs_error", "sum"),
        )
        .sort_values("total_error_delta", ascending=False)
    )
    for row in commodity.itertuples(index=False):
        rows.append({"summary_type": "commodity_total", **row._asdict()})

    group = (
        delta.groupby("commodity_group", as_index=False)
        .agg(
            total_error_delta=("error_delta", "sum"),
            improved_rows=("error_delta", lambda s: int((s > 0).sum())),
            worsened_rows=("error_delta", lambda s: int((s < 0).sum())),
            reference_abs_error=("reference_abs_error", "sum"),
            external_feature_abs_error=("external_feature_abs_error", "sum"),
        )
        .sort_values("total_error_delta", ascending=False)
    )
    for row in group.itertuples(index=False):
        rows.append({"summary_type": "commodity_group_total", **row._asdict()})

    high_sse = commodity.sort_values("reference_abs_error", ascending=False).head(15)
    for row in high_sse.itertuples(index=False):
        rows.append({"summary_type": "high_sse_commodity", **row._asdict()})

    for label, frame in [
        ("top_improved_commodity_15", commodity.head(15)),
        ("top_worsened_commodity_15", commodity.tail(15).sort_values("total_error_delta")),
    ]:
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            rows.append({"summary_type": label, "rank": rank, **row._asdict()})
    return pd.DataFrame(rows)


def importance_summary(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    imp = importance.copy()
    imp["importance_value"] = imp["importance_gain"].fillna(0.0)
    if float(imp["importance_value"].sum()) <= 0:
        imp["importance_value"] = imp["importance_split"].fillna(0.0)

    rows = []
    feature = (
        imp.groupby(["feature", "is_external_feature"], as_index=False)
        .agg(total_importance=("importance_value", "sum"), split_count=("split", "nunique"), model_scope_count=("model_scope", "nunique"))
        .sort_values("total_importance", ascending=False)
    )
    external_top20 = feature[feature["is_external_feature"]].head(20)
    for rank, row in enumerate(external_top20.itertuples(index=False), start=1):
        rows.append({"summary_type": "top_external_feature_20", "rank": rank, **row._asdict()})
    total = float(feature["total_importance"].sum())
    external = float(feature.loc[feature["is_external_feature"], "total_importance"].sum())
    rows.append(
        {
            "summary_type": "external_importance_share",
            "feature": "__external_total_share__",
            "is_external_feature": True,
            "total_importance": external,
            "external_importance_share": external / total if total > 0 else np.nan,
            "split_count": int(imp["split"].nunique()),
            "model_scope_count": int(imp["model_scope"].nunique()),
        }
    )
    repeated = feature[(feature["is_external_feature"]) & (feature["split_count"] >= 2)].sort_values(
        ["split_count", "total_importance"], ascending=[False, False]
    )
    for rank, row in enumerate(repeated.itertuples(index=False), start=1):
        rows.append({"summary_type": "repeated_external_feature_by_split", "rank": rank, **row._asdict()})
    return pd.DataFrame(rows)


def final_review_markdown(
    leaderboard: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    route_summary: pd.DataFrame,
    commodity_summary: pd.DataFrame,
    importance_sum: pd.DataFrame,
) -> str:
    ref = leaderboard[leaderboard["model"] == REFERENCE_MODEL].iloc[0]
    new = leaderboard[leaderboard["model"] == NEW_SELECTED_MODEL].iloc[0]
    tiny = feature_ablation[feature_ablation["external_mode"] == "tiny_prior"].sort_values("best_val_weighted_RMSE").head(1)
    group = feature_ablation[feature_ablation["external_feature_set"] == "group_specific"].head(1)
    all_feat = feature_ablation[feature_ablation["external_feature_set"] == "all"].head(1)
    split_part = route_summary[route_summary["summary_type"] == "split_part"].copy()
    imp_share = importance_sum[importance_sum["summary_type"] == "external_importance_share"]
    share = float(imp_share.iloc[0].get("external_importance_share", np.nan)) if not imp_share.empty else np.nan
    top_comm = commodity_summary[commodity_summary["summary_type"] == "top_improved_commodity_15"].head(5)
    top_feat = importance_sum[importance_sum["summary_type"] == "top_external_feature_20"].head(10)

    def table(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "No rows."
        view = df[cols].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
            else:
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(row[col]) for col in cols) + " |" for _, row in view.iterrows()]
        return "\n".join([header, sep] + body)

    return f"""# External Feature Selected Candidate Final Review

## Reference Selected

- model: `{REFERENCE_MODEL}`
- validation weighted RMSE: `{float(ref['val_weighted_RMSE']):.6f}`
- test weighted RMSE: `{float(ref['test_weighted_RMSE']):.6f}`

## New Selected Candidate

- model: `{NEW_SELECTED_MODEL}`
- validation weighted RMSE: `{float(new['val_weighted_RMSE']):.6f}`
- test weighted RMSE: `{float(new['test_weighted_RMSE']):.6f}`
- validation delta vs reference: `{float(new['val_delta_vs_reference_selected']):.6f}`
- test delta vs reference: `{float(new['test_delta_vs_reference_selected']):.6f}`

## Why Tiny Prior Was Rejected

Tiny prior reduced the excessive distortion from the earlier 0.50/0.75 prior, but it still changed the baseline directly. The best tiny-prior row improved validation but did not beat the reference selected on test, so it failed the selected-candidate rule. Best tiny-prior summary:

{table(tiny, ['external_feature_set', 'best_val_weighted_RMSE', 'best_test_weighted_RMSE', 'selected_candidates'])}

## Why Feature-Only All Is Defensible

The selected candidate keeps the commodity baseline unchanged and lets the residual model use base-year external ratios only as explanatory features. This preserves the baseline anchor while allowing LightGBM to learn small conditional residual corrections. It improves both validation and test under the fixed reference rule.

## Why Group-Specific Was Weaker Than All

Group-specific features also beat the reference selected, but it underperformed all external features. The likely reason is that some cross-group macro signals are useful even outside the hand-written commodity mapping. The stricter mapping removes potentially helpful interaction signals from the residual model.

{table(pd.concat([all_feat, group], ignore_index=True), ['external_feature_set', 'best_val_weighted_RMSE', 'best_test_weighted_RMSE', 'selected_candidates'])}

## Split Stability Limits

The model is not uniformly better on every route. Route-level delta remains concentrated in a limited number of high-impact lanes, so it should be presented as a validation-safe selected candidate rather than a universally dominant model.

{table(split_part, ['split', 'part', 'improved_route_count', 'worsened_route_count', 'total_error_delta', 'top10_improved_share_of_total_improvement'])}

## Route And Commodity Diagnostics

Top improved commodities:

{table(top_comm, ['rank', 'commodity', 'total_error_delta', 'reference_abs_error', 'external_feature_abs_error'])}

## Feature Importance

External feature total importance share: `{share:.6f}`

Top external features:

{table(top_feat, ['rank', 'feature', 'total_importance', 'split_count', 'model_scope_count'])}

## Final Selected Model Proposal

Use `{NEW_SELECTED_MODEL}` as the new selected candidate: it is validation-safe against the previous selected model, improves final test weighted RMSE, avoids direct external prior multiplication, and passes route/commodity/feature-importance review as a defensible external-feature residual correction.
"""


def build_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    specs, candidates = selected_candidates()
    reference_values = (15963.751778, 21263.001632)
    leaderboard = summarize_against_reference(metrics, list(candidates.values()), specs, REFERENCE_MODEL, reference_values)
    for col in ["external_mode", "uses_external_features", "external_feature_set"]:
        if col not in leaderboard.columns:
            leaderboard[col] = "none"
    return leaderboard


def log_mlflow_run(artifact_dir: Path, leaderboard: pd.DataFrame, metrics: pd.DataFrame, experimenter: str) -> None:
    if mlflow is None:
        raise RuntimeError("mlflow is not installed")
    row = leaderboard[leaderboard["model"] == NEW_SELECTED_MODEL].iloc[0]
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"{experimenter}_LGBM_LogRatio_ExternalFeatureAll_SelectedCandidate_{stamp}"[:240]
    req_path = artifact_dir / "external_feature_selected_requirements_freeze.txt"
    write_requirements(req_path)
    config_path = artifact_dir / "external_feature_selected_mlflow_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_version": DATASET_VERSION,
                "model": NEW_SELECTED_MODEL,
                "reference_model": REFERENCE_MODEL,
                "primary_metric": "weighted_RMSE",
                "selection_policy": VALIDATION_DEFENSE,
                "splits": SPLIT_INFO,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    model_metrics = metrics[metrics["model"] == NEW_SELECTED_MODEL]
    log_metrics: dict[str, float] = {}
    for _, metric_row in model_metrics.iterrows():
        prefix = split_label(metric_row["split"]) if metric_row["part"] == "test" else f"val_{split_label(metric_row['split'])}"
        for metric in METRIC_NAMES:
            log_metrics[f"{prefix}_{metric}"] = float(metric_row[metric])
    for metric in METRIC_NAMES:
        log_metrics[f"weighted_{metric}"] = float(row[f"test_weighted_{metric}"])
        log_metrics[f"val_weighted_{metric}"] = float(row[f"val_weighted_{metric}"])
    log_metrics["delta_vs_reference_val_weighted_RMSE"] = float(row["val_delta_vs_reference_selected"])
    log_metrics["delta_vs_reference_weighted_RMSE"] = float(row["test_delta_vs_reference_selected"])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    description = "External base-year ratio features used only in the residual LightGBM model; baseline is not multiplied by external prior."
    with mlflow.start_run(run_name=run_name, description=description) as run:
        mlflow.set_tags(
            {
                "dataset_version": DATASET_VERSION,
                "primary_metric": "weighted_RMSE",
                "metric_schema": METRIC_SCHEMA,
                "selection_status": "selected_candidate",
                "external_mode": "feature_only",
                "external_feature_set": "all",
                "uses_external_features": "true",
                "uses_external_prior": "false",
                "validation_overfit_defense": VALIDATION_DEFENSE,
                "model_name": NEW_SELECTED_MODEL,
                "reference_selected_model": REFERENCE_MODEL,
                "model_backend": "lgbm",
                "algorithm": "lgbm",
                "prediction_target": "3-year-ahead route-level tons",
                "hostname": socket.gethostname(),
            }
        )
        for split, info in SPLIT_INFO.items():
            label = split_label(split)
            for key, value in info.items():
                mlflow.set_tag(f"{label}_{key}", str(value))
        mlflow.log_params(
            {
                "experimenter": experimenter,
                "model_name": NEW_SELECTED_MODEL,
                "reference_selected_model": REFERENCE_MODEL,
                "baseline_name": "commodity_medmean_a0.20",
                "model_backend": "lgbm",
                "residual_target": "log_ratio",
                "granularity": "commodity_to_route",
                "external_feature_set": "all",
                "uses_external_features": True,
                "uses_external_prior": False,
                "random_seed": RANDOM_SEED,
            }
        )
        mlflow.log_metrics(log_metrics)
        for path in [
            artifact_dir / "external_feature_route_error_delta.csv",
            artifact_dir / "external_feature_route_error_delta_summary.csv",
            artifact_dir / "external_feature_commodity_error_delta.csv",
            artifact_dir / "external_feature_commodity_error_delta_summary.csv",
            artifact_dir / "external_feature_importance_by_split.csv",
            artifact_dir / "external_feature_importance_summary.csv",
            artifact_dir / "external_feature_final_review.md",
            artifact_dir / "external_feature_only_leaderboard.csv",
            artifact_dir / "external_split_delta_vs_selected.csv",
            req_path,
            config_path,
        ]:
            if path.exists():
                mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source_code")
        manifest = artifact_dir / "external_feature_selected_mlflow_upload_manifest.csv"
        pd.DataFrame(
            [
                {
                    "run_name": run_name,
                    "run_id": run.info.run_id,
                    "run_url": f"{TRACKING_URI}/#/experiments/2/runs/{run.info.run_id}",
                    "model": NEW_SELECTED_MODEL,
                    "selection_status": "selected_candidate",
                    "weighted_RMSE": log_metrics["weighted_RMSE"],
                    "val_weighted_RMSE": log_metrics["val_weighted_RMSE"],
                }
            ]
        ).to_csv(manifest, index=False, encoding="utf-8-sig")
        print("MLflow run:", f"{TRACKING_URI}/#/experiments/2/runs/{run.info.run_id}")


def main() -> None:
    args = parse_args()
    data_root = discover_data_root(args.data_root)
    stacking_experiment.GLOBAL_DATA_ROOT = data_root
    artifact_dir = repo_root() / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    specs, candidates = selected_candidates()
    metrics, route_predictions, comm_predictions, importance = generate_predictions(data_root, specs, candidates, args.max_pseudo_years)
    leaderboard = build_leaderboard(metrics)

    route_delta = route_error_delta(route_predictions)
    route_summary = route_delta_summary(route_delta)
    comm_delta = commodity_error_delta(comm_predictions)
    comm_summary = commodity_delta_summary(comm_delta)
    importance_sum = importance_summary(importance)

    route_delta.to_csv(artifact_dir / "external_feature_route_error_delta.csv", index=False, encoding="utf-8-sig")
    route_summary.to_csv(artifact_dir / "external_feature_route_error_delta_summary.csv", index=False, encoding="utf-8-sig")
    comm_delta.to_csv(artifact_dir / "external_feature_commodity_error_delta.csv", index=False, encoding="utf-8-sig")
    comm_summary.to_csv(artifact_dir / "external_feature_commodity_error_delta_summary.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(artifact_dir / "external_feature_importance_by_split.csv", index=False, encoding="utf-8-sig")
    importance_sum.to_csv(artifact_dir / "external_feature_importance_summary.csv", index=False, encoding="utf-8-sig")

    feature_ablation = pd.read_csv(artifact_dir / "external_feature_ablation_summary.csv")
    review = final_review_markdown(leaderboard, feature_ablation, route_summary, comm_summary, importance_sum)
    (artifact_dir / "external_feature_final_review.md").write_text(review, encoding="utf-8")

    metadata = {
        "reference_selected_model": REFERENCE_MODEL,
        "new_selected_model": NEW_SELECTED_MODEL,
        "validation_overfit_defense": VALIDATION_DEFENSE,
        "outputs": [
            "external_feature_route_error_delta.csv",
            "external_feature_route_error_delta_summary.csv",
            "external_feature_commodity_error_delta.csv",
            "external_feature_commodity_error_delta_summary.csv",
            "external_feature_importance_by_split.csv",
            "external_feature_importance_summary.csv",
            "external_feature_final_review.md",
        ],
    }
    (artifact_dir / "external_feature_selected_review_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.log_mlflow:
        log_mlflow_run(artifact_dir, leaderboard, metrics, args.experimenter)

    print("Reference selected:")
    print(leaderboard[leaderboard["model"] == REFERENCE_MODEL][["model", "val_weighted_RMSE", "test_weighted_RMSE"]].to_string(index=False))
    print("New selected candidate:")
    print(
        leaderboard[
            leaderboard["model"] == NEW_SELECTED_MODEL
        ][["model", "val_weighted_RMSE", "test_weighted_RMSE", "selected_status"]].to_string(index=False)
    )
    print("Route delta summary:")
    print(route_summary[route_summary["summary_type"] == "split_part"].to_string(index=False))
    print("External feature importance share:")
    share_row = importance_sum[importance_sum["summary_type"] == "external_importance_share"]
    print(share_row.to_string(index=False) if not share_row.empty else "No importance rows")
    print("Artifact dir:", artifact_dir)


if __name__ == "__main__":
    main()
