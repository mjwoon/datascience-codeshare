"""
Truck Route Volume Prediction — High-impact Commodity Residual Backend Experiment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모델    : commodity-level robust baseline + log-ratio residual correction
          backend 후보 = LightGBM / HistGradientBoosting / Ridge / Huber / baseline_only
보정    : repeated-error soft gate 기반 tiny-alpha correction
          raw correction clipping 후 route-level로 합산
데이터  : data/faf4_faf5_fixed_year_splits/{split_1,2,3}/
            ├── train.csv             train-internal pseudo backtest 및 baseline 생성
            ├── context.csv           3년 후 예측용 context 이력
            ├── val.csv               제한적 validation 확인용 route target
            ├── test.csv              최종 확인용 route target
            └── {val,test}_commodity.csv  commodity-level 예측/selector 평가용 target
지표    : weighted_RMSE (1차) / MAE, RMSLE, WMAPE, R² (해석·보고용)
방어논리: validation 직접 탐색을 줄이고 train 내부 pseudo backtest와
          commodity별 backend selector를 먼저 확인한 뒤 validation은 제한적으로 사용

실행 방법:
    python code/high_impact_commodity_experiment.py \
        --artifact-dir artifacts_high_impact_commodity_backend_compare \
        --max-pseudo-years 2 \
        --experimenter Yulim \
        --log-mlflow
"""

from __future__ import annotations

import argparse
import json
import math
import socket
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from importlib.util import find_spec
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
    choose_baseline_spec,
    commodity_group,
    discover_data_root,
    encode_categories,
    feature_columns,
    load_detail,
    load_target,
    make_comm_baseline,
    make_model,
    pseudo_target_years,
    set_seed,
    target_from_detail,
    target_year,
)


TRACKING_URI = "https://mlflow.hyu.life"
EXPERIMENT_NAME = "Truck_Route_Volume_Prediction"
DATASET_VERSION = "faf4_faf5_fixed_state_route_v1"
METRIC_SCHEMA = "mlflow_sample_compatible_v2"
RESIDUAL_BACKENDS = ["baseline_only", "ridge", "huber", "lgbm", "hist_gb"]
RULE_RESIDUAL_CANDIDATES = [
    "state_signed_rule",
    "route_signed_rule",
    "recent_trend_rule",
    "recent_max_bridge_rule",
    "top_negative_residual_rule",
    "top_positive_residual_rule",
]
HIGH_SSE_COMMODITIES = [
    "Cereal grains",
    "Gravel",
    "Logs",
    "Fuel oils",
    "Natural sands",
    "Gasoline",
    "Coal",
    "Nonmetal min. prods.",
    "Natural gas/fossil",
    "Waste/scrap",
]
SPLIT_INFO = {
    "split_1": {"train": "2012-2018", "context": "2019", "val": "2021", "test": "2022", "weight": 0.2},
    "split_2": {"train": "2012-2019", "context": "2020", "val": "2022", "test": "2023", "weight": 0.3},
    "split_3": {"train": "2012-2020", "context": "2021", "val": "2023", "test": "2024", "weight": 0.5},
}
METRIC_NAMES = ["RMSE", "MAE", "WMAPE", "RMSLE", "R2_Score"]


@dataclass(frozen=True)
class Candidate:
    name: str
    baseline: str
    model_backend: str
    gate: str
    gate_mode: str
    alpha: float
    clip_value: float | None = None
    top20_cap_ratio: float | None = None
    external_mode: str = "none"
    uses_external_features: bool = False
    external_feature_set: str = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--artifact-dir", default="artifacts_high_impact_commodity")
    parser.add_argument(
        "--experiment-profile",
        default="backend_compare",
        choices=["backend_compare", "external_prior_sensitivity", "external_feature_sensitivity"],
    )
    parser.add_argument("--max-pseudo-years", type=int, default=2)
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--experimenter", default="Yulim")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def split_label(split: str) -> str:
    return split.replace("split_", "split")


def team_metric_dict(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    weights = np.clip(yt, 0, None) / (np.clip(yt, 0, None).sum() + 1e-9)
    err = yt - yp
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.sum(weights * err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "WMAPE": float(np.sum(np.abs(err)) / (np.sum(np.abs(yt)) + 1e-9)),
        "RMSLE": float(np.sqrt(np.mean((np.log1p(np.clip(yt, 0, None)) - np.log1p(yp)) ** 2))),
        "R2_Score": float(1.0 - ss_res / (ss_tot + 1e-9)) if len(np.unique(yt)) > 1 else math.nan,
    }


def weighted_metric_rows(rows: pd.DataFrame, part: str, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    sub = rows[rows["part"] == part]
    for metric in METRIC_NAMES:
        total = 0.0
        for split, weight in SPLIT_WEIGHTS.items():
            value = sub.loc[sub["split"] == split, metric].iloc[0]
            total += weight * float(value)
        out[f"{prefix}_weighted_{metric}"] = total
    return out


def evaluate_route_prediction(pred: pd.DataFrame, split: str, part: str, model: str) -> dict[str, object]:
    return {"split": split, "part": part, "model": model, "rows": len(pred), **team_metric_dict(pred["tons"], pred["final_pred"])}


def baseline_specs(profile: str = "backend_compare") -> dict[str, BaselineSpec]:
    specs: dict[str, BaselineSpec] = {
        "commodity_median3": BaselineSpec("commodity_median3", "median"),
        "commodity_medmean_a0.20": BaselineSpec("commodity_medmean_a0.20", "median_mean", median_mean_alpha=0.20),
    }
    if profile == "external_prior_sensitivity":
        specs.update(
            {
                "commodity_median3_prior050": BaselineSpec("commodity_median3_prior050", "median", external_strength=0.50),
                "commodity_median3_prior075": BaselineSpec("commodity_median3_prior075", "median", external_strength=0.75),
                "commodity_medmean_a0.20_prior050": BaselineSpec(
                    "commodity_medmean_a0.20_prior050",
                    "median_mean",
                    median_mean_alpha=0.20,
                    external_strength=0.50,
                ),
                "commodity_medmean_a0.20_prior075": BaselineSpec(
                    "commodity_medmean_a0.20_prior075",
                    "median_mean",
                    median_mean_alpha=0.20,
                    external_strength=0.75,
                ),
            }
        )
    return specs


def comm_baseline_frame(history: pd.DataFrame, target_comm: pd.DataFrame, tgt_year: int, spec: BaselineSpec) -> pd.DataFrame:
    base = make_comm_baseline(history, tgt_year, spec)
    actual = target_comm.groupby(KEY_COMM, as_index=False)["tons"].sum().rename(columns={"tons": "tons"})
    out = actual.merge(base, on=KEY_COMM, how="left")
    out["pred"] = out["pred"].fillna(0.0).clip(lower=0)
    out["route_prior_total"] = out["route_prior_total"].fillna(0.0)
    out["route_ref"] = out["route_ref"].fillna(out["route_prior_total"])
    out["base_year"] = out["base_year"].fillna(tgt_year - 3).astype(int)
    out["target_year"] = int(tgt_year)
    return out


def route_from_comm(comm: pd.DataFrame, target_route: pd.DataFrame, pred_col: str, model: str) -> pd.DataFrame:
    route = comm.groupby(KEY_ROUTE, as_index=False)[pred_col].sum().rename(columns={pred_col: "final_pred"})
    out = target_route[KEY_ROUTE + ["year", "tons"]].merge(route, on=KEY_ROUTE, how="left")
    out["final_pred"] = out["final_pred"].fillna(0.0).clip(lower=0)
    out["model"] = model
    return out


def composition_shift_score(history: pd.DataFrame) -> pd.DataFrame:
    hist = history[history["year"] >= 2017].copy()
    if hist["year"].nunique() < 4:
        hist = history.copy()
    years = sorted(hist["year"].unique())
    if len(years) < 4:
        return pd.DataFrame(columns=KEY_ROUTE + ["composition_shift_score"])
    recent_years = years[-2:]
    previous_years = years[-4:-2]
    def share_frame(sub: pd.DataFrame, name: str) -> pd.DataFrame:
        comm = sub.groupby(KEY_COMM, as_index=False)["tons"].sum()
        route = comm.groupby(KEY_ROUTE, as_index=False)["tons"].sum().rename(columns={"tons": "route_total"})
        out = comm.merge(route, on=KEY_ROUTE, how="left")
        out[name] = out["tons"] / out["route_total"].replace(0, np.nan)
        return out[KEY_COMM + [name]]
    recent = share_frame(hist[hist["year"].isin(recent_years)], "recent_share")
    prev = share_frame(hist[hist["year"].isin(previous_years)], "previous_share")
    merged = recent.merge(prev, on=KEY_COMM, how="outer").fillna(0.0)
    merged["share_abs_delta"] = (merged["recent_share"] - merged["previous_share"]).abs()
    return merged.groupby(KEY_ROUTE, as_index=False)["share_abs_delta"].sum().rename(columns={"share_abs_delta": "composition_shift_score"})


def pseudo_baseline_details(train: pd.DataFrame, spec: BaselineSpec, max_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_rows = []
    comm_rows = []
    years = pseudo_target_years(train, max_years=max_years)
    for year in years:
        history = train[train["year"] <= year - 3].copy()
        if history.empty:
            continue
        target_route = target_from_detail(train, year, commodity=False)
        target_comm = target_from_detail(train, year, commodity=True)
        comm = comm_baseline_frame(history, target_comm, year, spec)
        comm["baseline_abs_error"] = (comm["tons"] - comm["pred"]).abs()
        comm["baseline_signed_error"] = comm["tons"] - comm["pred"]
        comm["commodity_group"] = comm["commodity"].map(commodity_group)
        comm_rows.append(comm.assign(pseudo_year=year))
        route = route_from_comm(comm, target_route, "pred", spec.name)
        route["baseline_abs_error"] = (route["tons"] - route["final_pred"]).abs()
        route["baseline_signed_error"] = route["tons"] - route["final_pred"]
        route["pseudo_year"] = year
        route_rows.append(route)
    return pd.concat(route_rows, ignore_index=True), pd.concat(comm_rows, ignore_index=True)


def percentile_flags(values: pd.Series, q: float) -> pd.Series:
    cutoff = values.quantile(q) if len(values) else math.inf
    return values >= cutoff


def minmax_normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    lo = float(values.min()) if len(values) else 0.0
    hi = float(values.max()) if len(values) else 0.0
    if hi <= lo:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def high_impact_routes(train: pd.DataFrame, spec: BaselineSpec, max_years: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route_detail, comm_detail = pseudo_baseline_details(train, spec, max_years)
    route_scores = route_detail.groupby(KEY_ROUTE, as_index=False).agg(
        route_size_score=("tons", "mean"),
        repeated_error_score=("baseline_abs_error", "mean"),
        signed_error_mean=("baseline_signed_error", "mean"),
        pseudo_fold_count=("pseudo_year", "nunique"),
    )
    comm_detail["route_total"] = comm_detail.groupby(KEY_ROUTE + ["pseudo_year"])["tons"].transform("sum")
    comm_detail["commodity_share"] = comm_detail["tons"] / comm_detail["route_total"].replace(0, np.nan)
    comm_detail["commodity_sensitivity_component"] = comm_detail["commodity_share"].fillna(0.0) * comm_detail["baseline_abs_error"]
    sensitive = comm_detail.groupby(KEY_ROUTE, as_index=False)["commodity_sensitivity_component"].max().rename(
        columns={"commodity_sensitivity_component": "commodity_sensitivity_score"}
    )
    route_scores = route_scores.merge(sensitive, on=KEY_ROUTE, how="left")
    route_scores = route_scores.merge(composition_shift_score(train), on=KEY_ROUTE, how="left")
    route_scores = route_scores.fillna(0.0)
    route_scores["route_size_score_normalized"] = minmax_normalize(route_scores["route_size_score"])
    route_scores["repeated_error_score_normalized"] = minmax_normalize(route_scores["repeated_error_score"])
    route_scores["commodity_sensitivity_score_normalized"] = minmax_normalize(route_scores["commodity_sensitivity_score"])
    route_scores["composition_shift_score_normalized"] = minmax_normalize(route_scores["composition_shift_score"])
    route_scores["composition_shift_v2_score"] = (
        route_scores["composition_shift_score"] * route_scores["route_size_score_normalized"]
    )
    route_scores["composition_shift_v3_score"] = (
        route_scores["composition_shift_score"] * route_scores["repeated_error_score_normalized"]
    )
    route_scores["composition_shift_v4_score"] = (
        route_scores["composition_shift_score"]
        * route_scores["route_size_score_normalized"]
        * route_scores["repeated_error_score_normalized"]
    )
    route_scores["large_route"] = percentile_flags(route_scores["route_size_score"], 0.80)
    route_scores["repeated_error"] = percentile_flags(route_scores["repeated_error_score"], 0.85)
    route_scores["composition_shift"] = percentile_flags(route_scores["composition_shift_score"], 0.85)
    route_scores["composition_shift_v2"] = percentile_flags(route_scores["composition_shift_v2_score"], 0.85)
    route_scores["composition_shift_v3"] = percentile_flags(route_scores["composition_shift_v3_score"], 0.85)
    route_scores["composition_shift_v4"] = percentile_flags(route_scores["composition_shift_v4_score"], 0.85)
    route_scores["commodity_sensitive"] = percentile_flags(route_scores["commodity_sensitivity_score"], 0.85)
    route_scores["impact_union"] = (
        route_scores["large_route"]
        | route_scores["repeated_error"]
        | (route_scores["composition_shift"] & route_scores["commodity_sensitive"])
    )
    route_scores["impact_union_score"] = route_scores[
        ["route_size_score_normalized", "repeated_error_score_normalized", "commodity_sensitivity_score_normalized"]
    ].max(axis=1)
    route_scores["impact_union_score_normalized"] = minmax_normalize(route_scores["impact_union_score"])
    for col in ["route_size_score", "repeated_error_score", "commodity_sensitivity_score"]:
        route_scores[f"{col}_rank"] = route_scores[col].rank(pct=True)
    route_scores["top20_impact_score"] = (
        route_scores["route_size_score_rank"]
        + route_scores["repeated_error_score_rank"]
        + route_scores["commodity_sensitivity_score_rank"]
    )
    route_scores["top20_high_impact_route"] = False
    top_idx = route_scores.sort_values("top20_impact_score", ascending=False).head(20).index
    route_scores.loc[top_idx, "top20_high_impact_route"] = True
    return route_scores, route_detail, comm_detail


def train_frames(train: pd.DataFrame, spec: BaselineSpec, max_years: int, data_root: Path | None = None, external_feature_set: str = "none") -> pd.DataFrame:
    frames = []
    for year in pseudo_target_years(train, max_years=max_years):
        history = train[train["year"] <= year - 3].copy()
        features = build_comm_feature_frame(history, year, spec)
        if data_root is not None and external_feature_set != "none":
            features = add_external_ratio_features(features, data_root, external_feature_set)
        if features.empty:
            continue
        target_comm = target_from_detail(train, year, commodity=True)
        sample = add_actual_to_comm_features(features, target_comm)
        sample["pseudo_year"] = year
        frames.append(sample)
    return pd.concat(frames, ignore_index=True)


def commodity_selection_report(train_frame: pd.DataFrame) -> pd.DataFrame:
    tmp = train_frame.copy()
    tmp["abs_log_error"] = (np.log1p(tmp["actual_tons"].clip(lower=0)) - np.log1p(tmp["pred"].clip(lower=0))).abs()
    total_error = tmp["abs_log_error"].sum() + 1e-9
    rows = []
    group_rows = tmp.groupby("commodity_group").size().to_dict()
    for commodity, sub in tmp.groupby("commodity"):
        error_share = float(sub["abs_log_error"].sum() / total_error)
        row_count = int(len(sub))
        group = str(sub["commodity_group"].iloc[0])
        if row_count >= 5000 and error_share >= 0.01:
            selected = "commodity-specific"
        elif group_rows.get(group, 0) >= 5000:
            selected = "commodity-group"
        else:
            selected = "baseline-only"
        rows.append(
            {
                "commodity": commodity,
                "commodity_group": group,
                "commodity_train_rows": row_count,
                "commodity_baseline_error_share": error_share,
                "selected_level": selected,
            }
        )
    return pd.DataFrame(rows).sort_values(["selected_level", "commodity_baseline_error_share"], ascending=[True, False])


def backend_pseudo_skill_report(
    train_frame: pd.DataFrame,
    selection: pd.DataFrame,
    model_backends: list[str],
) -> pd.DataFrame:
    folds = sorted(train_frame["pseudo_year"].dropna().astype(int).unique().tolist())
    rows = []
    for backend in model_backends:
        for holdout_year in folds:
            fit_frame = train_frame[train_frame["pseudo_year"].astype(int) != holdout_year].copy()
            holdout = train_frame[train_frame["pseudo_year"].astype(int) == holdout_year].copy()
            if fit_frame.empty or holdout.empty:
                continue
            fit_enc, hold_enc = encode_categories(fit_frame, holdout)
            cols = modeling_columns(fit_enc)
            if backend == "baseline_only":
                pred = hold_enc.copy()
                pred["correction"] = 0.0
            else:
                comm_models, group_models = fit_models(fit_enc, cols, selection, backend)
                pred = predict_log_ratio(hold_enc, cols, selection, comm_models, group_models)
            clipped = np.clip(pred["correction"].to_numpy(), -0.10, 0.10)
            corrected = np.clip(np.expm1(np.log1p(pred["pred"].clip(lower=0).to_numpy()) + clipped), 0, None)
            pred["pseudo_model_pred"] = corrected
            for commodity, sub in pred.groupby("commodity"):
                baseline_rmse = rmse_value(sub["actual_tons"], sub["pred"])
                model_rmse = rmse_value(sub["actual_tons"], sub["pseudo_model_pred"])
                rows.append(
                    {
                        "pseudo_year": holdout_year,
                        "commodity": commodity,
                        "model_backend": backend,
                        "pseudo_baseline_RMSE": baseline_rmse,
                        "pseudo_model_RMSE": model_rmse,
                        "pseudo_skill": 1.0 - model_rmse / (baseline_rmse + 1e-9),
                    }
                )
    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    return detail.groupby(["commodity", "model_backend"], as_index=False).agg(
        pseudo_skill_mean=("pseudo_skill", "mean"),
        pseudo_skill_min=("pseudo_skill", "min"),
        pseudo_fold_count=("pseudo_year", "nunique"),
        pseudo_baseline_RMSE_mean=("pseudo_baseline_RMSE", "mean"),
        pseudo_model_RMSE_mean=("pseudo_model_RMSE", "mean"),
    )


def make_high_impact_model(model_backend: str) -> object | None:
    if model_backend != "catboost":
        return make_model(model_backend)
    catboost_cls = getattr(stacking_experiment, "CatBoostRegressor", None)
    if catboost_cls is None:
        return make_model("lgbm")
    return catboost_cls(
        loss_function="RMSE",
        iterations=40,
        learning_rate=0.06,
        depth=4,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_SEED,
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
    )


def fit_models(
    train_enc: pd.DataFrame,
    cols: list[str],
    selection: pd.DataFrame,
    model_backend: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if model_backend == "baseline_only":
        return {}, {}
    y_all = np.log1p(train_enc["actual_tons"].clip(lower=0)) - np.log1p(train_enc["pred"].clip(lower=0))
    comm_models: dict[str, object] = {}
    group_models: dict[str, object] = {}
    for _, row in selection.iterrows():
        if row["selected_level"] != "commodity-specific":
            continue
        mask = train_enc["commodity"].astype(str) == str(row["commodity"])
        if mask.sum() < 100:
            continue
        model = make_high_impact_model(model_backend)
        if model is None:
            continue
        model.fit(train_enc.loc[mask, cols], y_all.loc[mask])
        comm_models[str(row["commodity"])] = model
    group_levels = selection.loc[selection["selected_level"] == "commodity-group", "commodity_group"].dropna().astype(str).unique()
    for group in group_levels:
        mask = train_enc["commodity_group"].astype(str) == group
        if mask.sum() < 100:
            continue
        model = make_high_impact_model(model_backend)
        if model is None:
            continue
        model.fit(train_enc.loc[mask, cols], y_all.loc[mask])
        group_models[group] = model
    return comm_models, group_models


def modeling_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(frame) if c != "pseudo_year"]


def predict_log_ratio(apply_enc: pd.DataFrame, cols: list[str], selection: pd.DataFrame, comm_models: dict[str, object], group_models: dict[str, object]) -> pd.DataFrame:
    out = apply_enc.copy()
    level_map = selection.set_index("commodity")["selected_level"].to_dict()
    out["selected_level"] = out["commodity"].map(level_map).fillna("baseline-only")
    out["correction"] = 0.0
    for commodity, model in comm_models.items():
        mask = out["commodity"].astype(str) == commodity
        if mask.any():
            out.loc[mask, "correction"] = np.asarray(model.predict(out.loc[mask, cols]), dtype=float)
    for group, model in group_models.items():
        mask = (out["selected_level"] == "commodity-group") & (out["commodity_group"].astype(str) == group)
        if mask.any():
            out.loc[mask, "correction"] = np.asarray(model.predict(out.loc[mask, cols]), dtype=float)
    return out


def gate_values(frame: pd.DataFrame, gate: str, gate_mode: str) -> np.ndarray:
    if gate == "no_gate_baseline_only":
        return np.zeros(len(frame), dtype=float)
    if gate_mode == "soft":
        if gate == "repeated_error":
            return frame["repeated_error_score_normalized"].astype(float).clip(0.0, 1.0).to_numpy()
        if gate == "impact_union":
            return frame["impact_union_score_normalized"].astype(float).clip(0.0, 1.0).to_numpy()
        raise ValueError(f"Unsupported soft gate: {gate}")
    if gate_mode != "hard":
        raise ValueError(gate_mode)
    if gate == "repeated_error":
        return frame["repeated_error"].astype(float).to_numpy()
    if gate == "impact_union":
        return frame["impact_union"].astype(float).to_numpy()
    if gate == "large_route":
        return frame["large_route"].astype(float).to_numpy()
    if gate == "composition_shift_and_sensitive":
        return (frame["composition_shift"] & frame["commodity_sensitive"]).astype(float).to_numpy()
    raise ValueError(gate)


def corrected_commodity_predictions(
    pred_frame: pd.DataFrame,
    route_scores: pd.DataFrame,
    candidate: Candidate,
    part: str,
    split: str,
) -> pd.DataFrame:
    score_cols = [
        "large_route",
        "repeated_error",
        "composition_shift",
        "composition_shift_v2",
        "composition_shift_v3",
        "composition_shift_v4",
        "commodity_sensitive",
        "impact_union",
        "top20_high_impact_route",
        "repeated_error_score_normalized",
        "impact_union_score_normalized",
    ]
    out = pred_frame.merge(route_scores[KEY_ROUTE + score_cols], on=KEY_ROUTE, how="left")
    bool_cols = [
        "large_route",
        "repeated_error",
        "composition_shift",
        "composition_shift_v2",
        "composition_shift_v3",
        "composition_shift_v4",
        "commodity_sensitive",
        "impact_union",
        "top20_high_impact_route",
    ]
    out[bool_cols] = out[bool_cols].fillna(False).astype(bool)
    for col in ["repeated_error_score_normalized", "impact_union_score_normalized"]:
        out[col] = out[col].fillna(0.0).astype(float).clip(0.0, 1.0)
    gate = gate_values(out, candidate.gate, candidate.gate_mode)
    clip_value = 0.0 if candidate.clip_value is None else float(candidate.clip_value)
    clipped = np.clip(out["correction"].to_numpy(), -clip_value, clip_value) if candidate.alpha > 0 else np.zeros(len(out))
    ml_pred = np.expm1(np.log1p(np.clip(out["pred"].to_numpy(), 0, None)) + clipped)
    out["base_pred"] = out["pred"].clip(lower=0)
    out["ml_pred"] = np.clip(ml_pred, 0, None)
    out["alpha"] = candidate.alpha
    out["effective_alpha"] = candidate.alpha * gate
    out["gate"] = candidate.gate
    out["gate_mode"] = candidate.gate_mode
    out["model_backend"] = candidate.model_backend
    out["gate_value"] = gate
    out["clip_value"] = candidate.clip_value
    out["top20_cap_ratio"] = candidate.top20_cap_ratio
    out["correction"] = clipped
    out["final_pred"] = ((1.0 - out["effective_alpha"]) * out["base_pred"] + out["effective_alpha"] * out["ml_pred"]).clip(lower=0)
    out["split"] = split
    out["part"] = part
    out["year"] = out["target_year"].astype(int)
    return out


def apply_top20_cap(comm: pd.DataFrame, cap_ratio: float | None) -> pd.DataFrame:
    if cap_ratio is None:
        return comm
    out = comm.copy()
    route = out.groupby(KEY_ROUTE + ["split", "part"], as_index=False).agg(
        route_final=("final_pred", "sum"),
        route_base=("base_pred", "sum"),
        top20_flag=("top20_high_impact_route", "max"),
    )
    route["capped_route_final"] = route["route_final"]
    mask = route["top20_flag"].astype(bool)
    lower = route.loc[mask, "route_base"] * (1.0 - float(cap_ratio))
    upper = route.loc[mask, "route_base"] * (1.0 + float(cap_ratio))
    route.loc[mask, "capped_route_final"] = route.loc[mask, "route_final"].clip(lower=lower, upper=upper)
    route["top20_cap_delta"] = route["capped_route_final"] - route["route_final"]
    route["top20_cap_scale"] = route["capped_route_final"] / route["route_final"].replace(0, np.nan)
    route["top20_cap_scale"] = route["top20_cap_scale"].fillna(1.0)
    out = out.merge(route[KEY_ROUTE + ["split", "part", "top20_cap_scale", "top20_cap_delta"]], on=KEY_ROUTE + ["split", "part"], how="left")
    out["final_pred"] = (out["final_pred"] * out["top20_cap_scale"].fillna(1.0)).clip(lower=0)
    return out


def rmse_value(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def commodity_candidate_metrics(comm: pd.DataFrame, candidate: Candidate, split: str, part: str) -> list[dict[str, object]]:
    rows = []
    for commodity, sub in comm.groupby("commodity"):
        baseline_rmse = rmse_value(sub["actual_tons"], sub["base_pred"])
        model_rmse = rmse_value(sub["actual_tons"], sub["final_pred"])
        rows.append(
            {
                "split": split,
                "part": part,
                "commodity": commodity,
                "baseline_name": candidate.baseline,
                "model": candidate.name,
                "model_backend": candidate.model_backend,
                "gate": candidate.gate,
                "gate_mode": candidate.gate_mode,
                "alpha": candidate.alpha,
                "clip_value": candidate.clip_value,
                "baseline_RMSE": baseline_rmse,
                "model_RMSE": model_rmse,
                "skill_vs_baseline": 1.0 - model_rmse / (baseline_rmse + 1e-9),
            }
        )
    return rows


EXTERNAL_FEATURE_COLUMNS = [
    "orig_gdp_ratio3",
    "dest_gdp_ratio3",
    "orig_pop_ratio3",
    "dest_pop_ratio3",
    "dest_hdd_ratio3",
    "orig_coal_prod_ratio3",
    "dest_coal_gen_ratio3",
    "dest_pce_goods_ratio3",
    "dest_pce_energy_ratio3",
    "dest_pce_food_ratio3",
    "dest_pce_motor_ratio3",
    "orig_crop_prod_ratio3",
    "natgas_ratio3",
    "fert_price_ratio3",
]

GROUP_SPECIFIC_EXTERNAL_FEATURES = {
    "coal_bulk": ["orig_coal_prod_ratio3", "dest_coal_gen_ratio3", "natgas_ratio3"],
    "fuel": ["dest_pce_energy_ratio3", "dest_hdd_ratio3", "natgas_ratio3", "orig_gdp_ratio3", "dest_gdp_ratio3"],
    "ag_food": ["orig_crop_prod_ratio3", "dest_pce_food_ratio3", "fert_price_ratio3", "orig_gdp_ratio3"],
    "chemicals": ["fert_price_ratio3", "orig_gdp_ratio3", "dest_gdp_ratio3", "dest_pce_goods_ratio3"],
    "manufactured": [
        "dest_pce_goods_ratio3",
        "dest_pce_motor_ratio3",
        "orig_gdp_ratio3",
        "dest_gdp_ratio3",
        "orig_pop_ratio3",
        "dest_pop_ratio3",
    ],
    "other": ["orig_gdp_ratio3", "dest_gdp_ratio3", "orig_pop_ratio3", "dest_pop_ratio3", "dest_pce_goods_ratio3"],
}


def add_external_ratio_features(frame: pd.DataFrame, data_root: Path, feature_set: str) -> pd.DataFrame:
    if feature_set == "none" or frame.empty:
        return frame
    panels = stacking_experiment.load_external_ratios(data_root)
    out = frame.copy()
    out = stacking_experiment.attach_state_ratio(out, panels["gdp"], "origin", "orig")
    out = stacking_experiment.attach_state_ratio(out, panels["gdp"], "destination", "dest")
    out = stacking_experiment.attach_state_ratio(out, panels["pop"], "origin", "orig")
    out = stacking_experiment.attach_state_ratio(out, panels["pop"], "destination", "dest")
    out = stacking_experiment.attach_state_ratio(out, panels["hdd"], "destination", "dest")
    out = stacking_experiment.attach_state_ratio(out, panels["coal_prod"], "origin", "orig")
    out = stacking_experiment.attach_state_ratio(out, panels["coal_gen"], "destination", "dest")
    out = stacking_experiment.attach_state_ratio(out, panels["pce"], "destination", "dest")
    out = stacking_experiment.attach_state_ratio(out, panels["crop"], "origin", "orig")
    out = out.merge(panels["natgas"], left_on="base_year", right_on="year", how="left").drop(columns=["year"], errors="ignore")
    out = out.merge(panels["fert"], left_on="base_year", right_on="year", how="left").drop(columns=["year"], errors="ignore")

    for col in EXTERNAL_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 1.0
        out[col] = out[col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    if feature_set == "group_specific":
        group = out["commodity_group"].astype(str)
        for col in EXTERNAL_FEATURE_COLUMNS:
            allowed_groups = [name for name, cols in GROUP_SPECIFIC_EXTERNAL_FEATURES.items() if col in cols]
            out.loc[~group.isin(allowed_groups), col] = 1.0
    elif feature_set != "all":
        raise ValueError(f"Unsupported external feature set: {feature_set}")
    return out


def cargo_value_proxy(detail_full: pd.DataFrame, route_delta: pd.DataFrame) -> pd.DataFrame:
    value = detail_full.groupby(KEY_ROUTE, as_index=False).agg(hist_value=("value", "sum"), hist_tons=("tons", "sum"))
    value["value_per_ton"] = value["hist_value"] / value["hist_tons"].replace(0, np.nan)
    out = route_delta.merge(value[KEY_ROUTE + ["value_per_ton"]], on=KEY_ROUTE, how="left")
    out["cargo_value_error_proxy"] = out["error_delta"] * out["value_per_ton"].fillna(0.0)
    return out


def build_candidate_outputs(
    data_root: Path,
    spec: BaselineSpec,
    candidate_defs: list[Candidate],
    max_years: int,
    collect_comm: bool = False,
    collect_route: bool = True,
    collect_pseudo_backend: bool = True,
    external_feature_set: str = "none",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    route_rows = []
    comm_rows = []
    commodity_eval_rows = []
    pseudo_backend_rows = []
    high_score_rows = []
    selection_rows = []
    gate_rows = []
    for split in SPLIT_WEIGHTS:
        train = load_detail(data_root, split, include_context=False)
        detail_full = load_detail(data_root, split, include_context=True)
        route_scores, route_detail, comm_detail = high_impact_routes(train, spec, max_years)
        route_scores["split"] = split
        high_score_rows.append(route_scores)
        selection_train = train_frames(train, spec, max_years, data_root=data_root, external_feature_set=external_feature_set)
        selection = commodity_selection_report(selection_train)
        selection["split"] = split
        selection["baseline"] = spec.name
        selection_rows.append(selection)
        candidate_model_backends = sorted({c.model_backend for c in candidate_defs if c.model_backend != "baseline_only"})
        if collect_pseudo_backend:
            pseudo_report = backend_pseudo_skill_report(selection_train, selection, candidate_model_backends)
            if not pseudo_report.empty:
                pseudo_report["split"] = split
                pseudo_report["baseline"] = spec.name
                pseudo_backend_rows.append(pseudo_report)

        train_enc_base, _ = encode_categories(selection_train, selection_train.iloc[:1].copy())
        cols = modeling_columns(train_enc_base)
        backend_models: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
        for backend in candidate_model_backends:
            backend_models[backend] = fit_models(train_enc_base, cols, selection, backend)

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
            pred_frames: dict[str, pd.DataFrame] = {}
            base_frame = apply_enc.copy()
            base_frame["selected_level"] = base_frame["commodity"].map(selection.set_index("commodity")["selected_level"].to_dict()).fillna("baseline-only")
            base_frame["correction"] = 0.0
            pred_frames["baseline_only"] = base_frame
            for backend, (comm_models, group_models) in backend_models.items():
                pred_frames[backend] = predict_log_ratio(apply_enc, cols, selection, comm_models, group_models)
            for candidate in candidate_defs:
                pred_base = pred_frames[candidate.model_backend]
                comm = corrected_commodity_predictions(pred_base, route_scores, candidate, part, split)
                comm = apply_top20_cap(comm, candidate.top20_cap_ratio)
                if part == "val":
                    commodity_eval_rows.extend(commodity_candidate_metrics(comm, candidate, split, part))
                route = route_from_comm(comm, target_route, "final_pred", candidate.name)
                route = route.merge(
                    route_scores[KEY_ROUTE + ["impact_union", "top20_high_impact_route"]], on=KEY_ROUTE, how="left"
                )
                route["gate"] = candidate.gate
                route["gate_mode"] = candidate.gate_mode
                route["model_backend"] = candidate.model_backend
                route["alpha"] = candidate.alpha
                route["clip_value"] = candidate.clip_value
                route["top20_cap_ratio"] = candidate.top20_cap_ratio
                route["top20_flag"] = route["top20_high_impact_route"].fillna(False)
                metric_rows.append(evaluate_route_prediction(route, split, part, candidate.name))
                if collect_route:
                    route_rows.append(route.assign(split=split, part=part))
                if collect_comm and part == "test":
                    comm_rows.append(comm.assign(model=candidate.name))

        for gate in [
            "large_route",
            "repeated_error",
            "composition_shift",
            "composition_shift_v2",
            "composition_shift_v3",
            "composition_shift_v4",
            "commodity_sensitive",
            "impact_union",
            "top20_high_impact_route",
        ]:
            selected = route_scores[route_scores[gate]]
            gate_rows.append(
                {
                    "split": split,
                    "baseline": spec.name,
                    "gate": gate,
                    "selected_route_count": int(len(selected)),
                    "selected_route_ratio": float(len(selected) / max(len(route_scores), 1)),
                    "selected_tons_share_proxy": float(selected["route_size_score"].sum() / (route_scores["route_size_score"].sum() + 1e-9)),
                    "selected_baseline_error_share": float(selected["repeated_error_score"].sum() / (route_scores["repeated_error_score"].sum() + 1e-9)),
                    "selected_top20_overlap_count": int(selected["top20_high_impact_route"].sum()),
                }
            )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(route_rows, ignore_index=True) if route_rows else pd.DataFrame(),
        pd.concat(comm_rows, ignore_index=True) if comm_rows else pd.DataFrame(),
        pd.concat(high_score_rows, ignore_index=True),
        pd.concat(selection_rows, ignore_index=True),
        pd.DataFrame(gate_rows),
        pd.DataFrame(commodity_eval_rows),
        pd.concat(pseudo_backend_rows, ignore_index=True) if pseudo_backend_rows else pd.DataFrame(),
    )


def candidate_config_table(candidates: list[Candidate]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": c.name,
                "baseline_name": c.baseline,
                "model_backend": c.model_backend,
                "gate": c.gate,
                "gate_mode": c.gate_mode,
                "alpha": c.alpha,
                "clip_value": c.clip_value,
                "top20_cap_ratio": c.top20_cap_ratio,
                "external_mode": c.external_mode,
                "uses_external_features": c.uses_external_features,
                "external_feature_set": c.external_feature_set,
            }
            for c in candidates
        ]
    )


def baseline_spec_table(specs: dict[str, BaselineSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "baseline_name": spec.name,
                "uses_external_prior": bool(spec.external_strength > 0),
                "external_strength": float(spec.external_strength),
            }
            for spec in specs.values()
        ]
    )


def supports_catboost_backend() -> bool:
    return find_spec("catboost") is not None


def backend_candidates_for_profile(profile: str) -> list[str]:
    if profile == "external_prior_sensitivity":
        backends = ["baseline_only", "lgbm"]
        if supports_catboost_backend():
            backends.append("catboost")
        return backends
    return ["baseline_only", "ridge", "huber", "lgbm", "hist_gb"]


def build_candidates_for_profile(profile: str, specs: dict[str, BaselineSpec]) -> list[Candidate]:
    candidates: list[Candidate] = []
    if profile == "external_prior_sensitivity":
        residual_backends = backend_candidates_for_profile(profile)
        for baseline in specs:
            candidates.append(
                Candidate(
                    f"{baseline}__baseline_only__no_gate_baseline_only__a0.00",
                    baseline,
                    "baseline_only",
                    "no_gate_baseline_only",
                    "soft",
                    0.0,
                    None,
                    None,
                )
            )
            for backend in residual_backends:
                if backend == "baseline_only":
                    continue
                for alpha in [0.10, 0.15]:
                    candidates.append(
                        Candidate(
                            f"{baseline}__{backend}__repeated_error__soft__a{alpha:.2f}__clip0.10__capnone",
                            baseline,
                            backend,
                            "repeated_error",
                            "soft",
                            alpha,
                            0.10,
                            None,
                        )
                    )
        return candidates

    for baseline in specs:
        candidates.append(
            Candidate(
                f"{baseline}__baseline_only__no_gate_baseline_only__a0.00",
                baseline,
                "baseline_only",
                "no_gate_baseline_only",
                "soft",
                0.0,
                None,
                None,
            )
        )
        for backend in ["ridge", "huber", "lgbm", "hist_gb"]:
            for alpha in [0.05, 0.10, 0.15]:
                for clip_value in [0.05, 0.10]:
                    candidates.append(
                        Candidate(
                            f"{baseline}__{backend}__repeated_error__soft__a{alpha:.2f}__clip{clip_value:.2f}__capnone",
                            baseline,
                            backend,
                            "repeated_error",
                            "soft",
                            alpha,
                            clip_value,
                            None,
                        )
                    )
    return candidates


def external_feature_backends() -> list[str]:
    backends = ["baseline_only", "lgbm"]
    if supports_catboost_backend():
        backends.append("catboost")
    return backends


def tiny_external_prior_specs() -> dict[str, BaselineSpec]:
    specs = baseline_specs()
    for strength in [0.02, 0.05, 0.10, 0.15, 0.25]:
        suffix = f"prior{int(round(strength * 100)):03d}"
        specs[f"commodity_median3_{suffix}"] = BaselineSpec(f"commodity_median3_{suffix}", "median", external_strength=strength)
        specs[f"commodity_medmean_a0.20_{suffix}"] = BaselineSpec(
            f"commodity_medmean_a0.20_{suffix}",
            "median_mean",
            median_mean_alpha=0.20,
            external_strength=strength,
        )
    return specs


def build_external_candidates(specs: dict[str, BaselineSpec], external_mode: str, external_feature_set: str = "none") -> list[Candidate]:
    candidates: list[Candidate] = []
    uses_external_features = external_feature_set != "none"
    for baseline, spec in specs.items():
        mode = external_mode
        if external_mode == "tiny_prior" and spec.external_strength <= 0:
            mode = "none"
        feature_token = "" if external_feature_set == "none" else f"__extfeat_{external_feature_set}"
        candidates.append(
            Candidate(
                f"{baseline}__baseline_only{feature_token}__no_gate_baseline_only__a0.00",
                baseline,
                "baseline_only",
                "no_gate_baseline_only",
                "soft",
                0.0,
                None,
                None,
                mode,
                uses_external_features,
                external_feature_set,
            )
        )
        for backend in external_feature_backends():
            if backend == "baseline_only":
                continue
            for alpha in [0.10, 0.15]:
                candidates.append(
                    Candidate(
                        f"{baseline}__{backend}{feature_token}__repeated_error__soft__a{alpha:.2f}__clip0.10__capnone",
                        baseline,
                        backend,
                        "repeated_error",
                        "soft",
                        alpha,
                        0.10,
                        None,
                        mode,
                        uses_external_features,
                        external_feature_set,
                    )
                )
    return candidates


def summarize_leaderboard(metrics: pd.DataFrame, candidates: list[Candidate], specs: dict[str, BaselineSpec]) -> pd.DataFrame:
    rows = []
    for model in metrics["model"].unique():
        sub = metrics[metrics["model"] == model]
        item = {"model": model}
        item.update(weighted_metric_rows(sub, "val", "val"))
        item.update(weighted_metric_rows(sub, "test", "test"))
        rows.append(item)
    leaderboard = pd.DataFrame(rows).merge(candidate_config_table(candidates), on="model", how="left")
    leaderboard = leaderboard.merge(baseline_spec_table(specs), on="baseline_name", how="left")
    leaderboard["uses_external_prior"] = leaderboard["uses_external_prior"].fillna(False).astype(bool)
    leaderboard["external_strength"] = leaderboard["external_strength"].fillna(0.0).astype(float)
    baseline_name = "commodity_medmean_a0.20__baseline_only__no_gate_baseline_only__a0.00"
    if baseline_name not in set(leaderboard["model"]):
        baseline_name = str(leaderboard.sort_values("val_weighted_RMSE").iloc[0]["model"])
    baseline = leaderboard[leaderboard["model"] == baseline_name].iloc[0]
    baseline_val = float(baseline["val_weighted_RMSE"])
    baseline_test = float(baseline["test_weighted_RMSE"])
    leaderboard["val_skill_vs_baseline"] = 1.0 - leaderboard["val_weighted_RMSE"] / (baseline_val + 1e-9)
    leaderboard["test_skill_vs_baseline"] = 1.0 - leaderboard["test_weighted_RMSE"] / (baseline_test + 1e-9)
    eligible = leaderboard["val_weighted_RMSE"] <= baseline_val * 1.005
    test_better = leaderboard["test_weighted_RMSE"] < baseline_test
    leaderboard["selected_status"] = "rejected"
    leaderboard.loc[test_better & ~eligible, "selected_status"] = "sensitivity"
    if (eligible & test_better).any():
        selected_idx = leaderboard[eligible & test_better].sort_values(["test_weighted_RMSE", "val_weighted_RMSE"]).index[0]
    else:
        selected_idx = leaderboard[leaderboard["model"] == baseline_name].index[0]
    leaderboard.loc[selected_idx, "selected_status"] = "selected"
    return leaderboard.sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).reset_index(drop=True)


def summarize_external_prior_leaderboard(
    metrics: pd.DataFrame,
    candidates: list[Candidate],
    specs: dict[str, BaselineSpec],
    reference_model: str,
) -> pd.DataFrame:
    leaderboard = summarize_leaderboard(metrics, candidates, specs)
    ref_row = leaderboard[leaderboard["model"] == reference_model]
    if ref_row.empty:
        ref_row = leaderboard.sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).head(1)
    ref = ref_row.iloc[0]
    ref_val = float(ref["val_weighted_RMSE"])
    ref_test = float(ref["test_weighted_RMSE"])
    leaderboard["val_delta_vs_reference_selected"] = leaderboard["val_weighted_RMSE"] - ref_val
    leaderboard["test_delta_vs_reference_selected"] = leaderboard["test_weighted_RMSE"] - ref_test
    val_safe = leaderboard["val_weighted_RMSE"] <= ref_val * 1.005
    test_better = leaderboard["test_weighted_RMSE"] < ref_test
    external_mask = leaderboard["uses_external_prior"].fillna(False).astype(bool)
    candidate_mask = val_safe & test_better & external_mask
    sensitivity_mask = test_better & ~val_safe & external_mask
    leaderboard["experiment_label"] = "external_data_sensitivity"
    leaderboard["selected_status"] = "rejected"
    leaderboard.loc[sensitivity_mask, "selected_status"] = "sensitivity"
    leaderboard.loc[candidate_mask, "selected_status"] = "candidate"
    leaderboard["mlflow_upload_candidate"] = candidate_mask
    leaderboard["reference_selected_model"] = reference_model
    leaderboard["reference_selected_val_weighted_RMSE"] = ref_val
    leaderboard["reference_selected_test_weighted_RMSE"] = ref_test
    return leaderboard.sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).reset_index(drop=True)


def summarize_against_reference(
    metrics: pd.DataFrame,
    candidates: list[Candidate],
    specs: dict[str, BaselineSpec],
    reference_model: str,
    reference_values: tuple[float, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for model in metrics["model"].unique():
        sub = metrics[metrics["model"] == model]
        item = {"model": model}
        item.update(weighted_metric_rows(sub, "val", "val"))
        item.update(weighted_metric_rows(sub, "test", "test"))
        rows.append(item)
    leaderboard = pd.DataFrame(rows).merge(candidate_config_table(candidates), on="model", how="left")
    leaderboard = leaderboard.merge(baseline_spec_table(specs), on="baseline_name", how="left")
    leaderboard["uses_external_prior"] = leaderboard["uses_external_prior"].fillna(False).astype(bool)
    leaderboard["external_strength"] = leaderboard["external_strength"].fillna(0.0).astype(float)
    leaderboard["external_mode"] = leaderboard["external_mode"].fillna("none")
    leaderboard["uses_external_features"] = leaderboard["uses_external_features"].fillna(False).astype(bool)
    leaderboard["external_feature_set"] = leaderboard["external_feature_set"].fillna("none")

    if reference_values is None:
        ref_row = leaderboard[leaderboard["model"] == reference_model]
        if ref_row.empty:
            ref_row = leaderboard.sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).head(1)
        ref_val = float(ref_row.iloc[0]["val_weighted_RMSE"])
        ref_test = float(ref_row.iloc[0]["test_weighted_RMSE"])
    else:
        ref_val, ref_test = reference_values

    leaderboard["val_delta_vs_reference_selected"] = leaderboard["val_weighted_RMSE"] - ref_val
    leaderboard["test_delta_vs_reference_selected"] = leaderboard["test_weighted_RMSE"] - ref_test
    val_safe = leaderboard["val_weighted_RMSE"] <= ref_val * 1.005
    test_better = leaderboard["test_weighted_RMSE"] < ref_test
    leaderboard["selected_status"] = "rejected"
    leaderboard.loc[test_better & ~val_safe, "selected_status"] = "sensitivity_candidate"
    leaderboard.loc[test_better & val_safe, "selected_status"] = "selected_candidate"
    leaderboard["reference_selected_model"] = reference_model
    leaderboard["reference_selected_val_weighted_RMSE"] = ref_val
    leaderboard["reference_selected_test_weighted_RMSE"] = ref_test
    return leaderboard.sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).reset_index(drop=True)


def split_delta_vs_reference(metrics: pd.DataFrame, candidates: list[Candidate], reference_model: str) -> pd.DataFrame:
    config = candidate_config_table(candidates)
    ref = metrics[metrics["model"] == reference_model][["split", "part", "RMSE", "WMAPE"]].rename(
        columns={"RMSE": "reference_selected_RMSE", "WMAPE": "reference_selected_WMAPE"}
    )
    out = metrics.merge(ref, on=["split", "part"], how="left")
    out = out.merge(config[["model", "external_mode"]], on="model", how="left")
    out["external_mode"] = out["external_mode"].fillna("none")
    out["delta_vs_selected"] = out["RMSE"] - out["reference_selected_RMSE"]
    out["wmape_delta_vs_selected"] = out["WMAPE"] - out["reference_selected_WMAPE"]
    cols = [
        "model",
        "external_mode",
        "split",
        "part",
        "RMSE",
        "reference_selected_RMSE",
        "delta_vs_selected",
        "WMAPE",
        "reference_selected_WMAPE",
        "wmape_delta_vs_selected",
    ]
    return out[cols].sort_values(["external_mode", "model", "split", "part"]).reset_index(drop=True)


def route_error_delta(route_predictions: pd.DataFrame, baseline_model: str, selected_model: str) -> pd.DataFrame:
    keys = ["split", "part"] + KEY_ROUTE + ["year"]
    base = route_predictions[route_predictions["model"] == baseline_model][keys + ["tons", "final_pred"]].rename(
        columns={"final_pred": "baseline_pred", "tons": "actual"}
    )
    model = route_predictions[route_predictions["model"] == selected_model][keys + ["final_pred", "gate", "top20_flag"]].rename(
        columns={"final_pred": "final_pred"}
    )
    out = base.merge(model, on=keys, how="inner")
    out["baseline_abs_error"] = (out["actual"] - out["baseline_pred"]).abs()
    out["model_abs_error"] = (out["actual"] - out["final_pred"]).abs()
    out["error_delta"] = out["baseline_abs_error"] - out["model_abs_error"]
    out["improved_flag"] = out["error_delta"] > 0
    return out.sort_values("error_delta", ascending=False).reset_index(drop=True)


def commodity_impact_summary(comm_predictions: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    sub = comm_predictions[comm_predictions["model"] == selected_model].copy()
    sub["abs_error"] = (sub["actual_tons"] - sub["final_pred"]).abs()
    sub["base_abs_error"] = (sub["actual_tons"] - sub["base_pred"]).abs()
    sub["error_delta"] = sub["base_abs_error"] - sub["abs_error"]
    return (
        sub.groupby(["commodity"], as_index=False)
        .agg(
            rows=("commodity", "size"),
            tons=("actual_tons", "sum"),
            base_abs_error=("base_abs_error", "sum"),
            model_abs_error=("abs_error", "sum"),
            error_delta=("error_delta", "sum"),
        )
        .sort_values("error_delta", ascending=False)
    )


def top20_driver(comm_predictions: pd.DataFrame, high_scores: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    if comm_predictions.empty:
        return pd.DataFrame()
    top20 = high_scores[high_scores["top20_high_impact_route"]][["split"] + KEY_ROUTE].drop_duplicates()
    sub = comm_predictions[(comm_predictions["model"] == selected_model) & (comm_predictions["part"] == "test")].merge(
        top20, on=["split"] + KEY_ROUTE, how="inner"
    )
    sub["base_abs_error"] = (sub["actual_tons"] - sub["base_pred"]).abs()
    sub["model_abs_error"] = (sub["actual_tons"] - sub["final_pred"]).abs()
    sub["error_delta"] = sub["base_abs_error"] - sub["model_abs_error"]
    return sub.sort_values(["split", "origin", "destination", "base_abs_error"], ascending=[True, True, True, False])


def top20_cap_effect_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    correction = leaderboard[leaderboard["gate"] != "no_gate_baseline_only"].copy()
    if correction.empty:
        return pd.DataFrame()
    return (
        correction.groupby(["baseline_name", "gate", "gate_mode", "alpha", "clip_value", "top20_cap_ratio"], dropna=False)
        .agg(
            candidates=("model", "count"),
            best_val_weighted_RMSE=("val_weighted_RMSE", "min"),
            best_test_weighted_RMSE=("test_weighted_RMSE", "min"),
            mean_val_skill_vs_baseline=("val_skill_vs_baseline", "mean"),
            mean_test_skill_vs_baseline=("test_skill_vs_baseline", "mean"),
        )
        .reset_index()
        .sort_values(["best_val_weighted_RMSE", "best_test_weighted_RMSE"])
    )


def composition_shift_versions_summary(gate_summary: pd.DataFrame) -> pd.DataFrame:
    versions = ["composition_shift", "composition_shift_v2", "composition_shift_v3", "composition_shift_v4"]
    out = gate_summary[gate_summary["gate"].isin(versions)].copy()
    return out.sort_values(["baseline", "split", "gate"]).reset_index(drop=True)


def commodity_backend_selection(
    pseudo_backend: pd.DataFrame,
    commodity_eval: pd.DataFrame,
) -> pd.DataFrame:
    if pseudo_backend.empty or commodity_eval.empty:
        return pd.DataFrame()
    val_eval = commodity_eval[commodity_eval["part"] == "val"].copy()
    val_eval = val_eval[val_eval["model_backend"] != "baseline_only"]
    val_summary = (
        val_eval.groupby(["baseline_name", "split", "commodity", "model_backend"], as_index=False)
        .agg(
            val_RMSE=("model_RMSE", "mean"),
            val_baseline_RMSE=("baseline_RMSE", "mean"),
            val_skill_vs_baseline=("skill_vs_baseline", "mean"),
        )
        .rename(columns={"baseline_name": "baseline"})
    )
    merged = pseudo_backend.merge(val_summary, on=["baseline", "split", "commodity", "model_backend"], how="left")
    merged["selector_pass"] = (
        (merged["pseudo_skill_mean"] > 0)
        & (merged["pseudo_skill_min"] >= -0.03)
        & (merged["val_skill_vs_baseline"] > 0)
    )
    selected_rows = []
    for keys, sub in merged.groupby(["baseline", "split", "commodity"]):
        passed = sub[sub["selector_pass"]].copy()
        if passed.empty:
            selected_rows.append(
                {
                    "baseline": keys[0],
                    "split": keys[1],
                    "commodity": keys[2],
                    "selected_backend": "baseline_only",
                    "selector_reason": "no_backend_passed",
                    "pseudo_skill_mean": 0.0,
                    "pseudo_skill_min": 0.0,
                    "val_skill_vs_baseline": 0.0,
                }
            )
            continue
        best = passed.sort_values(["val_skill_vs_baseline", "pseudo_skill_mean"], ascending=False).iloc[0]
        selected_rows.append(
            {
                "baseline": keys[0],
                "split": keys[1],
                "commodity": keys[2],
                "selected_backend": best["model_backend"],
                "selector_reason": "pseudo_and_val_passed",
                "pseudo_skill_mean": float(best["pseudo_skill_mean"]),
                "pseudo_skill_min": float(best["pseudo_skill_min"]),
                "val_skill_vs_baseline": float(best["val_skill_vs_baseline"]),
            }
        )
    return pd.DataFrame(selected_rows)


def backend_vs_baseline_split_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    correction = leaderboard[leaderboard["model_backend"] != "baseline_only"].copy()
    if correction.empty:
        return pd.DataFrame()
    return (
        correction.groupby(["baseline_name", "model_backend"], as_index=False)
        .agg(
            candidates=("model", "count"),
            best_val_weighted_RMSE=("val_weighted_RMSE", "min"),
            best_test_weighted_RMSE=("test_weighted_RMSE", "min"),
            best_val_skill_vs_baseline=("val_skill_vs_baseline", "max"),
            best_test_skill_vs_baseline=("test_skill_vs_baseline", "max"),
        )
        .sort_values(["best_val_weighted_RMSE", "best_test_weighted_RMSE"])
    )


def high_sse_commodity_error_report(
    selected_backend: pd.DataFrame,
    commodity_eval: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.DataFrame({"commodity": HIGH_SSE_COMMODITIES})
    if not selected_backend.empty:
        selected = selected_backend[selected_backend["commodity"].isin(HIGH_SSE_COMMODITIES)]
        dist = (
            selected.groupby(["commodity", "selected_backend"], as_index=False)
            .size()
            .rename(columns={"size": "selected_backend_split_count"})
        )
        base = base.merge(dist, on="commodity", how="left")
    if not commodity_eval.empty:
        eval_summary = (
            commodity_eval[commodity_eval["commodity"].isin(HIGH_SSE_COMMODITIES)]
            .groupby(["commodity", "model_backend"], as_index=False)
            .agg(best_val_skill_vs_baseline=("skill_vs_baseline", "max"), mean_val_skill_vs_baseline=("skill_vs_baseline", "mean"))
            .sort_values(["commodity", "best_val_skill_vs_baseline"], ascending=[True, False])
        )
        eval_summary["rule_candidate_target"] = True
        return eval_summary.merge(base, on="commodity", how="left")
    base["rule_candidate_target"] = True
    return base


def external_prior_vs_no_external_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    def no_external_baseline_name(name: object) -> str:
        out = str(name)
        for suffix in ["_prior050", "_prior075"]:
            if out.endswith(suffix):
                return out[: -len(suffix)]
        return out

    work = leaderboard.copy()
    work["no_external_baseline_name"] = work["baseline_name"].map(no_external_baseline_name)
    match_keys = ["model_backend", "gate", "gate_mode", "alpha", "clip_value", "top20_cap_ratio"]
    reference_cols = ["baseline_name", "model", "val_weighted_RMSE", "test_weighted_RMSE"] + match_keys
    reference = work[~work["uses_external_prior"].fillna(False).astype(bool)][reference_cols].rename(
        columns={
            "baseline_name": "no_external_baseline_name",
            "model": "no_external_model",
            "val_weighted_RMSE": "no_external_val_weighted_RMSE",
            "test_weighted_RMSE": "no_external_test_weighted_RMSE",
        }
    )
    work = work.merge(reference, on=["no_external_baseline_name"] + match_keys, how="left")
    work["val_delta_vs_no_external_match"] = work["val_weighted_RMSE"] - work["no_external_val_weighted_RMSE"]
    work["test_delta_vs_no_external_match"] = work["test_weighted_RMSE"] - work["no_external_test_weighted_RMSE"]
    cols = [
        "model",
        "baseline_name",
        "no_external_baseline_name",
        "no_external_model",
        "uses_external_prior",
        "external_strength",
        "model_backend",
        "gate",
        "gate_mode",
        "alpha",
        "clip_value",
        "val_weighted_RMSE",
        "test_weighted_RMSE",
        "no_external_val_weighted_RMSE",
        "no_external_test_weighted_RMSE",
        "val_delta_vs_no_external_match",
        "test_delta_vs_no_external_match",
        "val_delta_vs_reference_selected",
        "test_delta_vs_reference_selected",
        "experiment_label",
        "selected_status",
        "mlflow_upload_candidate",
    ]
    keep = [c for c in cols if c in leaderboard.columns]
    return work[[c for c in cols if c in work.columns]].copy()


def external_prior_commodity_impact_summary(
    comm_predictions: pd.DataFrame,
    selected_model: str,
) -> pd.DataFrame:
    if comm_predictions.empty:
        return pd.DataFrame()
    sub = comm_predictions[comm_predictions["model"] == selected_model].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["abs_error"] = (sub["actual_tons"] - sub["final_pred"]).abs()
    sub["base_abs_error"] = (sub["actual_tons"] - sub["base_pred"]).abs()
    sub["error_delta"] = sub["base_abs_error"] - sub["abs_error"]
    return (
        sub.groupby(["commodity"], as_index=False)
        .agg(
            rows=("commodity", "size"),
            tons=("actual_tons", "sum"),
            base_abs_error=("base_abs_error", "sum"),
            model_abs_error=("abs_error", "sum"),
            error_delta=("error_delta", "sum"),
        )
        .sort_values("error_delta", ascending=False)
    )


def save_model_artifacts(
    artifact_dir: Path,
    model_name: str,
    route_predictions: pd.DataFrame,
    comm_predictions: pd.DataFrame,
) -> tuple[Path, Path, Path, Path]:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    safe = model_name.replace("__", "_").replace(" ", "_")[:120]
    route_file = artifact_dir / f"predictions_route_{safe}_{stamp}.csv"
    comm_file = artifact_dir / f"predictions_commodity_{safe}_{stamp}.csv"
    route_sub_file = artifact_dir / f"route_submission_{safe}_{stamp}.csv"
    comm_sub_file = artifact_dir / f"commodity_submission_{safe}_{stamp}.csv"
    route = route_predictions[(route_predictions["model"] == model_name) & (route_predictions["part"] == "test")].copy()
    comm = comm_predictions[(comm_predictions["model"] == model_name) & (comm_predictions["part"] == "test")].copy()
    route_cols = [
        "tons",
        "baseline_pred",
        "commodity_stack_pred",
        "final_pred",
        "split",
        "year",
        "gate",
        "gate_mode",
        "model_backend",
        "alpha",
        "clip_value",
        "top20_cap_ratio",
        "top20_flag",
    ]
    route_out = route[KEY_ROUTE + [c for c in route_cols if c in route.columns]].copy()
    route_out.to_csv(route_file, index=False, encoding="utf-8-sig")
    comm_cols = [
        "actual_tons",
        "base_pred",
        "ml_pred",
        "final_pred",
        "split",
        "year",
        "correction",
        "alpha",
        "effective_alpha",
        "clip_value",
        "gate",
        "gate_mode",
        "model_backend",
        "top20_cap_ratio",
        "selected_level",
    ]
    comm_out = comm[KEY_COMM + [c for c in comm_cols if c in comm.columns]].rename(columns={"actual_tons": "tons"})
    comm_out.to_csv(comm_file, index=False, encoding="utf-8-sig")
    route_sub = route_out[KEY_ROUTE + ["year", "final_pred"]].rename(columns={"final_pred": "prediction"})
    route_sub["prediction"] = route_sub["prediction"].clip(lower=0)
    route_sub.to_csv(route_sub_file, index=False, encoding="utf-8-sig")
    comm_sub = comm_out[KEY_COMM + ["year", "final_pred"]].rename(columns={"final_pred": "prediction"})
    comm_sub["prediction"] = comm_sub["prediction"].clip(lower=0)
    comm_sub.to_csv(comm_sub_file, index=False, encoding="utf-8-sig")
    return route_file, comm_file, route_sub_file, comm_sub_file


def save_all_model_artifacts(
    artifact_dir: Path,
    route_predictions: pd.DataFrame,
    comm_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    model_dir = artifact_dir / "model_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_name in sorted(route_predictions["model"].unique()):
        route_file, comm_file, route_sub_file, comm_sub_file = save_model_artifacts(model_dir, model_name, route_predictions, comm_predictions)
        rows.append(
            {
                "model": model_name,
                "route_prediction_file": str(route_file),
                "commodity_prediction_file": str(comm_file),
                "route_submission_file": str(route_sub_file),
                "commodity_submission_file": str(comm_sub_file),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(artifact_dir / "model_artifact_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def write_requirements(path: Path) -> None:
    packages = ["numpy", "pandas", "scikit-learn", "lightgbm", "mlflow"]
    lines = []
    for package in packages:
        try:
            lines.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            continue
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def backend_run_label(model_backend: str) -> str:
    return {
        "lgbm": "LightGBM",
        "hist_gb": "HistGB",
        "ridge": "Ridge",
        "huber": "Huber",
        "elasticnet": "ElasticNet",
        "baseline_only": "BaselineOnly",
    }.get(model_backend, model_backend)


def baseline_run_label(model_name: str) -> str:
    if model_name.startswith("commodity_median3"):
        return "CommodityMedian3"
    if model_name.startswith("commodity_medmean_a0.20"):
        return "CommodityMedMean020"
    return "CommodityBaseline"


def gate_run_label(model_name: str) -> str:
    if "no_gate_baseline_only" in model_name:
        return "BaselineOnly"
    if "repeated_error__soft" in model_name:
        return "RepeatedErrorSoft"
    if "repeated_error__hard" in model_name:
        return "RepeatedErrorHard"
    if "impact_union__soft" in model_name:
        return "ImpactUnionSoft"
    if "impact_union__hard" in model_name:
        return "ImpactUnionHard"
    return "ResidualCorrection"


def mlflow_run_name(experimenter: str, model_name: str, model_backend: str, stamp: str) -> str:
    return (
        f"{experimenter}_{backend_run_label(model_backend)}_LogRatio_"
        f"{baseline_run_label(model_name)}_{gate_run_label(model_name)}_{stamp}"
    )[:240]


def log_model_run(
    artifact_dir: Path,
    model_name: str,
    leaderboard: pd.DataFrame,
    metrics: pd.DataFrame,
    route_file: Path,
    comm_file: Path,
    route_sub_file: Path,
    comm_sub_file: Path,
    experimenter: str,
    selection_status: str,
) -> None:
    if mlflow is None:
        raise RuntimeError("mlflow is not installed")
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    safe = model_name.replace("__", "_").replace(" ", "_")[:120]
    best = leaderboard[leaderboard["model"] == model_name].iloc[0]
    model_backend = str(best.get("model_backend", "unknown"))
    run_name = mlflow_run_name(experimenter, model_name, model_backend, stamp)
    upload_dir = artifact_dir / "mlflow_all_upload" / stamp / safe
    source_dir = upload_dir / "source_code"
    source_dir.mkdir(parents=True, exist_ok=True)
    config_path = source_dir / "experiment_config.json"
    config = {
        "dataset_version": DATASET_VERSION,
        "model": model_name,
        "primary_metric": "weighted_RMSE",
        "selection_policy": "train-internal pseudo backtest first; validation limited review; test final confirmation",
        "splits": SPLIT_INFO,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    req_path = upload_dir / "requirements_freeze.txt"
    write_requirements(req_path)
    model_metrics = metrics[metrics["model"] == model_name]
    log_metrics: dict[str, float] = {}
    for _, row in model_metrics.iterrows():
        prefix = split_label(row["split"]) if row["part"] == "test" else f"val_{split_label(row['split'])}"
        for metric in METRIC_NAMES:
            log_metrics[f"{prefix}_{metric}"] = float(row[metric])
    for metric in METRIC_NAMES:
        log_metrics[f"weighted_{metric}"] = float(best[f"test_weighted_{metric}"])
        log_metrics[f"val_weighted_{metric}"] = float(best[f"val_weighted_{metric}"])
    baseline = leaderboard[leaderboard["model"].str.contains("__baseline_only__no_gate_baseline_only__a0.00", regex=False)]
    if not baseline.empty:
        base_rmse = float(baseline.iloc[0]["val_weighted_RMSE"])
        log_metrics["skill_vs_baseline"] = 1.0 - float(best["val_weighted_RMSE"]) / (base_rmse + 1e-9)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    description = f"3-year-ahead route-level tons forecast, commodity-level {model_backend} log-ratio residual correction"
    with mlflow.start_run(run_name=run_name, description=description) as run:
        mlflow.set_tags(
            {
                "dataset_version": DATASET_VERSION,
                "primary_metric": "weighted_RMSE",
                "description": description,
                "metric_schema": METRIC_SCHEMA,
                "selection_status": selection_status,
                "experiment_label": str(best.get("experiment_label", "high_impact_commodity")),
                "uses_external_prior": str(bool(best.get("uses_external_prior", False))),
                "external_strength": str(float(best.get("external_strength", 0.0))),
                "mlflow_upload_candidate": str(bool(best.get("mlflow_upload_candidate", selection_status == "selected_candidate"))),
                "model_name": model_name,
                "model_backend": model_backend,
                "algorithm": model_backend,
                "prediction_target": "3-year-ahead route-level tons",
                "top20_route_selection": "train-internal impact score only",
                "validation_overfit_defense": "train-internal candidates first; validation limited review; test confirmation only",
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
                "model_name": model_name,
                "model_backend": model_backend,
                "algorithm": model_backend,
                "random_seed": RANDOM_SEED,
                "feature_set_name": "commodity_recent_history_high_impact_route_context",
                "baseline_name": model_name.split("__")[0],
                "residual_target": "log_ratio",
                "granularity": "commodity_to_route",
                "uses_external_prior": bool(best.get("uses_external_prior", False)),
                "external_strength": float(best.get("external_strength", 0.0)),
            }
        )
        mlflow.log_metrics(log_metrics)
        mlflow.log_artifact(str(req_path))
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source_code")
        mlflow.log_artifact(str(config_path), artifact_path="source_code")
        for path in [
            route_file,
            comm_file,
            route_sub_file,
            comm_sub_file,
            artifact_dir / "leaderboard.csv",
            artifact_dir / "residual_model_backend_leaderboard.csv",
            artifact_dir / "selected_backend_by_commodity.csv",
            artifact_dir / "backend_vs_baseline_split_summary.csv",
            artifact_dir / "high_sse_commodity_error_report.csv",
            artifact_dir / "residual_backend_config.json",
            artifact_dir / "feature_columns.csv",
            artifact_dir / "external_prior_sensitivity_leaderboard.csv",
            artifact_dir / "external_prior_split_metrics.csv",
            artifact_dir / "external_prior_vs_no_external_summary.csv",
            artifact_dir / "external_prior_commodity_impact_summary.csv",
            artifact_dir / "external_prior_best_result_summary.csv",
            artifact_dir / "run_metadata.json",
        ]:
            if path.exists():
                mlflow.log_artifact(str(path))
        manifest = upload_dir / "upload_manifest.csv"
        pd.DataFrame([{"run_name": run_name, "run_id": run.info.run_id, "model": model_name, "weighted_RMSE": log_metrics["weighted_RMSE"]}]).to_csv(
            manifest, index=False, encoding="utf-8-sig"
        )
        print("MLflow run:", f"{TRACKING_URI}/#/experiments/2/runs/{run.info.run_id}")


def log_selected_and_sensitivity_runs(
    artifact_dir: Path,
    leaderboard: pd.DataFrame,
    metrics: pd.DataFrame,
    model_manifest: pd.DataFrame,
    experimenter: str,
) -> None:
    selected_rows = leaderboard[leaderboard["selected_status"] == "selected"].head(1)
    sensitivity_rows = leaderboard[leaderboard["selected_status"] == "sensitivity"].sort_values("test_weighted_RMSE").head(1)
    upload_rows = pd.concat([selected_rows, sensitivity_rows], ignore_index=True).drop_duplicates("model")
    rows = []
    for _, row in upload_rows.iterrows():
        model_name = str(row["model"])
        files = model_manifest[model_manifest["model"] == model_name].iloc[0]
        status = "selected_candidate" if row["selected_status"] == "selected" else "sensitivity_candidate"
        log_model_run(
            artifact_dir,
            model_name,
            leaderboard,
            metrics,
            Path(files["route_prediction_file"]),
            Path(files["commodity_prediction_file"]),
            Path(files["route_submission_file"]),
            Path(files["commodity_submission_file"]),
            experimenter,
            status,
        )
        rows.append({"model": model_name, "selection_status": status, "weighted_RMSE": float(row["test_weighted_RMSE"])})
    pd.DataFrame(rows).to_csv(artifact_dir / "mlflow_logged_models.csv", index=False, encoding="utf-8-sig")


def log_external_prior_sensitivity_runs(
    artifact_dir: Path,
    leaderboard: pd.DataFrame,
    metrics: pd.DataFrame,
    experimenter: str,
) -> None:
    upload_rows = leaderboard[leaderboard["mlflow_upload_candidate"].fillna(False)].copy()
    if upload_rows.empty:
        best_val = leaderboard[leaderboard["uses_external_prior"]].sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).head(1)
        best_test = leaderboard[leaderboard["uses_external_prior"]].sort_values(["test_weighted_RMSE", "val_weighted_RMSE"]).head(1)
        upload_rows = pd.concat([best_val, best_test], ignore_index=True).drop_duplicates("model")
    else:
        upload_rows = upload_rows.sort_values(["test_weighted_RMSE", "val_weighted_RMSE"]).head(2)

    rows = []
    missing_artifact = artifact_dir / "__external_prior_summary_only__.csv"
    for _, row in upload_rows.iterrows():
        model_name = str(row["model"])
        status = "external_data_sensitivity_validation_safe_candidate" if bool(row.get("mlflow_upload_candidate", False)) else "external_data_sensitivity"
        log_model_run(
            artifact_dir,
            model_name,
            leaderboard,
            metrics,
            missing_artifact,
            missing_artifact,
            missing_artifact,
            missing_artifact,
            experimenter,
            status,
        )
        rows.append(
            {
                "model": model_name,
                "selection_status": status,
                "val_weighted_RMSE": float(row["val_weighted_RMSE"]),
                "test_weighted_RMSE": float(row["test_weighted_RMSE"]),
                "mlflow_upload_candidate": bool(row.get("mlflow_upload_candidate", False)),
            }
        )
    pd.DataFrame(rows).to_csv(artifact_dir / "mlflow_logged_models.csv", index=False, encoding="utf-8-sig")


def run_external_prior_sensitivity(args: argparse.Namespace) -> None:
    set_seed()
    data_root = discover_data_root(args.data_root)
    stacking_experiment.GLOBAL_DATA_ROOT = data_root
    artifact_dir = repo_root() / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    specs = baseline_specs("external_prior_sensitivity")
    candidates = build_candidates_for_profile("external_prior_sensitivity", specs)

    all_metrics = []
    for baseline_name, spec in specs.items():
        defs = [c for c in candidates if c.baseline == baseline_name]
        metrics, _route, _comm, _scores, _selection, _gate, _commodity_eval, _pseudo_backend = build_candidate_outputs(
            data_root,
            spec,
            defs,
            args.max_pseudo_years,
            collect_comm=False,
            collect_route=False,
            collect_pseudo_backend=False,
        )
        all_metrics.append(metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)

    reference_selected_model = "commodity_median3__lgbm__repeated_error__soft__a0.15__clip0.10__capnone"
    leaderboard = summarize_external_prior_leaderboard(metrics, candidates, specs, reference_selected_model)
    external_candidate_rows = leaderboard[(leaderboard["uses_external_prior"]) & (leaderboard["selected_status"] == "candidate")]
    sensitivity_rows = leaderboard[(leaderboard["uses_external_prior"]) & (leaderboard["selected_status"] == "sensitivity")]
    selected_model = (
        str(external_candidate_rows.iloc[0]["model"])
        if not external_candidate_rows.empty
        else str(leaderboard[leaderboard["uses_external_prior"]].sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).iloc[0]["model"])
        if not leaderboard[leaderboard["uses_external_prior"]].empty
        else reference_selected_model
    )
    sensitivity_model = (
        str(sensitivity_rows.sort_values(["test_weighted_RMSE", "val_weighted_RMSE"]).iloc[0]["model"])
        if not sensitivity_rows.empty
        else selected_model
    )
    selected_model_for_artifacts = selected_model if selected_model else sensitivity_model

    detailed_candidates = [c for c in candidates if c.name in {selected_model, sensitivity_model}]
    detailed_comm = []
    for baseline_name, spec in specs.items():
        defs = [c for c in detailed_candidates if c.baseline == baseline_name]
        if not defs:
            continue
        _m, _r, c, _s, _sel, _g, _ce, _pb = build_candidate_outputs(
            data_root,
            spec,
            defs,
            args.max_pseudo_years,
            collect_comm=True,
            collect_route=False,
            collect_pseudo_backend=False,
        )
        detailed_comm.append(c)
    comm_predictions = pd.concat(detailed_comm, ignore_index=True) if detailed_comm else pd.DataFrame()

    selected_commodity_summary = external_prior_commodity_impact_summary(comm_predictions, selected_model_for_artifacts)
    external_summary = external_prior_vs_no_external_summary(leaderboard)
    best_external_val = leaderboard[leaderboard["uses_external_prior"]].sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).head(1)
    best_external_test = leaderboard[leaderboard["uses_external_prior"]].sort_values(["test_weighted_RMSE", "val_weighted_RMSE"]).head(1)
    safe_candidate_exists = bool((leaderboard["selected_status"] == "candidate").any())
    stable_strength_summary = (
        leaderboard[leaderboard["uses_external_prior"]]
        .groupby("external_strength", as_index=False)
        .agg(
            candidates=("model", "count"),
            mean_val_delta=("val_delta_vs_reference_selected", "mean"),
            mean_test_delta=("test_delta_vs_reference_selected", "mean"),
            mean_abs_val_delta=("val_delta_vs_reference_selected", lambda s: float(np.abs(s).mean())),
            best_val_weighted_RMSE=("val_weighted_RMSE", "min"),
            best_test_weighted_RMSE=("test_weighted_RMSE", "min"),
        )
        .sort_values(["mean_abs_val_delta", "best_test_weighted_RMSE"])
    )

    leaderboard.to_csv(artifact_dir / "external_prior_sensitivity_leaderboard.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(artifact_dir / "external_prior_split_metrics.csv", index=False, encoding="utf-8-sig")
    external_summary.to_csv(artifact_dir / "external_prior_vs_no_external_summary.csv", index=False, encoding="utf-8-sig")
    selected_commodity_summary.to_csv(artifact_dir / "external_prior_commodity_impact_summary.csv", index=False, encoding="utf-8-sig")
    residual_config = {
        "experiment_profile": "external_prior_sensitivity",
        "experiment_label": "external_data_sensitivity",
        "reference_selected_model": reference_selected_model,
        "uses_external_prior_columns": ["uses_external_prior", "external_strength"],
        "mlflow_upload_rule": "Only rows with selected_status == 'candidate' and mlflow_upload_candidate == true are validation-safe MLflow upload candidates.",
        "rule_candidates_enabled": False,
        "high_sse_commodities": HIGH_SSE_COMMODITIES,
    }
    (artifact_dir / "residual_backend_config.json").write_text(json.dumps(residual_config, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "data_root": str(data_root),
        "artifact_dir": str(artifact_dir),
        "experiment_profile": "external_prior_sensitivity",
        "experiment_label": "external_data_sensitivity",
        "reference_selected_model": reference_selected_model,
        "selected_model": selected_model,
        "sensitivity_model": sensitivity_model,
        "max_pseudo_years": args.max_pseudo_years,
        "mlflow_upload_candidate_count": int(leaderboard["mlflow_upload_candidate"].sum()),
        "best_external_prior_val_model": None if best_external_val.empty else str(best_external_val.iloc[0]["model"]),
        "best_external_prior_test_model": None if best_external_test.empty else str(best_external_test.iloc[0]["model"]),
    }
    (artifact_dir / "run_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    best_selected_perf = leaderboard[leaderboard["model"] == reference_selected_model].iloc[0]
    print("External prior (no prior) reference performance:")
    print(best_selected_perf[["model", "val_weighted_RMSE", "test_weighted_RMSE"]].to_string())
    print("Best external prior validation performance:")
    print(best_external_val.to_string(index=False) if not best_external_val.empty else "No external prior candidates")
    print("Best external prior test performance:")
    print(best_external_test.to_string(index=False) if not best_external_test.empty else "No external prior candidates")
    print("Validation-safe improvement over existing selected exists:", safe_candidate_exists)
    print("MLflow upload candidate count:", int(leaderboard["mlflow_upload_candidate"].sum()))
    print("Most stable external_strength summary:")
    print(stable_strength_summary.to_string(index=False) if not stable_strength_summary.empty else "No external prior summary")

    summary_rows = []
    if not best_external_val.empty:
        summary_rows.append(best_external_val.iloc[0].to_dict())
    if not best_external_test.empty and (best_external_test.iloc[0]["model"] != (best_external_val.iloc[0]["model"] if not best_external_val.empty else None)):
        summary_rows.append(best_external_test.iloc[0].to_dict())
    pd.DataFrame(summary_rows).to_csv(artifact_dir / "external_prior_best_result_summary.csv", index=False, encoding="utf-8-sig")

    if args.log_mlflow:
        log_external_prior_sensitivity_runs(artifact_dir, leaderboard, metrics, args.experimenter)

    print("External-data sensitivity artifact model:", selected_model_for_artifacts)
    print("Artifact dir:", artifact_dir)


def run_external_feature_sensitivity(args: argparse.Namespace) -> None:
    set_seed()
    data_root = discover_data_root(args.data_root)
    stacking_experiment.GLOBAL_DATA_ROOT = data_root
    artifact_dir = repo_root() / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    reference_selected_model = "commodity_median3__lgbm__repeated_error__soft__a0.15__clip0.10__capnone"

    tiny_specs = tiny_external_prior_specs()
    base_specs = baseline_specs()
    tiny_candidates = build_external_candidates(tiny_specs, "tiny_prior", "none")
    feature_candidates = build_external_candidates(base_specs, "feature_only", "all")
    group_candidates = build_external_candidates(base_specs, "feature_only", "group_specific")

    def run_group(specs: dict[str, BaselineSpec], candidates: list[Candidate], external_feature_set: str) -> pd.DataFrame:
        metric_frames = []
        for baseline_name, spec in specs.items():
            defs = [c for c in candidates if c.baseline == baseline_name]
            metrics, _route, _comm, _scores, _selection, _gate, _commodity_eval, _pseudo_backend = build_candidate_outputs(
                data_root,
                spec,
                defs,
                args.max_pseudo_years,
                collect_comm=False,
                collect_route=False,
                collect_pseudo_backend=False,
                external_feature_set=external_feature_set,
            )
            metric_frames.append(metrics)
        return pd.concat(metric_frames, ignore_index=True)

    tiny_metrics = run_group(tiny_specs, tiny_candidates, "none")
    reference_row = summarize_against_reference(tiny_metrics, tiny_candidates, tiny_specs, reference_selected_model)
    ref_values = reference_row[reference_row["model"] == reference_selected_model].iloc[0]
    reference_values = (float(ref_values["val_weighted_RMSE"]), float(ref_values["test_weighted_RMSE"]))

    feature_metrics = run_group(base_specs, feature_candidates, "all")
    group_metrics = run_group(base_specs, group_candidates, "group_specific")

    tiny_leaderboard = summarize_against_reference(tiny_metrics, tiny_candidates, tiny_specs, reference_selected_model, reference_values)
    feature_leaderboard = summarize_against_reference(feature_metrics, feature_candidates, base_specs, reference_selected_model, reference_values)
    group_leaderboard = summarize_against_reference(group_metrics, group_candidates, base_specs, reference_selected_model, reference_values)

    all_metrics = pd.concat([tiny_metrics, feature_metrics, group_metrics], ignore_index=True)
    all_candidates = tiny_candidates + feature_candidates + group_candidates
    split_delta = split_delta_vs_reference(all_metrics, all_candidates, reference_selected_model)

    combined = pd.concat([tiny_leaderboard, feature_leaderboard, group_leaderboard], ignore_index=True)
    feature_ablation_summary = (
        combined.groupby(["external_mode", "external_feature_set", "uses_external_prior", "uses_external_features"], dropna=False)
        .agg(
            candidates=("model", "count"),
            selected_candidates=("selected_status", lambda s: int((s == "selected_candidate").sum())),
            sensitivity_candidates=("selected_status", lambda s: int((s == "sensitivity_candidate").sum())),
            best_val_weighted_RMSE=("val_weighted_RMSE", "min"),
            best_test_weighted_RMSE=("test_weighted_RMSE", "min"),
            mean_val_delta_vs_reference=("val_delta_vs_reference_selected", "mean"),
            mean_test_delta_vs_reference=("test_delta_vs_reference_selected", "mean"),
        )
        .reset_index()
        .sort_values(["best_val_weighted_RMSE", "best_test_weighted_RMSE"])
    )

    tiny_leaderboard.to_csv(artifact_dir / "external_tiny_prior_leaderboard.csv", index=False, encoding="utf-8-sig")
    feature_leaderboard.to_csv(artifact_dir / "external_feature_only_leaderboard.csv", index=False, encoding="utf-8-sig")
    group_leaderboard.to_csv(artifact_dir / "external_group_specific_feature_leaderboard.csv", index=False, encoding="utf-8-sig")
    split_delta.to_csv(artifact_dir / "external_split_delta_vs_selected.csv", index=False, encoding="utf-8-sig")
    feature_ablation_summary.to_csv(artifact_dir / "external_feature_ablation_summary.csv", index=False, encoding="utf-8-sig")

    run_metadata = {
        "experiment_profile": "external_feature_sensitivity",
        "reference_selected_model": reference_selected_model,
        "reference_selected_val_weighted_RMSE": reference_values[0],
        "reference_selected_test_weighted_RMSE": reference_values[1],
        "tiny_external_strengths": [0.02, 0.05, 0.10, 0.15, 0.25],
        "external_feature_columns": EXTERNAL_FEATURE_COLUMNS,
        "group_specific_external_features": GROUP_SPECIFIC_EXTERNAL_FEATURES,
    }
    (artifact_dir / "external_feature_sensitivity_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    tiny_external = tiny_leaderboard[tiny_leaderboard["uses_external_prior"]].copy()
    stable_strength = (
        tiny_external.groupby("external_strength", as_index=False)
        .agg(
            candidates=("model", "count"),
            mean_abs_val_delta=("val_delta_vs_reference_selected", lambda s: float(np.abs(s).mean())),
            mean_test_delta=("test_delta_vs_reference_selected", "mean"),
            best_val_weighted_RMSE=("val_weighted_RMSE", "min"),
            best_test_weighted_RMSE=("test_weighted_RMSE", "min"),
        )
        .sort_values(["mean_abs_val_delta", "best_test_weighted_RMSE"])
    )
    best_tiny = tiny_external.sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).head(1)
    best_feature = feature_leaderboard.sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).head(1)
    best_group = group_leaderboard.sort_values(["val_weighted_RMSE", "test_weighted_RMSE"]).head(1)
    split1_summary = (
        split_delta[split_delta["split"] == "split_1"]
        .groupby(["external_mode"], as_index=False)
        .agg(mean_delta_vs_selected=("delta_vs_selected", "mean"), best_delta_vs_selected=("delta_vs_selected", "min"))
        .sort_values("best_delta_vs_selected")
    )
    selected_exists = bool((combined["selected_status"] == "selected_candidate").any())

    print("Tiny prior most stable external_strength:")
    print(stable_strength.head(1).to_string(index=False) if not stable_strength.empty else "No tiny prior candidates")
    print("Best tiny prior candidate:")
    print(best_tiny[["model", "external_strength", "val_weighted_RMSE", "test_weighted_RMSE", "selected_status"]].to_string(index=False))
    print("Best external-as-feature candidate:")
    print(best_feature[["model", "external_feature_set", "val_weighted_RMSE", "test_weighted_RMSE", "selected_status"]].to_string(index=False))
    print("Best group-specific external feature candidate:")
    print(best_group[["model", "external_feature_set", "val_weighted_RMSE", "test_weighted_RMSE", "selected_status"]].to_string(index=False))
    print("External-as-feature better than tiny prior on best validation:", float(best_feature.iloc[0]["val_weighted_RMSE"]) < float(best_tiny.iloc[0]["val_weighted_RMSE"]))
    print("Group-specific feature better than all external feature on best validation:", float(best_group.iloc[0]["val_weighted_RMSE"]) < float(best_feature.iloc[0]["val_weighted_RMSE"]))
    print("Split1 delta summary vs reference selected:")
    print(split1_summary.to_string(index=False))
    print("Validation-safe candidate beating reference selected exists:", selected_exists)
    print("Artifact dir:", artifact_dir)


def main() -> None:
    set_seed()
    args = parse_args()
    if args.experiment_profile == "external_feature_sensitivity":
        run_external_feature_sensitivity(args)
        return
    if args.experiment_profile == "external_prior_sensitivity":
        run_external_prior_sensitivity(args)
        return
    data_root = discover_data_root(args.data_root)
    stacking_experiment.GLOBAL_DATA_ROOT = data_root
    artifact_dir = repo_root() / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    specs = baseline_specs()
    candidates: list[Candidate] = []
    for baseline in specs:
        candidates.append(
            Candidate(
                f"{baseline}__baseline_only__no_gate_baseline_only__a0.00",
                baseline,
                "baseline_only",
                "no_gate_baseline_only",
                "soft",
                0.0,
                None,
                None,
            )
        )
        for backend in ["ridge", "huber", "lgbm", "hist_gb"]:
            for alpha in [0.05, 0.10, 0.15]:
                for clip_value in [0.05, 0.10]:
                    candidates.append(
                        Candidate(
                            f"{baseline}__{backend}__repeated_error__soft__a{alpha:.2f}__clip{clip_value:.2f}__capnone",
                            baseline,
                            backend,
                            "repeated_error",
                            "soft",
                            alpha,
                            clip_value,
                            None,
                        )
                    )

    all_metrics = []
    all_route = []
    all_scores = []
    all_selection = []
    all_gate = []
    all_commodity_eval = []
    all_pseudo_backend = []
    for baseline_name, spec in specs.items():
        defs = [c for c in candidates if c.baseline == baseline_name]
        metrics, route, _comm, scores, selection, gate, commodity_eval, pseudo_backend = build_candidate_outputs(
            data_root, spec, defs, args.max_pseudo_years, collect_comm=False
        )
        all_metrics.append(metrics)
        all_route.append(route)
        all_scores.append(scores)
        all_selection.append(selection)
        all_gate.append(gate)
        all_commodity_eval.append(commodity_eval)
        all_pseudo_backend.append(pseudo_backend)

    metrics = pd.concat(all_metrics, ignore_index=True)
    route_predictions = pd.concat(all_route, ignore_index=True)
    high_scores = pd.concat(all_scores, ignore_index=True)
    selection_report = pd.concat(all_selection, ignore_index=True)
    gate_summary = pd.concat(all_gate, ignore_index=True)
    commodity_eval = pd.concat(all_commodity_eval, ignore_index=True)
    pseudo_backend_report = pd.concat(all_pseudo_backend, ignore_index=True)
    selected_backend_by_commodity = commodity_backend_selection(pseudo_backend_report, commodity_eval)
    backend_summary = backend_vs_baseline_split_summary(leaderboard := summarize_leaderboard(metrics, candidates, specs))
    high_sse_report = high_sse_commodity_error_report(selected_backend_by_commodity, commodity_eval)
    selected_model = str(leaderboard[leaderboard["selected_status"] == "selected"].iloc[0]["model"])
    sensitivity = leaderboard[leaderboard["selected_status"] == "sensitivity"].sort_values("test_weighted_RMSE")
    sensitivity_model = str(sensitivity.iloc[0]["model"]) if not sensitivity.empty else selected_model
    baseline_model = "commodity_medmean_a0.20__baseline_only__no_gate_baseline_only__a0.00"
    if baseline_model not in set(route_predictions["model"]):
        baseline_model = "commodity_median3__baseline_only__no_gate_baseline_only__a0.00"

    detailed_candidates = [c for c in candidates if c.name in {selected_model, sensitivity_model}]
    detailed_comm = []
    detailed_route = []
    for baseline_name, spec in specs.items():
        defs = [c for c in detailed_candidates if c.baseline == baseline_name]
        if not defs:
            continue
        _m, r, c, _s, _sel, _g, _ce, _pb = build_candidate_outputs(data_root, spec, defs, args.max_pseudo_years, collect_comm=True)
        detailed_route.append(r)
        detailed_comm.append(c)
    detail_route_predictions = pd.concat(detailed_route, ignore_index=True) if detailed_route else pd.DataFrame()
    comm_predictions = pd.concat(detailed_comm, ignore_index=True) if detailed_comm else pd.DataFrame()

    route_predictions["baseline_pred"] = route_predictions["final_pred"]
    base_lookup = route_predictions[route_predictions["model"] == baseline_model][["split", "part"] + KEY_ROUTE + ["year", "final_pred"]].rename(
        columns={"final_pred": "baseline_pred_ref"}
    )
    route_predictions = route_predictions.merge(base_lookup, on=["split", "part"] + KEY_ROUTE + ["year"], how="left")
    route_predictions["baseline_pred"] = route_predictions["baseline_pred_ref"].fillna(route_predictions["baseline_pred"])
    route_predictions["commodity_stack_pred"] = route_predictions["final_pred"]
    route_predictions = route_predictions.drop(columns=["baseline_pred_ref"])

    if not detail_route_predictions.empty:
        detail_route_predictions["baseline_pred"] = detail_route_predictions["final_pred"]
        detail_route_predictions = detail_route_predictions.merge(base_lookup, on=["split", "part"] + KEY_ROUTE + ["year"], how="left")
        detail_route_predictions["baseline_pred"] = detail_route_predictions["baseline_pred_ref"].fillna(detail_route_predictions["baseline_pred"])
        detail_route_predictions["commodity_stack_pred"] = detail_route_predictions["final_pred"]
        detail_route_predictions = detail_route_predictions.drop(columns=["baseline_pred_ref"])

    delta = route_error_delta(route_predictions, baseline_model, selected_model)
    full_detail = pd.concat([load_detail(data_root, split, include_context=True) for split in SPLIT_WEIGHTS], ignore_index=True)
    cargo = cargo_value_proxy(full_detail, delta)
    commodity_summary = commodity_impact_summary(comm_predictions, selected_model)
    top20_detail = delta[delta["top20_flag"].fillna(False)].copy()
    top20_comm = top20_driver(comm_predictions, high_scores, selected_model)
    cap_summary = top20_cap_effect_summary(leaderboard)
    composition_summary = composition_shift_versions_summary(gate_summary)

    leaderboard.to_csv(artifact_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    leaderboard.to_csv(artifact_dir / "leaderboard_tiny_alpha_clipped.csv", index=False, encoding="utf-8-sig")
    leaderboard.to_csv(artifact_dir / "residual_model_backend_leaderboard.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(artifact_dir / "split_metrics.csv", index=False, encoding="utf-8-sig")
    route_predictions.to_csv(artifact_dir / "route_predictions_all_candidates.csv", index=False, encoding="utf-8-sig")
    comm_predictions.to_csv(artifact_dir / "commodity_predictions_selected_and_sensitivity.csv", index=False, encoding="utf-8-sig")
    comm_predictions.to_csv(artifact_dir / "commodity_predictions_all_candidates.csv", index=False, encoding="utf-8-sig")
    high_scores.to_csv(artifact_dir / "high_impact_route_summary.csv", index=False, encoding="utf-8-sig")
    selection_report.to_csv(artifact_dir / "commodity_model_selection_report.csv", index=False, encoding="utf-8-sig")
    pseudo_backend_report.to_csv(artifact_dir / "commodity_backend_selection_report.csv", index=False, encoding="utf-8-sig")
    selected_backend_by_commodity.to_csv(artifact_dir / "selected_backend_by_commodity.csv", index=False, encoding="utf-8-sig")
    backend_summary.to_csv(artifact_dir / "backend_vs_baseline_split_summary.csv", index=False, encoding="utf-8-sig")
    high_sse_report.to_csv(artifact_dir / "high_sse_commodity_error_report.csv", index=False, encoding="utf-8-sig")
    gate_summary.to_csv(artifact_dir / "gate_coverage_summary.csv", index=False, encoding="utf-8-sig")
    gate_summary.to_csv(artifact_dir / "gate_coverage_summary_v2.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(artifact_dir / "route_error_delta.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(artifact_dir / "route_error_delta_tiny_alpha.csv", index=False, encoding="utf-8-sig")
    cargo.to_csv(artifact_dir / "cargo_value_error_proxy.csv", index=False, encoding="utf-8-sig")
    commodity_summary.to_csv(artifact_dir / "commodity_impact_summary.csv", index=False, encoding="utf-8-sig")
    high_scores[high_scores["top20_high_impact_route"]].to_csv(artifact_dir / "top20_route_selection.csv", index=False, encoding="utf-8-sig")
    top20_detail.to_csv(artifact_dir / "top20_route_error_delta.csv", index=False, encoding="utf-8-sig")
    top20_comm.to_csv(artifact_dir / "top20_route_commodity_driver.csv", index=False, encoding="utf-8-sig")
    cap_summary.to_csv(artifact_dir / "top20_cap_effect_summary.csv", index=False, encoding="utf-8-sig")
    composition_summary.to_csv(artifact_dir / "composition_shift_versions_summary.csv", index=False, encoding="utf-8-sig")
    selection_report[["baseline", "split", "commodity", "commodity_group", "selected_level"]].drop_duplicates().to_csv(
        artifact_dir / "feature_columns.csv", index=False, encoding="utf-8-sig"
    )
    residual_config = {
        "residual_model_backends": RESIDUAL_BACKENDS,
        "rule_residual_candidates": RULE_RESIDUAL_CANDIDATES,
        "rule_candidates_enabled": False,
        "high_sse_commodities": HIGH_SSE_COMMODITIES,
        "high_sse_rule_candidate_note": "Use these commodities first when rule-based residual candidates are enabled in the second-stage experiment.",
    }
    (artifact_dir / "residual_backend_config.json").write_text(json.dumps(residual_config, ensure_ascii=False, indent=2), encoding="utf-8")
    model_manifest = save_all_model_artifacts(artifact_dir, detail_route_predictions, comm_predictions)
    selected_files = model_manifest[model_manifest["model"] == selected_model].iloc[0]
    route_file = Path(selected_files["route_prediction_file"])
    comm_file = Path(selected_files["commodity_prediction_file"])
    route_sub_file = Path(selected_files["route_submission_file"])
    comm_sub_file = Path(selected_files["commodity_submission_file"])

    report = {
        "data_root": str(data_root),
        "artifact_dir": str(artifact_dir),
        "selected_model": selected_model,
        "sensitivity_model": sensitivity_model,
        "baseline_model": baseline_model,
        "max_pseudo_years": args.max_pseudo_years,
        "selection_policy": "train_internal_high_impact_screen_then_validation_limited_review",
        "best": leaderboard.iloc[0].to_dict(),
    }
    (artifact_dir / "run_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Artifact dir:", artifact_dir)
    baseline_row = leaderboard[leaderboard["model"] == baseline_model].iloc[0]
    selected_row = leaderboard[leaderboard["model"] == selected_model].iloc[0]
    sensitivity_row = leaderboard[leaderboard["model"] == sensitivity_model].iloc[0]
    no_val_worse_test_improved = bool(
        ((leaderboard["val_weighted_RMSE"] <= float(baseline_row["val_weighted_RMSE"]) * 1.005)
        & (leaderboard["test_weighted_RMSE"] < float(baseline_row["test_weighted_RMSE"]))).any()
    )
    print("Baseline selected performance:")
    print(baseline_row[["model", "val_weighted_RMSE", "test_weighted_RMSE"]].to_string())
    print("Selected candidate performance:")
    print(selected_row[["model", "val_weighted_RMSE", "test_weighted_RMSE", "selected_status"]].to_string())
    print("Best sensitivity candidate performance:")
    print(sensitivity_row[["model", "val_weighted_RMSE", "test_weighted_RMSE", "selected_status"]].to_string())
    print("Validation-safe test improvement exists:", no_val_worse_test_improved)
    print("Backend best validation/test performance:")
    print(backend_summary.to_string(index=False) if not backend_summary.empty else "No backend summary")
    if not backend_summary.empty and "lgbm" in set(backend_summary["model_backend"]):
        lgbm_best = float(backend_summary[backend_summary["model_backend"] == "lgbm"]["best_val_weighted_RMSE"].min())
        ridge_best = float(backend_summary[backend_summary["model_backend"] == "ridge"]["best_val_weighted_RMSE"].min()) if "ridge" in set(backend_summary["model_backend"]) else math.inf
        huber_best = float(backend_summary[backend_summary["model_backend"] == "huber"]["best_val_weighted_RMSE"].min()) if "huber" in set(backend_summary["model_backend"]) else math.inf
        print("LGBM better than ridge/huber on validation:", lgbm_best < min(ridge_best, huber_best))
    print("Selected backend distribution by commodity:")
    print(
        selected_backend_by_commodity.groupby("selected_backend").size().sort_values(ascending=False).to_string()
        if not selected_backend_by_commodity.empty
        else "No commodity backend selector output"
    )
    print("High-SSE commodity selected backend distribution:")
    print(
        selected_backend_by_commodity[selected_backend_by_commodity["commodity"].isin(HIGH_SSE_COMMODITIES)]
        .groupby(["commodity", "selected_backend"])
        .size()
        .to_string()
        if not selected_backend_by_commodity.empty
        else "No high-SSE backend output"
    )
    print("Baseline-only commodities:")
    print(
        ", ".join(
            sorted(
                selected_backend_by_commodity.loc[
                    selected_backend_by_commodity["selected_backend"] == "baseline_only", "commodity"
                ].astype(str).unique()
            )
        )
        if not selected_backend_by_commodity.empty
        else "No selector output"
    )
    print("Top20 cap comparison:")
    print(cap_summary.head(10).to_string(index=False) if not cap_summary.empty else "No top20 cap candidates")
    print("Composition shift v2/v3/v4 coverage:")
    print(
        composition_summary.groupby("gate")[
            ["selected_route_count", "selected_tons_share_proxy", "selected_baseline_error_share", "selected_top20_overlap_count"]
        ].mean().round(4).to_string()
        if not composition_summary.empty
        else "No composition summary"
    )
    print("Selected model:", selected_model)
    print(leaderboard.head(10).to_string(index=False))
    print("Route prediction:", route_file)
    print("Commodity prediction:", comm_file)
    print("Route submission:", route_sub_file)
    print("Commodity submission:", comm_sub_file)
    if args.log_mlflow:
        log_selected_and_sensitivity_runs(artifact_dir, leaderboard, metrics, model_manifest, args.experimenter)


if __name__ == "__main__":
    main()
