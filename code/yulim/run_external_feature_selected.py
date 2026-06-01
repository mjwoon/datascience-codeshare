from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import commodity_stacking_experiment as stacking_experiment
from commodity_stacking_experiment import (
    BaselineSpec,
    DETAIL_COLS,
    KEY_COMM,
    KEY_ROUTE,
    RANDOM_SEED,
    SPLIT_WEIGHTS,
    add_actual_to_comm_features,
    build_comm_feature_frame,
    encode_categories,
    feature_columns,
    read_csv,
    set_seed,
)
from external_feature_selected_review import NEW_SELECTED_MODEL
from high_impact_commodity_experiment import (
    Candidate,
    EXTERNAL_FEATURE_COLUMNS,
    add_external_ratio_features,
    corrected_commodity_predictions,
    evaluate_route_prediction,
    fit_models,
    high_impact_routes,
    modeling_columns,
    predict_log_ratio,
    team_metric_dict,
    train_frames,
    weighted_metric_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root-direct",
        default="data/faf4_faf5_fixed_year_splits",
        help="Directory that directly contains split_1, split_2, split_3.",
    )
    parser.add_argument(
        "--external-data-root",
        default="data",
        help="Root directory that contains additional_dataset/ for external ratios.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts_external_feature_selected",
    )
    parser.add_argument("--max-pseudo-years", type=int, default=2)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def candidate_spec() -> tuple[BaselineSpec, Candidate]:
    spec = BaselineSpec("commodity_medmean_a0.20", "median_mean", median_mean_alpha=0.20)
    candidate = Candidate(
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
    )
    return spec, candidate


def split_dir(split_root_direct: Path, split: str) -> Path:
    path = split_root_direct / split
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_detail_direct(split_root_direct: Path, split: str, include_context: bool) -> pd.DataFrame:
    sp = split_dir(split_root_direct, split)
    train = read_csv(sp / "train.csv", usecols=DETAIL_COLS)
    frames = [train]
    if include_context:
        frames.append(read_csv(sp / "context.csv", usecols=DETAIL_COLS))
    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["year"].astype(int)
    return out


def load_target_direct(split_root_direct: Path, split: str, part: str, commodity: bool = False) -> pd.DataFrame:
    suffix = "_commodity" if commodity else ""
    path = split_dir(split_root_direct, split) / f"{part}{suffix}.csv"
    out = read_csv(path)
    out["year"] = out["year"].astype(int)
    return out


def target_year_direct(target: pd.DataFrame) -> int:
    years = sorted(target["year"].dropna().astype(int).unique().tolist())
    if len(years) != 1:
        raise ValueError(f"Expected one target year, got {years}")
    return int(years[0])


def commodity_selection_report_local(train_frame: pd.DataFrame) -> pd.DataFrame:
    mod = __import__("high_impact_commodity_experiment")
    return mod.commodity_selection_report(train_frame)


def run_model(
    split_root_direct: Path,
    external_data_root: Path,
    artifact_dir: Path,
    max_pseudo_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec, candidate = candidate_spec()
    metric_rows: list[dict[str, object]] = []
    route_rows: list[pd.DataFrame] = []
    comm_rows: list[pd.DataFrame] = []

    stacking_experiment.GLOBAL_DATA_ROOT = external_data_root
    stacking_experiment.EXTERNAL_RATIO_CACHE = None

    for split in SPLIT_WEIGHTS:
        train = load_detail_direct(split_root_direct, split, include_context=False)
        detail_full = load_detail_direct(split_root_direct, split, include_context=True)
        route_scores, _route_detail, _comm_detail = high_impact_routes(train, spec, max_pseudo_years)
        selection_train = train_frames(train, spec, max_pseudo_years, data_root=external_data_root, external_feature_set="all")
        selection = commodity_selection_report_local(selection_train)

        train_enc_base, _ = encode_categories(selection_train, selection_train.iloc[:1].copy())
        cols = modeling_columns(train_enc_base)
        comm_models, group_models = fit_models(train_enc_base, cols, selection, candidate.model_backend)

        score_cols = [
            "impact_union",
            "top20_high_impact_route",
            "repeated_error_score",
            "commodity_sensitivity_score",
            "composition_shift_score",
        ]
        for part in ["val", "test"]:
            target_route = load_target_direct(split_root_direct, split, part, commodity=False)
            target_comm = load_target_direct(split_root_direct, split, part, commodity=True)
            tgt_year = target_year_direct(target_route)
            history = detail_full[detail_full["year"] <= tgt_year - 3].copy()
            apply_features = build_comm_feature_frame(history, tgt_year, spec)
            apply_features = add_external_ratio_features(apply_features, external_data_root, "all")
            apply_features = add_actual_to_comm_features(apply_features, target_comm)
            train_enc, apply_enc = encode_categories(selection_train, apply_features)
            cols = modeling_columns(train_enc)
            pred_frame = predict_log_ratio(apply_enc, cols, selection, comm_models, group_models)
            comm = corrected_commodity_predictions(pred_frame, route_scores, candidate, part, split)
            route_pred = comm.groupby(KEY_ROUTE, as_index=False)["final_pred"].sum()
            route = target_route[KEY_ROUTE + ["year", "tons"]].merge(route_pred, on=KEY_ROUTE, how="left")
            route["final_pred"] = route["final_pred"].fillna(0.0).clip(lower=0)
            route["model"] = candidate.name
            route = route.merge(route_scores[KEY_ROUTE + score_cols], on=KEY_ROUTE, how="left")
            route["top20_flag"] = route["top20_high_impact_route"].fillna(False).astype(bool)
            route["impact_union"] = route["impact_union"].fillna(False).astype(bool)
            for col in ["repeated_error_score", "commodity_sensitivity_score", "composition_shift_score"]:
                route[col] = route[col].fillna(0.0)
            metric_rows.append(evaluate_route_prediction(route, split, part, candidate.name))
            route_rows.append(route.assign(split=split, part=part))

            comm_out = comm[
                KEY_COMM
                + [
                    "year",
                    "actual_tons",
                    "base_pred",
                    "ml_pred",
                    "final_pred",
                    "selected_level",
                    "commodity_group",
                    "model_backend",
                    "gate",
                    "gate_mode",
                    "effective_alpha",
                    "correction",
                ]
            ].copy()
            comm_out["model"] = candidate.name
            comm_rows.append(comm_out.assign(split=split, part=part))

    metrics = pd.DataFrame(metric_rows)
    route_predictions = pd.concat(route_rows, ignore_index=True)
    commodity_predictions = pd.concat(comm_rows, ignore_index=True)

    val_rows = metrics[metrics["part"] == "val"]
    test_rows = metrics[metrics["part"] == "test"]
    leaderboard = pd.DataFrame(
        [
            {
                "model": candidate.name,
                **weighted_metric_rows(val_rows, "val", "val"),
                **weighted_metric_rows(test_rows, "test", "test"),
                "baseline_name": candidate.baseline,
                "model_backend": candidate.model_backend,
                "gate": candidate.gate,
                "gate_mode": candidate.gate_mode,
                "alpha": candidate.alpha,
                "clip_value": candidate.clip_value,
                "top20_cap_ratio": candidate.top20_cap_ratio,
                "external_mode": candidate.external_mode,
                "uses_external_features": candidate.uses_external_features,
                "external_feature_set": candidate.external_feature_set,
                "uses_external_prior": False,
                "external_strength": 0.0,
            }
        ]
    )

    leaderboard.to_csv(artifact_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(artifact_dir / "split_metrics.csv", index=False, encoding="utf-8-sig")
    route_predictions.to_csv(artifact_dir / "route_predictions.csv", index=False, encoding="utf-8-sig")
    commodity_predictions.to_csv(artifact_dir / "commodity_predictions.csv", index=False, encoding="utf-8-sig")
    return leaderboard, metrics, route_predictions


def write_submission_package(
    artifact_dir: Path,
    route_predictions: pd.DataFrame,
    leaderboard: pd.DataFrame,
    split_root_direct: Path,
) -> None:
    safe_name = "commodity_medmean_a0.20_lgbm_extfeat_all_repeated_error_soft_a0.15_clip0.10_capnone"
    route_sub = (
        route_predictions[route_predictions["part"] == "test"][KEY_ROUTE + ["year", "final_pred"]]
        .rename(columns={"final_pred": "prediction"})
        .sort_values(["year", "origin", "destination"])
        .reset_index(drop=True)
    )
    route_sub["prediction"] = route_sub["prediction"].clip(lower=0.0)
    route_sub_path = artifact_dir / f"route_submission_{safe_name}.csv"
    route_sub.to_csv(route_sub_path, index=False, encoding="utf-8-sig")

    package_dir = artifact_dir / "scoring_web_hyu_life_package"
    upload_dir = package_dir / "upload"
    review_dir = package_dir / "review"
    metadata_dir = package_dir / "metadata"
    upload_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    primary_upload = upload_dir / "route_submission_external_feature_selected.csv"
    backup_upload = upload_dir / f"route_submission_{safe_name}.csv"
    shutil.copy2(route_sub_path, primary_upload)
    shutil.copy2(route_sub_path, backup_upload)
    shutil.copy2(artifact_dir / "leaderboard.csv", review_dir / "leaderboard.csv")
    shutil.copy2(artifact_dir / "split_metrics.csv", review_dir / "split_metrics.csv")

    manifest = {
        "split_root_direct": str(split_root_direct.resolve()),
        "primary_upload_file": str(primary_upload.resolve()),
        "backup_upload_file": str(backup_upload.resolve()),
        "rows": int(len(route_sub)),
        "years": sorted(route_sub["year"].astype(int).unique().tolist()),
        "columns": route_sub.columns.tolist(),
        "model": NEW_SELECTED_MODEL,
        "val_weighted_RMSE": float(leaderboard.iloc[0]["val_weighted_RMSE"]),
        "test_weighted_RMSE": float(leaderboard.iloc[0]["test_weighted_RMSE"]),
    }
    (metadata_dir / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = package_dir / "README_scoring_web_upload.md"
    readme.write_text(
        "\n".join(
            [
                "# Scoring Web Upload Package",
                "",
                "Upload site: https://scoring-web.hyu.life/",
                "",
                f"Model: `{NEW_SELECTED_MODEL}`",
                f"Primary upload file: `upload/{primary_upload.name}`",
                f"Backup upload file: `upload/{backup_upload.name}`",
                "",
                "Submission schema:",
                "- origin",
                "- destination",
                "- year",
                "- prediction",
                "",
                f"Rows: {len(route_sub)}",
                f"Years: {sorted(route_sub['year'].astype(int).unique().tolist())}",
                "",
                f"Validation weighted RMSE: {float(leaderboard.iloc[0]['val_weighted_RMSE']):.6f}",
                f"Test weighted RMSE: {float(leaderboard.iloc[0]['test_weighted_RMSE']):.6f}",
            ]
        ),
        encoding="utf-8",
    )


def write_run_metadata(
    artifact_dir: Path,
    split_root_direct: Path,
    external_data_root: Path,
    leaderboard: pd.DataFrame,
) -> None:
    metadata = {
        "split_root_direct": str(split_root_direct.resolve()),
        "external_data_root": str(external_data_root.resolve()),
        "model": NEW_SELECTED_MODEL,
        "random_seed": RANDOM_SEED,
        "external_feature_columns": EXTERNAL_FEATURE_COLUMNS,
        "val_weighted_RMSE": float(leaderboard.iloc[0]["val_weighted_RMSE"]),
        "test_weighted_RMSE": float(leaderboard.iloc[0]["test_weighted_RMSE"]),
        "note": "Rerun of best external-feature selected model on the repository split root.",
    }
    (artifact_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    set_seed()
    args = parse_args()
    root = repo_root()
    split_root_direct = (root / args.split_root_direct).resolve()
    external_data_root = (root / args.external_data_root).resolve()
    artifact_dir = (root / args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    leaderboard, metrics, route_predictions = run_model(split_root_direct, external_data_root, artifact_dir, args.max_pseudo_years)
    write_submission_package(artifact_dir, route_predictions, leaderboard, split_root_direct)
    write_run_metadata(artifact_dir, split_root_direct, external_data_root, leaderboard)

    print("Artifact dir:", artifact_dir)
    print("Leaderboard:")
    print(leaderboard.to_string(index=False))
    print("Validation/Test split metrics:")
    print(metrics.to_string(index=False))
    print("Route submission:", artifact_dir / "scoring_web_hyu_life_package" / "upload" / "route_submission_external_feature_selected.csv")


if __name__ == "__main__":
    main()
