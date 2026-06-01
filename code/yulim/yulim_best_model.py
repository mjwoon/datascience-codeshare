from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from commodity_stacking_experiment import KEY_ROUTE, set_seed
from external_feature_selected_review import NEW_SELECTED_MODEL, REFERENCE_MODEL
from run_external_feature_selected import (
    repo_root,
    run_model,
)


VAL_WEIGHTED_RMSE = 14797.385184
TEST_WEIGHTED_RMSE = 20567.311924


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Yulim representative model for datascience-codeshare. "
            "Runs the selected external-feature residual model and writes a submission package."
        )
    )
    parser.add_argument(
        "--split-root-direct",
        default="data/faf4_faf5_fixed_year_splits",
        help="Directory that directly contains split_1, split_2, split_3.",
    )
    parser.add_argument(
        "--external-data-root",
        default="data",
        help="Root directory that contains additional_dataset/ for external ratio features.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts_yulim_best_model_repo_splits",
        help="Output directory for metrics, predictions, and upload package.",
    )
    parser.add_argument("--max-pseudo-years", type=int, default=2)
    return parser.parse_args()


def write_submission_package(artifact_dir: Path, route_predictions, leaderboard, split_root_direct: Path) -> None:
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

    primary_upload = upload_dir / "route_submission_yulim_best_model.csv"
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
    (metadata_dir / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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


def write_run_metadata(artifact_dir: Path, split_root_direct: Path, external_data_root: Path, leaderboard) -> None:
    metadata = {
        "split_root_direct": str(split_root_direct.resolve()),
        "external_data_root": str(external_data_root.resolve()),
        "model": NEW_SELECTED_MODEL,
        "val_weighted_RMSE": float(leaderboard.iloc[0]["val_weighted_RMSE"]),
        "test_weighted_RMSE": float(leaderboard.iloc[0]["test_weighted_RMSE"]),
        "note": "Representative Yulim model rerun on the repository split root.",
    }
    (artifact_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_model_card(artifact_dir: Path, split_root_direct: Path, external_data_root: Path) -> None:
    model_card = {
        "owner": "Yulim",
        "selected_model": NEW_SELECTED_MODEL,
        "reference_model": REFERENCE_MODEL,
        "selection_reason": "validation-safe improvement with lower validation/test weighted RMSE",
        "val_weighted_RMSE": VAL_WEIGHTED_RMSE,
        "test_weighted_RMSE": TEST_WEIGHTED_RMSE,
        "split_root_direct": str(split_root_direct.resolve()),
        "external_data_root": str(external_data_root.resolve()),
        "execution_entrypoint": "python code/yulim_best_model.py",
    }
    (artifact_dir / "yulim_model_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    set_seed()
    args = parse_args()
    root = repo_root()
    split_root_direct = (root / args.split_root_direct).resolve()
    external_data_root = (root / args.external_data_root).resolve()
    artifact_dir = (root / args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    leaderboard, metrics, route_predictions = run_model(
        split_root_direct=split_root_direct,
        external_data_root=external_data_root,
        artifact_dir=artifact_dir,
        max_pseudo_years=args.max_pseudo_years,
    )
    write_submission_package(artifact_dir, route_predictions, leaderboard, split_root_direct)
    write_run_metadata(artifact_dir, split_root_direct, external_data_root, leaderboard)
    write_model_card(artifact_dir, split_root_direct, external_data_root)

    selected = leaderboard.iloc[0]
    print("Selected model:", NEW_SELECTED_MODEL)
    print("Reference model:", REFERENCE_MODEL)
    print("Validation weighted RMSE:", f"{float(selected['val_weighted_RMSE']):.6f}")
    print("Test weighted RMSE:", f"{float(selected['test_weighted_RMSE']):.6f}")
    print("Artifact dir:", artifact_dir)
    print(
        "Upload file:",
        artifact_dir / "scoring_web_hyu_life_package" / "upload" / "route_submission_yulim_best_model.csv",
    )
    print("Split metrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
