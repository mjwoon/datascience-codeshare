"""
config.py — Global constants and split definitions for the integrated FAF pipeline.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "faf4_faf5_fixed_year_splits"
# actual data lives in additional_dataset (not additional_data)
EXTERNAL_DIR = REPO_ROOT / "data" / "additional_dataset"

SPLITS = {
    "split_1": {"context": 2019, "val": 2021, "test": 2022, "weight": 0.2},
    "split_2": {"context": 2020, "val": 2022, "test": 2023, "weight": 0.3},
    "split_3": {"context": 2021, "val": 2023, "test": 2024, "weight": 0.5},
}

RANDOM_SEED = 42
MIN_TARGET_YEAR = 2017   # applies to target year (not base year)
COVID_SKIP_YEARS = {2020}
CORRECTION_CLIP = 0.10
SELECTION_MARGIN = 0.15

KEY_COMM = ["origin", "destination", "commodity"]
KEY_ROUTE = ["origin", "destination"]

COMM_GROUPS = {
    "bulk": {"Gravel", "Nonmetal min. prods.", "Building stone", "Metallic ores", "Coal", "Waste/scrap", "Natural sands"},
    "fuel": {"Gasoline", "Fuel oils", "Natural gas/fossil", "Crude petroleum"},
    "ag_food": {
        "Cereal grains", "Other ag prods.", "Animal feed", "Meat/seafood",
        "Milled grain prods.", "Other foodstuffs", "Alcoholic beverages",
        "Tobacco prods.", "Live animals/fish",
    },
    "chemicals": {"Basic chemicals", "Chemical prods.", "Fertilizers", "Pharmaceuticals", "Plastics/rubber"},
    "manufactured": {
        "Articles base metal", "Base metals", "Machinery", "Electronics",
        "Motorized vehicles", "Transport equip.", "Precision instruments",
        "Furniture", "Textiles/leather", "Wood prods.", "Paper articles",
        "Printed prods.", "Misc. mfg. prods.",
    },
}
# any commodity not in above → "other"
COMMODITY_TO_GROUP = {comm: grp for grp, comms in COMM_GROUPS.items() for comm in comms}


def get_commodity_group(commodity: str) -> str:
    return COMMODITY_TO_GROUP.get(commodity, "other")
