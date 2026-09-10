"""Download-independent preprocessing for the Kaggle used-car price dataset.

The script expects the Kaggle archive to have been downloaded into data/raw.
It writes model-ready CSV files and a machine-readable profile to data/processed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

DATASET_COLUMNS = [
    "record_id",
    "Name",
    "Location",
    "Year",
    "Kilometers_Driven",
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
    "Mileage",
    "Engine",
    "Power",
    "Seats",
    "New_Price",
    "Price",
]
TARGET = "Price"
NUMERIC_COLUMNS = [
    "Year",
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats",
    "New_Price",
]
CATEGORICAL_COLUMNS = ["Name", "Location", "Fuel_Type", "Transmission", "Owner_Type"]


def _numeric_value(value: object, column: str | None = None) -> float:
    """Extract the first numeric value from a value such as '1,582 CC'."""
    if pd.isna(value):
        return float("nan")
    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else float("nan")


def load_raw(path: Path, has_target: bool) -> pd.DataFrame:
    expected = DATASET_COLUMNS if has_target else DATASET_COLUMNS[:-1]
    frame = pd.read_csv(path)
    # The source's first column is an exported DataFrame index, not a predictor.
    frame = frame.iloc[:, : len(expected)].copy()
    frame.columns = expected
    return frame


def clean(frame: pd.DataFrame, has_target: bool) -> pd.DataFrame:
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in result:
            result[column] = result[column].map(
                lambda value: (
                    _numeric_value(value, column) * 100
                    if column == "New_Price"
                    and isinstance(value, str)
                    and "cr" in value.lower()
                    else _numeric_value(value, column)
                )
            )
    # Zero seats and zero mileage are invalid measurements in this dataset.
    result.loc[result["Seats"] <= 0, "Seats"] = float("nan")
    result.loc[result["Mileage"] <= 0, "Mileage"] = float("nan")
    if has_target:
        result[TARGET] = result[TARGET].map(_numeric_value)
        result = result.dropna(subset=[TARGET])
    for column in NUMERIC_COLUMNS:
        if column in result:
            result[column] = result[column].fillna(result[column].median())
    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].fillna("Unknown").astype("string")
    return result


def profile(raw: pd.DataFrame, cleaned: pd.DataFrame, has_target: bool) -> dict:
    duplicate_count = int(raw.duplicated().sum())
    missing = {
        str(column): int(value)
        for column, value in raw.isna().sum().items()
        if int(value) > 0
    }
    return {
        "raw_records": int(len(raw)),
        "cleaned_records": int(len(cleaned)),
        "raw_columns": int(len(raw.columns)),
        "predictor_count": int(len(raw.columns) - (2 if has_target else 1)),
        "target": TARGET if has_target else None,
        "missing_values_by_column": missing,
        "duplicate_rows": duplicate_count,
        "cleaning_actions": [
            "Dropped the exported source index column by renaming it record_id.",
            "Converted unit-bearing numeric strings to numbers.",
            "Treated zero Mileage and zero Seats as invalid and imputed them.",
            "Imputed numeric missing values with training-column medians.",
            "Filled missing categorical values with Unknown.",
        ],
    }


def run(raw_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_train = load_raw(raw_dir / "train-data.csv", has_target=True)
    raw_test = load_raw(raw_dir / "test-data.csv", has_target=False)
    train = clean(raw_train, has_target=True)
    test = clean(raw_test, has_target=False)
    train.to_csv(output_dir / "train-clean.csv", index=False)
    test.to_csv(output_dir / "test-clean.csv", index=False)
    report = profile(raw_train, train, has_target=True)
    report["test_records"] = int(len(raw_test))
    report["test_missing_values_by_column"] = {
        str(column): int(value)
        for column, value in raw_test.isna().sum().items()
        if int(value) > 0
    }
    (output_dir / "data_profile.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    report = run(args.raw_dir, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
