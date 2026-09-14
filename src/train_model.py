"""Train and serialize the production price prediction pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering import add_features

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "train-data.csv"
MODELS_DIR = BASE_DIR / "models"

NUMERIC_COLUMNS = [
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats",
    "vehicle_age",
    "mileage_per_year",
    "power_per_cc",
    "owner_rank",
    "age_mileage_interaction",
]
CATEGORICAL_COLUMNS = ["Location", "Fuel_Type", "Transmission"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def _number(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else float("nan")


def load_training_frame(path: Path = RAW_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path).iloc[:, :14].copy()
    frame.columns = [
        "record_id", "Name", "Location", "Year", "Kilometers_Driven",
        "Fuel_Type", "Transmission", "Owner_Type", "Mileage", "Engine",
        "Power", "Seats", "New_Price", "Price",
    ]
    for column in ["Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]:
        frame[column] = frame[column].map(_number)
    frame.loc[frame["Seats"] <= 0, "Seats"] = np.nan
    frame.loc[frame["Mileage"] <= 0, "Mileage"] = np.nan
    frame["Price"] = frame["Price"].map(_number)
    frame = frame.dropna(subset=["Price"])
    for column in ["Location", "Fuel_Type", "Transmission", "Owner_Type"]:
        frame[column] = frame[column].fillna("Unknown").astype(str)
    return add_features(frame)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            learning_rate=0.20312640661491188,
            max_depth=3,
            n_estimators=364,
            random_state=42,
        )),
    ])


def train_and_save() -> dict:
    frame = load_training_frame()
    pipeline = build_pipeline()
    pipeline.fit(frame[FEATURE_COLUMNS], np.log1p(frame["Price"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "final_pipeline.joblib")
    metadata = {
        "model_name": "GradientBoosting",
        "target_transform": "log1p",
        "training_rows": int(len(frame)),
        "feature_columns_numeric": NUMERIC_COLUMNS,
        "feature_columns_categorical": CATEGORICAL_COLUMNS,
        "hyperparameters": {
            "learning_rate": 0.20312640661491188,
            "max_depth": 3,
            "n_estimators": 364,
        },
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))