"""Feature engineering for the used-car price prediction model.

This module implements the project's 5 core, genuine feature engineering
techniques (as distinct from preprocessing/cleaning, which lives in
src/data_preprocessing.py). Every function here is a pure, deterministic
row-wise transformation with NO fitted statistics (no means/medians/encoders
learned from data), so these can be safely applied identically to training
data, validation data, and single records at FastAPI inference time without
any risk of data leakage.

Input contract: functions in this module expect a dataframe that has ALREADY
been through src.data_preprocessing.clean() — i.e. Year, Kilometers_Driven,
Engine, Power, Seats are numeric, and Owner_Type/Fuel_Type/Transmission are
clean strings with no missing values.
"""

from __future__ import annotations

import pandas as pd

# Fixed reference year for age calculations. This MUST be a constant, not
# datetime.now(), so that a model trained today produces the same vehicle_age
# for a given Year at inference time next month as it did during training.
# Update this constant and retrain if the project extends well past 2026.
REFERENCE_YEAR = 2026

# Ownership is genuinely ordinal (each additional owner generally correlates
# with more wear/depreciation), so we preserve that order deliberately rather
# than one-hot encoding it as an unordered category.
OWNER_TYPE_ORDER = {
    "First": 1,
    "Second": 2,
    "Third": 3,
    "Fourth & Above": 4,
}


def add_vehicle_age(frame: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    """Technique 1: Vehicle age.

    vehicle_age = reference_year - Year

    Reason: raw Year is just a label; price depreciation relates to how OLD
    the car is, not which calendar year it happens to be. Expected
    relationship with Price: strong negative (older -> cheaper).
    """
    result = frame.copy()
    result["vehicle_age"] = reference_year - result["Year"]
    # Guard against future-dated Year values producing a negative age.
    result["vehicle_age"] = result["vehicle_age"].clip(lower=0)
    return result


def add_mileage_per_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Technique 2: Mileage per year (usage intensity).

    mileage_per_year = Kilometers_Driven / max(vehicle_age, 1)

    Reason: raw Kilometers_Driven conflates age and usage intensity. This
    ratio isolates how heavily the car was actually driven, independent of
    how old it is. Expected relationship: negative (higher usage intensity
    -> lower price, holding age constant).

    Requires vehicle_age to already exist (call add_vehicle_age first).
    """
    if "vehicle_age" not in frame.columns:
        raise ValueError("add_vehicle_age must be applied before add_mileage_per_year")
    result = frame.copy()
    safe_age = result["vehicle_age"].clip(lower=1)  # avoid divide-by-zero for age=0
    result["mileage_per_year"] = result["Kilometers_Driven"] / safe_age
    return result


def add_power_per_cc(frame: pd.DataFrame) -> pd.DataFrame:
    """Technique 3: Power-to-engine ratio (performance density).

    power_per_cc = Power (bhp) / Engine (CC)

    Reason: Power and Engine size are correlated but represent different
    things; this ratio approximates performance tuning/efficiency (e.g. a
    turbocharged small engine vs. a large naturally-aspirated one), a real
    factor in how buyers value a car beyond raw displacement. Expected
    relationship: positive (higher performance density -> higher price).

    Assumes Power/Engine have already been cleaned to numeric, non-zero
    values by src.data_preprocessing.clean(). Guards against any residual
    zero Engine values to avoid divide-by-zero.
    """
    result = frame.copy()
    safe_engine = result["Engine"].replace(0, pd.NA)
    result["power_per_cc"] = result["Power"] / safe_engine
    # If Engine was zero (shouldn't happen post-cleaning, but defensive),
    # fall back to 0 rather than propagating NaN into the model.
    result["power_per_cc"] = result["power_per_cc"].fillna(0.0)
    return result


def add_ownership_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Technique 4: Ownership recency encoding (ordinal, not one-hot).

    Maps Owner_Type -> {First: 1, Second: 2, Third: 3, Fourth & Above: 4}.

    Reason: this is a deliberate choice to preserve the real-world ORDER of
    ownership count (more owners generally -> more wear/depreciation),
    rather than treating the categories as unordered labels the way one-hot
    encoding would. Expected relationship: negative (more prior owners ->
    lower price).

    Unrecognised/unseen Owner_Type values map to NaN rather than silently
    guessing an order - this should be surfaced, not hidden, if the API
    ever receives an unexpected value.
    """
    result = frame.copy()
    result["owner_rank"] = result["Owner_Type"].map(OWNER_TYPE_ORDER)
    return result


def add_age_mileage_interaction(frame: pd.DataFrame) -> pd.DataFrame:
    """Technique 5: Age x mileage interaction.

    age_mileage_interaction = vehicle_age * Kilometers_Driven

    Reason: EDA is expected to show that age and mileage don't just
    independently reduce price - their combination has a compounding
    depreciation effect that a linear model can't capture without an
    explicit interaction term. Expected relationship: negative (larger
    combined value -> lower price).

    Requires vehicle_age to already exist (call add_vehicle_age first).
    """
    if "vehicle_age" not in frame.columns:
        raise ValueError("add_vehicle_age must be applied before add_age_mileage_interaction")
    result = frame.copy()
    result["age_mileage_interaction"] = result["vehicle_age"] * result["Kilometers_Driven"]
    return result


def add_features(frame: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    """Apply all 5 feature engineering techniques in the correct order.

    This is the single entry point that both the training notebook and the
    FastAPI prediction service should call, so that training-time and
    inference-time feature engineering can never drift apart.
    """
    result = add_vehicle_age(frame, reference_year=reference_year)
    result = add_mileage_per_year(result)
    result = add_power_per_cc(result)
    result = add_ownership_rank(result)
    result = add_age_mileage_interaction(result)
    return result


ENGINEERED_FEATURE_COLUMNS = [
    "vehicle_age",
    "mileage_per_year",
    "power_per_cc",
    "owner_rank",
    "age_mileage_interaction",
]