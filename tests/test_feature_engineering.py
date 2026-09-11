import pandas as pd
import pytest

from src.feature_engineering import (
    add_age_mileage_interaction,
    add_features,
    add_mileage_per_year,
    add_ownership_rank,
    add_power_per_cc,
    add_vehicle_age,
    ENGINEERED_FEATURE_COLUMNS,
)


def _sample_frame() -> pd.DataFrame:
    """A small, already-cleaned-style frame mirroring data_preprocessing output."""
    return pd.DataFrame(
        {
            "record_id": [0, 1, 2],
            "Year": [2010, 2020, 2026],
            "Kilometers_Driven": [72000, 20000, 0],
            "Engine": [998.0, 1582.0, 1200.0],
            "Power": [58.16, 126.2, 90.0],
            "Owner_Type": ["First", "Second", "Fourth & Above"],
        }
    )


def test_add_vehicle_age_computes_difference_from_reference_year():
    frame = _sample_frame()
    result = add_vehicle_age(frame, reference_year=2026)
    assert result.loc[0, "vehicle_age"] == 16
    assert result.loc[1, "vehicle_age"] == 6
    assert result.loc[2, "vehicle_age"] == 0


def test_add_vehicle_age_clips_negative_age_to_zero():
    frame = pd.DataFrame({"Year": [2030]})
    result = add_vehicle_age(frame, reference_year=2026)
    assert result.loc[0, "vehicle_age"] == 0


def test_add_mileage_per_year_requires_vehicle_age_first():
    frame = _sample_frame()
    with pytest.raises(ValueError, match="add_vehicle_age must be applied"):
        add_mileage_per_year(frame)


def test_add_mileage_per_year_divides_by_age_with_floor_of_one():
    frame = add_vehicle_age(_sample_frame(), reference_year=2026)
    result = add_mileage_per_year(frame)
    assert result.loc[0, "mileage_per_year"] == pytest.approx(72000 / 16)
    # Row 2 has vehicle_age == 0, so the floor of 1 must be applied
    # (72000 km / 0 years would otherwise be a divide-by-zero).
    assert result.loc[2, "mileage_per_year"] == pytest.approx(0 / 1)


def test_add_power_per_cc_computes_ratio():
    frame = _sample_frame()
    result = add_power_per_cc(frame)
    assert result.loc[0, "power_per_cc"] == pytest.approx(58.16 / 998.0)
    assert result.loc[1, "power_per_cc"] == pytest.approx(126.2 / 1582.0)


def test_add_power_per_cc_handles_zero_engine_defensively():
    frame = pd.DataFrame({"Power": [100.0], "Engine": [0.0]})
    result = add_power_per_cc(frame)
    assert result.loc[0, "power_per_cc"] == 0.0


def test_add_ownership_rank_maps_known_categories_in_order():
    frame = _sample_frame()
    result = add_ownership_rank(frame)
    assert result.loc[0, "owner_rank"] == 1  # First
    assert result.loc[1, "owner_rank"] == 2  # Second
    assert result.loc[2, "owner_rank"] == 4  # Fourth & Above


def test_add_ownership_rank_maps_unseen_category_to_nan():
    frame = pd.DataFrame({"Owner_Type": ["Fifth"]})
    result = add_ownership_rank(frame)
    assert pd.isna(result.loc[0, "owner_rank"])


def test_add_age_mileage_interaction_requires_vehicle_age_first():
    frame = _sample_frame()
    with pytest.raises(ValueError, match="add_vehicle_age must be applied"):
        add_age_mileage_interaction(frame)


def test_add_age_mileage_interaction_multiplies_age_and_km():
    frame = add_vehicle_age(_sample_frame(), reference_year=2026)
    result = add_age_mileage_interaction(frame)
    assert result.loc[0, "age_mileage_interaction"] == 16 * 72000
    assert result.loc[2, "age_mileage_interaction"] == 0 * 0


def test_add_features_applies_all_five_techniques_in_correct_order():
    frame = _sample_frame()
    result = add_features(frame, reference_year=2026)
    for column in ENGINEERED_FEATURE_COLUMNS:
        assert column in result.columns
    # Spot-check one derived value to confirm the pipeline ran end-to-end,
    # not just that the columns exist.
    assert result.loc[0, "vehicle_age"] == 16
    assert result.loc[0, "mileage_per_year"] == pytest.approx(72000 / 16)
    assert result.loc[0, "age_mileage_interaction"] == 16 * 72000


def test_add_features_does_not_mutate_the_input_frame():
    frame = _sample_frame()
    original_columns = list(frame.columns)
    add_features(frame)
    # The original frame passed in should be untouched (functions use .copy()).
    assert list(frame.columns) == original_columns