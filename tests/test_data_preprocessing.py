import pandas as pd

from src.data_preprocessing import clean, profile


def test_clean_converts_units_and_imputes_invalid_values():
    raw = pd.DataFrame(
        {
            "record_id": [0, 1],
            "Name": ["A", "B"],
            "Location": ["Mumbai", "Pune"],
            "Year": ["2010", "2011"],
            "Kilometers_Driven": ["72000", "41000"],
            "Fuel_Type": ["CNG", "Diesel"],
            "Transmission": ["Manual", "Manual"],
            "Owner_Type": ["First", "First"],
            "Mileage": ["26.6 km/kg", "0 kmpl"],
            "Engine": ["998 CC", ""],
            "Power": ["58.16 bhp", "null bhp"],
            "Seats": ["5.0", "0.0"],
            "New_Price": ["", "1 Cr"],
            "Price": ["1.75", "12.5"],
        }
    )
    cleaned = clean(raw, has_target=True)
    assert cleaned[["Mileage", "Engine", "Power", "Seats", "New_Price"]].isna().sum().sum() == 0
    assert cleaned.loc[0, "Mileage"] == 26.6
    assert cleaned.loc[1, "Seats"] == 5.0
    assert cleaned.loc[1, "New_Price"] == 100.0


def test_profile_counts_duplicates_and_missing_values():
    raw = pd.DataFrame({"record_id": [0, 1], "Price": [1.0, 2.0]})
    result = profile(raw, raw, has_target=True)
    assert result["raw_records"] == 2
    assert result["duplicate_rows"] == 0
