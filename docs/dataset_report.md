# Dataset and preprocessing report

## 1. Source and suitability

The selected real-world problem is dealership vehicle price optimization:
predicting a used vehicle's resale price so a dealer can make consistent
valuation and pricing decisions. The source is Kaggle's **Used Cars Price
Prediction** dataset, published by **Avi Kasliwal**:
<https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction>.
The Kaggle dataset reference is `avikasliwal/used-cars-price-prediction`;
the downloaded version is version 2, which supplies CSV files.

The labeled training file is `train-data.csv`. The separate
`test-data.csv` is an unlabeled holdout and is not used to calculate training
statistics.

## 2. Records, features, and target

| Item | Value |
|---|---:|
| Training records before cleaning | 6,019 |
| Test records (unlabeled) | 1,234 |
| Raw training columns | 14 (including exported index and target) |
| Usable predictor features | 12 |
| Target variable | `Price` |
| Target meaning | Vehicle resale price, in Indian lakh units |

The exported first column is an unnamed row index and is not a vehicle
attribute. It is retained only as `record_id` in the cleaned files for
traceability and must not be supplied to a model.

### Data dictionary

| Feature | Semantic type | Description |
|---|---|---|
| `Name` | categorical | Vehicle make, model, and variant |
| `Location` | categorical | Indian city where the vehicle is listed |
| `Year` | integer | Manufacturing year |
| `Kilometers_Driven` | integer | Distance driven |
| `Fuel_Type` | categorical | Petrol, Diesel, CNG, LPG, or Electric |
| `Transmission` | categorical | Manual or Automatic |
| `Owner_Type` | categorical | Number of previous owners |
| `Mileage` | float | Efficiency; source values include `kmpl` and `km/kg` |
| `Engine` | float | Engine displacement in CC |
| `Power` | float | Engine power in bhp |
| `Seats` | float/integer | Number of seats |
| `New_Price` | float | Original/new price; source values include Lakh and Cr |
| `Price` | float | **Target** resale price in lakh |

Although CSV parsing initially represents most fields as text, the semantic
types above are the types used by preprocessing: numeric measurements are
converted to floating-point values and categorical fields are retained as
strings for encoding.

## 3. Data quality audit

The audit was run on the 6,019-row training file before imputation.

| Column | Missing values | Missing percentage |
|---|---:|---:|
| `Mileage` | 2 | 0.03% |
| `Engine` | 36 | 0.60% |
| `Power` | 36 | 0.60% |
| `Seats` | 42 | 0.70% |
| `New_Price` | 5,195 | 86.31% |
| All other columns | 0 | 0.00% |

There are **0 exact duplicate training rows**. Additional issues found:

* The first CSV column is an exported index rather than a predictive feature.
* `Mileage`, `Engine`, `Power`, and `New_Price` contain unit-bearing strings
  such as `19.67 kmpl`, `1582 CC`, `126.2 bhp`, and `8.61 Lakh`.
* `Power` contains the literal placeholder `null` in addition to 36 blank
  values; the placeholder occurs 107 times and is also treated as missing.
* One row has zero seats and 68 rows have zero mileage; these are invalid
  measurements and are treated as missing.
* `New_Price` is too sparse to be reliable without imputation, so its missing
  values are filled using the training median. The high missingness should be
  considered when interpreting model importance.

## 4. Cleaning and preprocessing performed

`python -m src.data_preprocessing` performs the following reproducible steps:

1. Reads the Kaggle CSVs and removes the exported index from the predictor
   set, while naming it `record_id`.
2. Extracts numeric values from unit-bearing strings and converts them to
   numeric values; `New_Price` values in Crore are converted to Lakh
   (`1 Cr = 100 Lakh`), and unparseable values become missing.
3. Converts zero `Mileage` and zero `Seats` to missing.
4. Drops rows with a missing target (none are expected in the training file).
5. Imputes numeric missing values with each training column's median.
6. Fills missing categorical values with `Unknown`.
7. Writes model-ready CSV files and a JSON profile under `data/processed/`.

The imputation is intentionally implemented in the reusable source module so
the same rules can later be placed in a fitted scikit-learn pipeline and
applied consistently by the API.
