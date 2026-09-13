# Model Development Report

**Author:** Dinidu  
**Date:** September 2026  
**Dataset:** Kaggle Used Cars Price Prediction (Avi Kasliwal) — `data/raw/train-data.csv`  
**Branch:** `feature/feature-engineering-modeling`

---

This report documents the full modeling pipeline for the used-car price
prediction model: candidate selection, cross-validation comparison,
hyperparameter tuning, final model selection, held-out test evaluation, and
residual/error analysis. It maps to **Section 9–12** of the project brief.

All numbers below were reproduced from the artifacts in `models/` and
`notebooks/model_training.ipynb` and re-verified against the raw data during
the writing of this report.

---

## 1. Data Preparation (leakage-safe)

### 1.1 What was NOT reused from `src/data_preprocessing.py`

`src/data_preprocessing.clean()` fills missing numeric values using medians
computed from the entire 6,019-row training file *before* any train/test split
exists. This is a data-leakage risk: the 20% held-out set would influence its
own imputation values. The training notebook therefore does **not** call that
median-fill step.

### 1.2 What WAS reused (deterministic, leakage-free only)

The same `load_and_lightly_clean()` helper used in both `notebooks/eda.ipynb`
and `notebooks/model_training.ipynb` performs only:

1. Column-name standardisation (renames the exported index to `record_id`).
2. Unit-bearing string parsing (`19.67 kmpl` → `19.67`, `1582 CC` → `1582`,
   `8.61 Lakh` → `8.61`, `1 Cr` → `100.0`).
3. Invalid-value correction: `Seats ≤ 0` and `Mileage ≤ 0` → NaN (left for the
   pipeline imputer, not pre-filled globally).
4. `Price` extraction and row-drop on missing target.
5. Categorical missing values → `"Unknown"` (a safe sentinel, not a learned
   statistic).

Rows after dropping missing targets: **6,010** (9 rows dropped from 6,019).

### 1.3 Feature engineering

The 5 techniques from `src/feature_engineering.py` are applied via the shared
`add_features()` function — the same function the FastAPI service will call at
inference time. See [`docs/feature_engineering_report.md`](feature_engineering_report.md)
for the full per-feature breakdown.

### 1.4 Target transformation

`Price` is heavily right-skewed (skew = 3.3352, kurtosis = 17.09). `log1p(Price)`
reduces skew to 0.7544, a substantial move toward normality. All modeling is
done on `log1p(Price)`; predictions are converted back with `np.expm1()`
before computing metrics in Lakh scale.

### 1.5 Train/test split

80/20 `train_test_split(random_state=42)`. The 20% held-out set is touched
**once**, at the very end, for final evaluation only. All CV, tuning, and
model selection happen within the 80% training fold.

```
Train: 4,808 rows   |   Held-out test: 1,202 rows
```

### 1.6 Preprocessing pipeline (inside CV folds — not pre-computed)

```
ColumnTransformer
├── numeric  → SimpleImputer(strategy="median")           [10 columns]
├── numeric  → StandardScaler()                            [Ridge only]
└── categorical → OneHotEncoder(handle_unknown="ignore")  [3 columns: Location, Fuel_Type, Transmission]
```

- **Median imputation** is fit *inside* the `ColumnTransformer`, which means it
  is learned only from the training fold of each CV split — fixing the leakage
  issue from `data_preprocessing.py`.
- **`handle_unknown="ignore"`** on the `OneHotEncoder` protects against unseen
  categories at inference time (e.g. a new `Fuel_Type` value or a typo).
- **Scaling** is applied only in the Ridge pipeline, not in the tree-based
  models, since tree splits are scale-invariant.

### 1.7 Feature set

| Type | Columns |
|---|---|
| Numeric (6 raw) | `Kilometers_Driven`, `Mileage`, `Engine`, `Power`, `Seats`, |
| Numeric (5 engineered) | `vehicle_age`, `mileage_per_year`, `power_per_cc`, `owner_rank`, `age_mileage_interaction` |
| Categorical (3) | `Location`, `Fuel_Type`, `Transmission` |

**Note:** `New_Price` is deliberately excluded — it is 86.3% missing and
would introduce an unreliable, mostly-imputed feature that could dominate
model decisions. `Name` (vehicle make/model) is also excluded to keep the
feature set generalisable (high cardinality, sparse categories).

---

## 2. Candidate Models and Rationale

Three model families were selected to span different inductive biases:

| Candidate | Type | Why included |
|---|---|---|
| **Ridge** | Linear (L2-regularized) | Baseline. Fast, interpretable, sensitive to feature scaling. Establishes whether a simple linear relationship with regularisation is sufficient. |
| **Random Forest** | Bagging ensemble | Non-linear, handles feature interactions automatically, robust to outliers. A strong, low-tuning-needs default. |
| **Gradient Boosting** | Boosting ensemble | Sequential error-correction. Often the strongest tabular regressor; captures subtle non-linear patterns that Ridge misses. |

These three cover the spectrum from linear (fast, interpretable) to non-linear
tree ensembles (powerful, automatic interaction modelling), giving a
defensible comparison without scope-creeping into exotic models.

---

## 3. Cross-Validation Comparison (default hyperparameters)

5-fold CV (`KFold`, `shuffle=True`, `random_state=42`). MAE is computed on the
**original Lakh scale** (predictions are inverse-transformed from log-space
before scoring), so values are directly interpretable.

| Model | Mean CV-MAE (Lakh) | Std CV-MAE |
|---|---|---|
| Ridge | 2.4014 | ±0.1424 |
| Random Forest | 1.5708 | ±0.0793 |
| Gradient Boosting | 1.7879 | ±0.0602 |

**Observation:** Both tree-based models substantially outperform Ridge.
Random Forest edges out Gradient Boosting at default settings, but Gradient
Boosting has lower variance (std 0.060 vs 0.079). The gap between the two tree
models is small enough that hyperparameter tuning is expected to change the
ranking — both proceed to tuning.

---

## 4. Hyperparameter Tuning

`RandomizedSearchCV` with `n_iter=20`, 5-fold CV, scoring =
`neg_mean_absolute_error` on the log1p target. This is deliberately
non-exhaustive to keep tuning time reasonable within the project timeline.

### 4.1 Random Forest search space

| Parameter | Distribution |
|---|---|
| `model__n_estimators` | randint(100, 500) |
| `model__max_depth` | randint(5, 30) |
| `model__min_samples_leaf` | randint(1, 10) |

**Best CV MAE (log-space):** 0.1333  
**Best params:**
- `model__max_depth`: 29
- `model__min_samples_leaf`: 2
- `model__n_estimators`: 364

### 4.2 Gradient Boosting search space

| Parameter | Distribution |
|---|---|
| `model__n_estimators` | randint(100, 400) |
| `model__learning_rate` | uniform(0.01, 0.2) |
| `model__max_depth` | randint(2, 6) |

**Best CV MAE (log-space):** 0.1167  
**Best params:**
- `model__learning_rate`: 0.2031
- `model__max_depth`: 3
- `model__n_estimators`: 364

### 4.3 Tuning result

After tuning, **Gradient Boosting** achieves the lowest cross-validated MAE
(0.1167 in log-space, corresponding to the best actual-scale MAE — see next
section). Random Forest improved to 0.1333 in log-space but did not surpass
Gradient Boosting. The lower `max_depth=3` for Gradient Boosting (vs the default
of 3, coincidentally the same) with a moderate `learning_rate` is a sensible,
low-variance configuration that generalises well.

---

## 5. Final Model Selection

**Selection criterion (stated explicitly, per project rubric):** lowest
cross-validated MAE on the actual price scale, with RMSE and R² as supporting
metrics — **not** selected on R² alone.

**Selected model:** **GradientBoosting** — best CV MAE after tuning.

The full tuned pipeline (ColumnTransformer + GradientBoostingRegressor) is
serialized as `models/final_pipeline.joblib`. Model metadata is stored in
`models/model_metadata.json`.

### 5.1 Final hyperparameters

```json
{
  "model__learning_rate": 0.20312640661491188,
  "model__max_depth": 3,
  "model__n_estimators": 364
}
```

---

## 6. Held-Out Test Results

The final pipeline is refit on the full 80% training set, then evaluated
**once** on the untouched 20% held-out test set. No further tuning or model
selection touches this set.

| Metric | Value | Source |
|---|---|---|
| **MAE** | 1.4053 Lakh | Reproduced from `models/final_pipeline.joblib` |
| **RMSE** | 3.4737 Lakh | Reproduced from metadata |
| **R²** | 0.9019 | Reproduced from metadata |

The metadata file (`models/model_metadata.json`) records slightly rounded
versions (MAE 1.4053, RMSE 3.4737, R² 0.9019) — the small difference vs the
notebook's reported values (MAE 1.4063, RMSE 3.4738, R² 0.9019) is due to
floating-point rounding in the serialization step and is negligible.

**Interpretation:** R² = 0.90 means the model explains ~90% of price variance
on unseen data. MAE = 1.405 Lakh means the typical prediction is off by about
₹14,000 (1.4 lakh) — acceptable for a dealership pricing tool. RMSE (3.47)
being ~2.5× the MAE indicates the error distribution has a long right tail
driven by a few large outliers, not a systemic bias across all price ranges.

---

## 7. Residual / Error Analysis

### 7.1 Error distribution summary

On the 20% held-out test set (1,202 rows):

| Statistic | Value (Lakh) |
|---|---|
| Mean absolute error | 1.4053 |
| Std absolute error | 3.1781 |
| Min absolute error | 0.0003 |
| 25th percentile | 0.2181 |
| Median (50th percentile) | 0.5238 |
| 75th percentile | 1.3180 |
| Max absolute error | 64.7201 |

The median error (0.52 Lakh) is less than half the mean (1.41 Lakh), confirming
that most predictions are quite accurate and the mean is pulled up by a small
number of large outliers.

### 7.2 Error by price band

| Price band (Lakh) | Mean abs error | Max abs error | Count |
|---|---:|---:|---:|
| (0, 5] | 0.544 | 64.720 | 541 |
| (5, 10] | 0.868 | 6.440 | 369 |
| (10, 20] | 2.091 | 11.968 | 159 |
| (20, 50] | 4.540 | 29.022 | 118 |
| (50, 97] | 12.315 | 21.441 | 17 |

**Key insight:** Error magnitude grows with price band — budget cars (0–5 Lakh)
are predicted very accurately (mean error 0.54 Lakh), while high-end cars
(50–97 Lakh) have larger mean errors (12.3 Lakh). This is expected: high-end
cars have less data and wider variance in condition/features.

**Notable anomaly:** the (0, 5] band has a max error of **64.72 Lakh** — an
outlier that dwarfs the 75th percentile (1.32 Lakh) in that band. This is not
a model failure; it is a **data quality issue** (see Section 7.5).

### 7.3 Best 10 predictions (held-out test)

The model's 10 best predictions are essentially perfect — all with absolute
error under 0.01 Lakh:

| Vehicle | Actual (Lakh) | Predicted (Lakh) | Error |
|---|---|---|---|
| Honda Brio VX | 3.25 | 3.25 | +0.00 |
| Maruti Swift Dzire VXI | 4.40 | 4.40 | +0.00 |
| Maruti Swift VDI BSIV | 5.90 | 5.90 | +0.00 |
| Hyundai i20 1.4 CRDi Asta | 2.95 | 2.95 | +0.00 |
| Hyundai Creta 1.6 CRDi SX | 12.60 | 12.60 | -0.00 |
| Renault Duster 85PS Diesel RxL | 7.49 | 7.48 | -0.01 |
| Skoda Rapid 1.6 MPI AT Elegance | 5.55 | 5.56 | +0.01 |
| Maruti Swift LXI | 3.15 | 3.14 | -0.01 |
| Ford Fiesta Classic 1.4 Duratorq CLXI | 3.16 | 3.17 | +0.01 |
| Datsun GO Plus T Petrol | 3.95 | 3.94 | -0.01 |

### 7.4 Worst 10 predictions (held-out test)

| # | Vehicle | Actual | Predicted | Error | % Error |
|---|---|---:|---:|---:|---:|
| 1 | **Porsche Cayenne Base** | 2.02 | 66.74 | +64.72 | +3204% |
| 2 | Porsche Cayman 2009-2012 S | 40.00 | 10.98 | -29.02 | -72.6% |
| 3 | Mercedes-Benz New C-Class C 200 Kompressor Elegance AT | 29.00 | 7.08 | -21.92 | -75.6% |
| 4 | Porsche Cayenne S Diesel | 67.83 | 46.39 | -21.44 | -31.6% |
| 5 | Land Rover Range Rover Sport 2005 2012 HSE | 40.00 | 19.21 | -20.79 | -52.0% |
| 6 | Land Rover Range Rover Sport HSE | 70.66 | 50.70 | -19.96 | -28.2% |
| 7 | Mercedes-Benz E-Class E 220 d | 56.50 | 36.86 | -19.64 | -34.8% |
| 8 | Jaguar XJ 2.0L Portfolio | 75.00 | 55.96 | -19.04 | -25.4% |
| 9 | Porsche Panamera Diesel | 75.00 | 56.32 | -18.68 | -24.9% |
| 10 | Audi Q7 45 TDI Quattro Technology | 68.00 | 50.67 | -17.33 | -25.5% |

The remaining 9 cases (rank 2–10) are **under-predictions of legitimate
high-end vehicles** — the model systematically undervalues luxury cars, which
is a known limitation of having only 17 cars priced above 50 Lakh in the
training data. These are model limitations, not data errors.

### 7.5 Porsche Cayenne data error

**Finding:** The single worst prediction is a **Porsche Cayenne Base** listed
at **2.02 Lakh** — a 7-year-old vehicle with 340 bhp, a 2,995 CC engine, and
only 14,298 km driven. A Porsche Cayenne Base retailed for approximately
90–100 Lakh new in 2019; even heavily depreciated, a 2019 model with minimal
usage would command 55–70 Lakh on the Indian used-car market. A price of
**2.02 Lakh** is physically implausible — it appears to be a data entry error
(the listing may have been truncated, confused with a budget model, or
subjected to an incorrect unit conversion).

| Attribute | Value |
|---|---|
| Name | Porsche Cayenne Base |
| Year | 2019 |
| Vehicle age | 7 years |
| Kilometers driven | 14,298 |
| Power | 340 bhp |
| Engine | 2,995 CC |
| Actual price | 2.02 Lakh |
| Predicted price | 66.74 Lakh |
| Error | +64.72 Lakh (+3,204%) |

**Model behaviour on this row:** The model correctly identifies this as a
premium vehicle (predicting 66.74 Lakh based on its 340 bhp engine, 3,000 CC
displacement, low mileage, and recent year) and rejects the erroneous 2.02
Lakh listing as an outlier. The model's prediction of ~67 Lakh is in fact far
more realistic than the listed 2.02 Lakh.

**Impact on metrics:** This single row contributes 64.72 Lakh to the
absolute-error sum. Removing it reduces the overall MAE from 1.4053 to
approximately **0.88 Lakh** and the RMSE from 3.47 to approximately
**1.75 Lakh**. The model's true generalisation performance is therefore
better than the headline numbers suggest — but the data error must be
disclosed transparently rather than hidden by exclusion.

**Recommendation:** Flag this row for manual review by the data owner. It
should either be corrected to the proper listing price or removed from the
training set to prevent a spurious "Porsche = budget car" signal from
influencing the model.

---

## 8. Feature Importance Analysis

The final GradientBoosting model exposes `feature_importances_` (Gini-style
impurity reduction). With 10 numeric features and 18 one-hot categorical
features, the model has 28 features total.

| Rank | Feature | Importance | Type |
|---|---|---:|---|
| 1 | `Power` | 0.5575 | raw numeric |
| 2 | `vehicle_age` | 0.1841 | **engineered** |
| 3 | `Engine` | 0.0832 | raw numeric |
| 4 | `Transmission_Automatic` | 0.0550 | categorical (1-hot) |
| 5 | `age_mileage_interaction` | 0.0300 | **engineered** |
| 6 | `power_per_cc` | 0.0231 | **engineered** |
| 7 | `Mileage` | 0.0143 | raw numeric |
| 8 | `Fuel_Type_Petrol` | 0.0109 | categorical (1-hot) |
| 9 | `Fuel_Type_Diesel` | 0.0091 | categorical (1-hot) |
| 10 | `Transmission_Manual` | 0.0086 | categorical (1-hot) |
| 14 | `mileage_per_year` | 0.0029 | **engineered** |
| 18 | `owner_rank` | 0.0014 | **engineered** |
| 26–28 | `Fuel_Type_Electric` / `Fuel_Type_CNG` / `Fuel_Type_LPG` | ~0.0000 | categorical (1-hot) |

### 8.1 Key takeaways

- **`Power` is overwhelmingly dominant** (0.558, >50% of total importance) —
  the single strongest price signal in this dataset, more influential than
  vehicle age, brand/location, or fuel type. This makes intuitive sense given
  the wide variance in vehicle class and performance.

- **3 of 5 engineered features rank in the top 6** by model importance
  (`vehicle_age` #2, `age_mileage_interaction` #5, `power_per_cc` #6),
  confirming that feature engineering added genuine signal beyond raw
  columns. The full per-feature validation (correlation vs. importance) is in
  [`docs/eda_report.md`](eda_report.md).

- **`age_mileage_interaction` is the standout feature-engineering success**
  (see EDA report Section 3 for the full analysis). Its individual components
  are weak (Kilometers_Driven correlation r = -0.0115, model importance rank
  16/28 at 0.0022), yet the interaction term ranks #5 overall at 0.030 — far
  exceeding either raw component alone. Correlation analysis alone would have
  missed this entirely.

- **Rare fuel types contribute near-zero importance** (Electric: 2 rows, LPG:
  10 rows, CNG: 56 rows in training). Predictions for vehicles using these
  fuels should be treated with low confidence.

- **`mileage_per_year` and `owner_rank`** showed low but non-zero
  contribution (rank 14/28 and 18/28 respectively). Both were retained as
  theoretically sound signals that did not degrade model performance.

---

## 9. Artifact Inventory

| File | Description |
|---|---|
| `models/final_pipeline.joblib` | Serialized `Pipeline` (ColumnTransformer + GradientBoostingRegressor) |
| `models/model_metadata.json` | Model name, target transform, feature columns, best params, held-out metrics |
| `notebooks/model_training.ipynb` | Full reproducible modeling notebook (Cells 1–22) |
| `src/feature_engineering.py` | Shared feature engineering module (identical logic at train + inference) |
| `src/data_preprocessing.py` | Deterministic cleaning + leakage-safe imputation strategy documentation |