# Car Price Prediction — End-to-End Project Journey

This document consolidates the complete journey recorded in the project notebooks and supporting source files. It explains the business problem, dataset, exploratory analysis, preprocessing, feature engineering, proposed algorithms, algorithm comparison, final decision, error analysis, and the path from experiment to application.

## 1. Problem Statement

Used-car prices depend on several interacting factors, including vehicle age, distance driven, engine power, transmission, location, fuel type, and ownership history. A dealership needs a consistent starting estimate when accepting, pricing, or negotiating a used vehicle.

The project therefore treats used-car pricing as a **supervised regression problem**:

- **Input:** vehicle attributes such as year, kilometres driven, mileage, engine size, power, seats, fuel type, transmission, owner type, and location.
- **Output:** predicted resale price in Indian lakh units, where 1 lakh = ₹100,000.
- **Goal:** produce a useful, repeatable estimate while documenting the decisions made during data preparation and model selection.
- **Success measures:** low Mean Absolute Error (MAE), low Root Mean Squared Error (RMSE), strong R² on unseen data, and a pipeline that can be reused for inference.

The final solution is not presented as a guaranteed transaction price. Inspection, accident history, local demand, vehicle condition, and negotiation can still change the final market price.

## 2. Solution Overview

The project journey follows this pipeline:

```text
Kaggle CSV files
      ↓
Raw-data audit and deterministic cleaning
      ↓
Exploratory Data Analysis
      ↓
Five feature-engineering transformations
      ↓
Leakage-aware train/test split
      ↓
Ridge, Random Forest, and Gradient Boosting comparison
      ↓
Randomized hyperparameter search
      ↓
Gradient Boosting selection and held-out evaluation
      ↓
Serialized pipeline, metadata, and application handoff
```

The selected experimental model is a tuned **Gradient Boosting Regressor**. It is wrapped in a scikit-learn pipeline containing numeric imputation and categorical one-hot encoding; the target is transformed externally with `log1p(Price)` and converted back with `np.expm1()` for evaluation and inference.

## 3. Dataset Overview

### 3.1 Source and files

The data is the Kaggle **Used Cars Price Prediction** dataset published by Avi Kasliwal:

- Dataset reference: `avikasliwal/used-cars-price-prediction`
- Training file: `data/raw/train-data.csv`
- Unlabeled test file: `data/raw/test-data.csv`
- Processed outputs: `data/processed/train-clean.csv`, `data/processed/test-clean.csv`, and `data/processed/data_profile.json`

| File | Records | Columns | Target |
|---|---:|---:|---|
| Raw training data | 6,019 | 14 | `Price` |
| Raw test data | 1,234 | 13 | No target |
| Processed training data | 6,019 | 14 including `record_id` | `Price` |

The first source column is an exported DataFrame index rather than a vehicle attribute. It is renamed to `record_id` for traceability and is not supplied to the model.

### 3.2 Features and target

The dataset contains 12 usable predictor fields after excluding the exported index and target:

| Feature | Meaning |
|---|---|
| `Name` | Vehicle make, model, and variant |
| `Location` | Indian city where the vehicle is listed |
| `Year` | Manufacturing year |
| `Kilometers_Driven` | Distance driven in kilometres |
| `Fuel_Type` | Petrol, Diesel, CNG, LPG, or Electric |
| `Transmission` | Manual or Automatic |
| `Owner_Type` | First, Second, Third, or Fourth & Above owner |
| `Mileage` | Fuel efficiency, with values such as `kmpl` or `km/kg` |
| `Engine` | Engine displacement in CC |
| `Power` | Engine power in bhp |
| `Seats` | Number of seats |
| `New_Price` | Original price, with Lakh or Crore units |
| `Price` | **Target resale price in lakh** |

Two fields were deliberately excluded from the modeling feature set:

- `Name` has high cardinality and many sparse vehicle variants.
- `New_Price` is approximately 86.3% missing, making it unreliable as a mostly imputed predictor.

### 3.3 Data quality audit

The raw training file had no exact duplicate rows. The main quality issues were:

| Issue | Observation | Treatment |
|---|---:|---|
| Missing `Mileage` | 2 values, 0.03% | Parsed or left for model-fold imputation |
| Missing `Engine` | 36 values, 0.60% | Parsed or left for model-fold imputation |
| Missing `Power` | 36 blanks plus 107 literal `null` placeholders | Treated as missing |
| Missing `Seats` | 42 values, 0.70% | Invalid zero values also treated as missing |
| Missing `New_Price` | 5,195 values, 86.31% | Excluded from model features |
| Zero mileage | 68 rows | Converted to missing |
| Zero seats | 1 row | Converted to missing |

The source also contains unit-bearing strings, for example `19.67 kmpl`, `1582 CC`, `126.2 bhp`, and `8.61 Lakh`. Cleaning extracts the numeric value and converts Crore values to lakh where required.

## 4. Exploratory Data Analysis

The EDA notebook (`notebooks/eda.ipynb`) uses deterministic parsing without global median imputation. This preserves the real missingness and outlier pattern for analysis.

### 4.1 Target distribution

`Price` is heavily right-skewed:

| Measure | Value |
|---|---:|
| Raw Price skewness | 3.3352 |
| Raw Price kurtosis | 17.0922 |
| `log1p(Price)` skewness | 0.7544 |

The long upper tail is caused by a small number of expensive vehicles. Training on `log1p(Price)` reduces the skew and makes large-price errors less dominant. Predictions are converted back to lakh with `np.expm1()` before reporting metrics.

### 4.2 Important relationships

The strongest observed relationships with price were:

| Feature | Correlation with Price | Interpretation |
|---|---:|---|
| `Power` | **+0.7697** | Strong positive relationship |
| `Engine` | +0.6584 | Larger engines tend to be in more expensive vehicles |
| `power_per_cc` | +0.3961 | Performance density adds useful signal |
| `Mileage` | -0.3333 | Higher efficiency is associated with lower price in this dataset |
| `vehicle_age` | -0.3053 | Older cars tend to be cheaper |
| `owner_rank` | -0.0976 | More previous owners generally means a lower price |
| `age_mileage_interaction` | -0.0944 | Combined age and usage captures depreciation |
| `Kilometers_Driven` | -0.0115 | Almost no standalone linear relationship |

The weak standalone correlation of `Kilometers_Driven` is important. Kilometres alone does not explain price well, but kilometres combined with vehicle age can reveal usage intensity and depreciation.

### 4.3 Categorical findings

- Diesel (3,205 rows) and Petrol (2,746 rows) dominate the dataset.
- CNG has 56 rows, LPG has 10 rows, and Electric has only 2 rows.
- The data covers 11 locations, with Mumbai having the most records and Ahmedabad the fewest.
- Automatic vehicles generally have higher prices than manual vehicles.
- Median price declines from First owner to Fourth & Above owner, supporting ordinal encoding of ownership.

Rare fuel categories have too few examples for the model to learn reliable category-specific behavior. Predictions for Electric, LPG, and some CNG vehicles should therefore be treated with lower confidence.

### 4.4 Outliers

Four training rows exceed 500,000 kilometres. The most extreme is a 2017 BMW X5 recorded with 6,500,000 km, approximately 2,900 km per day. This is almost certainly a data-entry error, but it was retained during the experiment because tree models are comparatively robust to extreme numeric values.

The held-out evaluation also exposed a 2019 Porsche Cayenne listed at 2.02 lakh despite having 340 bhp, a 2,995 CC engine, and only 14,298 km. The model predicted approximately 66.74 lakh. This is best interpreted as a source-data error rather than a normal model failure.

## 5. Cleaning and Leakage Decisions

The notebooks separate deterministic cleaning from statistical fitting:

1. Rename the exported index to `record_id`.
2. Parse unit-bearing values into numbers.
3. Convert `1 Cr` to `100 Lakh` where applicable.
4. Convert invalid zero mileage and zero seats to missing values.
5. Drop rows without a usable target.
6. Fill missing categorical values with the sentinel `Unknown`.
7. Leave numeric imputation to a scikit-learn pipeline fitted inside each training fold.

This last point matters. The reusable `src/data_preprocessing.py` currently computes numeric medians over the full training file before a split. If those values are used for an 80/20 experiment, the held-out data has indirectly influenced the training imputers. The experimental notebook therefore bypasses that median-fill step and fits `SimpleImputer(strategy="median")` inside cross-validation.

The production training script (`src/train_model.py`) currently fits on the full available frame and uses its own pipeline. The notebook experiment and production trainer should be aligned before the artifact is treated as a strict reproduction of the reported held-out evaluation.

## 6. Feature Engineering

The shared function `src.feature_engineering.add_features()` creates five deterministic features. The same function is intended for training and inference, preventing train/inference logic from drifting apart.

| Feature | Formula | Reason | Observed result |
|---|---|---|---|
| `vehicle_age` | `2026 - Year`, clipped at zero | Depreciation depends on age rather than calendar year | Correlation -0.3053; model importance rank 2 |
| `mileage_per_year` | `Kilometers_Driven / max(vehicle_age, 1)` | Separates usage intensity from vehicle age | Correlation 0.0419; rank 14 |
| `power_per_cc` | `Power / Engine` | Captures performance density beyond raw engine size | Correlation 0.3961; rank 6 |
| `owner_rank` | First=1, Second=2, Third=3, Fourth & Above=4 | Preserves the ordinal ownership relationship | Correlation -0.0976; rank 18 |
| `age_mileage_interaction` | `vehicle_age × Kilometers_Driven` | Represents compounding depreciation from age and use | Importance rank 5 |

Three of the five engineered features rank in the top six model features: `vehicle_age`, `age_mileage_interaction`, and `power_per_cc`. The interaction is the clearest feature-engineering success because its components are weak individually but become useful together.

## 7. Experimental Design

The notebook records the following modeling design:

- Modeling rows after target cleaning: **6,010** in the notebook run.
- Split: 80% training and 20% held-out test.
- Training fold: 4,808 rows.
- Held-out test: 1,202 rows.
- Random state: 42.
- Validation: 5-fold shuffled cross-validation inside the training fold.
- Target used for fitting: `log1p(Price)`.
- Metrics reported on the original lakh scale: MAE, RMSE, and R².
- The held-out test is used only once, after model selection and tuning.

There is a reproducibility discrepancy in the current repository: `data/processed/data_profile.json` and `src/train_model.py` report or use 6,019 rows, while the notebook reports 6,010 rows after dropping missing targets. The notebook path also contains a Colab-style input path (`/content/train-data.csv`). These differences should be resolved before rerunning the experiment for a final report.

## 8. Proposed Algorithms

Three algorithm families were proposed to cover different levels of complexity and different assumptions about the data.

| Algorithm | Family | Why it was proposed |
|---|---|---|
| Ridge Regression | Linear, L2-regularized baseline | Fast, interpretable, and useful for testing whether a simple regularized relationship is sufficient |
| Random Forest Regressor | Bagging tree ensemble | Handles nonlinear relationships and interactions, is robust to outliers, and needs relatively little tuning |
| Gradient Boosting Regressor | Boosting tree ensemble | Builds trees sequentially to correct earlier errors and often performs strongly on structured tabular data |

Ridge receives numeric scaling because linear models are sensitive to feature scale. The tree models do not require scaling because splits are based on feature ordering rather than distance.

## 9. Algorithm Comparison

### 9.1 Default-model cross-validation

The first comparison used default hyperparameters and 5-fold cross-validation. MAE was calculated after inverse-transforming predictions to the original lakh scale.

| Model | Mean CV-MAE (lakh) | Standard deviation |
|---|---:|---:|
| Ridge | **2.4014** | ±0.1424 |
| Random Forest | **1.5708** | ±0.0793 |
| Gradient Boosting | **1.7879** | ±0.0602 |

Both tree ensembles substantially outperformed the linear baseline. Random Forest was slightly better at default settings, while Gradient Boosting had lower variance. Both tree models were therefore sent to hyperparameter tuning.

### 9.2 Hyperparameter tuning

`RandomizedSearchCV` tested 20 combinations for each tree model using 5-fold cross-validation. The search optimized negative MAE on the log-transformed target.

| Model | Best CV MAE in log space | Best parameters |
|---|---:|---|
| Random Forest | 0.1333 | `n_estimators=364`, `max_depth=29`, `min_samples_leaf=2` |
| Gradient Boosting | **0.1167** | `n_estimators=364`, `learning_rate=0.2031264066`, `max_depth=3` |

After tuning, Gradient Boosting had the lowest cross-validated error.

### 9.3 Final decision

**Final selected algorithm: Gradient Boosting Regressor.**

The decision was based primarily on the lowest cross-validated MAE after tuning, with RMSE and R² used as supporting evidence. Gradient Boosting was selected because it produced the best tuned validation error while retaining the nonlinear interaction modeling needed for this dataset.

Final tuned hyperparameters:

```json
{
  "learning_rate": 0.20312640661491188,
  "max_depth": 3,
  "n_estimators": 364
}
```

## 10. Held-Out Evaluation

The notebook refits the selected pipeline on the full 80% training fold and evaluates it once on the untouched 20% test fold.

| Metric | Notebook result |
|---|---:|
| MAE | **1.4063 lakh** |
| RMSE | **3.4738 lakh** |
| R² | **0.9019** |

The written model-development report rounds these to approximately MAE 1.4053 lakh, RMSE 3.4737 lakh, and R² 0.9019. The current `models/model_metadata.json` stores the model name, target transform, feature lists, training-row count, and hyperparameters, but it does not currently store held-out metrics.

Interpretation:

- MAE of roughly 1.4 lakh means the average absolute prediction error is about ₹140,000.
- R² of 0.9019 means the model explains approximately 90% of the variance in the held-out prices.
- RMSE is much larger than MAE, showing that a small number of large errors affect the result.

## 11. Error and Feature Analysis

### 11.1 Error distribution

On the notebook held-out set:

| Statistic | Absolute error (lakh) |
|---|---:|
| Mean | 1.4053 |
| Median | 0.5238 |
| 75th percentile | 1.3180 |
| Maximum | 64.7201 |

The median error is less than half the mean error. Most predictions are therefore fairly close, while a few extreme records pull the mean and RMSE upward.

### 11.2 Error by price band

| Actual price band | Mean absolute error | Count |
|---|---:|---:|
| 0–5 lakh | 0.544 | 541 |
| 5–10 lakh | 0.868 | 369 |
| 10–20 lakh | 2.091 | 159 |
| 20–50 lakh | 4.540 | 118 |
| 50 lakh and above | 12.315 | 17 |

The model is most accurate for common budget vehicles. Errors increase for expensive cars because the training data contains very few high-end examples. The highest-price band has only 17 vehicles, so the model has limited evidence for luxury-market pricing.

### 11.3 Most important features

The final Gradient Boosting model uses 28 transformed features: 10 numeric features and 18 one-hot categorical features.

| Rank | Feature | Importance |
|---|---|---:|
| 1 | `Power` | 0.5575 |
| 2 | `vehicle_age` | 0.1841 |
| 3 | `Engine` | 0.0832 |
| 4 | `Transmission_Automatic` | 0.0550 |
| 5 | `age_mileage_interaction` | 0.0300 |
| 6 | `power_per_cc` | 0.0231 |

`Power` contributes more than half of the model's measured feature importance. This makes the model effective on the available data but also means predictions are highly sensitive to the quality of the power field.

## 12. Limitations and Lessons Learned

1. **Rare categories:** Electric, LPG, and CNG vehicles are underrepresented, so predictions for those categories are less reliable.
2. **Luxury-market sparsity:** only a small number of cars exceed 50 lakh, causing systematic underprediction for some luxury vehicles.
3. **Data quality:** the Porsche Cayenne row and the 6.5-million-km BMW record should be reviewed or corrected before retraining.
4. **Feature dependence:** the model relies heavily on `Power`; incorrect power values can strongly affect predictions.
5. **Excluded fields:** `Name` and `New_Price` were excluded for sparsity and missingness, so the model does not use make/model identity or original price directly.
6. **Weak engineered features:** `mileage_per_year` and `owner_rank` are theoretically sensible but contributed little on this dataset.
7. **Reference year:** `vehicle_age` uses the fixed year 2026. The constant and model must be updated if the project is used much later.
8. **Target transformation:** inference must apply `np.expm1()`; skipping the inverse transform would return invalid log-scale values.
9. **Reproducibility:** notebook row counts, input paths, preprocessing, and production-training behavior are not fully aligned.
10. **Interpretation:** predictions are decision support, not a substitute for physical inspection or market negotiation.

## 13. Application Handoff

The repository includes a Vite and React frontend that collects vehicle details and calls a `/api/predict` endpoint. The intended backend flow is:

1. FastAPI validates the JSON request with Pydantic.
2. The service creates a one-row DataFrame with training-compatible column names.
3. `add_features()` computes the same five engineered features used during training.
4. The serialized pipeline imputes and encodes the row.
5. Gradient Boosting predicts `log1p(Price)`.
6. `np.expm1()` converts the result back to lakh.
7. The response returns `predicted_price_lakh` and `model_name`.

The saved artifacts are:

- `models/final_pipeline.joblib` — preprocessing and trained model.
- `models/model_metadata.json` — model configuration and feature schema.
- `notebooks/eda.ipynb` — exploratory analysis.
- `notebooks/model_training.ipynb` — comparison, tuning, evaluation, and feature importance.
- `src/feature_engineering.py` — shared feature transformation logic.
- `src/train_model.py` — production-style training script.

### Current repository status

The frontend production build completes successfully. The data-preprocessing and feature-engineering tests also pass: 14 tests passed in the focused run. The full pytest collection is currently blocked because this checkout contains API documentation and `tests/test_api.py`, but no `api/` source directory. The API should therefore be treated as a documented/intended handoff rather than a currently runnable backend in this snapshot.

The preprocessing module also still performs full-file median imputation, while the notebook intentionally avoids that step for leakage-safe evaluation. Aligning these two paths is the most important next engineering cleanup before a final deployment.

## 14. Final Conclusion

The project moved from a raw, inconsistent used-car CSV to a documented regression pipeline. EDA showed that price is highly skewed and that power, engine size, vehicle age, and age-mileage interaction carry the strongest signals. Five deterministic features were added, three algorithm families were compared, and Gradient Boosting was selected after cross-validation and randomized tuning.

The notebook's held-out result was approximately **1.41 lakh MAE, 3.47 lakh RMSE, and 0.902 R²**. The result is promising for common used vehicles, but the model is less reliable for rare fuel types and high-end cars. The remaining work is to reconcile the notebook and production paths, resolve the missing API implementation, correct or flag bad source records, and retrain with a fully leakage-safe and reproducible pipeline.
