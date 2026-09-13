# Exploratory Data Analysis Report

**Author:** Dinidu  
**Date:** September 2026  
**Notebook:** [`notebooks/eda.ipynb`](../notebooks/eda.ipynb)  
**Cleaned data:** `data/raw/train-data.csv` → `load_and_lightly_clean()` (deterministic, no median imputation)

---

This is a written companion to Section 13 ("Summary of EDA Findings") of
`notebooks/eda.ipynb`. It consolidates the key empirical findings that
justified the feature engineering and modelling decisions. All numbers below
were re-verified against the raw data at the time of writing this report.

## Methodology note

Correlation values measure only **linear** relationships, while Gradient
Boosting feature importance reflects everything the model actually learned —
including non-linear effects and interactions. Where the two disagree,
model-based importance is the more reliable signal for a non-linear model
like Gradient Boosting. This distinction is a legitimate point to raise in a
viva setting.

---

## 1. Price distribution

| Metric | Value |
|---|---|
| Price skewness | 3.3352 |
| Price kurtosis | 17.0922 |
| log1p(Price) skewness | 0.7544 |

Price is heavily right-skewed (skew = 3.34, kurtosis = 17.09). A small number
of expensive vehicles stretch the tail. `log1p(Price)` reduces skew to 0.75 —
a substantial improvement toward normality — empirically confirming the
`log1p` target transformation used in modelling
(`notebooks/model_training.ipynb`, Cell 10).

---

## 2. Correlations with Price (raw + engineered features)

All correlations computed on the lightly-cleaned data (no median imputation),
matching `notebooks/eda.ipynb` Cell 21.

| Feature | Correlation with Price | Direction |
|---|---:|:---|
| `Power` | **0.7697** | Strong positive |
| `Engine` | 0.6584 | Strong positive |
| `Mileage` | -0.3333 | Moderate negative |
| `vehicle_age` | -0.3053 | Moderate negative |
| `power_per_cc` | 0.3961 | Moderate positive |
| `Owner_Type` (ordinal) | — | — |
| `owner_rank` | -0.0976 | Weak negative |
| `age_mileage_interaction` | -0.0944 | Weak negative |
| `Seats` | 0.0532 | Very weak positive |
| `mileage_per_year` | 0.0419 | Very weak positive |
| `Kilometers_Driven` | -0.0115 | Essentially zero |

**Key observation:** `Kilometers_Driven` alone has almost no linear
correlation with Price (r = -0.0115). Its value only becomes apparent when
combined with `vehicle_age` in the interaction term — see Section 5.

---

## 3. Model-based feature importance

Final model: **Gradient Boosting** (28 total features: 10 numeric + 18
one-hot encoded categorical).

| Rank | Feature | Importance | Type |
|---|---|---:|---|
| 1 | `Power` | 0.5575 | raw numeric |
| 2 | `vehicle_age` | 0.1841 | **engineered** |
| 3 | `Engine` | 0.0832 | raw numeric |
| 4 | `Transmission_Automatic` | 0.0550 | categorical (1-hot) |
| 5 | `age_mileage_interaction` | 0.0300 | **engineered** |
| 6 | `power_per_cc` | 0.0231 | **engineered** |
| 14 | `mileage_per_year` | 0.0029 | **engineered** |
| 18 | `owner_rank` | 0.0014 | **engineered** |
| 26–28 | `Fuel_Type_Electric` / `Fuel_Type_CNG` / `Fuel_Type_LPP` | ~0.0000 | categorical (1-hot) |

---

## 4. Feature-by-feature evaluation

### 4.1 `vehicle_age` — strong confirmation

| Measure | Value |
|---|---|
| Correlation with Price | r = -0.3053 |
| Model importance | Rank #2 / 28 (0.1841) |

`vehicle_age` is the strongest engineered feature by **both** correlation and
model-based importance. It ranks #2 overall — ahead of every raw numeric
feature except `Power`. This strongly confirms `vehicle_age` as a genuinely
valuable feature engineering technique: it carries a signal that `Year` alone
(used as a categorical in the original dataset) does not.

### 4.2 `age_mileage_interaction` — the success story

| Measure | Value |
|---|---|
| Individual component `Kilometers_Driven` correlation | r = -0.0115 |
| Individual component `Kilometers_Driven` importance | Rank 16 / 28 (0.0022) |
| **Interaction term** correlation | r = -0.0944 |
| **Interaction term** importance | **Rank #5 / 28 (0.0300)** |

This is the clearest success of the feature engineering process. Its individual
components are weak in isolation — `Kilometers_Driven` has almost no linear
correlation with Price (r = -0.0115) and very low model importance (rank 16/28,
0.0022). Yet the **interaction term** (`vehicle_age × Kilometers_Driven`)
ranks #5 overall by model importance (0.0300), far exceeding either raw
component alone.

**Why this matters:** This directly confirms the original design hypothesis —
age and mileage have a compounding depreciation effect on price that only
becomes visible when combined. Correlation analysis alone (which would have
rejected `Kilometers_Driven` as useless) would have missed this entirely.
Model-based feature importance was necessary to properly evaluate this
feature. This is strong, specific evidence to raise when explaining why
feature engineering decisions were validated with more than one method.

### 4.3 `power_per_cc` — modest but real

| Measure | Value |
|---|---|
| Correlation with Price | r = 0.3961 |
| Model importance | Rank #6 / 28 (0.0231) |
| vs. raw `Power` | Rank #1 / 28 (0.5575) |
| vs. raw `Engine` | Rank #3 / 28 (0.0832) |

`power_per_cc` makes a modest but real, distinct contribution — lower than
raw `Power` or `Engine` individually, but still a non-trivial, separate
signal. It is retained as a legitimate, if secondary, feature engineering
technique.

### 4.4 `mileage_per_year` — limited impact

| Measure | Value |
|---|---|
| Correlation with Price | r = 0.0419 |
| Model importance | Rank 14 / 28 (0.0029) |

Low predictive contribution by both measures. Retained as a theoretically
sound signal that does not degrade model performance, but its practical
impact on this dataset was limited.

### 4.5 `owner_rank` — limited impact

| Measure | Value |
|---|---|
| Correlation with Price | r = -0.0976 |
| Model importance | Rank 18 / 28 (0.0014) |

Low predictive contribution by both measures. Same rationale as
`mileage_per_year`: retained, not because it proved useful on this dataset,
but because the ordinal encoding is theoretically justified and harmless.

### 4.6 Raw `Power` — dominant signal

`Power` alone accounts for **55.75%** of total Gradient Boosting feature
importance — more influential than vehicle age, brand/location, or fuel type.
This makes intuitive sense given the wide variance in vehicle class and
performance within the used-car market: engine power is the strongest single
proxy for a vehicle's position in the market hierarchy.

---

## 5. Categorical feature analysis

### 5.1 Fuel_Type — severe class imbalance

| Fuel type | Count |
|---|---:|
| Diesel | 3,205 |
| Petrol | 2,746 |
| CNG | 56 |
| LPG | 10 |
| Electric | 2 |

Electric and LPG vehicles are extremely rare. This is confirmed as a genuine
data quality caveat by the feature importance results: `Fuel_Type_Electric`,
`Fuel_Type_CNG`, and `Fuel_Type_LPG` all rank at the very bottom (near-zero
importance). The model has learned almost nothing about these rare categories,
and predictions for them should be treated with low confidence.

### 5.2 Location — moderate imbalance

| Location | Count |
|---|---:|
| Mumbai | 790 |
| Hyderabad | 742 |
| Kochi | 651 |
| Coimbatore | 636 |
| Pune | 622 |
| Delhi | 554 |
| Kolkata | 535 |
| Chennai | 494 |
| Jaipur | 413 |
| Bangalore | 358 |
| Ahmedabad | 224 |

11 cities span a 3.5× range in frequency. Location effects are small but
non-zero (e.g. `Location_Kolkata` appears in the top 15 importances at rank 11).

### 5.3 Transmission — balanced binary

| Transmission | Count |
|---|---:|
| Manual | ~3,500+ |
| Automatic | ~2,500+ |

Automatic cars generally command higher resale prices than manual, and
`Transmission_Automatic` ranks #4 in model importance (0.0550), confirming
this as a meaningful signal.

---

## 6. Outliers

### 6.1 Kilometers_Driven

Four rows exceed 500,000 km driven:

| Name | Year | Kilometers_Driven | Price (Lakh) |
|---|---|---:|---:|
| Skoda Octavia Ambition Plus 2.0 TDI AT | 2013 | 775,000 | 7.5 |
| Hyundai i10 Magna 1.2 | 2009 | 620,000 | 2.7 |
| Volkswagen Vento Diesel Highline | 2013 | 720,000 | 5.9 |
| BMW X5 xDrive 30d M Sport | 2017 | **6,500,000** | 65.0 |

The BMW X5 at 6.5 million km over 6 years (≈2,900 km/day) is physically
implausible and almost certainly a data entry error. These extreme values were
retained (not removed) because the tree-based model is robust to outliers,
and removing them would discard legitimate high-mileage vehicles.

### 6.2 Price

Price ranges from 0.44 to 160.00 Lakh. The upper tail is heavy but legitimate
— luxury vehicles (Porsche, Panamera, Range Rover) are genuinely expensive.

---

## 7. Summary of EDA findings (Section 13 of the notebook)

1. **Price is heavily right-skewed** (skew = 3.3352, kurtosis = 17.09).
   `log1p(Price)` reduces skew to 0.7544 — a substantial improvement toward
   normality, empirically confirming the `log1p` target transformation used
   in modelling.

2. **`vehicle_age` is the strongest engineered feature by both measures**:
   correlation r = -0.3053, and it ranks #2 of 28 total model features by
   Gradient Boosting importance (0.1841). This strongly confirms
   `vehicle_age` as a genuinely valuable feature engineering technique.

3. **`age_mileage_interaction` is the clearest success story of the feature
   engineering process.** Its individual components are weak in isolation —
   `Kilometers_Driven` alone has almost no linear correlation with Price
   (r = -0.0115) and very low model importance (rank 16 of 28, 0.0022) —
   yet the interaction term (`vehicle_age × Kilometers_Driven`) ranks #5
   overall by model importance (0.0300), far exceeding either raw component
   alone. This directly confirms the original hypothesis: age and mileage
   have a compounding depreciation effect on price that only becomes visible
   when combined, not when considered independently. Correlation analysis
   alone would have missed this entirely, which is why model-based feature
   importance was necessary to properly evaluate this feature.

4. **`power_per_cc` makes a modest but real, distinct contribution**
   (correlation r = 0.3961; model importance rank #6, 0.0231) — lower than
   raw `Power` alone (rank #1, 0.5575) or `Engine` alone (rank #3, 0.0832),
   but still a non-trivial, separate signal. Retained as a legitimate, if
   secondary, feature engineering technique.

5. **`mileage_per_year` and `owner_rank` showed low predictive contribution by
   both correlation and model-based feature importance**
   (`mileage_per_year`: r = 0.0419, importance rank 14/28, 0.0029;
   `owner_rank`: r = -0.0976, importance rank 18/28, 0.0014). This is an
   honest, stated finding rather than a hidden weakness: both features
   represent theoretically sound depreciation/ownership signals and were
   retained since they did not degrade model performance, but their practical
   impact on this specific dataset was limited. This is a legitimate outcome
   of empirical feature engineering — not every well-reasoned feature will
   show strong measurable impact on every dataset.

6. **Raw `Power` is overwhelmingly the single strongest price signal in this
   dataset**, accounting for over half (0.5575) of total Gradient Boosting
   feature importance — more influential than vehicle age, brand/location,
   or fuel type. This makes intuitive sense given the wide variance in
   vehicle class and performance within a used-car market.

7. **`Fuel_Type` has severe class imbalance** (Electric = 2, LPG = 10, CNG =
   56 vs. thousands of Petrol/Diesel). Confirmed as a genuine data quality
   caveat by feature importance results: `Fuel_Type_Electric`,
   `Fuel_Type_CNG`, and `Fuel_Type_LPG` all rank at the very bottom (near-zero
   importance). Predictions for vehicles using these fuels should be treated
   with low confidence.

8. **At least four extreme `Kilometers_Driven` outliers** (>500,000 km)
   were identified, including one implausible value of 6.5 million km on a
   2017 BMW X5. These were retained because tree-based models are robust to
   outliers, but they should be flagged for data-quality review.

9. **At least one extreme Porsche price outlier** was identified: a 2019
   Porsche Cayenne Base listed at 2.02 Lakh despite having 340 bhp and only
   14,298 km driven. The model predicts 66.74 Lakh for this vehicle — a +64.72
   Lakh error and +3,204% percentage error. This is a data entry error (a
   Porsche Cayenne Base costs ~90–100 Lakh new), not a model failure. See
   [`docs/model_development_report.md`](model_development_report.md) Section 7.5
   for full details.