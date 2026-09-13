# Feature Engineering Report

**Author:** Dinidu  
**Date:** September 2026  
**Source module:** [`src/feature_engineering.py`](../src/feature_engineering.py)  
**Tests:** [`tests/test_feature_engineering.py`](../tests/test_feature_engineering.py) — 12 tests, all passing  
**Maps to:** Section 5 of the project brief

---

This report documents the 5 feature engineering techniques implemented in
`src/feature_engineering.py`. Each technique is a **pure, deterministic,
row-wise transformation with no fitted statistics** — meaning the same
function is called identically at training time and at FastAPI inference
time, eliminating train/inference drift and data leakage.

**Input contract:** functions expect a dataframe that has already passed
through `src.data_preprocessing.clean()` — i.e. `Year`, `Kilometers_Driven`,
`Engine`, `Power`, `Seats` are numeric, and `Owner_Type` is a clean string
with no missing values.

---

## Summary Table

| # | Feature | Original columns used | Formula | Leakage risk |
|---|---|---|---|---|
| 1 | `vehicle_age` | `Year` | `REFERENCE_YEAR - Year` | None |
| 2 | `mileage_per_year` | `vehicle_age`, `Kilometers_Driven` | `Kilometers_Driven / max(vehicle_age, 1)` | None |
| 3 | `power_per_cc` | `Power`, `Engine` | `Power / Engine` | None |
| 4 | `owner_rank` | `Owner_Type` | Ordinal map | None |
| 5 | `age_mileage_interaction` | `vehicle_age`, `Kilometers_Driven` | `vehicle_age × Kilometers_Driven` | None |

---

## Technique 1: Vehicle Age

### Original columns

`Year` (integer, manufacturing year of the vehicle)

### Transformation

```python
vehicle_age = REFERENCE_YEAR - Year
result["vehicle_age"] = result["vehicle_age"].clip(lower=0)
```

`REFERENCE_YEAR` is a **module-level constant** (`2026`), not `datetime.now()`.
This ensures a model trained today produces the same `vehicle_age` for a given
`Year` at inference time next month as it did during training. Negative ages
(future-dated vehicles) are clipped to 0.

### Reason

Raw `Year` is just a calendar label; price depreciation relates to how **old**
the car is, not which calendar year it happens to be. `vehicle_age` converts
an arbitrary label into a meaningful, monotonically-decreasing quantity.

### Expected relationship with Price

**Strong negative** — older cars are cheaper. Confirmed empirically:
correlation r = -0.3053.

### Leakage risk

**None.** No statistics are learned from data. The only "parameter" is the
fixed `REFERENCE_YEAR` constant, which is hardcoded and identical at train and
inference time.

### Inference reproduction

At FastAPI inference time, the Pydantic request contains `year: int`. The
prediction service builds a 1-row DataFrame with column `Year` before calling
`add_features()`, so the transformation is automatically identical.

### Test coverage

- `test_add_vehicle_age_computes_difference_from_reference_year` — verifies
  `Year=2010` → `vehicle_age=16` when `reference_year=2026`.
- `test_add_vehicle_age_clips_negative_age_to_zero` — verifies `Year=2030`
  → `vehicle_age=0` (clipped from -4).

---

## Technique 2: Mileage Per Year (Usage Intensity)

### Original columns

`Kilometers_Driven` (integer), `vehicle_age` (from Technique 1)

### Transformation

```python
safe_age = vehicle_age.clip(lower=1)  # avoid divide-by-zero
mileage_per_year = Kilometers_Driven / safe_age
```

Requires `vehicle_age` to already exist (i.e. `add_vehicle_age` must be
called first). A divide-by-zero guard (`clip(lower=1)`) handles age-0 vehicles.

### Reason

Raw `Kilometers_Driven` conflates age and usage intensity. A 10-year-old car
with 50,000 km is in better shape than a 2-year-old car with 50,000 km. This
ratio isolates **how heavily the car was actually driven**, independent of
its age — a signal useful for both linear and tree-based models.

### Expected relationship with Price

**Negative** — higher usage intensity → lower price, holding age constant.
Empirically: correlation r = 0.0419 (weak linear, but the feature is retained
as a theoretically sound signal that does not degrade tree model performance).

### Leakage risk

**None.** Pure arithmetic on already-known values. No statistics are fitted.

### Inference reproduction

Same as Technique 1 — `Kilometers_Driven` comes from the request,
`vehicle_age` is computed first within `add_features()`, then this ratio
follows deterministically.

### Test coverage

- `test_add_mileage_per_year_requires_vehicle_age_first` — asserts `ValueError`
  if `vehicle_age` column is missing.
- `test_add_mileage_per_year_divides_by_age_with_floor_of_one` — verifies
  correct division and the age-0 floor behaviour.

---

## Technique 3: Power-to-Engine Ratio (Performance Density)

### Original columns

`Power` (float, brake horsepower), `Engine` (float, CC displacement)

### Transformation

```python
safe_engine = Engine.replace(0, pd.NA)
power_per_cc = Power / safe_engine
power_per_cc = power_per_cc.fillna(0.0)  # defensive fallback for any residual zero
```

A defensive guard replaces any residual zero `Engine` values with NaN (avoiding
divide-by-zero), then fills the result with 0.0. In practice, `Engine` is
cleaned to numeric, non-zero values by `src/data_preprocessing.clean()`, so
this guard is belt-and-suspenders.

### Reason

`Power` and `Engine` are correlated but represent different things. A
turbocharged 1.5 L engine can produce more power than a large naturally
aspirated 3.0 L engine. This ratio approximates **performance tuning /
efficiency** — a real factor in how buyers value a car beyond raw displacement.

### Expected relationship with Price

**Positive** — higher performance density → higher price.
Empirically: correlation r = 0.3961; model importance rank #6 (0.0231). While
weaker than raw `Power` (rank #1, importance 0.5575) or `Engine` (rank #3,
importance 0.0832), it provides a distinct, non-redundant signal.

### Leakage risk

**None.** Pure ratio of two input values. No statistics learned.

### Inference reproduction

`Power` and `Engine` both come from the Pydantic request. The transformation
is applied identically in the `add_features()` call within the FastAPI
prediction service.

### Test coverage

- `test_add_power_per_cc_computes_ratio` — verifies `Power=58.16, Engine=998`
  → `power_per_cc = 0.0583`.
- `test_add_power_per_cc_handles_zero_engine_defensively` — verifies that
  `Engine=0` produces `power_per_cc=0.0` (not NaN/inf).

---

## Technique 4: Ownership Rank (Ordinal Encoding)

### Original columns

`Owner_Type` (categorical string: "First", "Second", "Third", "Fourth & Above")

### Transformation

```python
OWNER_TYPE_ORDER = {"First": 1, "Second": 2, "Third": 3, "Fourth & Above": 4}
owner_rank = Owner_Type.map(OWNER_TYPE_ORDER)
```

Unrecognised or unseen `Owner_Type` values map to **NaN** (not silently
guessed). This surfaces unexpected values at inference time rather than
hiding them — the pipeline's median imputer will then handle the NaN.

### Reason

Ownership count is **genuinely ordinal** — each additional prior owner
generally correlates with more wear and depreciation. This is a deliberate
choice to preserve that ordering, rather than one-hot encoding it as an
unordered category (which would force the model to learn the relationship
from scratch and could not express the monotonic trend).

### Expected relationship with Price

**Negative** — more previous owners → lower price.
Empirically: correlation r = -0.0976 (weak linear, but confirms the
directional hypothesis).

### Leakage risk

**None.** A static lookup dictionary. No statistics learned from data. The
same map is used at train and inference time.

### Inference reproduction

`Owner_Type` comes from the Pydantic request as a string. The `.map()`
transformation is applied identically via `add_features()`. If a frontend
sends an unexpected value (e.g. "Fifth"), it produces NaN, which the
pipeline's `SimpleImputer` handles — a graceful, visible failure mode, not a
silent misclassification.

### Test coverage

- `test_add_ownership_rank_maps_known_categories_in_order` — verifies
  First→1, Second→2, Fourth & Above→4.
- `test_add_ownership_rank_maps_unseen_category_to_nan` — verifies that
  an unseen value ("Fifth") maps to NaN.

---

## Technique 5: Age × Mileage Interaction

### Original columns

`vehicle_age` (from Technique 1), `Kilometers_Driven` (integer)

### Transformation

```python
age_mileage_interaction = vehicle_age * Kilometers_Driven
```

Requires `vehicle_age` to already exist (i.e. `add_vehicle_age` must be
called first).

### Reason

EDA is expected to show that age and mileage don't just independently reduce
price — their combination has a **compounding depreciation effect** that a
linear model cannot capture without an explicit interaction term. This
feature makes that compounding effect directly visible to the model.

### Expected relationship with Price

**Negative** — larger combined value → lower price.

### Leakage risk

**None.** Pure product of two already-known values. No statistics fitted.

### Inference reproduction

Both `vehicle_age` and `Kilometers_Driven` are derived from request fields
(`Year` and `kilometers_driven`) within the `add_features()` call, so the
interaction is computed identically at inference time.

### Test coverage

- `test_add_age_mileage_interaction_requires_vehicle_age_first` — asserts
  `ValueError` if `vehicle_age` column is missing.
- `test_add_age_mileage_interaction_multiplies_age_and_km` — verifies
  `16 × 72000 = 1,152,000`.

---

## Ordering and Composition

All 5 techniques are composed into a single entry point:

```python
def add_features(frame: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    result = add_vehicle_age(frame, reference_year=reference_year)
    result = add_mileage_per_year(result)
    result = add_power_per_cc(result)
    result = add_ownership_rank(result)
    result = add_age_mileage_interaction(result)
    return result
```

The ordering matters: Techniques 2 and 5 depend on `vehicle_age` from
Technique 1. The `add_features()` function enforces this ordering with
explicit `ValueError` guards.

**This is the single function that both the training notebook and the FastAPI
prediction service call**, so training-time and inference-time feature
engineering can never drift apart — the most critical design decision for
preventing data leakage and train/inference drift.

## Engineered Feature List

```python
ENGINEERED_FEATURE_COLUMNS = [
    "vehicle_age",
    "mileage_per_year",
    "power_per_cc",
    "owner_rank",
    "age_mileage_interaction",
]
```