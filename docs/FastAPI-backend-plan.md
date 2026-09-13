# FastAPI Backend — Implementation Plan

---

## 1. Architecture (confirmed, matches project brief)

```
Client (frontend / tests)
        │
        ▼
   FastAPI app (api/main.py)
        │
        ▼
   Pydantic request model (validates raw vehicle input)
        │
        ▼
   Prediction service (api/prediction_service.py)
        │   1. Build a 1-row DataFrame from the request
        │   2. Apply src.feature_engineering.add_features()   <- SAME function used in training
        │   3. Feed into the loaded final_pipeline (preprocessing + model)
        │   4. Inverse-transform prediction: np.expm1(...)     <- undo the log1p target transform
        ▼
   JSON response ({"predicted_price": ...})
```

**Critical rule (already established earlier in this project):** the API
must call `add_features()` from `src/feature_engineering.py` — never
re-implement vehicle_age/mileage_per_year/etc. logic separately inside the
API. This is the single biggest leakage/drift risk mentioned in the
assignment rubric (Section 13) and it's already solved by having feature
engineering live in a shared, importable module.

---

## 2. New folder/file structure to create

```
api/
├── __init__.py
├── main.py                 # FastAPI app, routes: /predict, /health
├── schemas.py               # Pydantic request/response models
├── prediction_service.py    # loads pipeline once, runs the prediction flow
└── config.py                 # paths to model artifacts (no hardcoded paths in main.py)

tests/
└── test_api.py               # (Kavindu's task, but stub can be added here)
```

Add to `requirements.txt`:

```
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.0
```

---

## 3. Step-by-step build order

### Step 1 — `api/config.py`

Centralize paths so nothing is hardcoded inline in the app:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
```

### Step 2 — `api/schemas.py`

Define the **exact** request contract. This must match the raw columns a
user/frontend would realistically know about a car — NOT the engineered
features (those are computed server-side).

Required input fields (pre-feature-engineering, matches
`FEATURE_COLUMNS_NUMERIC`/`CATEGORICAL` minus engineered ones):

- `year: int`
- `kilometers_driven: float`
- `fuel_type: str` (Petrol / Diesel / CNG / LPG / Electric)
- `transmission: str` (Manual / Automatic)
- `owner_type: str` (First / Second / Third / Fourth & Above)
- `mileage: float` (already-parsed numeric kmpl/km-per-kg value)
- `engine: float` (CC)
- `power: float` (bhp)
- `seats: float`
- `location: str`

Use Pydantic `Field` constraints (e.g. `year: int = Field(ge=1980, le=2030)`,
`kilometers_driven: float = Field(ge=0)`) so invalid input is rejected
with a clean 422 before it ever reaches the model.

Response schema:

```python
class PredictionResponse(BaseModel):
    predicted_price_lakh: float
    model_name: str
```

### Step 3 — `api/prediction_service.py`

- Load `final_pipeline.joblib` and `model_metadata.json` **once at module
  import time** (not per-request — reloading a model on every call is slow
  and wasteful).
- Function `predict_price(request: PredictRequest) -> float`:
  1. Convert the Pydantic request into a single-row `pandas.DataFrame`
     with column names matching what `add_features()` expects
     (`Year`, `Kilometers_Driven`, `Engine`, `Power`, `Owner_Type`, etc. —
     note the case/naming must match exactly what
     `src/feature_engineering.py` expects).
  2. Call `add_features(df)`.
  3. Call `final_pipeline.predict(engineered_df)` — this returns a
     **log1p-scale** prediction (per your training decision).
  4. Apply `np.expm1(...)` to convert back to actual Lakh scale.
  5. Return the float.

### Step 4 — `api/main.py`

```python
from fastapi import FastAPI, HTTPException
from api.schemas import PredictRequest, PredictionResponse
from api.prediction_service import predict_price, MODEL_METADATA

app = FastAPI(title="Used Car Price Prediction API")

@app.get("/health")
def health():
    return {"status": "ok", "model_name": MODEL_METADATA["model_name"]}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    try:
        price = predict_price(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PredictionResponse(
        predicted_price_lakh=round(price, 2),
        model_name=MODEL_METADATA["model_name"],
    )
```

### Step 5 — Run locally and verify

```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` — FastAPI's auto-generated Swagger UI.
Manually test `/predict` with a real row from your dataset (e.g. the
Hyundai Creta example from earlier) and confirm the prediction is close to
what the notebook produced for the same row — this is your sanity check
that nothing drifted between training and inference.

### Step 6 — Lock the contract and hand off

Once Step 5 passes, write down (in this plan or a shared doc) the exact
request/response JSON example, e.g.:

```json
POST /predict
{
  "year": 2015,
  "kilometers_driven": 41000,
  "fuel_type": "Diesel",
  "transmission": "Manual",
  "owner_type": "First",
  "mileage": 19.67,
  "engine": 1582,
  "power": 126.2,
  "seats": 5.0,
  "location": "Pune"
}

Response:
{
  "predicted_price_lakh": 12.3,
  "model_name": "GradientBoosting"
}
```

Send this to Sampath and Kavindu immediately — this is the blocker they've
been waiting on.

---

## 4. Validation & error handling checklist

- [ ] Missing required field → Pydantic returns 422 automatically
- [ ] Wrong type (e.g. `"year": "abc"`) → 422 automatically
- [ ] Out-of-range values (`year: 3000`) → 422 via `Field` constraints
- [ ] Unseen categorical value (e.g. a typo'd Fuel_Type) → should NOT crash;
      `OneHotEncoder(handle_unknown="ignore")` in the saved pipeline already
      handles this gracefully — verify this with a test call
- [ ] `/health` returns 200 even before any `/predict` call has been made
- [ ] Model fails to load at startup → app should fail loudly/immediately,
      not silently serve broken predictions

---

## 5. Explicit non-goals for this task (don't scope-creep)

- No authentication/API keys — out of scope for this assignment
- No database — stateless prediction only
- No retraining endpoint — model is a static artifact loaded at startup
- No batch prediction endpoint unless time permits after the core flow works

---

## 6. Commit plan (small, meaningful commits — same pattern as before)

1. `Add FastAPI app skeleton with config and schemas`
2. `Implement prediction service using shared feature engineering module`
3. `Add /predict and /health endpoints`
4. `Add requirements for FastAPI/uvicorn/pydantic`
5. `Document API request/response contract for frontend integration`

Push to a new branch: `feature/fastapi-backend` (or continue on
`feature/feature-engineering-modeling` if you'd rather keep one PR for your
whole individual contribution — either is defensible, but a separate
branch makes the PR review cleaner).
