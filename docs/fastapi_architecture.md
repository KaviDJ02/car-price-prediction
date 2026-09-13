# FastAPI Backend Architecture

**Status:** Implemented and verified  
**Date:** September 2026  
**Branch:** `feature/feature-engineering-modeling`

This document describes the actual final design of the FastAPI prediction
service, superseding the implementation plan in
[`docs/FastAPI-backend-plan.md`](FastAPI-backend-plan.md).

---

## 1. Architecture

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
        │      (column names capitalised to match training: Year, Kilometers_Driven, etc.)
        │   2. Apply src.feature_engineering.add_features()  ← SAME function used in training
        │   3. Feed into the loaded final_pipeline (ColumnTransformer + GradientBoosting)
        │   4. Inverse-transform prediction: np.expm1(...)  ← undo the log1p target transform
        ▼
   JSON response ({"predicted_price_lakh": ..., "model_name": ...})
```

**Critical design decision:** the API calls `add_features()` from
`src/feature_engineering.py` — it never re-implements feature engineering
logic separately. This eliminates the train/inference drift risk flagged in
the assignment rubric (Section 13).

---

## 2. File structure

```
api/
├── __init__.py            # package marker
├── main.py                # FastAPI app, routes: GET /health, POST /predict
├── schemas.py             # Pydantic request/response models
├── prediction_service.py  # loads pipeline once, runs the prediction flow
└── config.py              # centralized paths to model artifacts
```

---

## 3. Endpoints

### `GET /health`

Health check endpoint. Returns 200 regardless of whether `/predict` has been
called, as long as the model metadata loaded successfully.

**Response (200):**
```json
{
  "status": "ok",
  "model_name": "GradientBoosting"
}
```

### `POST /predict`

Predicts the resale price of a used vehicle.

**Request body:**
```json
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
```

**Response (200):**
```json
{
  "predicted_price_lakh": 9.7,
  "model_name": "GradientBoosting"
}
```

---

## 4. Request schema validation

Defined in `api/schemas.py` using Pydantic v2 with `Field` constraints.

| Field | Type | Constraints |
|---|---|---|
| `year` | int | `ge=1980, le=2030` |
| `kilometers_driven` | float | `ge=0` |
| `fuel_type` | str | (any non-empty string) |
| `transmission` | str | (any non-empty string) |
| `owner_type` | str | (any non-empty string) |
| `mileage` | float | `ge=0` |
| `engine` | float | `gt=0` |
| `power` | float | `ge=0` |
| `seats` | float | `ge=0` |
| `location` | str | (any non-empty string) |

**Important:** input fields use the raw vehicle attributes (post-cleaning but
pre-feature-engineering). Engineered features (`vehicle_age`,
`mileage_per_year`, `power_per_cc`, `owner_rank`,
`age_mileage_interaction`) are computed server-side and are never exposed in
the request contract.

---

## 5. Error handling

| Scenario | HTTP status | Behaviour |
|---|---|---|
| Missing required field | 422 | Pydantic validation error auto-returned |
| Wrong type (e.g. `"year": "abc"`) | 422 | Pydantic type validation error |
| Out-of-range value (e.g. `year: 3000`) | 422 | Pydantic `Field` constraint violation |
| Unseen categorical value (e.g. a typo'd `Fuel_Type`) | 200 | `OneHotEncoder(handle_unknown="ignore")` in the saved pipeline handles this gracefully — verified with a test call |
| Model fails to load at startup | App fails to start | Model is loaded at module import time; `api/prediction_service.py` opens `models/final_pipeline.joblib` and `models/model_metadata.json` at import, so a missing/corrupt model fails the app immediately rather than serving broken predictions |
| Unexpected error during prediction | 500 | Caught and returned as `HTTPException(status_code=500, detail=str(exc))` |

---

## 6. Model loading strategy

The model pipeline and metadata are loaded **once at module import time**
(lazily, inside `_load_model()`), not per-request. This avoids the
performance penalty of reloading a multi-MB joblib artifact on every API
call.

```python
# api/prediction_service.py
_model = None

def _load_model():
    global _model
    if _model is None:
        import joblib
        _model = joblib.load(MODEL_PATH)
    return _model
```

---

## 7. Inference flow (request-to-response)

1. **Request received:** FastAPI deserializes JSON → `PredictRequest` (Pydantic
   validation runs automatically).
2. **DataFrame construction:** `_request_to_dataframe()` maps the Pydantic
   fields to the capitalised column names that `add_features()` expects
   (`Year`, `Kilometers_Driven`, `Mileage`, `Engine`, `Power`, `Seats`,
   `Owner_Type`, `Location`, `Fuel_Type`, `Transmission`).
3. **Feature engineering:** `add_features(df)` applies all 5 techniques
   (vehicle_age, mileage_per_year, power_per_cc, owner_rank,
   age_mileage_interaction) — the exact same function used during training.
4. **Prediction:** `final_pipeline.predict(engineered_df)` produces a
   `log1p(Price)` prediction.
5. **Inverse transform:** `np.expm1(...)` converts the prediction back to
   the original Lakh scale.
6. **Response:** `PredictionResponse` with `predicted_price_lakh` (rounded
   to 2 decimal places) and `model_name`.

---

## 8. Running locally

```bash
uvicorn api.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for the auto-generated Swagger UI, or
`http://127.0.0.1:8000/redoc` for ReDoc.

---

## 9. Requirements

Added to `requirements.txt`:

```
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.0
```

---

## 10. Validation checklist (all passed)

- [x] Missing required field → 422 returned by Pydantic
- [x] Wrong type (`"year": "abc"`) → 422 returned by Pydantic
- [x] Out-of-range values (`year: 3000`) → 422 via `Field` constraints
- [x] Unseen categorical value (typo'd `Fuel_Type: "DieSEL"`) → 200 returned,
      `OneHotEncoder(handle_unknown="ignore")` handles gracefully
- [x] `/health` returns 200 even before any `/predict` call
- [x] Model fails to load at startup → app fails loudly (module-level import)
- [x] Sanity check: Hyundai Creta (2015, Diesel, 41,000 km) → predicted
      9.70 Lakh, consistent with model training results