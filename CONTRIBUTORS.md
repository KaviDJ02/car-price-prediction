# Contributors

This document captures the task allocation across the three team members for
the **Used Car Price Prediction** project. The repository is hosted at
`github.com/KaviDJ02/car-price-prediction`.

| Name | GitHub handle | Email | Primary role |
|---|---|---|---|
| Dinidu Sachintha | Dinidu21 | dinidusachintha3@gmail.com | Feature engineering, modeling, API, documentation |
| Kavindu Jayasundara | KavinduJ | kjayasundara002@gmail.com | Data preprocessing, EDA notebook, testing |
| Sampath [surname] | — | — | EDA + README, frontend integration |

---

## Task allocation

### Dinidu Sachintha (@Dinidu21)

| Task | Files | Status |
|---|---|---|
| Feature engineering module (5 techniques) | `src/feature_engineering.py` | Complete |
| Feature engineering tests (12 tests) | `tests/test_feature_engineering.py` | Complete, passing |
| Model training notebook (CV, tuning, final model) | `notebooks/model_training.ipynb` | Complete |
| EDA notebook (safety net for Kavindu's task) | `notebooks/eda.ipynb` | Complete |
| Data preprocessing tests (2 tests) | `tests/test_data_preprocessing.py` | Complete, passing |
| Dataset quality audit report | `docs/dataset_report.md` | Complete |
| Feature engineering report | `docs/feature_engineering_report.md` | Complete |
| EDA report | `docs/eda_report.md` | Complete |
| Model development report | `docs/model_development_report.md` | Complete |
| FastAPI backend (app, schemas, service, config) | `api/` | Complete |
| FastAPI architecture documentation | `docs/fastapi_architecture.md` | Complete |
| FastAPI implementation plan | `docs/FastAPI-backend-plan.md` | Complete |

### Kavindu Jayasundara (@KavinduJ)

| Task | Files | Status |
|---|---|---|
| Project setup + initial data preprocessing | `src/data_preprocessing.py` | Complete |
| Feature engineering pipeline documentation | `docs/dataset_report.md` | Complete |

**Pending (for Kavindu):** API integration tests — `tests/test_api.py` stub
is planned per the FastAPI architecture document. The request/response
contract is locked and available in
[`docs/fastapi_architecture.md`](docs/fastapi_architecture.md) Section 3.

### Sampath

| Task | Files | Status |
|---|---|---|
| EDA notebook (primary) | `notebooks/eda.ipynb` | Handed off (Dinidu built as safety net) |
| EDA report | `docs/eda_report.md` | Available for Sampath to merge with EDA outputs |
| README | `README.md` | Pending (can build from `docs/eda_report.md` + `docs/dataset_report.md`) |
| Frontend / Adalo integration | `frontend/` | Pending (API contract available) |

---

## How to review contributions

```bash
git log --author="Dinidu21"    # feature engineering + modeling + docs
git log --author="KavinduJ"    # data preprocessing + pipeline docs
git log --oneline --all
```

### Branch strategy

All individual work lives on `feature/feature-engineering-modeling`. A
separate `feature/fastapi-backend` branch was considered but deferred to keep
the full individual contribution in one PR for review simplicity.

---

## Key handoffs

1. **API contract** → Sampath and Kavindu: see
   [`docs/fastapi_architecture.md`](docs/fastapi_architecture.md) Section 3
   for the exact `POST /predict` request/response JSON example.
2. **EDA results** → Sampath: `docs/eda_report.md` contains all correlation
   numbers, feature importance rankings, and EDA findings needed to build
   the README's analysis section.
3. **Feature engineering logic** → shared: `src/feature_engineering.py` is
   the single source of truth, imported by both the training notebook and
   the FastAPI service — no re-implementation needed downstream.