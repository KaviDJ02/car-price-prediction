# Dealership Vehicle Price Optimization

This project prepares a used-vehicle dataset for a regression service that
estimates resale price for dealership staff.

## Dataset

The data is the Kaggle **Used Cars Price Prediction** dataset by Avi Kasliwal:
<https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction>.
Raw CSV files are intentionally ignored by Git. Download them with:

```python
import kagglehub
path = kagglehub.dataset_download("avikasliwal/used-cars-price-prediction")
```

Copy `train-data.csv` and `test-data.csv` from the returned directory to
`data/raw/`, then run:

```bash
python -m src.data_preprocessing
```

The cleaner writes `data/processed/train-clean.csv`,
`data/processed/test-clean.csv`, and `data/processed/data_profile.json`.
Detailed dataset documentation and the observed quality audit are in
[`docs/dataset_report.md`](docs/dataset_report.md).

## Feature engineering

Apply the deterministic feature pipeline to a cleaned dataframe with
`src.feature_engineering.add_features()`:

```python
import pandas as pd

from src.feature_engineering import add_features

train = add_features(pd.read_csv("data/processed/train-clean.csv"))
test = add_features(pd.read_csv("data/processed/test-clean.csv"))
```

The pipeline adds these five features:

- `vehicle_age`
- `mileage_per_year`
- `power_per_cc`
- `owner_rank`
- `age_mileage_interaction`

Run the focused feature-engineering tests with:

```bash
python -m pytest tests/test_feature_engineering.py -q
```

## Full-stack application

The project includes a FastAPI backend and a Vite + React frontend. The API
loads the shared feature-engineering pipeline and serves `GET /health` and
`POST /predict`. The frontend provides the appraisal form at `http://localhost:5173`.

Create the ignored model artifacts before starting the API:

```bash
python -m src.train_model
python -m uvicorn api.main:app --reload
```

In a second terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` to use the appraisal workspace. The Vite
development proxy forwards `/api/predict` to the FastAPI `/predict` endpoint.
Run the backend tests with:

```bash
python -m pytest tests -q
```

## Project layout

```text
data/          ignored raw and processed CSV files
docs/          data dictionary and quality report
notebooks/     planned EDA, preprocessing, feature engineering and modelling work
src/           preprocessing, feature engineering and model training code
models/        ignored fitted model artifacts
api/           FastAPI prediction service
frontend/      Vite + React appraisal interface
tests/         automated tests
```
