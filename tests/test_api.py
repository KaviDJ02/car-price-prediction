from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)

PAYLOAD = {
    "year": 2015,
    "kilometers_driven": 41000,
    "fuel_type": "Diesel",
    "transmission": "Manual",
    "owner_type": "First",
    "mileage": 19.67,
    "engine": 1582,
    "power": 126.2,
    "seats": 5,
    "location": "Pune",
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_price():
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    assert response.json()["predicted_price_lakh"] > 0
    assert response.json()["model_name"] == "GradientBoosting"


def test_predict_rejects_out_of_range_year():
    response = client.post("/predict", json={**PAYLOAD, "year": 3000})
    assert response.status_code == 422


def test_predict_accepts_unseen_categories():
    response = client.post("/predict", json={**PAYLOAD, "fuel_type": "Hydrogen"})
    assert response.status_code == 200
