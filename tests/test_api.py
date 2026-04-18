from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.main import app
from src.hotel_booking_ml.inference.sample_payload import EXAMPLE_BOOKING

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_ready"] is True


def test_example_endpoint() -> None:
    response = client.get("/example")
    assert response.status_code == 200
    payload = response.json()
    assert payload["threshold"] == 0.5
    assert payload["booking"]["hotel"] == EXAMPLE_BOOKING["hotel"]


def test_predict_endpoint() -> None:
    response = client.post(
        "/predict",
        json={"booking": EXAMPLE_BOOKING, "threshold": 0.5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["result"]["prediction"] in [0, 1]
    assert 0.0 <= payload["result"]["cancel_probability"] <= 1.0


def test_predict_schema_validation() -> None:
    bad_payload = {"booking": {**EXAMPLE_BOOKING, "lead_time": -5}, "threshold": 2.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
