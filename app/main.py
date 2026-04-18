from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.schemas import PredictionRequest, PredictionResponse
from src.hotel_booking_ml.config import BEST_MODEL_PATH
from src.hotel_booking_ml.inference.predict import predict_single
from src.hotel_booking_ml.inference.sample_payload import EXAMPLE_BOOKING

app = FastAPI(
    title="Hotel Booking Cancellation API",
    version="3.0.0",
    description="Hotel booking cancel bo‘lish ehtimolini qaytaruvchi FastAPI servis.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Hotel Booking Cancellation API ishlayapti.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": BEST_MODEL_PATH.exists(),
        "model_path": str(BEST_MODEL_PATH),
    }


@app.get("/example")
def example() -> dict[str, Any]:
    return {
        "booking": EXAMPLE_BOOKING,
        "threshold": 0.5,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = predict_single(request.booking.model_dump(), threshold=request.threshold)
        return PredictionResponse(
            success=True,
            threshold=request.threshold,
            result=result,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
