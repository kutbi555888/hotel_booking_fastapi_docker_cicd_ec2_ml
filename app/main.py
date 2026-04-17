from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.hotel_booking_ml.inference.predict import predict_single
from src.hotel_booking_ml.inference.sample_payload import EXAMPLE_BOOKING

app = FastAPI(
    title="Hotel Booking Cancellation API",
    version="2.0.0",
    description="Hotel booking cancel bo‘lish ehtimolini qaytaruvchi FastAPI servis.",
)

class PredictionRequest(BaseModel):
    booking: dict[str, Any] = Field(..., description="Bitta booking yozuvi")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hotel Booking Cancellation API ishlayapti. /docs ga kiring."}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/example")
def example() -> dict[str, Any]:
    return EXAMPLE_BOOKING

@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        result = predict_single(request.booking, threshold=request.threshold)
        return {
            "success": True,
            "threshold": request.threshold,
            "result": result,
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
