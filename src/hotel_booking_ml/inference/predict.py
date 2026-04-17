from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.hotel_booking_ml.config import BEST_MODEL_PATH

def load_model(model_path=BEST_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model topilmadi: {model_path}. Avval training scriptlarini ishga tushiring."
        )
    return joblib.load(model_path)

def predict_records(records: list[dict[str, Any]], model=None, threshold: float = 0.5) -> list[dict[str, Any]]:
    if model is None:
        model = load_model()

    frame = pd.DataFrame(records)
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    outputs: list[dict[str, Any]] = []
    for row_pred, row_proba in zip(predictions, probabilities, strict=True):
        outputs.append(
            {
                "prediction": int(row_pred),
                "cancel_probability": float(row_proba),
            }
        )
    return outputs

def predict_single(record: dict[str, Any], model=None, threshold: float = 0.5) -> dict[str, Any]:
    return predict_records([record], model=model, threshold=threshold)[0]
