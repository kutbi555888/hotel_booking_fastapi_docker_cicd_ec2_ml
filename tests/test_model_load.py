from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hotel_booking_ml.config import BEST_MODEL_PATH
from src.hotel_booking_ml.inference.predict import load_model, predict_single
from src.hotel_booking_ml.inference.sample_payload import EXAMPLE_BOOKING


def test_best_model_file_exists() -> None:
    assert BEST_MODEL_PATH.exists()


def test_best_model_can_load_and_predict() -> None:
    model = load_model()
    assert hasattr(model, "predict_proba")

    result = predict_single(EXAMPLE_BOOKING, model=model, threshold=0.5)
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["cancel_probability"] <= 1.0
