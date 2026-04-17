import json
from pathlib import Path
from typing import Any

from src.hotel_booking_ml.config import (
    DATA_DIR,
    EDA_REPORT_DIR,
    ENGINEERED_DATA_DIR,
    EVALUATION_REPORT_DIR,
    FINAL_DATA_DIR,
    MODELS_DIR,
    MODELS_REPORT_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)

def ensure_project_directories() -> None:
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ENGINEERED_DATA_DIR,
        FINAL_DATA_DIR,
        REPORTS_DIR,
        EDA_REPORT_DIR,
        MODELS_REPORT_DIR,
        EVALUATION_REPORT_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
